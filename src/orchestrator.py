"""
src/orchestrator.py
===================
Multi-bot orchestrator with route-bot for single-GPU deployments.

The orchestrator:
  - Connects all child bot Matrix clients (lightweight sync — no LLM).
  - Loads a small route-bot LLM for message classification.
  - Maintains a shared message buffer and a priority queue ordered by
    ``server_timestamp`` so the oldest unanswered message is always handled
    first.
  - When a bot should respond, swaps the route-bot LLM out and loads the
    child bot's LLM, generates a response, then swaps back.
  - Only one LLM is in memory at a time.

Usage
-----
    from src.orchestrator import Orchestrator
    from src.fleet_config import FleetConfig

    fleet = FleetConfig("bots/fleet_config.yaml")
    orch = Orchestrator(fleet)
    await orch.start()          # blocks
"""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import sys
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

from nio import MatrixRoom, RoomMessageText  # type: ignore[import-untyped]

from src.child_bot_context import ChildBotContext
from src.config_manager import ConfigManager
from src.fleet_config import FleetConfig
from src.llm import LLMBackend
from src.scheduler import Scheduler

logger = logging.getLogger(__name__)

_HISTORY_SIZE = 50  # per-room buffer depth


@dataclass(order=True)
class PendingMessage:
    """An incoming message awaiting evaluation, ordered by server timestamp."""
    timestamp: int                              # event.server_timestamp (ms)
    event_id: str = field(compare=False)
    room_id: str = field(compare=False)
    room_alias: Optional[str] = field(compare=False, default=None)
    sender_id: str = field(compare=False, default="")
    sender_name: str = field(compare=False, default="")
    text: str = field(compare=False, default="")
    source_bot: str = field(compare=False, default="")  # which child client received it


class Orchestrator:
    """Manages multiple bots with a single shared LLM slot.

    Parameters
    ----------
    fleet_config:
        Parsed :class:`FleetConfig` pointing to the route-bot and child bot
        config files.
    """

    def __init__(self, fleet_config: FleetConfig) -> None:
        self._fleet = fleet_config

        # Load route-bot config (LLM only — no matrix validation)
        route_cfg = ConfigManager(
            fleet_config.route_bot_config_path, route_bot_only=True
        )
        self._route_llm = LLMBackend(route_cfg.llm)
        self._route_cfg = route_cfg

        # Build child bot contexts
        self._bots: Dict[str, ChildBotContext] = {}
        for entry in fleet_config.child_bots:
            cfg = ConfigManager(entry.config_path)
            cfg.ensure_directories()
            ctx = ChildBotContext(
                name=entry.name,
                config=cfg,
                message_callback=self._on_message,
            )
            self._bots[entry.name] = ctx

        # Shared state
        self._room_history: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=_HISTORY_SIZE)
        )
        # OrderedDict preserves insertion order for correct oldest-first eviction
        self._seen_events: OrderedDict[str, None] = OrderedDict()
        self._seen_events_max = 5000

        self._pending: asyncio.PriorityQueue[PendingMessage] = asyncio.PriorityQueue()
        self._llm_lock = asyncio.Lock()

        # Currently loaded child LLM (None when route-bot is loaded)
        self._active_child_llm: Optional[LLMBackend] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect all Matrix clients, load route-bot LLM, and run."""
        logger.info("Loading route-bot LLM …")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._route_llm.load)
        logger.info("Route-bot LLM ready.")

        # Connect all child Matrix clients
        for name, ctx in self._bots.items():
            logger.info("Connecting Matrix client for bot %r …", name)
            await ctx.matrix_client.login()
            await ctx.matrix_client.set_display_name(ctx.config.bot.display_name)
            rooms = await ctx.matrix_client.get_joined_rooms()
            logger.info("Bot %r is in %d room(s): %s", name, len(rooms), rooms)

        # Start schedulers (temperature + memory compaction per bot)
        scheduler_tasks = []
        for name, ctx in self._bots.items():
            sched = Scheduler(ctx.temp_ctrl, ctx.memory, ctx.config)
            scheduler_tasks.append(asyncio.create_task(sched.run()))

        # Start the evaluation loop
        eval_task = asyncio.create_task(self._evaluation_loop())

        # Start thoughts runners
        thoughts_tasks = []
        for name, ctx in self._bots.items():
            interval = getattr(ctx.config.logging, "thoughts_interval_seconds", 3600)
            thoughts_tasks.append(
                asyncio.create_task(
                    self._run_thoughts_periodically(name, ctx, interval)
                )
            )

        # Start Matrix sync loops for all child bots (these block individually
        # but run concurrently as tasks)
        sync_tasks = []
        for name, ctx in self._bots.items():
            sync_tasks.append(
                asyncio.create_task(self._run_sync(name, ctx))
            )

        try:
            # Wait for any sync task to finish (shouldn't happen normally)
            await asyncio.gather(*sync_tasks)
        finally:
            eval_task.cancel()
            for t in scheduler_tasks + thoughts_tasks:
                t.cancel()
            all_tasks = [eval_task] + scheduler_tasks + thoughts_tasks
            await asyncio.gather(*all_tasks, return_exceptions=True)
            # Close all Matrix clients
            for ctx in self._bots.values():
                try:
                    await ctx.matrix_client.close()
                except Exception:
                    pass

    async def _run_sync(self, name: str, ctx: ChildBotContext) -> None:
        """Run the Matrix sync loop for a single child bot."""
        try:
            await ctx.matrix_client.sync_forever()
        except Exception:
            logger.exception("Sync loop for bot %r crashed", name)
            raise

    async def _run_thoughts_periodically(
        self, name: str, ctx: ChildBotContext, interval: int
    ) -> None:
        """Periodically run thoughts.py for a child bot."""
        data_dir = str(Path(ctx.config.memory.dossier_dir).parent)
        while True:
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "thoughts.py",
                    "--data-dir", data_dir,
                    "--config", str(ctx.config._path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if stdout:
                    logger.info("[%s thoughts] %s", name, stdout.decode().strip())
                if stderr:
                    logger.error("[%s thoughts] %s", name, stderr.decode().strip())
            except Exception:
                logger.exception("Error running thoughts for bot %r", name)
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Message ingestion (non-blocking)
    # ------------------------------------------------------------------

    async def _on_message(
        self, bot_name: str, room: MatrixRoom, event: RoomMessageText
    ) -> None:
        """Unified callback for all child Matrix clients.

        Deduplicates by event_id, buffers in room history, and enqueues for
        evaluation.  Never blocks on the LLM lock.
        """
        event_id: str = event.event_id
        if event_id in self._seen_events:
            return
        self._seen_events[event_id] = None
        # Evict oldest entries when the cap is exceeded
        while len(self._seen_events) > self._seen_events_max:
            self._seen_events.popitem(last=False)

        sender_id: str = event.sender
        sender_name: str = room.user_name(sender_id) or sender_id
        text: str = getattr(event, "body", "") or ""
        room_id: str = room.room_id

        if not text.strip():
            return

        # Skip bot's own messages (check all child bot usernames)
        all_bot_usernames = {
            ctx.config.matrix.username for ctx in self._bots.values()
        }
        if sender_id in all_bot_usernames:
            return

        # Skip special commands (handled later in single-bot mode only)
        text_lower = text.lower().strip()
        if re.match(r"^\S+\s+--", text_lower):
            return

        # Buffer in room history
        self._room_history[room_id].append((sender_name, text))

        # Tick message counters for all bots eligible in this room
        room_alias = getattr(room, "canonical_alias", None)
        for ctx in self._bots.values():
            if ctx.is_room_eligible(room_id, room_alias):
                ctx.tick_message(room_id)

        # Enqueue for evaluation (sorted by server_timestamp)
        msg = PendingMessage(
            timestamp=event.server_timestamp,
            event_id=event_id,
            room_id=room_id,
            room_alias=room_alias,
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            source_bot=bot_name,
        )
        await self._pending.put(msg)

    # ------------------------------------------------------------------
    # Evaluation loop (processes queue oldest-first)
    # ------------------------------------------------------------------

    async def _evaluation_loop(self) -> None:
        """Continuously drain the pending queue, evaluate each message, and
        dispatch responses.  Always processes oldest messages first."""
        while True:
            msg = await self._pending.get()
            try:
                await self._evaluate_and_dispatch(msg)
            except Exception:
                logger.exception(
                    "Error evaluating message %s in room %s",
                    msg.event_id, msg.room_id,
                )

    async def _evaluate_and_dispatch(self, msg: PendingMessage) -> None:
        """Evaluate one message: rule-based checks, then route-bot LLM if needed."""
        room_id = msg.room_id
        room_alias = msg.room_alias
        text = msg.text
        sender_id = msg.sender_id
        sender_name = msg.sender_name

        # Find eligible bots for this room
        eligible: List[ChildBotContext] = [
            ctx for ctx in self._bots.values()
            if ctx.is_room_eligible(room_id, room_alias)
        ]
        if not eligible:
            return

        # --- Phase 1: Rule-based routing ---
        chosen: Optional[ChildBotContext] = None

        # Check trigger names first
        for ctx in eligible:
            if ctx.is_triggered(text):
                chosen = ctx
                break

        # If no trigger, check chime-in probability
        if chosen is None:
            for ctx in eligible:
                if ctx.should_chime_in(room_id):
                    chosen = ctx
                    break

        # --- Phase 2: LLM-based routing (if no rule triggered) ---
        if chosen is None:
            chosen = await self._route_bot_classify(
                room_id, text, sender_name, eligible
            )

        if chosen is None:
            return  # no bot should respond

        # --- Phase 3: Dispatch response ---
        await self._dispatch_response(chosen, msg)

    # ------------------------------------------------------------------
    # Route-bot LLM classification
    # ------------------------------------------------------------------

    async def _route_bot_classify(
        self,
        room_id: str,
        text: str,
        sender_name: str,
        eligible: List[ChildBotContext],
    ) -> Optional[ChildBotContext]:
        """Use the route-bot LLM to decide which bot (if any) should respond."""
        loop = asyncio.get_event_loop()

        # Build recent transcript
        history = list(self._room_history.get(room_id, []))[-6:]
        transcript = "\n".join(f"{name}: {msg}" for name, msg in history)

        # Build bot descriptions
        bot_descriptions = []
        for ctx in eligible:
            passive = room_id in ctx.passive_channels
            desc = (
                f"- {ctx.name} ({ctx.config.bot.display_name}): "
                f"triggers={ctx.config.bot.trigger_names}, "
                f"passive={'yes' if passive else 'no'}"
            )
            # Include a brief persona summary (first 100 chars)
            persona_snippet = ctx.persona[:100].replace("\n", " ")
            desc += f", personality: {persona_snippet}…"
            bot_descriptions.append(desc)

        bot_list_str = "\n".join(bot_descriptions)
        bot_names = [ctx.name for ctx in eligible]

        prompt = (
            "You are a message router for a group chat with multiple bots. "
            "Given the recent transcript, decide which bot (if any) should respond "
            "to the last message.\n\n"
            f"Available bots:\n{bot_list_str}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Reply with ONLY the bot name ({', '.join(bot_names)}) or \"none\" "
            "if no bot should respond. Do not explain."
        )

        try:
            async with self._llm_lock:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._route_llm.generate(
                        system_prompt="You are a message routing assistant. Reply with only a bot name or 'none'.",
                        user_message=prompt,
                        temperature=0.1,
                    ),
                )
        except Exception:
            logger.exception("Route-bot LLM classification failed")
            return None

        answer = result.strip().lower()
        # Match against bot names
        for ctx in eligible:
            if ctx.name.lower() in answer:
                logger.info(
                    "Route-bot selected %r for message in %s", ctx.name, room_id
                )
                return ctx

        logger.debug("Route-bot returned %r — no bot selected", answer)
        return None

    # ------------------------------------------------------------------
    # LLM swap and response dispatch
    # ------------------------------------------------------------------

    async def _dispatch_response(
        self, ctx: ChildBotContext, msg: PendingMessage
    ) -> None:
        """Swap LLMs and generate a response from the chosen child bot."""
        loop = asyncio.get_event_loop()

        async with self._llm_lock:
            # --- Unload route-bot LLM ---
            logger.info(
                "Swapping LLM: unloading route-bot, loading %r …", ctx.name
            )
            await loop.run_in_executor(None, self._route_llm.unload)

            # --- Load child bot LLM ---
            child_llm = LLMBackend(ctx.config.llm)
            try:
                await loop.run_in_executor(None, child_llm.load)
                logger.info("Child bot %r LLM loaded.", ctx.name)

                # --- Build prompt and generate ---
                reply = await loop.run_in_executor(
                    None,
                    self._build_and_generate,
                    child_llm,
                    ctx,
                    msg,
                )

                if reply:
                    await ctx.matrix_client.send_message(msg.room_id, reply)
                    ctx.record_post(msg.room_id)

                    # Buffer bot's own message in history
                    self._room_history[msg.room_id].append(
                        (ctx.config.bot.display_name, reply)
                    )

                    # Record interaction in dossier
                    try:
                        ctx.memory.record_interaction(
                            user_id=msg.sender_id,
                            display_name=msg.sender_name,
                            message=msg.text,
                            bot_response=reply,
                            room_id=msg.room_id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to record interaction for %s (bot %s)",
                            msg.sender_id, ctx.name,
                        )
                else:
                    logger.warning(
                        "Bot %r returned empty response — skipping.", ctx.name
                    )

            finally:
                # --- Always unload child and reload route-bot ---
                await loop.run_in_executor(None, child_llm.unload)
                del child_llm
                gc.collect()
                logger.info("Reloading route-bot LLM …")
                await loop.run_in_executor(None, self._route_llm.load)
                logger.info("Route-bot LLM back online.")

    def _build_and_generate(
        self,
        llm: LLMBackend,
        ctx: ChildBotContext,
        msg: PendingMessage,
    ) -> str:
        """Build the child bot's prompt and generate a reply.  Runs in a thread pool."""
        # Dossier context
        dossier_ctx = ctx.memory.get_dossier_context(
            msg.sender_id,
            max_chars=ctx.config.bot.dossier_token_budget * 4,
        )
        system_parts = [ctx.persona]
        if dossier_ctx:
            system_parts.append(
                f"\n--- What you remember about {msg.sender_name} ---\n{dossier_ctx}"
            )
        system_prompt = "\n".join(system_parts)

        # Conversation history (use latest from shared buffer — includes
        # messages that arrived during LLM swap)
        history = list(self._room_history.get(msg.room_id, []))
        limit = ctx.config.bot.conversation_history_limit
        recent = history[-limit:]

        messages: List[dict] = []
        for spkr, m in recent[:-1]:
            role = "assistant" if spkr == ctx.config.bot.display_name else "user"
            messages.append({"role": role, "content": m})
        # Current message is the final user turn
        messages.append({"role": "user", "content": msg.text})

        raw_reply = llm.generate(
            system_prompt=system_prompt,
            messages=messages,
            temperature=ctx.temp_ctrl.temperature,
        )

        # Strip any leading 'name: ' prefix
        reply = re.sub(r"^\s*\w+: ", "", raw_reply, count=1, flags=re.IGNORECASE)
        return reply
