"""
src/bot.py
==========
Core bot logic: decides when and how to respond to Matrix messages.

Decision flow for each incoming message
-----------------------------------------
1. Is the message from a room that the bot is allowed to be in?  (handled by
   MatrixClient; we only reach this point if yes)
2. Does the message contain one of the bot's trigger names / nicknames?
   → Always respond.
3. Is the bot in a cooldown period for this room (message count or time)?
   → Skip chime-in check.
4. Roll random float against ``chime_in_probability``.
   → If it lands below the threshold, respond even without a trigger.

When responding, the bot:
  - Loads the sender's dossier context from MemoryManager.
  - Builds a prompt from system persona + dossier + recent room history.
  - Calls LLMBackend.generate() with the current temperature.
  - Sends the reply back to the room.
  - Records the interaction in the dossier.

Usage
-----
    from src.bot import Bot
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    bot = Bot(cfg)
    await bot.start()          # blocks — runs Matrix sync + scheduler
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from nio import MatrixRoom, RoomMessageText  # type: ignore[import-untyped]

from src.config_manager import ConfigManager
from src.llm import LLMBackend
from src.matrix_client import MatrixClient
from src.memory_manager import MemoryManager
from src.scheduler import Scheduler
from src.temperature_controller import TemperatureController

logger = logging.getLogger(__name__)

# Number of recent (sender, text) pairs kept per room for context
_HISTORY_SIZE = 30


class Bot:
    """The main bot class that ties all components together.

    Parameters
    ----------
    config:
        A fully initialised :class:`~src.config_manager.ConfigManager`.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._cfg = config

        # Load persona text and replace {NAME} with bot_name
        persona_path = Path(config.bot.persona_file)
        bot_name = getattr(config, "bot_name", config.bot.display_name)
        if persona_path.exists():
            persona_text = persona_path.read_text(encoding="utf-8").strip()
            self._persona = persona_text.replace("{NAME}", bot_name)
        else:
            logger.warning(
                "Persona file not found at %s — using built-in default.", persona_path
            )
            self._persona = (
                f"You are a witty, sarcastic bot named {bot_name} living in a Matrix chat server. "
                "Keep responses short and punchy."
            )

        self._llm = LLMBackend(config.llm)
        self._memory = MemoryManager(config.memory)
        self._temp_ctrl = TemperatureController(
            config.temperature_controller,
            config.llm.temperature,
            config.bot.chime_in_probability,
        )
        self._matrix = MatrixClient(config.matrix, message_callback=self._on_message)
        self._scheduler = Scheduler(self._temp_ctrl, self._memory, config)

        # Per-room message history: room_id → deque of (sender_display, text)
        self._room_history: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=_HISTORY_SIZE)
        )
        # Per-room cooldown tracking
        self._last_bot_post_time: Dict[str, float] = {}
        self._messages_since_last_post: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load the model, log in to Matrix, and start all async loops."""
        logger.info("Loading LLM model …")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._llm.load)
        logger.info("LLM ready.")

        async with self._matrix:
            await self._matrix.set_display_name(self._cfg.bot.display_name)
            rooms = await self._matrix.get_joined_rooms()
            logger.info("Bot is in %d room(s): %s", len(rooms), rooms)

            scheduler_task = asyncio.create_task(self._scheduler.run())
            try:
                await self._matrix.sync_forever()
            finally:
                scheduler_task.cancel()
                try:
                    await scheduler_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Handle an incoming Matrix message."""
        sender_id: str = event.sender
        sender_name: str = room.user_name(sender_id) or sender_id
        text: str = getattr(event, "body", "") or ""
        room_id: str = room.room_id

        if not text.strip():
            return

        # Update room history
        self._room_history[room_id].append((sender_name, text))
        self._messages_since_last_post[room_id] += 1

        # Passive channel logic
        passive_channels = set(self._cfg.matrix.passive_channels)
        if room_id in passive_channels or room.canonical_alias in passive_channels:
            # Use LLM to classify if bot is being addressed (yes/maybe/no)
            response_type = await self._classify_addressed(text, sender_name, room_id)
            logger.info(f"Passive channel: classified response={response_type!r} for message '{text}'")
            if response_type == "yes":
                should_respond = True
            elif response_type == "maybe":
                should_respond = self._cfg.bot.respond_on_maybe
            else:
                should_respond = False
        else:
            triggered = self._is_triggered(text)
            should_respond = triggered or self._should_chime_in(room_id)

        if not should_respond:
            return

        logger.info(
            "Responding in %s to %s (triggered=%s)", room_id, sender_name, 'passive' if room_id in passive_channels else triggered
        )

        # Build prompt and generate reply (off the event loop to avoid blocking)
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(
            None,
            self._build_and_generate,
            sender_id,
            sender_name,
            text,
            room_id,
        )

        if not reply:
            logger.warning("LLM returned empty response — skipping send.")
            return

        await self._matrix.send_message(room_id, reply)

    async def _classify_addressed(self, text: str, sender_name: str, room_id: str) -> str:
        """
        Use LLM to classify if the message is directed at the bot.
        Returns: 'yes', 'maybe', or 'no'.
        """
        # Gather last 4 messages for context
        history = list(self._room_history[room_id])[-4:] if room_id and room_id in self._room_history else [(sender_name, text)]
        transcript = "\n".join([f"{name}: {msg}" for name, msg in history])

        # Prompt with yes/maybe/no and diverse examples
        prompt = (
            "You are an AI assistant in a group chat. Given the transcript of the last few messages, answer with one word: \"yes\", \"no\", or \"maybe\" to indicate whether you should reply to the last message.\n"
            "- \"yes\" if you are being directly addressed or it would be natural for you to respond.\n"
            "- \"maybe\" if it's ambiguous, group-oriented, or a casual message where a reply could be appropriate.\n"
            "- \"no\" if you should not reply.\n"
            "\n"
            "Examples:\n"
            "Transcript:\nAlice: Can anyone help?\nYou: \nAnswer: yes\n"
            "Transcript:\nBob: Anyone around?\nYou: \nAnswer: maybe\n"
            "Transcript:\nCharlie: I love pizza.\nYou: \nAnswer: maybe\n"
            "Transcript:\nAlice: Good morning, everyone!\nYou: \nAnswer: yes\n"
            "Transcript:\nBob: What’s up, everyone?\nYou: \nAnswer: yes\n"
            "Transcript:\nAlice: Anyone want to play a game?\nBob: I'm in!\nCharlie: What game?\nYou: \nAnswer: yes\n"
            "Transcript:\nAlice: Can someone explain this error?\nBob: Sorry, maybe a bot knows?\nYou: \nAnswer: yes\n"
            "Transcript:\nCharlie: I just finished my project!\nYou: \nAnswer: yes\n"
            "Transcript:\nBob: Hi, everyone, let's keep the chat friendly.\nYou: \nAnswer: yes\n"
            "Transcript:\nAlice: I love pizza.\nBob: Me too!\nCharlie: Same here.\nYou: \nAnswer: yes\n"
            f"Transcript:\n{transcript}\nAnswer:"
        )
        # Use the LLM to generate a response
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._llm.generate(prompt)
        )
        answer = result.strip().lower()
        if answer.startswith("yes"):
            return "yes"
        elif answer.startswith("maybe"):
            return "maybe"
        elif answer.startswith("no"):
            return "no"
        else:
            # fallback: try to find yes/maybe/no anywhere in output
            if "yes" in answer:
                return "yes"
            elif "maybe" in answer:
                return "maybe"
            elif "no" in answer:
                return "no"
            else:
                logger.warning(f"Could not parse yes/maybe/no from LLM output: {result}")
                return "no"

        # Update cooldown state
        self._last_bot_post_time[room_id] = time.monotonic()
        self._messages_since_last_post[room_id] = 0

        # Persist interaction to dossier
        try:
            self._memory.record_interaction(
                user_id=sender_id,
                display_name=sender_name,
                message=text,
                bot_response=reply,
                room_id=room_id,
            )
        except Exception:
            logger.exception("Failed to record interaction for %s", sender_id)

    # ------------------------------------------------------------------
    # Trigger detection
    # ------------------------------------------------------------------

    def _is_triggered(self, text: str) -> bool:
        """Return True if any of the bot's trigger names appear in the message."""
        lower = text.lower()
        return any(name in lower for name in self._cfg.bot.trigger_names)

    # ------------------------------------------------------------------
    # Chime-in decision
    # ------------------------------------------------------------------

    def _should_chime_in(self, room_id: str) -> bool:
        """Return True if the bot should spontaneously join the conversation."""
        # Time-based cooldown
        last_post = self._last_bot_post_time.get(room_id, 0)
        if time.monotonic() - last_post < self._cfg.bot.chime_in_cooldown_seconds:
            return False

        # Message-count cooldown
        if (
            self._messages_since_last_post[room_id]
            < self._cfg.bot.chime_in_cooldown_messages
        ):
            return False

        # Probabilistic roll using the current mood's chime_in_probability
        return random.random() < self._temp_ctrl.chime_in_probability

    # ------------------------------------------------------------------
    # Prompt construction & generation
    # ------------------------------------------------------------------

    def _build_and_generate(
        self,
        sender_id: str,
        sender_name: str,
        text: str,
        room_id: str,
    ) -> str:
        """Build the LLM prompt and call generate().  Runs in a thread pool."""
        # --- System prompt = persona + optional dossier context ---
        dossier_ctx = self._memory.get_dossier_context(
            sender_id, max_chars=self._cfg.bot.dossier_token_budget * 4
        )
        system_parts = [self._persona]
        if dossier_ctx:
            system_parts.append(
                f"\n--- What you remember about {sender_name} ---\n{dossier_ctx}"
            )
        system_prompt = "\n".join(system_parts)

        # --- Conversation history as message list ---
        history = list(self._room_history[room_id])
        limit = self._cfg.bot.conversation_history_limit
        recent = history[-limit:]

        messages: List[dict] = []
        for spkr, msg in recent[:-1]:  # exclude the triggering message
            role = "assistant" if spkr == self._cfg.bot.display_name else "user"
            messages.append({"role": role, "content": f"{spkr}: {msg}"})

        # The current message is the final user turn
        messages.append({"role": "user", "content": f"{sender_name}: {text}"})

        return self._llm.generate(
            system_prompt=system_prompt,
            messages=messages,
            temperature=self._temp_ctrl.temperature,
        )
