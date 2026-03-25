"""
src/child_bot_context.py
========================
Lightweight per-bot state container used by the multi-bot orchestrator.

Holds everything the orchestrator needs to manage a child bot **except** a
loaded LLM model — the LLM is loaded on-demand (one at a time) by the
orchestrator to guarantee only one model is in GPU memory at any point.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

from src.config_manager import ConfigManager
from src.matrix_client import MatrixClient
from src.memory_manager import MemoryManager
from src.temperature_controller import TemperatureController


class ChildBotContext:
    """Per-bot state — config, memory, temperature, Matrix client, persona.

    Parameters
    ----------
    name:
        Short identifier for this bot (e.g. ``"alice"``).
    config:
        Fully loaded :class:`ConfigManager` for this bot.
    message_callback:
        Async callback ``(bot_name, room, event) -> None`` the orchestrator
        provides so it can intercept messages from this bot's Matrix client.
    """

    def __init__(
        self,
        name: str,
        config: ConfigManager,
        message_callback,
    ) -> None:
        self.name = name
        self.config = config

        # Load persona text
        persona_path = Path(config.bot.persona_file)
        bot_name = config.bot.display_name
        if persona_path.exists():
            persona_text = persona_path.read_text(encoding="utf-8").strip()
            self.persona = persona_text.replace("{NAME}", bot_name)
        else:
            self.persona = (
                f"You are a witty, sarcastic bot named {bot_name} "
                "living in a Matrix chat server. Keep responses short and punchy."
            )

        self.memory = MemoryManager(config.memory)
        self.temp_ctrl = TemperatureController(
            config.temperature_controller,
            config.llm.temperature,
            config.bot.chime_in_probability,
        )

        # Wrap the orchestrator callback so it passes the bot name
        async def _cb(room, event):
            await message_callback(name, room, event)

        self.matrix_client = MatrixClient(config.matrix, message_callback=_cb)

        # Per-room cooldown state
        self.last_post_time: Dict[str, float] = {}
        self.messages_since_post: Dict[str, int] = defaultdict(int)

    @property
    def allowed_rooms(self) -> Set[str]:
        """Set of room IDs/aliases this bot is allowed to operate in."""
        return set(self.config.matrix.allowed_rooms)

    @property
    def passive_channels(self) -> Set[str]:
        """Set of rooms where this bot is in passive mode."""
        return set(self.config.matrix.passive_channels)

    def is_triggered(self, text: str) -> bool:
        """Return True if any of this bot's trigger names appear in *text*."""
        lower = text.lower()
        return any(name in lower for name in self.config.bot.trigger_names)

    def should_chime_in(self, room_id: str) -> bool:
        """Return True if this bot should spontaneously join the conversation."""
        import random

        last = self.last_post_time.get(room_id, 0)
        if time.monotonic() - last < self.config.bot.chime_in_cooldown_seconds:
            return False
        if self.messages_since_post[room_id] < self.config.bot.chime_in_cooldown_messages:
            return False
        return random.random() < self.temp_ctrl.chime_in_probability

    def record_post(self, room_id: str) -> None:
        """Update cooldown tracking after the bot has posted in a room."""
        self.last_post_time[room_id] = time.monotonic()
        self.messages_since_post[room_id] = 0

    def tick_message(self, room_id: str) -> None:
        """Increment the message counter for chime-in cooldown tracking."""
        self.messages_since_post[room_id] += 1

    def is_room_eligible(self, room_id: str, room_alias: str | None = None) -> bool:
        """Return True if this bot is allowed to interact in the given room."""
        allowed = self.config.matrix.allowed_rooms
        if not allowed:
            return True
        if room_id in allowed:
            return True
        if room_alias and room_alias in allowed:
            return True
        return False
