"""
src/scheduler.py
================
Background task scheduler that runs periodic jobs without blocking the async
Matrix sync loop.

Jobs registered:
  1. **Temperature roll** — calls ``TemperatureController.roll_new_mood()``
     every N minutes (configurable).
  2. **Memory compaction** — calls ``MemoryManager.compact_all()`` every N hours
     (configurable).

The scheduler runs in a separate ``asyncio`` task so it doesn't block the bot.

Usage
-----
    from src.scheduler import Scheduler

    scheduler = Scheduler(temp_controller, memory_manager, cfg)
    task = asyncio.create_task(scheduler.run())
    # ... later ...
    task.cancel()
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_manager import ConfigManager
    from src.memory_manager import MemoryManager
    from src.temperature_controller import TemperatureController

logger = logging.getLogger(__name__)


class Scheduler:
    """Runs periodic background jobs for the bot.

    Parameters
    ----------
    temp_controller:
        The bot's :class:`~src.temperature_controller.TemperatureController`.
    memory_manager:
        The bot's :class:`~src.memory_manager.MemoryManager`.
    config:
        The full :class:`~src.config_manager.ConfigManager` instance.
    """

    def __init__(
        self,
        temp_controller: "TemperatureController",
        memory_manager: "MemoryManager",
        config: "ConfigManager",
    ) -> None:
        self._tc = temp_controller
        self._mm = memory_manager
        self._cfg = config

    async def run(self) -> None:
        """Start all background tasks and run until cancelled."""
        tasks = [
            asyncio.create_task(self._temperature_job()),
            asyncio.create_task(self._compaction_job()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    async def _temperature_job(self) -> None:
        """Roll a new mood on the configured interval."""
        interval_seconds = (
            self._cfg.temperature_controller.change_interval_minutes * 60
        )
        logger.info(
            "Temperature job started — rolling mood every %d minutes.",
            self._cfg.temperature_controller.change_interval_minutes,
        )
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                snap = self._tc.roll_new_mood()
                logger.info(
                    "🎭 New mood: %s (temp=%.3f, chime_in=%.2f)",
                    snap.label,
                    snap.temperature,
                    snap.chime_in_probability,
                )
            except Exception:
                logger.exception("Error in temperature job")

    async def _compaction_job(self) -> None:
        """Compact user dossiers on the configured interval."""
        interval_seconds = (
            self._cfg.memory.compaction_interval_hours * 3600
        )
        logger.info(
            "Memory compaction job started — running every %d hours.",
            self._cfg.memory.compaction_interval_hours,
        )
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                count = self._mm.compact_all()
                if count:
                    logger.info(
                        "🗜  Memory compaction complete — %d dossier(s) compacted.", count
                    )
            except Exception:
                logger.exception("Error in compaction job")
