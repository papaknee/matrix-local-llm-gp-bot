"""
src/temperature_controller.py
==============================
Manages the bot's "mood" — a runtime LLM temperature and chime-in probability
that change on a schedule to make the bot feel alive and unpredictable.

The controller keeps its own internal state; the scheduler (src/scheduler.py)
calls ``roll_new_mood()`` on the configured interval.

Usage
-----
    from src.temperature_controller import TemperatureController
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    tc = TemperatureController(cfg.temperature_controller, cfg.llm.temperature)

    # Get current mood values
    print(tc.temperature)
    print(tc.chime_in_probability)

    # Force a mood roll (also called automatically by the scheduler)
    tc.roll_new_mood()
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.config_manager import BotConfig, TemperatureControllerConfig

logger = logging.getLogger(__name__)


@dataclass
class MoodSnapshot:
    """Records what mood the bot was in at a particular moment."""

    timestamp: datetime
    temperature: float
    chime_in_probability: float
    label: str  # human-readable mood name


# Mood labels keyed by approximate temperature range
_MOOD_LABELS = [
    (0.0, 0.35, "zen"),
    (0.35, 0.55, "calm"),
    (0.55, 0.75, "balanced"),
    (0.75, 0.90, "chatty"),
    (0.90, 1.00, "hyper"),
    (1.00, 1.10, "feisty"),
    (1.10, float("inf"), "unhinged"),
]


def _temperature_to_label(temp: float) -> str:
    for lo, hi, label in _MOOD_LABELS:
        if lo <= temp < hi:
            return label
    return "mysterious"


class TemperatureController:
    """Controls the bot's runtime temperature and spontaneous chime-in rate.

    Parameters
    ----------
    tc_config:
        The ``temperature_controller`` section of the config.
    initial_temperature:
        Starting temperature (taken from ``llm.temperature``).
    initial_chime_in:
        Starting chime-in probability (taken from ``bot.chime_in_probability``).
    """

    def __init__(
        self,
        tc_config: TemperatureControllerConfig,
        initial_temperature: float,
        initial_chime_in: float = 0.08,
    ) -> None:
        self._cfg = tc_config
        self._temperature: float = initial_temperature
        self._chime_in: float = initial_chime_in
        self._history: list[MoodSnapshot] = []

        # Record initial mood
        self._record_snapshot()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """Current LLM sampling temperature."""
        return self._temperature

    @property
    def chime_in_probability(self) -> float:
        """Current probability that the bot spontaneously joins a conversation."""
        return self._chime_in

    @property
    def current_mood_label(self) -> str:
        """Human-readable mood label for the current temperature."""
        return _temperature_to_label(self._temperature)

    @property
    def history(self) -> list[MoodSnapshot]:
        """Read-only list of recent mood snapshots (most recent last)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Mood roll
    # ------------------------------------------------------------------

    def roll_new_mood(self) -> MoodSnapshot:
        """Pick a new temperature (and optionally chime-in probability) at random.

        The new values are drawn uniformly from the ranges defined in the
        config.  A log message is emitted so the mood change is visible.

        Returns
        -------
        MoodSnapshot
            The newly applied mood snapshot.
        """
        if not self._cfg.enabled:
            logger.debug("Temperature controller is disabled; mood roll skipped.")
            return self._history[-1] if self._history else self._record_snapshot()

        old_temp = self._temperature
        self._temperature = round(
            random.uniform(self._cfg.min_temperature, self._cfg.max_temperature), 3
        )

        if self._cfg.randomise_chime_in:
            self._chime_in = round(
                random.uniform(self._cfg.min_chime_in, self._cfg.max_chime_in), 3
            )

        snap = self._record_snapshot()
        logger.info(
            "🎭 Mood swing! %.3f → %.3f (%s) | chime_in=%.2f",
            old_temp,
            self._temperature,
            snap.label,
            self._chime_in,
        )
        return snap

    # ------------------------------------------------------------------
    # Direct setters (used by tests / manual overrides)
    # ------------------------------------------------------------------

    def set_temperature(self, value: float) -> None:
        """Manually set the temperature without triggering a full mood roll."""
        if not (0.0 < value <= 2.0):
            raise ValueError(f"Temperature must be in (0, 2], got {value}")
        self._temperature = value
        self._record_snapshot()

    def set_chime_in_probability(self, value: float) -> None:
        """Manually override the chime-in probability."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Chime-in probability must be in [0, 1], got {value}")
        self._chime_in = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_snapshot(self) -> MoodSnapshot:
        snap = MoodSnapshot(
            timestamp=datetime.now(timezone.utc),
            temperature=self._temperature,
            chime_in_probability=self._chime_in,
            label=_temperature_to_label(self._temperature),
        )
        self._history.append(snap)
        # Keep history bounded
        if len(self._history) > 200:
            self._history = self._history[-100:]
        return snap
