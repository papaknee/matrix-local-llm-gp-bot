"""
src/fleet_config.py
===================
Loads and validates the multi-bot fleet configuration (``bots/fleet_config.yaml``).

The fleet config references a route-bot (LLM-only, no Matrix account) and one
or more child bots, each with their own standard ``config.yaml``.

Usage
-----
    from src.fleet_config import FleetConfig

    fleet = FleetConfig("bots/fleet_config.yaml")
    print(fleet.route_bot_config_path)
    for bot in fleet.child_bots:
        print(bot.name, bot.config_path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class BotEntry:
    """A single child bot in the fleet."""
    name: str
    config_path: str


@dataclass
class FleetConfig:
    """Parsed and validated fleet configuration.

    Parameters
    ----------
    fleet_config_path:
        Path to the ``fleet_config.yaml`` file (e.g. ``bots/fleet_config.yaml``).

    Raises
    ------
    FileNotFoundError
        If the fleet config or any referenced config file does not exist.
    ValueError
        If required fields are missing or invalid.
    """

    route_bot_config_path: str = ""
    child_bots: List[BotEntry] = field(default_factory=list)

    def __init__(self, fleet_config_path: str = "bots/fleet_config.yaml") -> None:
        path = Path(fleet_config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fleet config not found: {path}\n"
                "Run the setup wizard in multi-bot mode to create it:\n\n"
                "    python setup_wizard.py\n"
            )

        with path.open("r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # Parse route-bot
        route_section = raw.get("route_bot", {})
        self.route_bot_config_path = route_section.get("config_path", "")

        # Parse child bots
        self.child_bots = []
        for entry in raw.get("bots", []):
            if isinstance(entry, dict) and "name" in entry and "config_path" in entry:
                self.child_bots.append(
                    BotEntry(name=entry["name"], config_path=entry["config_path"])
                )

        self._validate()

    def _validate(self) -> None:
        errors: list[str] = []

        if not self.route_bot_config_path:
            errors.append("route_bot.config_path is required")
        elif not Path(self.route_bot_config_path).exists():
            errors.append(
                f"route_bot config not found: {self.route_bot_config_path}"
            )

        if not self.child_bots:
            errors.append("At least one child bot is required under 'bots'")

        seen_names: set[str] = set()
        for bot in self.child_bots:
            if bot.name in seen_names:
                errors.append(f"Duplicate bot name: {bot.name!r}")
            seen_names.add(bot.name)

            if not Path(bot.config_path).exists():
                errors.append(
                    f"Config not found for bot {bot.name!r}: {bot.config_path}"
                )

        if errors:
            raise ValueError(
                "Fleet configuration errors:\n  - " + "\n  - ".join(errors)
            )

    def add_bot(self, name: str, config_path: str) -> None:
        """Add a new child bot entry (in memory only — call :meth:`save` to persist)."""
        for bot in self.child_bots:
            if bot.name == name:
                raise ValueError(f"Bot {name!r} already exists in the fleet")
        self.child_bots.append(BotEntry(name=name, config_path=config_path))

    def remove_bot(self, name: str) -> None:
        """Remove a child bot by name (in memory only — call :meth:`save` to persist)."""
        original = len(self.child_bots)
        self.child_bots = [b for b in self.child_bots if b.name != name]
        if len(self.child_bots) == original:
            raise ValueError(f"Bot {name!r} not found in the fleet")

    def save(self, fleet_config_path: str = "bots/fleet_config.yaml") -> None:
        """Write the current fleet config back to disk."""
        data = {
            "route_bot": {"config_path": self.route_bot_config_path},
            "bots": [
                {"name": b.name, "config_path": b.config_path}
                for b in self.child_bots
            ],
        }
        path = Path(fleet_config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
