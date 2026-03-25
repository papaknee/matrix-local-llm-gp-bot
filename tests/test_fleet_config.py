"""
tests/test_fleet_config.py
==========================
Tests for src/fleet_config.py — fleet configuration loading, validation,
add/remove bot, and save/load round-trip.
"""

import textwrap
from pathlib import Path

import pytest

from src.fleet_config import BotEntry, FleetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fleet(tmp_path: Path, content: str) -> str:
    """Write a fleet_config.yaml and return its path."""
    p = tmp_path / "fleet_config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def _write_dummy_config(tmp_path: Path, name: str = "config.yaml") -> str:
    """Create a minimal YAML file that FleetConfig can reference."""
    p = tmp_path / name
    p.write_text("matrix:\n  homeserver: x\n", encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


class TestFleetConfigLoading:
    def test_valid_config(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        bot_cfg = _write_dummy_config(tmp_path, "bot1.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "{bot_cfg}"
        """)
        fleet = FleetConfig(fleet_yaml)
        assert fleet.route_bot_config_path == route_cfg
        assert len(fleet.child_bots) == 1
        assert fleet.child_bots[0].name == "alice"
        assert fleet.child_bots[0].config_path == bot_cfg

    def test_multiple_bots(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        b1 = _write_dummy_config(tmp_path, "b1.yaml")
        b2 = _write_dummy_config(tmp_path, "b2.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "{b1}"
              - name: bob
                config_path: "{b2}"
        """)
        fleet = FleetConfig(fleet_yaml)
        assert len(fleet.child_bots) == 2
        assert fleet.child_bots[1].name == "bob"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Fleet config not found"):
            FleetConfig(str(tmp_path / "nonexistent.yaml"))


class TestFleetConfigValidation:
    def test_missing_route_bot_config(self, tmp_path):
        bot_cfg = _write_dummy_config(tmp_path, "bot.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: ""
            bots:
              - name: alice
                config_path: "{bot_cfg}"
        """)
        with pytest.raises(ValueError, match="route_bot.config_path is required"):
            FleetConfig(fleet_yaml)

    def test_route_bot_config_missing_file(self, tmp_path):
        bot_cfg = _write_dummy_config(tmp_path, "bot.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "/nonexistent/route.yaml"
            bots:
              - name: alice
                config_path: "{bot_cfg}"
        """)
        with pytest.raises(ValueError, match="route_bot config not found"):
            FleetConfig(fleet_yaml)

    def test_no_child_bots(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots: []
        """)
        with pytest.raises(ValueError, match="At least one child bot"):
            FleetConfig(fleet_yaml)

    def test_duplicate_bot_names(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        b1 = _write_dummy_config(tmp_path, "b1.yaml")
        b2 = _write_dummy_config(tmp_path, "b2.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "{b1}"
              - name: alice
                config_path: "{b2}"
        """)
        with pytest.raises(ValueError, match="Duplicate bot name.*alice"):
            FleetConfig(fleet_yaml)

    def test_child_bot_config_not_found(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "/nonexistent/alice.yaml"
        """)
        with pytest.raises(ValueError, match="Config not found for bot.*alice"):
            FleetConfig(fleet_yaml)

    def test_malformed_bot_entries_ignored(self, tmp_path):
        """Bot entries missing 'name' or 'config_path' are silently skipped."""
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        bot_cfg = _write_dummy_config(tmp_path, "bot.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "{bot_cfg}"
              - oops: not_a_bot
        """)
        fleet = FleetConfig(fleet_yaml)
        assert len(fleet.child_bots) == 1


# ---------------------------------------------------------------------------
# Add / remove bots
# ---------------------------------------------------------------------------


class TestFleetConfigMutation:
    def _make_fleet(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        b1 = _write_dummy_config(tmp_path, "b1.yaml")
        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "{b1}"
        """)
        return FleetConfig(fleet_yaml)

    def test_add_bot(self, tmp_path):
        fleet = self._make_fleet(tmp_path)
        fleet.add_bot("bob", "/some/path.yaml")
        assert len(fleet.child_bots) == 2
        assert fleet.child_bots[-1].name == "bob"

    def test_add_duplicate_raises(self, tmp_path):
        fleet = self._make_fleet(tmp_path)
        with pytest.raises(ValueError, match="already exists"):
            fleet.add_bot("alice", "/any.yaml")

    def test_remove_bot(self, tmp_path):
        fleet = self._make_fleet(tmp_path)
        fleet.add_bot("bob", "/some/path.yaml")
        fleet.remove_bot("bob")
        assert len(fleet.child_bots) == 1
        assert fleet.child_bots[0].name == "alice"

    def test_remove_nonexistent_raises(self, tmp_path):
        fleet = self._make_fleet(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            fleet.remove_bot("nobody")


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestFleetConfigSave:
    def test_save_and_reload(self, tmp_path):
        route_cfg = _write_dummy_config(tmp_path, "route.yaml")
        b1 = _write_dummy_config(tmp_path, "b1.yaml")
        b2 = _write_dummy_config(tmp_path, "b2.yaml")

        fleet_yaml = _write_fleet(tmp_path, f"""
            route_bot:
              config_path: "{route_cfg}"
            bots:
              - name: alice
                config_path: "{b1}"
        """)

        fleet = FleetConfig(fleet_yaml)
        fleet.add_bot("bob", str(b2))

        out_path = str(tmp_path / "out.yaml")
        fleet.save(out_path)

        reloaded = FleetConfig(out_path)
        assert reloaded.route_bot_config_path == route_cfg
        assert len(reloaded.child_bots) == 2
        assert reloaded.child_bots[0].name == "alice"
        assert reloaded.child_bots[1].name == "bob"


# ---------------------------------------------------------------------------
# BotEntry dataclass
# ---------------------------------------------------------------------------


class TestBotEntry:
    def test_fields(self):
        entry = BotEntry(name="test", config_path="/a/b.yaml")
        assert entry.name == "test"
        assert entry.config_path == "/a/b.yaml"
