"""
tests/test_child_bot_context.py
===============================
Tests for src/child_bot_context.py — per-bot state, trigger detection,
chime-in logic, cooldown tracking, and room eligibility.
"""

import time
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.child_bot_context import ChildBotContext


# ---------------------------------------------------------------------------
# Helpers — build a minimal ConfigManager for a child bot
# ---------------------------------------------------------------------------

CHILD_YAML = """
matrix:
  homeserver: "https://example.org"
  username: "@alice:example.org"
  password: "secret"
  allowed_rooms:
    - "!room1:example.org"
    - "#general:example.org"
  passive_channels:
    - "!passive:example.org"

llm:
  backend: "llamacpp"
  model_path: "models/test.gguf"
  hardware_mode: "cpu"

bot:
  display_name: "Alice"
  trigger_names:
    - "alice"
    - "hey alice"
  chime_in_probability: 0.5
  chime_in_cooldown_messages: 3
  chime_in_cooldown_seconds: 10

temperature_controller:
  enabled: true
  change_interval_minutes: 10
  min_temperature: 0.3
  max_temperature: 0.9

memory:
  dossier_dir: "{dossier_dir}"
  archive_dir: "{archive_dir}"

logging:
  level: "DEBUG"
"""


def _make_context(tmp_path, persona_text=None, cb=None):
    """Create a ChildBotContext with a real ConfigManager, mocked MatrixClient."""
    from src.config_manager import ConfigManager

    dossier_dir = str(tmp_path / "dossiers")
    archive_dir = str(tmp_path / "archives")

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        CHILD_YAML.format(dossier_dir=dossier_dir, archive_dir=archive_dir),
        encoding="utf-8",
    )

    # Write persona file
    persona_path = tmp_path / "persona.txt"
    if persona_text is not None:
        persona_path.write_text(persona_text, encoding="utf-8")

    cfg = ConfigManager(str(cfg_path))
    cfg.bot.persona_file = str(persona_path)

    callback = cb or AsyncMock()

    # Patch MatrixClient to avoid real network calls
    with patch("src.child_bot_context.MatrixClient") as MockMC:
        MockMC.return_value = MagicMock()
        ctx = ChildBotContext(
            name="alice",
            config=cfg,
            message_callback=callback,
        )
    return ctx


# ---------------------------------------------------------------------------
# Persona loading
# ---------------------------------------------------------------------------


class TestPersonaLoading:
    def test_persona_loaded_from_file(self, tmp_path):
        ctx = _make_context(tmp_path, persona_text="I am {NAME}, a helpful bot.")
        assert "Alice" in ctx.persona
        assert "{NAME}" not in ctx.persona

    def test_fallback_persona_when_file_missing(self, tmp_path):
        ctx = _make_context(tmp_path, persona_text=None)
        assert "Alice" in ctx.persona


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------


class TestTriggerDetection:
    def test_trigger_exact(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert ctx.is_triggered("alice can you help?")

    def test_trigger_phrase(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert ctx.is_triggered("hey alice, what's up?")

    def test_no_trigger(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert not ctx.is_triggered("anyone around?")

    def test_trigger_case_insensitive(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert ctx.is_triggered("ALICE help me")


# ---------------------------------------------------------------------------
# Chime-in logic
# ---------------------------------------------------------------------------


class TestChimeIn:
    def test_no_chime_in_during_cooldown(self, tmp_path):
        ctx = _make_context(tmp_path)
        room = "!room1:example.org"
        ctx.record_post(room)
        # Just posted — cooldown active, messages_since_post is 0
        assert not ctx.should_chime_in(room)

    def test_no_chime_in_insufficient_messages(self, tmp_path):
        ctx = _make_context(tmp_path)
        room = "!room1:example.org"
        # Simulate posting a long time ago
        ctx.last_post_time[room] = time.monotonic() - 999
        ctx.messages_since_post[room] = 1  # less than 3
        assert not ctx.should_chime_in(room)

    def test_chime_in_possible_when_conditions_met(self, tmp_path):
        ctx = _make_context(tmp_path)
        room = "!room1:example.org"
        ctx.last_post_time[room] = time.monotonic() - 999
        ctx.messages_since_post[room] = 100
        # With 0.5 probability and conditions met, we should see a chime-in
        # at least once out of many tries
        got_chime = any(ctx.should_chime_in(room) for _ in range(100))
        assert got_chime


# ---------------------------------------------------------------------------
# Cooldown tracking
# ---------------------------------------------------------------------------


class TestCooldownTracking:
    def test_record_post_resets_counter(self, tmp_path):
        ctx = _make_context(tmp_path)
        room = "!room1:example.org"
        ctx.tick_message(room)
        ctx.tick_message(room)
        assert ctx.messages_since_post[room] == 2
        ctx.record_post(room)
        assert ctx.messages_since_post[room] == 0

    def test_tick_message_increments(self, tmp_path):
        ctx = _make_context(tmp_path)
        room = "!room1:example.org"
        for _ in range(5):
            ctx.tick_message(room)
        assert ctx.messages_since_post[room] == 5


# ---------------------------------------------------------------------------
# Room eligibility
# ---------------------------------------------------------------------------


class TestRoomEligibility:
    def test_allowed_room_by_id(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert ctx.is_room_eligible("!room1:example.org")

    def test_allowed_room_by_alias(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert ctx.is_room_eligible("!unknown:x", "#general:example.org")

    def test_room_not_allowed(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert not ctx.is_room_eligible("!other:example.org")

    def test_empty_allowed_rooms_means_all(self, tmp_path):
        ctx = _make_context(tmp_path)
        ctx.config.matrix.allowed_rooms = []
        assert ctx.is_room_eligible("!anything:example.org")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_allowed_rooms_property(self, tmp_path):
        ctx = _make_context(tmp_path)
        rooms = ctx.allowed_rooms
        assert "!room1:example.org" in rooms
        assert "#general:example.org" in rooms

    def test_passive_channels_property(self, tmp_path):
        ctx = _make_context(tmp_path)
        assert "!passive:example.org" in ctx.passive_channels
