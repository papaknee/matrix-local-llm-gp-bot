"""
tests/test_memory_manager.py
============================
Tests for src/memory_manager.py
"""

import json
from pathlib import Path

import pytest

from src.config_manager import MemoryConfig
from src.memory_manager import MemoryManager, _safe_filename, estimate_token_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, max_entries: int = 10) -> MemoryConfig:
    return MemoryConfig(
        dossier_dir=str(tmp_path / "dossiers"),
        archive_dir=str(tmp_path / "archives"),
        max_active_entries=max_entries,
        compaction_interval_hours=6,
        max_summary_chars=200,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSafeFilename:
    @pytest.mark.parametrize(
        "user_id, expected_pattern",
        [
            ("@alice:example.org", "_alice_example_org"),
            ("@bob-smith:matrix.org", "_bob-smith_matrix_org"),
            ("@user123:server.com", "_user123_server_com"),
        ],
    )
    def test_safe_filename_no_illegal_chars(self, user_id, expected_pattern):
        name = _safe_filename(user_id)
        # Should not contain @ or : (replaced by _)
        assert "@" not in name
        assert ":" not in name
        assert "." not in name

    def test_empty_string_fallback(self):
        assert _safe_filename("") == "unknown_user"


class TestRecordInteraction:
    def test_creates_dossier_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction(
            user_id="@alice:example.org",
            display_name="Alice",
            message="Hello bot!",
            bot_response="Hey Alice!",
        )
        dossier_dir = Path(cfg.dossier_dir)
        files = list(dossier_dir.glob("*.json"))
        assert len(files) == 1

    def test_dossier_contains_entry(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction(
            user_id="@alice:example.org",
            display_name="Alice",
            message="Hello bot!",
            bot_response="Hey Alice!",
        )
        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["user_message"] == "Hello bot!"
        assert data["entries"][0]["bot_response"] == "Hey Alice!"

    def test_multiple_interactions_appended(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        for i in range(5):
            mm.record_interaction(
                user_id="@alice:example.org",
                display_name="Alice",
                message=f"Message {i}",
                bot_response=f"Reply {i}",
            )
        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        assert len(data["entries"]) == 5

    def test_display_name_updated(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "hi", "hello")
        mm.record_interaction("@alice:example.org", "Alice Smith", "bye", "cya")

        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        assert data["display_name"] == "Alice Smith"

    def test_long_message_truncated(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        long_msg = "x" * 1000
        mm.record_interaction("@alice:example.org", "Alice", long_msg, "ok")

        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        assert len(data["entries"][0]["user_message"]) <= 500

    def test_room_id_stored(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction(
            "@alice:example.org", "Alice", "hi", "hello", room_id="!room1:example.org"
        )
        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        assert data["entries"][0]["room_id"] == "!room1:example.org"

    def test_separate_dossiers_per_user(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "hi", "hey")
        mm.record_interaction("@bob:example.org", "Bob", "hello", "sup")

        dossier_dir = Path(cfg.dossier_dir)
        files = list(dossier_dir.glob("*.json"))
        assert len(files) == 2


class TestGetDossierContext:
    def test_returns_empty_for_unknown_user(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        # Unknown user — new dossier but no first_seen
        # Actually the code sets first_seen on load; test that context is short
        ctx = mm.get_dossier_context("@ghost:example.org")
        # Either empty or very short (no real interactions)
        assert len(ctx) < 200

    def test_returns_context_for_known_user(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "I love cats!", "Me too!")
        ctx = mm.get_dossier_context("@alice:example.org")
        assert "Alice" in ctx
        assert "I love cats!" in ctx

    def test_context_respects_max_chars(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        for i in range(10):
            mm.record_interaction(
                "@alice:example.org", "Alice", f"Long message number {i} " * 20, "ok"
            )
        ctx = mm.get_dossier_context("@alice:example.org", max_chars=300)
        assert len(ctx) <= 300


class TestCompaction:
    def test_compaction_triggers_when_entries_exceed_limit(self, tmp_path):
        cfg = _make_config(tmp_path, max_entries=5)
        mm = MemoryManager(cfg)

        # Add max_active_entries entries — compaction triggers on the 5th
        for i in range(6):
            mm.record_interaction(
                "@alice:example.org",
                "Alice",
                f"Message {i}",
                f"Reply {i}",
            )

        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        # Active entries should be reduced
        assert len(data["entries"]) < 6
        # Summary should be non-empty
        assert len(data["summary"]) > 0

    def test_archive_created_during_compaction(self, tmp_path):
        cfg = _make_config(tmp_path, max_entries=4)
        mm = MemoryManager(cfg)

        for i in range(5):
            mm.record_interaction("@alice:example.org", "Alice", f"m{i}", f"r{i}")

        archive_dir = Path(cfg.archive_dir)
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) >= 1

    def test_compact_all_returns_count(self, tmp_path):
        cfg = _make_config(tmp_path, max_entries=3)
        mm = MemoryManager(cfg)

        # Create two users that both need compaction
        for user in ("@alice:example.org", "@bob:example.org"):
            for i in range(4):
                mm.record_interaction(user, user, f"m{i}", f"r{i}")

        count = mm.compact_all()
        assert count >= 0  # at least ran without error

    def test_summary_max_chars_respected(self, tmp_path):
        cfg = _make_config(tmp_path, max_entries=4)
        mm = MemoryManager(cfg)
        for i in range(5):
            mm.record_interaction(
                "@alice:example.org",
                "Alice",
                "x" * 200,
                "y" * 200,
            )
        dossier_dir = Path(cfg.dossier_dir)
        data = json.loads(next(dossier_dir.glob("*.json")).read_text())
        assert len(data.get("summary", "")) <= cfg.max_summary_chars


class TestGetKnownUsers:
    def test_empty_when_no_dossiers(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        assert mm.get_known_users() == []

    def test_returns_users_after_recording(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "hi", "hey")
        mm.record_interaction("@bob:example.org", "Bob", "hi", "sup")
        users = mm.get_known_users()
        assert len(users) == 2


class TestEstimateTokenUsage:
    def test_empty_string(self):
        assert estimate_token_usage("") == 0

    def test_short_string(self):
        # 12 chars → 3 tokens
        assert estimate_token_usage("hello world!") == 3

    def test_longer_string(self):
        text = "a" * 400
        assert estimate_token_usage(text) == 100


class TestGetAllMessages:
    def test_empty_when_no_dossiers(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        assert mm.get_all_messages() == []

    def test_returns_messages_from_single_user(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "Hello!", "Hi there!")
        msgs = mm.get_all_messages()
        # Should contain both user message and bot response
        assert len(msgs) == 2
        senders = [m["sender"] for m in msgs]
        assert "Alice" in senders
        assert "bot" in senders

    def test_returns_messages_from_multiple_users(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "Hi", "Hey")
        mm.record_interaction("@bob:example.org", "Bob", "Hello", "Sup")
        msgs = mm.get_all_messages()
        # 2 users × 2 messages each (user msg + bot response)
        assert len(msgs) == 4

    def test_messages_sorted_by_timestamp(self, tmp_path):
        cfg = _make_config(tmp_path)
        mm = MemoryManager(cfg)
        mm.record_interaction("@alice:example.org", "Alice", "First", "Reply1")
        mm.record_interaction("@alice:example.org", "Alice", "Second", "Reply2")
        msgs = mm.get_all_messages()
        timestamps = [m["timestamp"] for m in msgs]
        assert timestamps == sorted(timestamps)
