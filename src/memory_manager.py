"""
src/memory_manager.py
=====================
Manages per-user dossier files that the bot uses to remember who it has
talked to and what it knows about them.

Each user gets a JSON file in ``data/dossiers/<user_id_safe>.json``.
The file contains:
  - ``user_id``       — full Matrix user ID (e.g. @alice:example.org)
  - ``display_name``  — last known display name
  - ``summary``       — compacted text summary of older interactions
  - ``entries``       — list of recent interaction dicts (newest last)
  - ``first_seen``    — ISO timestamp of first interaction
  - ``last_seen``     — ISO timestamp of most recent interaction

When the number of entries exceeds ``max_active_entries``, the oldest half
is collapsed into the summary text and archived to ``data/archives/``.

Usage
-----
    from src.memory_manager import MemoryManager
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    mm = MemoryManager(cfg.memory)

    mm.record_interaction(
        user_id="@alice:example.org",
        display_name="Alice",
        message="I love Python!",
        bot_response="Same tbh.",
    )

    context = mm.get_dossier_context("@alice:example.org", max_chars=500)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_manager import MemoryConfig

logger = logging.getLogger(__name__)

# Characters that are illegal in filenames on most OSes.
_UNSAFE_CHARS = re.compile(r"[^\w\-]")


def estimate_token_usage(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def _safe_filename(user_id: str) -> str:
    """Convert a Matrix user ID into a filesystem-safe filename stem."""
    return _UNSAFE_CHARS.sub("_", user_id).strip("_") or "unknown_user"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryManager:
    """Handles reading, writing, and compacting per-user dossier files.

    Parameters
    ----------
    config:
        The ``memory`` section of the bot configuration.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._cfg = config
        self._dossier_dir = Path(config.dossier_dir)
        self._archive_dir = Path(config.archive_dir)
        self._dossier_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        user_id: str,
        display_name: str,
        message: str,
        bot_response: str,
        room_id: Optional[str] = None,
        extra_notes: Optional[str] = None,
    ) -> None:
        """Record a new interaction with a user and persist it to disk.

        Parameters
        ----------
        user_id:
            Full Matrix user ID (e.g. ``@alice:example.org``).
        display_name:
            The user's current display name.
        message:
            The user's message text.
        bot_response:
            The bot's reply text.
        room_id:
            Optional room ID where the interaction happened.
        extra_notes:
            Any additional notes to attach to this entry (e.g. detected topics).
        """
        dossier = self._load(user_id)

        entry: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "room_id": room_id,
            "user_message": message[:500],   # cap to avoid bloat
            "bot_response": bot_response[:500],
        }
        if extra_notes:
            entry["notes"] = extra_notes[:200]

        dossier["entries"].append(entry)
        dossier["display_name"] = display_name
        dossier["last_seen"] = _now_iso()

        self._compact_if_needed(dossier)
        self._save(user_id, dossier)

    def get_dossier_context(self, user_id: str, max_chars: int = 600) -> str:
        """Return a text snippet summarising what the bot knows about a user.

        The snippet is designed to fit inside the LLM prompt without eating
        the entire context window.

        Parameters
        ----------
        user_id:
            Full Matrix user ID.
        max_chars:
            Hard cap on the returned string length.

        Returns
        -------
        str
            Human-readable context text, or an empty string if the user is
            unknown.
        """
        dossier = self._load(user_id)
        if not dossier["first_seen"] and not dossier["entries"]:
            return ""

        lines: List[str] = []

        name = dossier.get("display_name") or user_id
        first = dossier.get("first_seen", "")
        last = dossier.get("last_seen", "")
        lines.append(f"User: {name} ({user_id})")
        if first:
            lines.append(f"  First seen: {first[:10]}  Last seen: {last[:10]}")

        if dossier.get("summary"):
            lines.append(f"  Summary: {dossier['summary']}")

        # Include a few of the most recent entries
        recent = dossier["entries"][-5:]
        if recent:
            lines.append("  Recent interactions:")
            for e in recent:
                ts = e.get("timestamp", "")[:16]
                um = e.get("user_message", "")[:120]
                br = e.get("bot_response", "")[:120]
                lines.append(f"    [{ts}] {name}: {um}")
                lines.append(f"           Bot: {br}")

        text = "\n".join(lines)
        return text[:max_chars]

    def get_known_users(self) -> List[str]:
        """Return a list of all known Matrix user IDs."""
        users: List[str] = []
        for path in self._dossier_dir.glob("*.json"):
            try:
                data = self._load_path(path)
                uid = data.get("user_id")
                if uid:
                    users.append(uid)
            except Exception:
                logger.debug("Could not read user_id from %s", path)
        return users

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Return all recorded messages from every dossier, sorted by timestamp.

        Each item is a dict with keys ``timestamp``, ``sender``, and ``text``.
        Both user messages and bot responses are included.
        """
        all_msgs: List[Dict[str, Any]] = []
        for path in self._dossier_dir.glob("*.json"):
            try:
                data = self._load_path(path)
                display_name = data.get("display_name") or data.get("user_id", "unknown")
                for entry in data.get("entries", []):
                    ts = entry.get("timestamp", "")
                    user_msg = entry.get("user_message", "")
                    if user_msg:
                        all_msgs.append(
                            {"timestamp": ts, "sender": display_name, "text": user_msg}
                        )
                    bot_resp = entry.get("bot_response", "")
                    if bot_resp:
                        all_msgs.append(
                            {"timestamp": ts, "sender": "bot", "text": bot_resp}
                        )
            except Exception:
                logger.debug("Could not read messages from %s", path)
        all_msgs.sort(key=lambda m: m.get("timestamp", ""))
        return all_msgs

    def compact_all(self) -> int:
        """Run compaction on all dossiers that need it.

        Returns
        -------
        int
            Number of dossiers that were compacted.
        """
        count = 0
        for path in self._dossier_dir.glob("*.json"):
            try:
                dossier = self._load_path(path)
                if len(dossier["entries"]) >= self._cfg.max_active_entries:
                    user_id = dossier.get("user_id", path.stem)
                    self._compact_if_needed(dossier)
                    self._save(user_id, dossier)
                    count += 1
            except Exception:
                logger.exception("Failed to compact dossier %s", path)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dossier_path(self, user_id: str) -> Path:
        return self._dossier_dir / f"{_safe_filename(user_id)}.json"

    def _load(self, user_id: str) -> Dict[str, Any]:
        path = self._dossier_path(user_id)
        if path.exists():
            return self._load_path(path)
        return {
            "user_id": user_id,
            "display_name": "",
            "summary": "",
            "entries": [],
            "first_seen": _now_iso(),
            "last_seen": "",
        }

    @staticmethod
    def _load_path(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Ensure expected keys exist (backwards compat)
        data.setdefault("summary", "")
        data.setdefault("entries", [])
        data.setdefault("first_seen", "")
        data.setdefault("last_seen", "")
        return data

    def _save(self, user_id: str, dossier: Dict[str, Any]) -> None:
        dossier["user_id"] = user_id
        path = self._dossier_path(user_id)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(dossier, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)

    def _compact_if_needed(self, dossier: Dict[str, Any]) -> None:
        """If entries exceed the limit, collapse the oldest half into summary."""
        entries = dossier["entries"]
        if len(entries) < self._cfg.max_active_entries:
            return

        # Archive the full current state first
        self._archive(dossier)

        # Split: compact oldest half, keep newest half active
        split = len(entries) // 2
        to_compact = entries[:split]
        dossier["entries"] = entries[split:]

        # Build new summary by prepending new data to existing summary
        new_summary_parts: List[str] = []
        if dossier.get("summary"):
            new_summary_parts.append(dossier["summary"])

        for e in to_compact:
            ts = e.get("timestamp", "")[:10]
            um = e.get("user_message", "")[:80]
            br = e.get("bot_response", "")[:80]
            new_summary_parts.append(f"[{ts}] said: {um!r} → bot: {br!r}")

        combined = " | ".join(new_summary_parts)
        # Trim to max_summary_chars
        dossier["summary"] = combined[: self._cfg.max_summary_chars]

        logger.info(
            "Compacted dossier for %s: %d entries archived.",
            dossier.get("user_id", "?"),
            split,
        )

    def _archive(self, dossier: Dict[str, Any]) -> None:
        """Save a timestamped snapshot of the dossier to the archive directory."""
        user_id = dossier.get("user_id", "unknown")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = f"{_safe_filename(user_id)}_{ts}.json"
        archive_path = self._archive_dir / fname
        with archive_path.open("w", encoding="utf-8") as fh:
            json.dump(dossier, fh, indent=2, ensure_ascii=False)
        logger.debug("Archived dossier snapshot → %s", archive_path)
