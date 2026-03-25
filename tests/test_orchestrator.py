"""
tests/test_orchestrator.py
==========================
Tests for src/orchestrator.py — PendingMessage ordering, deduplication,
and message queue FIFO guarantees.

Heavy integration tests (actual Matrix connections, LLM loading) are not run
here; the focus is on the orchestration logic that can be unit-tested.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator import PendingMessage


# ---------------------------------------------------------------------------
# PendingMessage ordering
# ---------------------------------------------------------------------------


class TestPendingMessageOrdering:
    """Verify that PendingMessage sorts by server_timestamp (oldest first)."""

    def test_ordering_by_timestamp(self):
        m1 = PendingMessage(timestamp=100, event_id="e1", room_id="!r")
        m2 = PendingMessage(timestamp=200, event_id="e2", room_id="!r")
        m3 = PendingMessage(timestamp=50, event_id="e3", room_id="!r")
        assert sorted([m1, m2, m3]) == [m3, m1, m2]

    def test_equal_timestamps(self):
        m1 = PendingMessage(timestamp=100, event_id="e1", room_id="!r1")
        m2 = PendingMessage(timestamp=100, event_id="e2", room_id="!r2")
        # Both equal — no crash on comparison
        assert sorted([m1, m2]) in ([m1, m2], [m2, m1])

    def test_comparison_operators(self):
        early = PendingMessage(timestamp=1, event_id="e1", room_id="!r")
        late = PendingMessage(timestamp=999, event_id="e2", room_id="!r")
        assert early < late
        assert late > early
        assert not (early > late)


# ---------------------------------------------------------------------------
# PriorityQueue FIFO guarantee
# ---------------------------------------------------------------------------


class TestPriorityQueueOrdering:
    """Ensure the asyncio.PriorityQueue pops oldest messages first."""

    @pytest.mark.asyncio
    async def test_queue_pops_oldest_first(self):
        q: asyncio.PriorityQueue[PendingMessage] = asyncio.PriorityQueue()

        # Insert out of order
        await q.put(PendingMessage(timestamp=300, event_id="e3", room_id="!r"))
        await q.put(PendingMessage(timestamp=100, event_id="e1", room_id="!r"))
        await q.put(PendingMessage(timestamp=200, event_id="e2", room_id="!r"))

        first = await q.get()
        second = await q.get()
        third = await q.get()

        assert first.event_id == "e1"
        assert second.event_id == "e2"
        assert third.event_id == "e3"

    @pytest.mark.asyncio
    async def test_queue_handles_mixed_rooms(self):
        q: asyncio.PriorityQueue[PendingMessage] = asyncio.PriorityQueue()

        await q.put(PendingMessage(timestamp=500, event_id="e5", room_id="!room_a"))
        await q.put(PendingMessage(timestamp=100, event_id="e1", room_id="!room_b"))
        await q.put(PendingMessage(timestamp=300, event_id="e3", room_id="!room_a"))

        results = []
        while not q.empty():
            results.append(await q.get())

        assert [m.event_id for m in results] == ["e1", "e3", "e5"]


# ---------------------------------------------------------------------------
# PendingMessage fields
# ---------------------------------------------------------------------------


class TestPendingMessageFields:
    def test_default_fields(self):
        m = PendingMessage(timestamp=42, event_id="e1", room_id="!r")
        assert m.sender_id == ""
        assert m.sender_name == ""
        assert m.text == ""
        assert m.source_bot == ""
        assert m.room_alias is None

    def test_all_fields(self):
        m = PendingMessage(
            timestamp=42,
            event_id="e1",
            room_id="!r",
            room_alias="#test:x",
            sender_id="@user:x",
            sender_name="User",
            text="hello",
            source_bot="alice",
        )
        assert m.room_alias == "#test:x"
        assert m.sender_id == "@user:x"
        assert m.text == "hello"
        assert m.source_bot == "alice"
