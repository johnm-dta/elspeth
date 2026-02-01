# tests/plugins/batching/test_row_reorder_buffer.py
"""Tests for RowReorderBuffer - the core FIFO ordering component."""

from __future__ import annotations

import threading
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.plugins.batching.row_reorder_buffer import (
    RowReorderBuffer,
    ShutdownError,
)


class TestRowReorderBufferBasics:
    """Basic functionality tests."""

    def test_single_row_submit_complete_release(self):
        """Single row flows through correctly."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        ticket = buffer.submit("row-1")
        assert ticket.sequence == 0
        assert ticket.row_id == "row-1"

        buffer.complete(ticket, "result-1")

        entry = buffer.wait_for_next_release(timeout=1.0)
        assert entry.result == "result-1"
        assert entry.row_id == "row-1"
        assert entry.sequence == 0

    def test_fifo_ordering_sequential_complete(self):
        """Rows completed in order are released in order."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        t1 = buffer.submit("row-1")
        t2 = buffer.submit("row-2")
        t3 = buffer.submit("row-3")

        buffer.complete(t1, "result-1")
        buffer.complete(t2, "result-2")
        buffer.complete(t3, "result-3")

        assert buffer.wait_for_next_release(timeout=1.0).result == "result-1"
        assert buffer.wait_for_next_release(timeout=1.0).result == "result-2"
        assert buffer.wait_for_next_release(timeout=1.0).result == "result-3"

    def test_fifo_ordering_reverse_complete(self):
        """Rows completed in reverse order still release in submission order."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        t1 = buffer.submit("row-1")
        t2 = buffer.submit("row-2")
        t3 = buffer.submit("row-3")

        # Complete in REVERSE order
        buffer.complete(t3, "result-3")
        buffer.complete(t2, "result-2")
        buffer.complete(t1, "result-1")

        # Still release in submission order
        assert buffer.wait_for_next_release(timeout=1.0).result == "result-1"
        assert buffer.wait_for_next_release(timeout=1.0).result == "result-2"
        assert buffer.wait_for_next_release(timeout=1.0).result == "result-3"

    def test_metrics_accurate(self):
        """Metrics reflect actual buffer state."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        assert buffer.pending_count == 0

        t1 = buffer.submit("row-1")
        assert buffer.pending_count == 1

        t2 = buffer.submit("row-2")
        assert buffer.pending_count == 2

        buffer.complete(t2, "result-2")  # Complete out of order
        metrics = buffer.get_metrics()
        assert metrics["completed_waiting"] == 1

        buffer.complete(t1, "result-1")
        buffer.wait_for_next_release(timeout=1.0)
        assert buffer.pending_count == 1


class TestBackpressure:
    """Backpressure tests."""

    def test_submit_blocks_when_full(self):
        """Submit blocks when max_pending reached."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=2)

        t1 = buffer.submit("row-1")
        buffer.submit("row-2")

        # Third submit should block
        submit_completed = threading.Event()

        def try_submit():
            buffer.submit("row-3")
            submit_completed.set()

        thread = threading.Thread(target=try_submit)
        thread.start()

        # Should not complete immediately (blocked)
        assert not submit_completed.wait(timeout=0.2)

        # Release one row
        buffer.complete(t1, "result-1")
        buffer.wait_for_next_release(timeout=1.0)

        # Now submit should complete
        assert submit_completed.wait(timeout=1.0)
        thread.join()

    def test_submit_timeout(self):
        """Submit times out if buffer stays full."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=1)

        buffer.submit("row-1")

        with pytest.raises(TimeoutError, match="buffer space"):
            buffer.submit("row-2", timeout=0.1)


class TestShutdown:
    """Shutdown and error handling tests."""

    def test_shutdown_wakes_submit_waiters(self):
        """Shutdown wakes threads blocked on submit."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=1)

        buffer.submit("row-1")

        exception_caught = threading.Event()
        caught_exception = [None]

        def try_submit():
            try:
                buffer.submit("row-2")
            except ShutdownError as e:
                caught_exception[0] = e
                exception_caught.set()

        thread = threading.Thread(target=try_submit)
        thread.start()

        time.sleep(0.1)  # Let thread block
        buffer.shutdown()

        assert exception_caught.wait(timeout=1.0)
        assert "shut down" in str(caught_exception[0])
        thread.join()

    def test_shutdown_wakes_release_waiters(self):
        """Shutdown wakes threads blocked on wait_for_next_release."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        exception_caught = threading.Event()

        def try_release():
            try:
                buffer.wait_for_next_release()
            except ShutdownError:
                exception_caught.set()

        thread = threading.Thread(target=try_release)
        thread.start()

        time.sleep(0.1)  # Let thread block
        buffer.shutdown()

        assert exception_caught.wait(timeout=1.0)
        thread.join()

    def test_double_complete_raises(self):
        """Completing the same ticket twice raises ValueError."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        ticket = buffer.submit("row-1")
        buffer.complete(ticket, "result-1")

        with pytest.raises(ValueError, match="already completed"):
            buffer.complete(ticket, "result-2")


class TestConcurrency:
    """Concurrent access tests."""

    def test_concurrent_complete_fifo_maintained(self):
        """FIFO ordering maintained with concurrent completions."""
        buffer: RowReorderBuffer[int] = RowReorderBuffer(max_pending=100)
        num_rows = 50

        # Submit all tickets
        tickets = [buffer.submit(f"row-{i}") for i in range(num_rows)]

        # Complete from multiple threads in random order
        import random

        def completer(indices: list[int]):
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for i in shuffled:
                time.sleep(random.uniform(0.0001, 0.001))
                buffer.complete(tickets[i], i)

        # Split indices across threads
        threads = []
        chunk_size = num_rows // 5
        for t in range(5):
            start = t * chunk_size
            end = start + chunk_size if t < 4 else num_rows
            indices = list(range(start, end))
            thread = threading.Thread(target=completer, args=(indices,))
            threads.append(thread)
            thread.start()

        # Release and verify order
        results = []
        for _ in range(num_rows):
            entry = buffer.wait_for_next_release(timeout=10.0)
            results.append(entry.result)

        for t in threads:
            t.join()

        # Results should be in submission order (0, 1, 2, ...)
        assert results == list(range(num_rows))


class TestEviction:
    """Tests for evicting abandoned entries (timeout/retry scenarios)."""

    def test_evict_removes_entry_and_advances_release_sequence(self):
        """Evicting an entry allows subsequent entries to be released.

        This is the core retry scenario:
        1. Row A submitted (seq 0)
        2. Row A times out, retry happens
        3. Row A retry submitted (seq 1)
        4. Row A retry completes
        5. Without eviction, seq 1 is blocked waiting for seq 0
        6. With eviction, seq 0 is removed and seq 1 can be released
        """
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        # Original attempt
        ticket_original = buffer.submit("row-A-original")
        assert ticket_original.sequence == 0

        # Retry attempt (gets new sequence)
        ticket_retry = buffer.submit("row-A-retry")
        assert ticket_retry.sequence == 1

        # Retry completes
        buffer.complete(ticket_retry, "retry-result")

        # Without eviction, release would block forever waiting for seq 0
        # With eviction, we can skip seq 0 and release seq 1
        evicted = buffer.evict(ticket_original)
        assert evicted is True

        # Now seq 1 should be releasable
        entry = buffer.wait_for_next_release(timeout=1.0)
        assert entry.result == "retry-result"
        assert entry.sequence == 1

    def test_evict_already_completed_returns_false(self):
        """Evicting an already-completed entry returns False."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        ticket = buffer.submit("row-1")
        buffer.complete(ticket, "result-1")

        # Already completed - evict should fail
        assert buffer.evict(ticket) is False

    def test_evict_already_released_returns_false(self):
        """Evicting an already-released entry returns False."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        ticket = buffer.submit("row-1")
        buffer.complete(ticket, "result-1")
        buffer.wait_for_next_release(timeout=1.0)

        # Already released - evict should fail (not in pending)
        assert buffer.evict(ticket) is False

    def test_evict_advances_past_multiple_evicted(self):
        """Evicting multiple entries advances past all of them."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=10)

        t0 = buffer.submit("row-0")
        t1 = buffer.submit("row-1")
        t2 = buffer.submit("row-2")
        t3 = buffer.submit("row-3")

        # Complete only t3
        buffer.complete(t3, "result-3")

        # Evict t0, t1, t2
        assert buffer.evict(t0) is True
        assert buffer.evict(t1) is True
        assert buffer.evict(t2) is True

        # t3 should now be releasable
        entry = buffer.wait_for_next_release(timeout=1.0)
        assert entry.result == "result-3"
        assert entry.sequence == 3

    def test_evict_releases_backpressure(self):
        """Evicting an entry frees up a slot for new submissions."""
        buffer: RowReorderBuffer[str] = RowReorderBuffer(max_pending=2)

        t0 = buffer.submit("row-0")
        buffer.submit("row-1")  # Fill buffer, t1 not needed

        # Buffer is full - third submit would block
        # Evict t0 to free up space
        buffer.evict(t0)

        # Now we should be able to submit another row
        t2 = buffer.submit("row-2", timeout=0.1)  # Would timeout if still full
        assert t2.sequence == 2


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(completion_order=st.permutations(range(20)))
    @settings(max_examples=50, deadline=10000)
    def test_fifo_invariant_any_completion_order(self, completion_order: list[int]):
        """PROPERTY: Release order always matches submission order."""
        buffer: RowReorderBuffer[int] = RowReorderBuffer(max_pending=20)

        # Submit all
        tickets = [buffer.submit(f"row-{i}") for i in range(20)]

        # Complete in given order
        for i in completion_order:
            buffer.complete(tickets[i], i)

        # Release all
        results = []
        for _ in range(20):
            entry = buffer.wait_for_next_release(timeout=5.0)
            results.append(entry.result)

        # INVARIANT: results == [0, 1, 2, ..., 19]
        assert results == list(range(20))
