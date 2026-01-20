# tests/plugins/llm/test_reorder_buffer.py
"""Tests for reorder buffer that maintains submission order."""

import time

from elspeth.plugins.llm.reorder_buffer import ReorderBuffer


class TestReorderBufferBasic:
    """Test basic reorder buffer operations."""

    def test_empty_buffer_has_no_ready_results(self) -> None:
        """Empty buffer should have no ready results."""
        buffer = ReorderBuffer[str]()

        assert buffer.get_ready_results() == []
        assert buffer.pending_count == 0

    def test_single_result_emitted_immediately(self) -> None:
        """Single result should be available immediately."""
        buffer = ReorderBuffer[str]()

        idx = buffer.submit()
        buffer.complete(idx, "result_0")

        results = buffer.get_ready_results()
        assert len(results) == 1
        assert results[0].result == "result_0"
        assert buffer.pending_count == 0


class TestReorderBufferOrdering:
    """Test that results are emitted in submission order."""

    def test_out_of_order_completion_reordered(self) -> None:
        """Results completing out of order should be emitted in order."""
        buffer = ReorderBuffer[str]()

        # Submit 5 items
        indices = [buffer.submit() for _ in range(5)]
        assert indices == [0, 1, 2, 3, 4]

        # Complete in order: 2, 0, 4, 1, 3
        buffer.complete(2, "result_2")
        assert buffer.get_ready_results() == []  # Can't emit yet

        buffer.complete(0, "result_0")
        ready = buffer.get_ready_results()
        assert len(ready) == 1
        assert ready[0].result == "result_0"

        buffer.complete(4, "result_4")
        assert buffer.get_ready_results() == []  # Still waiting for 1

        buffer.complete(1, "result_1")
        # Now 1 and 2 can be emitted
        ready = buffer.get_ready_results()
        assert len(ready) == 2
        assert ready[0].result == "result_1"
        assert ready[1].result == "result_2"

        buffer.complete(3, "result_3")
        # Now 3 and 4 can be emitted
        ready = buffer.get_ready_results()
        assert len(ready) == 2
        assert ready[0].result == "result_3"
        assert ready[1].result == "result_4"

        assert buffer.pending_count == 0

    def test_in_order_completion_immediate(self) -> None:
        """Results completing in order should emit immediately."""
        buffer = ReorderBuffer[str]()

        for i in range(3):
            idx = buffer.submit()
            buffer.complete(idx, f"result_{i}")
            ready = buffer.get_ready_results()
            assert len(ready) == 1
            assert ready[0].result == f"result_{i}"

        assert buffer.pending_count == 0


class TestReorderBufferTiming:
    """Test timing metadata for audit trail."""

    def test_entry_has_submit_timestamp(self) -> None:
        """Buffer entries should record submit timestamp."""
        buffer = ReorderBuffer[str]()

        before = time.perf_counter()
        idx = buffer.submit()
        after = time.perf_counter()

        buffer.complete(idx, "result")
        ready = buffer.get_ready_results()

        assert len(ready) == 1
        assert before <= ready[0].submit_timestamp <= after

    def test_entry_has_complete_timestamp(self) -> None:
        """Buffer entries should record complete timestamp."""
        buffer = ReorderBuffer[str]()

        idx = buffer.submit()
        time.sleep(0.01)  # Small delay

        before = time.perf_counter()
        buffer.complete(idx, "result")
        after = time.perf_counter()

        ready = buffer.get_ready_results()

        assert len(ready) == 1
        assert before <= ready[0].complete_timestamp <= after
        # Complete should be after submit
        assert ready[0].complete_timestamp > ready[0].submit_timestamp

    def test_entry_tracks_buffer_wait_time(self) -> None:
        """Entry should track time spent waiting in buffer."""
        buffer = ReorderBuffer[str]()

        # Submit two items
        idx0 = buffer.submit()
        idx1 = buffer.submit()

        # Complete second first (will wait in buffer)
        buffer.complete(idx1, "result_1")
        time.sleep(0.02)  # Wait while 1 is buffered

        # Complete first (releases both)
        buffer.complete(idx0, "result_0")
        ready = buffer.get_ready_results()

        assert len(ready) == 2
        # First item shouldn't have waited much
        assert ready[0].buffer_wait_ms < 50
        # Second item waited while first was pending
        assert ready[1].buffer_wait_ms >= 15  # At least 20ms minus some tolerance
