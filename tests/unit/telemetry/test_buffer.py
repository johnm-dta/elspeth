# tests/unit/telemetry/test_buffer.py
"""Unit tests for BoundedBuffer telemetry event buffering.

Tests cover:
- Basic append and pop_batch behavior
- Correct overflow counting (critical: check was_full BEFORE append)
- Aggregate logging every 100 drops (Warning Fatigue prevention)
- Property-based tests for buffer invariants
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.events import TelemetryEvent
from elspeth.telemetry.buffer import BoundedBuffer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed timestamp for deterministic tests."""
    return datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def base_run_id() -> str:
    """Fixed run ID for tests."""
    return "run-test-buffer"


def make_event(run_id: str, timestamp: datetime | None = None) -> TelemetryEvent:
    """Create a test TelemetryEvent."""
    return TelemetryEvent(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
    )


# =============================================================================
# Basic Behavior Tests
# =============================================================================


class TestBoundedBufferBasics:
    """Tests for basic append and pop_batch behavior."""

    def test_empty_buffer_length(self) -> None:
        """New buffer has length 0."""
        buffer = BoundedBuffer(max_size=100)
        assert len(buffer) == 0

    def test_append_increases_length(self, base_run_id: str) -> None:
        """Appending events increases buffer length."""
        buffer = BoundedBuffer(max_size=100)
        event = make_event(base_run_id)

        buffer.append(event)
        assert len(buffer) == 1

        buffer.append(event)
        assert len(buffer) == 2

    def test_pop_batch_returns_events_in_fifo_order(self, base_run_id: str) -> None:
        """pop_batch returns events in FIFO order (oldest first)."""
        buffer = BoundedBuffer(max_size=100)
        events = [make_event(f"run-{i}") for i in range(5)]

        for event in events:
            buffer.append(event)

        batch = buffer.pop_batch(max_count=5)
        assert batch == events

    def test_pop_batch_removes_events(self, base_run_id: str) -> None:
        """pop_batch removes events from the buffer."""
        buffer = BoundedBuffer(max_size=100)
        for i in range(10):
            buffer.append(make_event(f"run-{i}"))

        assert len(buffer) == 10
        batch = buffer.pop_batch(max_count=3)
        assert len(batch) == 3
        assert len(buffer) == 7

    def test_pop_batch_respects_max_count(self, base_run_id: str) -> None:
        """pop_batch returns at most max_count events."""
        buffer = BoundedBuffer(max_size=100)
        for i in range(10):
            buffer.append(make_event(f"run-{i}"))

        batch = buffer.pop_batch(max_count=3)
        assert len(batch) == 3

    def test_pop_batch_returns_all_if_fewer_than_max_count(self, base_run_id: str) -> None:
        """pop_batch returns all events if fewer than max_count available."""
        buffer = BoundedBuffer(max_size=100)
        for i in range(3):
            buffer.append(make_event(f"run-{i}"))

        batch = buffer.pop_batch(max_count=10)
        assert len(batch) == 3

    def test_pop_batch_empty_buffer(self) -> None:
        """pop_batch returns empty list on empty buffer."""
        buffer = BoundedBuffer(max_size=100)
        batch = buffer.pop_batch(max_count=10)
        assert batch == []

    def test_default_max_size(self) -> None:
        """Default max_size is 10,000."""
        buffer = BoundedBuffer()
        # Access internal deque's maxlen to verify default
        assert buffer._buffer.maxlen == 10_000

    def test_max_size_must_be_positive(self) -> None:
        """max_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BoundedBuffer(max_size=0)
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BoundedBuffer(max_size=-1)


# =============================================================================
# Overflow Counting Tests
# =============================================================================


class TestOverflowCounting:
    """Tests for correct overflow counting.

    Critical: The deque auto-evicts DURING append, so we must check
    was_full BEFORE calling append to count correctly.
    """

    def test_no_drops_when_under_capacity(self, base_run_id: str) -> None:
        """No drops when buffer is under capacity."""
        buffer = BoundedBuffer(max_size=10)
        for i in range(10):
            buffer.append(make_event(f"run-{i}"))

        assert buffer.dropped_count == 0

    def test_drops_counted_when_over_capacity(self, base_run_id: str) -> None:
        """Drops are counted when buffer exceeds capacity."""
        buffer = BoundedBuffer(max_size=5)
        for i in range(8):
            buffer.append(make_event(f"run-{i}"))

        # 8 events added to buffer of size 5 = 3 drops
        assert buffer.dropped_count == 3

    def test_dropped_count_formula(self, base_run_id: str) -> None:
        """dropped_count == max(0, events_added - max_size)."""
        max_size = 10
        events_added = 25
        buffer = BoundedBuffer(max_size=max_size)

        for i in range(events_added):
            buffer.append(make_event(f"run-{i}"))

        expected_drops = max(0, events_added - max_size)
        assert buffer.dropped_count == expected_drops

    def test_buffer_length_never_exceeds_max_size(self, base_run_id: str) -> None:
        """Buffer length is always <= max_size."""
        buffer = BoundedBuffer(max_size=5)
        for i in range(100):
            buffer.append(make_event(f"run-{i}"))

        assert len(buffer) == 5

    def test_oldest_events_dropped(self, base_run_id: str) -> None:
        """Oldest events are dropped when buffer overflows."""
        buffer = BoundedBuffer(max_size=3)
        events = [make_event(f"run-{i}") for i in range(5)]

        for event in events:
            buffer.append(event)

        # Should have events run-2, run-3, run-4 (oldest two dropped)
        batch = buffer.pop_batch(max_count=3)
        assert [e.run_id for e in batch] == ["run-2", "run-3", "run-4"]


# =============================================================================
# Aggregate Logging Tests
# =============================================================================


class TestAggregateLogging:
    """Tests for aggregate logging every 100 drops (Warning Fatigue prevention)."""

    def test_no_logging_when_no_drops(self, base_run_id: str) -> None:
        """No warning logged when there are no drops."""
        with patch("elspeth.telemetry.buffer.logger") as mock_logger:
            buffer = BoundedBuffer(max_size=10)
            for i in range(10):
                buffer.append(make_event(f"run-{i}"))

            mock_logger.warning.assert_not_called()

    def test_no_logging_for_fewer_than_100_drops(self, base_run_id: str) -> None:
        """No warning logged for fewer than 100 drops."""
        with patch("elspeth.telemetry.buffer.logger") as mock_logger:
            buffer = BoundedBuffer(max_size=10)
            # 109 events = 99 drops (not enough to trigger log)
            for i in range(109):
                buffer.append(make_event(f"run-{i}"))

            assert buffer.dropped_count == 99
            mock_logger.warning.assert_not_called()

    def test_logging_at_100_drops(self, base_run_id: str) -> None:
        """Warning logged exactly when 100 drops occur."""
        with patch("elspeth.telemetry.buffer.logger") as mock_logger:
            buffer = BoundedBuffer(max_size=10)
            # 110 events = 100 drops (triggers log)
            for i in range(110):
                buffer.append(make_event(f"run-{i}"))

            assert buffer.dropped_count == 100
            mock_logger.warning.assert_called_once()

            # Verify log message content
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "Telemetry buffer overflow - events dropped"
            assert call_args[1]["dropped_since_last_log"] == 100
            assert call_args[1]["dropped_total"] == 100
            assert call_args[1]["buffer_size"] == 10

    def test_logging_at_every_100_drops(self, base_run_id: str) -> None:
        """Warning logged at every 100 drop milestone."""
        with patch("elspeth.telemetry.buffer.logger") as mock_logger:
            buffer = BoundedBuffer(max_size=10)
            # 310 events = 300 drops (triggers 3 logs)
            for i in range(310):
                buffer.append(make_event(f"run-{i}"))

            assert buffer.dropped_count == 300
            assert mock_logger.warning.call_count == 3

    def test_logging_includes_hint(self, base_run_id: str) -> None:
        """Warning log includes actionable hint."""
        with patch("elspeth.telemetry.buffer.logger") as mock_logger:
            buffer = BoundedBuffer(max_size=10)
            for i in range(110):
                buffer.append(make_event(f"run-{i}"))

            call_args = mock_logger.warning.call_args
            assert "hint" in call_args[1]
            assert "buffer size" in call_args[1]["hint"].lower()

    def test_log_interval_constant(self) -> None:
        """LOG_INTERVAL is 100."""
        assert BoundedBuffer._LOG_INTERVAL == 100


# =============================================================================
# Property-Based Tests (Hypothesis)
# =============================================================================


class TestBoundedBufferProperties:
    """Property-based tests for buffer invariants using Hypothesis."""

    @given(
        max_size=st.integers(min_value=1, max_value=1000),
        num_events=st.integers(min_value=0, max_value=2000),
    )
    @settings(max_examples=100)
    def test_length_never_exceeds_max_size(self, max_size: int, num_events: int) -> None:
        """Property: len(buffer) <= max_size always."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(make_event(f"run-{i}"))

        assert len(buffer) <= max_size

    @given(
        max_size=st.integers(min_value=1, max_value=1000),
        num_events=st.integers(min_value=0, max_value=2000),
    )
    @settings(max_examples=100)
    def test_dropped_count_formula_holds(self, max_size: int, num_events: int) -> None:
        """Property: dropped_count == max(0, events_added - max_size)."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(make_event(f"run-{i}"))

        expected_drops = max(0, num_events - max_size)
        assert buffer.dropped_count == expected_drops

    @given(
        max_size=st.integers(min_value=1, max_value=100),
        num_events=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=100)
    def test_length_plus_drops_equals_events_added(self, max_size: int, num_events: int) -> None:
        """Property: len(buffer) + dropped_count == events_added when no pops."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(make_event(f"run-{i}"))

        # For n events: either all fit (len=n, drops=0) or buffer is full (len=max, drops=n-max)
        # In both cases: len + drops = n
        assert len(buffer) + buffer.dropped_count == num_events

    @given(
        max_size=st.integers(min_value=1, max_value=100),
        num_events=st.integers(min_value=0, max_value=200),
        pop_count=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=100)
    def test_pop_batch_returns_at_most_available(self, max_size: int, num_events: int, pop_count: int) -> None:
        """Property: pop_batch returns at most min(pop_count, len(buffer)) events."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(make_event(f"run-{i}"))

        initial_length = len(buffer)
        batch = buffer.pop_batch(max_count=pop_count)

        assert len(batch) <= pop_count
        assert len(batch) <= initial_length
        assert len(batch) == min(pop_count, initial_length)

    @given(
        max_size=st.integers(min_value=1, max_value=100),
        num_events=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=100)
    def test_buffer_preserves_fifo_order(self, max_size: int, num_events: int) -> None:
        """Property: Events are returned in FIFO order (oldest first)."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(make_event(f"run-{i}"))

        batch = buffer.pop_batch(max_count=len(buffer))

        # The batch should be the last max_size events in order
        dropped = max(0, num_events - max_size)
        expected_run_ids = [f"run-{i}" for i in range(dropped, num_events)]
        actual_run_ids = [e.run_id for e in batch]

        assert actual_run_ids == expected_run_ids


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_buffer_size_one(self, base_run_id: str) -> None:
        """Buffer with max_size=1 works correctly."""
        buffer = BoundedBuffer(max_size=1)
        events = [make_event(f"run-{i}") for i in range(5)]

        for event in events:
            buffer.append(event)

        assert len(buffer) == 1
        assert buffer.dropped_count == 4

        batch = buffer.pop_batch(max_count=10)
        assert len(batch) == 1
        assert batch[0].run_id == "run-4"

    def test_pop_batch_zero_count(self, base_run_id: str) -> None:
        """pop_batch with max_count=0 returns empty list."""
        buffer = BoundedBuffer(max_size=10)
        buffer.append(make_event(base_run_id))

        batch = buffer.pop_batch(max_count=0)
        assert batch == []
        assert len(buffer) == 1

    def test_multiple_pop_batches(self, base_run_id: str) -> None:
        """Multiple pop_batch calls work correctly."""
        buffer = BoundedBuffer(max_size=100)
        for i in range(10):
            buffer.append(make_event(f"run-{i}"))

        batch1 = buffer.pop_batch(max_count=3)
        batch2 = buffer.pop_batch(max_count=3)
        batch3 = buffer.pop_batch(max_count=10)

        assert [e.run_id for e in batch1] == ["run-0", "run-1", "run-2"]
        assert [e.run_id for e in batch2] == ["run-3", "run-4", "run-5"]
        assert [e.run_id for e in batch3] == ["run-6", "run-7", "run-8", "run-9"]
        assert len(buffer) == 0

    def test_interleaved_append_and_pop(self, base_run_id: str) -> None:
        """Interleaved append and pop operations work correctly."""
        buffer = BoundedBuffer(max_size=5)

        # Add 3
        for i in range(3):
            buffer.append(make_event(f"run-{i}"))
        assert len(buffer) == 3

        # Pop 2
        batch = buffer.pop_batch(max_count=2)
        assert len(batch) == 2
        assert len(buffer) == 1

        # Add 6 more (buffer size 5, so should overflow)
        for i in range(3, 9):
            buffer.append(make_event(f"run-{i}"))

        # Buffer had 1, added 6, max is 5 -> dropped 2
        assert len(buffer) == 5
        assert buffer.dropped_count == 2
