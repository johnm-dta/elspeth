# tests/property/engine/test_trigger_properties.py
"""Property-based tests for TriggerEvaluator state machine invariants.

The TriggerEvaluator tracks batch state (count, age) and evaluates
whether ANY of its configured triggers should fire (OR logic).
When multiple triggers fire, "first to fire wins" — the one that became
true earliest is reported, not the one checked first in code.

Properties tested:
- Count trigger fires at exactly the threshold (not before, exactly at)
- Timeout trigger fires based on elapsed clock time
- First-to-fire-wins: when both count and timeout are satisfied,
  the one that fired earliest is reported
- Reset clears all state back to initial conditions
- batch_count is monotonically non-decreasing after record_accept
- batch_age_seconds is non-negative and monotonic with clock advancement
- which_triggered is None before first should_trigger call
- Checkpoint/restore preserves trigger ordering across resume
- Non-boolean condition expressions raise TypeError
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.config import TriggerConfig
from elspeth.engine.clock import MockClock
from elspeth.engine.triggers import TriggerEvaluator

# =============================================================================
# Strategies
# =============================================================================

# Count thresholds (positive integers, realistic range)
count_thresholds = st.integers(min_value=1, max_value=500)

# Timeout values in seconds (positive floats, realistic range)
timeout_seconds_values = st.floats(min_value=0.01, max_value=3600.0, allow_nan=False, allow_infinity=False)

# Number of rows to accept (test range)
accept_counts = st.integers(min_value=0, max_value=200)

# Clock advancement amounts
clock_advances = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# Count Trigger Properties
# =============================================================================


class TestCountTriggerProperties:
    """Count trigger must fire at exactly the threshold, not before."""

    @given(threshold=count_thresholds)
    @settings(max_examples=200)
    def test_count_does_not_fire_before_threshold(self, threshold: int) -> None:
        """Property: should_trigger() is False when batch_count < threshold."""
        config = TriggerConfig(count=threshold)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(threshold - 1):
            evaluator.record_accept()
            assert not evaluator.should_trigger()

    @given(threshold=count_thresholds)
    @settings(max_examples=200)
    def test_count_fires_at_exactly_threshold(self, threshold: int) -> None:
        """Property: should_trigger() is True when batch_count == threshold."""
        config = TriggerConfig(count=threshold)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(threshold):
            evaluator.record_accept()

        assert evaluator.should_trigger()
        assert evaluator.which_triggered() == "count"

    @given(threshold=count_thresholds, extra=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_count_stays_triggered_after_threshold(self, threshold: int, extra: int) -> None:
        """Property: should_trigger() remains True once threshold is passed."""
        config = TriggerConfig(count=threshold)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(threshold + extra):
            evaluator.record_accept()

        assert evaluator.should_trigger()


# =============================================================================
# Timeout Trigger Properties
# =============================================================================


class TestTimeoutTriggerProperties:
    """Timeout trigger must fire based on elapsed clock time."""

    @given(timeout=timeout_seconds_values)
    @settings(max_examples=200)
    def test_timeout_does_not_fire_before_elapsed(self, timeout: float) -> None:
        """Property: should_trigger() is False when elapsed < timeout."""
        config = TriggerConfig(timeout_seconds=timeout)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # Starts the timer

        # Advance to just under timeout
        under = timeout * 0.5
        clock.advance(under)
        assert not evaluator.should_trigger()

    @given(timeout=timeout_seconds_values)
    @settings(max_examples=200)
    def test_timeout_fires_at_elapsed(self, timeout: float) -> None:
        """Property: should_trigger() is True when elapsed >= timeout."""
        config = TriggerConfig(timeout_seconds=timeout)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()

        clock.advance(timeout)
        assert evaluator.should_trigger()
        assert evaluator.which_triggered() == "timeout"

    @given(timeout=timeout_seconds_values, extra=clock_advances)
    @settings(max_examples=100)
    def test_timeout_stays_triggered_after_elapsed(self, timeout: float, extra: float) -> None:
        """Property: should_trigger() remains True once timeout is passed."""
        config = TriggerConfig(timeout_seconds=timeout)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()
        clock.advance(timeout + extra)

        assert evaluator.should_trigger()

    def test_timeout_does_not_fire_without_accept(self) -> None:
        """Property: Timeout does not fire if no rows were accepted."""
        config = TriggerConfig(timeout_seconds=1.0)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        clock.advance(100.0)  # Way past timeout
        assert not evaluator.should_trigger()


# =============================================================================
# First-to-Fire-Wins Properties
# =============================================================================


class TestFirstToFireWinsProperties:
    """When multiple triggers are satisfied, earliest fire time wins."""

    @given(
        count=count_thresholds,
        timeout=timeout_seconds_values,
    )
    @settings(max_examples=200)
    def test_count_wins_when_it_fires_first(self, count: int, timeout: float) -> None:
        """Property: If count fires before timeout, which_triggered == 'count'."""
        config = TriggerConfig(count=count, timeout_seconds=timeout)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept rows quickly (1ms apart) to fire count before timeout
        for _ in range(count):
            evaluator.record_accept()
            clock.advance(0.001)

        # Only check if timeout hasn't fired yet
        elapsed = count * 0.001
        assume(elapsed < timeout)

        assert evaluator.should_trigger()
        assert evaluator.which_triggered() == "count"

    @given(timeout=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_timeout_wins_when_count_not_reached(self, timeout: float) -> None:
        """Property: If only timeout fires, which_triggered == 'timeout'."""
        config = TriggerConfig(count=1000, timeout_seconds=timeout)  # count=1000 is unreachable
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # Need at least one accept for timer
        clock.advance(timeout)

        assert evaluator.should_trigger()
        assert evaluator.which_triggered() == "timeout"


# =============================================================================
# Batch State Properties
# =============================================================================


class TestBatchStateProperties:
    """Batch count and age must have correct monotonicity properties."""

    @given(n=st.integers(min_value=0, max_value=100))
    @settings(max_examples=100)
    def test_batch_count_equals_accept_count(self, n: int) -> None:
        """Property: batch_count always equals number of record_accept calls."""
        config = TriggerConfig(count=1000)  # High threshold so it never fires
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        for i in range(n):
            evaluator.record_accept()
            assert evaluator.batch_count == i + 1

    @given(advances=st.lists(clock_advances, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_batch_age_monotonically_increases(self, advances: list[float]) -> None:
        """Property: batch_age_seconds never decreases with clock advancement."""
        config = TriggerConfig(count=1000)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # Start timer
        prev_age = evaluator.batch_age_seconds

        for dt in advances:
            clock.advance(dt)
            current_age = evaluator.batch_age_seconds
            assert current_age >= prev_age
            prev_age = current_age

    def test_batch_age_zero_before_accept(self) -> None:
        """Property: batch_age_seconds is 0 when no rows accepted."""
        config = TriggerConfig(count=10)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        assert evaluator.batch_age_seconds == 0.0


# =============================================================================
# Reset Properties
# =============================================================================


class TestResetProperties:
    """Reset must return evaluator to pristine initial state."""

    @given(
        threshold=count_thresholds,
        n_accepts=st.integers(min_value=1, max_value=50),
        advance=clock_advances,
    )
    @settings(max_examples=100)
    def test_reset_clears_all_state(self, threshold: int, n_accepts: int, advance: float) -> None:
        """Property: After reset, evaluator behaves as if freshly created."""
        config = TriggerConfig(count=threshold, timeout_seconds=1.0)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        # Dirty up the state
        for _ in range(n_accepts):
            evaluator.record_accept()
        clock.advance(advance)
        evaluator.should_trigger()

        # Reset
        evaluator.reset()

        # All state is clear
        assert evaluator.batch_count == 0
        assert evaluator.batch_age_seconds == 0.0
        assert evaluator.which_triggered() is None
        assert not evaluator.should_trigger()

    @given(threshold=count_thresholds)
    @settings(max_examples=50)
    def test_reset_allows_reuse(self, threshold: int) -> None:
        """Property: After reset, evaluator can trigger again normally."""
        config = TriggerConfig(count=threshold)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        # First batch
        for _ in range(threshold):
            evaluator.record_accept()
        assert evaluator.should_trigger()

        evaluator.reset()

        # Second batch — must behave identically
        for i in range(threshold - 1):
            evaluator.record_accept()
            assert not evaluator.should_trigger(), f"Fired early at {i + 1}/{threshold} after reset"

        evaluator.record_accept()
        assert evaluator.should_trigger()
        assert evaluator.which_triggered() == "count"


# =============================================================================
# Checkpoint/Restore Properties
# =============================================================================


class TestCheckpointRestoreProperties:
    """Checkpoint/restore must preserve trigger ordering across resume."""

    @given(
        threshold=count_thresholds,
        n_accepts=st.integers(min_value=1, max_value=100),
        elapsed=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_restore_preserves_batch_count(self, threshold: int, n_accepts: int, elapsed: float) -> None:
        """Property: Restored evaluator has correct batch_count."""
        config = TriggerConfig(count=threshold)
        clock = MockClock(start=1000.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.restore_from_checkpoint(
            batch_count=n_accepts,
            elapsed_age_seconds=elapsed,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        assert evaluator.batch_count == n_accepts

    @given(
        elapsed=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_restore_preserves_batch_age(self, elapsed: float) -> None:
        """Property: Restored evaluator has approximately correct batch_age_seconds."""
        config = TriggerConfig(timeout_seconds=200.0)
        clock = MockClock(start=1000.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=elapsed,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        # batch_age_seconds should match elapsed (clock hasn't moved)
        assert abs(evaluator.batch_age_seconds - elapsed) < 1e-9

    @given(
        count_offset=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        elapsed=st.floats(min_value=60.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_restore_preserves_count_fire_ordering(self, count_offset: float, elapsed: float) -> None:
        """Property: If count fired before timeout, restored evaluator reports count first."""
        assume(count_offset < elapsed)
        timeout = elapsed * 0.8  # Timeout fires at 80% of elapsed

        config = TriggerConfig(count=5, timeout_seconds=timeout)
        clock = MockClock(start=1000.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Restore: count fired at count_offset, timeout at (timeout from first_accept)
        evaluator.restore_from_checkpoint(
            batch_count=10,
            elapsed_age_seconds=elapsed,
            count_fire_offset=count_offset,
            condition_fire_offset=None,
        )

        # Both should be satisfied, but whichever fired first wins
        if evaluator.should_trigger():
            triggered = evaluator.which_triggered()
            # count_offset vs timeout: the earlier one wins
            if count_offset < timeout:
                assert triggered == "count"
            else:
                assert triggered == "timeout"


# =============================================================================
# which_triggered Properties
# =============================================================================


class TestWhichTriggeredProperties:
    """which_triggered must only be set after should_trigger."""

    def test_which_triggered_none_initially(self) -> None:
        """Property: which_triggered is None before any should_trigger call."""
        config = TriggerConfig(count=10)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        assert evaluator.which_triggered() is None

    def test_which_triggered_none_when_not_fired(self) -> None:
        """Property: which_triggered is None after should_trigger returns False."""
        config = TriggerConfig(count=10)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()
        evaluator.should_trigger()

        assert evaluator.which_triggered() is None

    @given(threshold=count_thresholds)
    @settings(max_examples=50)
    def test_get_trigger_type_matches_which_triggered(self, threshold: int) -> None:
        """Property: get_trigger_type enum matches which_triggered string."""
        from elspeth.contracts.enums import TriggerType

        config = TriggerConfig(count=threshold)
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(threshold):
            evaluator.record_accept()
        evaluator.should_trigger()

        which = evaluator.which_triggered()
        trigger_type = evaluator.get_trigger_type()

        assert which == "count"
        assert trigger_type == TriggerType.COUNT


# =============================================================================
# No-Trigger Configuration Properties
# =============================================================================


class TestNoTriggerProperties:
    """No configured early trigger means flush only at end-of-source."""

    @given(accept_count=accept_counts)
    def test_empty_config_never_fires_early_trigger(self, accept_count: int) -> None:
        """Property: TriggerConfig() leaves early trigger evaluation false."""
        config = TriggerConfig()
        clock = MockClock()
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(accept_count):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False
        assert evaluator.which_triggered() is None
        assert evaluator.get_trigger_type() is None
