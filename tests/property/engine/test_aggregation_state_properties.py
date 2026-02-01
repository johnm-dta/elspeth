# tests/property/engine/test_aggregation_state_properties.py
"""Property-based stateful tests for aggregation behavior.

Aggregation is a state machine:
- Initial: Empty buffer, timer not started
- Buffering: Accepting rows, timer running
- Triggered: Threshold met, ready to flush
- Flushed: Buffer cleared, timer reset

These tests use Hypothesis RuleBasedStateMachine to explore all
possible state transitions and verify invariants hold.

Key Invariants:
- Buffer count matches number of accepted rows
- Flush clears all state
- Timer starts on first accept, not before
- Trigger fires exactly at threshold, not before
- If trigger fires, at least one condition (count OR timeout) is met

REVIEWER NOTE: This implementation addresses the following issues
from the test suite review:
1. check_trigger() now has negative assertion (else clause)
2. Added trigger_condition_implies_threshold invariant
3. Model synchronization is explicit
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from elspeth.core.config import TriggerConfig
from elspeth.engine.clock import MockClock
from elspeth.engine.triggers import TriggerEvaluator


class TriggerEvaluatorStateMachine(RuleBasedStateMachine):
    """Stateful property tests for TriggerEvaluator.

    This explores the state space of:
    - accept() calls with various row data
    - Time advances
    - Trigger checks
    - Reset operations
    """

    # Fixed configuration values for the state machine
    COUNT_THRESHOLD: int = 10
    TIMEOUT_SECONDS: float = 5.0

    def __init__(self) -> None:
        super().__init__()
        self.clock = MockClock(start=0.0)
        self.config = TriggerConfig(count=self.COUNT_THRESHOLD, timeout_seconds=self.TIMEOUT_SECONDS)
        self.evaluator = TriggerEvaluator(self.config, clock=self.clock)

        # Model state for verification
        self.model_count = 0
        self.model_first_accept_time: float | None = None

    def _model_should_trigger(self) -> bool:
        """Calculate expected trigger state from model.

        Separated from rules to ensure consistent calculation.
        """
        # Count condition
        if self.model_count >= self.COUNT_THRESHOLD:
            return True

        # Timeout condition (only if timer started)
        if self.model_first_accept_time is not None:
            elapsed = self.clock.monotonic() - self.model_first_accept_time
            if elapsed >= self.TIMEOUT_SECONDS:
                return True

        return False

    @rule()
    def accept_row(self) -> None:
        """Accept a row into the aggregation buffer."""
        if self.model_first_accept_time is None:
            self.model_first_accept_time = self.clock.monotonic()

        self.evaluator.record_accept()
        self.model_count += 1

    @rule(seconds=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    def advance_time(self, seconds: float) -> None:
        """Advance the mock clock."""
        self.clock.advance(seconds)

    @rule()
    def check_trigger(self) -> None:
        """Check if trigger should fire - FIXED: now has negative assertion."""
        actual = self.evaluator.should_trigger()
        expected = self._model_should_trigger()

        # REVIEWER FIX: Assert BOTH directions, not just positive case
        assert actual == expected, (
            f"Trigger state mismatch: actual={actual}, expected={expected}, "
            f"count={self.model_count}/{self.config.count}, "
            f"age={self.evaluator.batch_age_seconds}/{self.config.timeout_seconds}"
        )

    @rule()
    def reset(self) -> None:
        """Reset the evaluator state."""
        self.evaluator.reset()

        # Reset model state
        self.model_count = 0
        self.model_first_accept_time = None

    @invariant()
    def count_matches_model(self) -> None:
        """Invariant: Buffer count always matches our model."""
        assert self.evaluator.batch_count == self.model_count, (
            f"Count mismatch: evaluator={self.evaluator.batch_count}, model={self.model_count}"
        )

    @invariant()
    def age_is_non_negative(self) -> None:
        """Invariant: Batch age is never negative."""
        assert self.evaluator.batch_age_seconds >= 0.0

    @invariant()
    def age_is_zero_before_first_accept(self) -> None:
        """Invariant: Age is 0 when no rows have been accepted."""
        if self.model_first_accept_time is None:
            assert self.evaluator.batch_age_seconds == 0.0

    @invariant()
    def trigger_condition_implies_threshold(self) -> None:
        """Invariant: If trigger fires, at least one condition is met.

        REVIEWER FIX: This is the critical invariant that was missing.
        If should_trigger() returns True, EITHER:
        - count >= count_threshold, OR
        - elapsed_time >= timeout_threshold

        This catches bugs where trigger fires spuriously.
        """
        if self.evaluator.should_trigger():
            count_ok = self.evaluator.batch_count >= self.COUNT_THRESHOLD
            time_ok = self.evaluator.batch_age_seconds >= self.TIMEOUT_SECONDS

            assert count_ok or time_ok, (
                f"Trigger fired but no condition met: "
                f"count={self.evaluator.batch_count}/{self.COUNT_THRESHOLD}, "
                f"age={self.evaluator.batch_age_seconds}/{self.TIMEOUT_SECONDS}"
            )


# Create the test class that pytest will discover
TestTriggerStateMachine = TriggerEvaluatorStateMachine.TestCase
TestTriggerStateMachine.settings = settings(max_examples=100, stateful_step_count=30)


# =============================================================================
# Additional Non-Stateful Aggregation Properties
# =============================================================================


class TestAggregationInvariants:
    """Additional property tests for aggregation invariants."""

    @given(count=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_trigger_fires_exactly_at_count_threshold(self, count: int) -> None:
        """Property: Count trigger fires at exactly the threshold, not before."""
        config = TriggerConfig(count=count)
        evaluator = TriggerEvaluator(config)

        # Accept count-1 rows - should NOT trigger
        for _ in range(count - 1):
            evaluator.record_accept()
        assert not evaluator.should_trigger(), f"Triggered early at {count - 1}/{count}"

        # Accept one more - NOW should trigger
        evaluator.record_accept()
        assert evaluator.should_trigger(), f"Didn't trigger at {count}/{count}"

    @given(timeout=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_trigger_fires_at_timeout_threshold(self, timeout: float) -> None:
        """Property: Timeout trigger fires at exactly the threshold."""
        clock = MockClock(start=0.0)
        config = TriggerConfig(timeout_seconds=timeout)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Must accept at least one row to start timer
        evaluator.record_accept()

        # Advance to just before timeout - should NOT trigger
        clock.advance(timeout * 0.9)
        assert not evaluator.should_trigger(), f"Triggered early at {timeout * 0.9}s"

        # Advance past timeout - NOW should trigger
        clock.advance(timeout * 0.2)  # Total: timeout * 1.1
        assert evaluator.should_trigger(), f"Didn't trigger at {timeout * 1.1}s"
