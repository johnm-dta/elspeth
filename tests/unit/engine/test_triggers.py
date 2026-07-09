# tests/unit/engine/test_triggers.py
"""Tests for TriggerEvaluator."""


class TestTriggerEvaluator:
    """Tests for TriggerEvaluator class.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    The evaluator checks ALL configured triggers with OR logic.
    """

    def test_count_trigger_not_reached(self) -> None:
        """Count trigger returns False when threshold not reached."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        for _ in range(50):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False

    def test_count_trigger_reached(self) -> None:
        """Count trigger returns True when threshold reached."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        for _ in range(100):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

    def test_count_trigger_exceeded(self) -> None:
        """Count trigger returns True when threshold exceeded."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        for _ in range(150):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

    def test_timeout_trigger_not_reached(self) -> None:
        """Timeout trigger returns False when time not exceeded."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config)

        evaluator.record_accept()
        assert evaluator.should_trigger() is False

    def test_timeout_trigger_reached(self) -> None:
        """Timeout trigger returns True when time exceeded."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(timeout_seconds=0.01)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()
        clock.advance(0.02)  # Advance past timeout
        assert evaluator.should_trigger() is True

    def test_condition_trigger_not_met(self) -> None:
        """Condition trigger returns False when condition not met."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        # Use row['batch_count'] syntax per ExpressionParser security model
        config = TriggerConfig(condition="row['batch_count'] >= 50")
        evaluator = TriggerEvaluator(config)

        for _ in range(30):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False

    def test_condition_trigger_met(self) -> None:
        """Condition trigger returns True when condition met."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(condition="row['batch_count'] >= 50")
        evaluator = TriggerEvaluator(config)

        for _ in range(50):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

    def test_condition_trigger_with_age(self) -> None:
        """Condition trigger can use batch_age_seconds."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            condition="row['batch_count'] >= 10 and row['batch_age_seconds'] > 0.01",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(15):
            evaluator.record_accept()

        clock.advance(0.02)  # Advance past condition threshold
        assert evaluator.should_trigger() is True

    def test_combined_count_and_timeout_count_wins(self) -> None:
        """Combined triggers: count fires first (OR logic)."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=10, timeout_seconds=3600.0)
        evaluator = TriggerEvaluator(config)

        for _ in range(10):
            evaluator.record_accept()

        result = evaluator.should_trigger()
        assert result is True
        assert evaluator.which_triggered() == "count"

    def test_combined_count_and_timeout_timeout_wins(self) -> None:
        """Combined triggers: timeout fires first (OR logic)."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=1000, timeout_seconds=0.01)
        evaluator = TriggerEvaluator(config, clock=clock)

        for _ in range(5):
            evaluator.record_accept()

        clock.advance(0.02)  # Advance past timeout

        result = evaluator.should_trigger()
        assert result is True
        assert evaluator.which_triggered() == "timeout"

    def test_combined_all_triggers_count_wins(self) -> None:
        """Combined count + timeout + condition: count fires first."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(
            count=5,
            timeout_seconds=3600.0,
            condition="row['batch_count'] >= 1000",
        )
        evaluator = TriggerEvaluator(config)

        for _ in range(5):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

    def test_combined_none_fire_yet(self) -> None:
        """Combined triggers: none fire until at least one condition met."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(
            count=100,
            timeout_seconds=3600.0,
        )
        evaluator = TriggerEvaluator(config)

        for _ in range(10):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False
        assert evaluator.which_triggered() is None

    def test_reset_clears_state(self) -> None:
        """Reset clears batch count and timer."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        for _ in range(100):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

        evaluator.reset()

        assert evaluator.should_trigger() is False
        assert evaluator.batch_count == 0

    def test_batch_count_property(self) -> None:
        """batch_count property returns current count."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        assert evaluator.batch_count == 0

        for _ in range(42):
            evaluator.record_accept()

        assert evaluator.batch_count == 42

    def test_batch_age_seconds_property(self) -> None:
        """batch_age_seconds returns time since first accept."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        assert evaluator.batch_age_seconds == 0.0

        evaluator.record_accept()
        clock.advance(0.5)  # Advance time by 500ms

        assert evaluator.batch_age_seconds == 0.5


class TestTriggerFirstToFireWins:
    """Tests for 'first to fire wins' semantics per plugin-protocol.md:1211.

    When multiple triggers are satisfied simultaneously, the one that
    FIRED FIRST (became true earliest) should be reported, not the one
    checked first in code order.

    Bug: P2-2026-01-22-trigger-type-priority-misreports-first-fire
    """

    def test_timeout_fires_before_count_reports_timeout(self) -> None:
        """When timeout elapsed before count reached, report TIMEOUT.

        Scenario:
        - Count=100, Timeout=1.0s
        - Accept 99 rows at t=0
        - Time advances to t=1.1s (timeout fired at t=1.0s)
        - Accept row 100 at t=1.1s (count fires now)
        - Both conditions are now true, but timeout fired FIRST

        Expected: which_triggered() == "timeout"
        Bug behavior: which_triggered() == "count" (checked first in code)
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=100, timeout_seconds=1.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept 99 rows quickly at t=0
        for _ in range(99):
            evaluator.record_accept()

        # Timeout hasn't fired yet (only 99 rows, age ~0)
        assert evaluator.should_trigger() is False

        # Time advances past timeout threshold
        clock.advance(1.1)  # Now at t=1.1s, timeout fired at t=1.0s

        # Accept the 100th row - count threshold now reached at t=1.1s
        evaluator.record_accept()

        # Both triggers are satisfied, but timeout fired first (at t=1.0s)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "timeout", (
            "Timeout fired at t=1.0s, count fired at t=1.1s. Per 'first to fire wins' contract, should report 'timeout', not 'count'."
        )

    def test_count_fires_before_timeout_reports_count(self) -> None:
        """When count reached before timeout elapsed, report COUNT.

        Scenario:
        - Count=10, Timeout=5.0s
        - Accept 10 rows quickly at t=0.1s
        - Count fires at t=0.1s, timeout would fire at t=5.0s

        Expected: which_triggered() == "count"
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10, timeout_seconds=5.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept 10 rows quickly
        clock.advance(0.1)
        for _ in range(10):
            evaluator.record_accept()

        # Count reached at t=0.1s, timeout would fire at t=5.0s
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

    def test_condition_fires_before_timeout_reports_condition(self) -> None:
        """When condition becomes true before timeout, report CONDITION.

        Scenario:
        - Condition: batch_count >= 5
        - Timeout: 2.0s
        - Accept 5 rows at t=0.5s (condition fires)
        - Time advances to t=2.5s (timeout also fires)

        Expected: which_triggered() == "condition"
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            timeout_seconds=2.0,
            condition="row['batch_count'] >= 5",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept 5 rows at t=0.5s
        clock.advance(0.5)
        for _ in range(5):
            evaluator.record_accept()

        # Condition fires at t=0.5s
        # Time advances past timeout
        clock.advance(2.0)  # Now at t=2.5s

        # Both condition (t=0.5s) and timeout (t=2.0s) have fired
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition", (
            "Condition fired at t=0.5s, timeout fired at t=2.0s. Per 'first to fire wins' contract, should report 'condition'."
        )

    def test_timeout_fires_before_condition_reports_timeout(self) -> None:
        """When timeout elapses before condition becomes true, report TIMEOUT.

        Scenario:
        - Condition: batch_count >= 100
        - Timeout: 1.0s
        - Accept 50 rows at t=0 (condition not yet true)
        - Time advances to t=1.5s (timeout fires at t=1.0s)
        - Accept 50 more rows at t=1.5s (condition now fires)

        Expected: which_triggered() == "timeout"
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            timeout_seconds=1.0,
            condition="row['batch_count'] >= 100",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept 50 rows at t=0 (condition not yet true)
        for _ in range(50):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False

        # Time advances past timeout
        clock.advance(1.5)  # Timeout fires at t=1.0s

        # Accept 50 more rows - condition now true at t=1.5s
        for _ in range(50):
            evaluator.record_accept()

        # Timeout (t=1.0s) fired before condition (t=1.5s)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "timeout", (
            "Timeout fired at t=1.0s, condition fired at t=1.5s. Per 'first to fire wins' contract, should report 'timeout'."
        )


class TestConditionFireTimeSampling:
    """Time-based condition fire times are sampled-at-evaluation (elspeth-06df383e4a).

    A poll-driven engine only KNOWS condition truth at observation time. The
    old code backdated a newly-true condition to the previous (known-false)
    check time — or first accept — letting a condition beat a timeout whose
    real fire instant came first, corrupting the TriggerType audit value and
    the persisted condition_fire_offset.
    """

    def test_backdated_condition_cannot_steal_timeout_win(self) -> None:
        """Panel scenario: timeout=10, condition age>=15, false check at t=5,
        next check at t=20. Timeout (t=10) must win, not a condition
        backdated to the known-false t=5."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            timeout_seconds=10.0,
            condition="row['batch_age_seconds'] >= 15",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # t=0

        clock.advance(5.0)  # t=5: neither trigger ready
        assert evaluator.should_trigger() is False

        clock.advance(15.0)  # t=20: timeout fired at t=10, condition observed true now
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "timeout", (
            "Timeout fired at t=10; condition was first OBSERVED true at t=20. "
            "Backdating the condition to the known-false t=5 check must not steal the win."
        )

    def test_condition_fire_offset_reflects_observation_time(self) -> None:
        """The persisted checkpoint offset is the observation instant, never a
        backdated known-false check time."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(condition="row['batch_age_seconds'] >= 15")
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # t=0

        clock.advance(5.0)  # t=5: condition false, records a check
        assert evaluator.should_trigger() is False

        clock.advance(15.0)  # t=20: condition first observed true
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition"
        assert evaluator.get_condition_fire_offset() == 20.0

    def test_condition_true_on_first_evaluation_uses_observation_time(self) -> None:
        """With no prior false check, the fire time is still the observation
        instant — not first_accept_time."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            timeout_seconds=10.0,
            condition="row['batch_age_seconds'] >= 15",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # t=0

        clock.advance(20.0)  # t=20: first evaluation ever; both fired
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "timeout", (
            "Condition first observed true at t=20 must not be backdated to first accept (t=0)."
        )


class TestTriggerConditionLatching:
    """Tests for P1-2026-02-05: Condition trigger must latch once fired.

    Window-based conditions (e.g., batch_age_seconds < 0.5) can become
    false after the window closes. Once _condition_fire_time is set,
    should_trigger() must honor it regardless of current evaluation.
    """

    def test_window_condition_latched_after_window_closes(self) -> None:
        """Condition that was true at accept time still triggers after window closes.

        Scenario:
        - Condition: batch_age_seconds < 0.5 (true in first 500ms)
        - Accept row at t=0 (condition true, fire time latched)
        - should_trigger() called at t=1.0 (condition now false)

        Expected: should_trigger() returns True (latched fire time honored)
        Bug behavior: should_trigger() returned False (re-evaluated, found false)
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(condition="row['batch_age_seconds'] < 0.5")
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept at t=0 — condition is true (0.0 < 0.5), fire time latched
        evaluator.record_accept()
        assert evaluator._condition_fire_time is not None

        # Advance past the window
        clock.advance(1.0)  # Now at t=1.0, condition is false (1.0 < 0.5 = False)

        # should_trigger() must honor the latched fire time
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition"

    def test_latched_condition_reports_original_fire_time(self) -> None:
        """Latched condition uses the original fire time, not current time.

        This matters for "first to fire wins" when combined with other triggers.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            timeout_seconds=2.0,
            condition="row['batch_age_seconds'] < 0.5",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept at t=0 — condition fires immediately
        evaluator.record_accept()

        # Advance past both condition window AND timeout
        clock.advance(3.0)  # Now at t=3.0

        # Both have fired, but condition fired at t=0 (before timeout at t=2.0)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition", "Condition latched at t=0.0, timeout at t=2.0. Condition should win."

    def test_unlatched_condition_still_evaluated(self) -> None:
        """Condition that hasn't fired yet is still re-evaluated at should_trigger() time.

        This is the non-bug case: time-dependent conditions that become true
        between accepts (e.g., batch_age_seconds >= 5) should be detected.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(condition="row['batch_age_seconds'] >= 5.0")
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept at t=0 — condition is false (0.0 >= 5.0 = False)
        evaluator.record_accept()
        assert evaluator._condition_fire_time is None

        # Advance past threshold
        clock.advance(6.0)

        # should_trigger() re-evaluates and finds condition is now true
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition"


class TestTriggerConditionBooleanValidation:
    """Tests for P2-2026-01-31: Trigger condition must return boolean.

    Per CLAUDE.md Three-Tier Trust Model: trigger config is "our data" (Tier 1).
    Non-boolean results should be rejected, not silently coerced with bool().
    """

    def test_non_boolean_condition_rejected_at_config_time(self) -> None:
        """Non-boolean expressions should be rejected at config validation.

        A condition like 'row["batch_count"]' (returns int) or
        'row["batch_count"] + 1' (returns int) should fail validation.
        """
        import pytest

        from elspeth.core.config import TriggerConfig

        # Integer expression - should be rejected
        with pytest.raises(ValueError, match="boolean"):
            TriggerConfig(condition="row['batch_count']")

        # Arithmetic expression - should be rejected
        with pytest.raises(ValueError, match="boolean"):
            TriggerConfig(condition="row['batch_count'] + 1")

    def test_boolean_condition_accepted(self) -> None:
        """Boolean expressions should pass validation."""
        from elspeth.core.config import TriggerConfig

        # Comparison - returns bool
        config = TriggerConfig(condition="row['batch_count'] >= 50")
        assert config.condition == "row['batch_count'] >= 50"

        # Boolean operator - returns bool
        config = TriggerConfig(condition="row['batch_count'] >= 10 and row['batch_age_seconds'] > 1.0")
        assert config.condition is not None

        # Unary not - returns bool
        config = TriggerConfig(condition="not row['batch_count'] >= 100")
        assert config.condition is not None

    def test_ternary_with_boolean_branches_accepted(self) -> None:
        """Ternary expressions returning booleans should pass."""
        from elspeth.core.config import TriggerConfig

        config = TriggerConfig(condition="True if row['batch_count'] > 10 else False")
        assert config.condition is not None

    def test_non_boolean_runtime_raises(self) -> None:
        """If a non-boolean somehow reaches runtime, it should raise.

        This is defense-in-depth: even if config validation is bypassed,
        the runtime should reject non-boolean results instead of coercing.
        """
        import pytest

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        # Bypass config validation by constructing directly
        # (simulates a bug in validation or manual construction)
        config = TriggerConfig.__new__(TriggerConfig)
        object.__setattr__(config, "count", None)
        object.__setattr__(config, "timeout_seconds", None)
        object.__setattr__(config, "condition", "row['batch_count']")  # Returns int

        evaluator = TriggerEvaluator(config)

        # Should raise on first condition evaluation (in record_accept or should_trigger)
        # The condition is first evaluated in record_accept() when tracking fire times
        with pytest.raises(TypeError, match="condition must return bool"):
            evaluator.record_accept()


class TestTriggerCheckpointRestore:
    """Tests for P2-2026-02-01: Trigger fire times must be preserved on resume.

    When checkpoint restore reconstructs batches, the "first to fire wins"
    ordering must be preserved from before the crash.
    """

    def test_count_fire_time_preserved_on_restore(self) -> None:
        """Count trigger fire time offset should be restored from checkpoint.

        Scenario:
        - Pre-crash: First accept at t=0, count fires at t=2s (5 rows, offset=2s)
        - Checkpoint stores: elapsed_age_seconds=5s, count_fire_offset=2s
        - Resume at t=100s
        - Expected: count fire time should have offset 2s from restored first_accept

        Bug behavior: count_fire_time = current time during restore (loses ordering)
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=5, timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # First accept at t=0
        evaluator.record_accept()
        assert evaluator._first_accept_time == 0.0

        # Accept 4 more rows at t=2s - count fires
        clock.advance(2.0)
        for _ in range(4):
            evaluator.record_accept()

        # Verify count fired at t=2s (offset 2s from first accept)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

        # Get fire time offset for checkpoint
        count_fire_offset = evaluator.get_count_fire_offset()
        assert count_fire_offset == 2.0  # Fired 2s after first accept at t=0

        # Simulate time passing and crash at t=5s
        clock.advance(3.0)  # Now at t=5s
        elapsed_age = evaluator.get_age_seconds()
        assert elapsed_age == 5.0

        # --- CRASH AND RESUME ---
        # Create new evaluator at t=100s
        clock2 = MockClock(start=100.0)
        evaluator2 = TriggerEvaluator(config, clock=clock2)

        # Restore using the new API that preserves fire times
        evaluator2.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=elapsed_age,
            count_fire_offset=count_fire_offset,
            condition_fire_offset=None,
        )

        # Count should still be reported as firing first
        # (count fired at offset 2s, timeout would fire at offset 10s)
        assert evaluator2.should_trigger() is True
        assert evaluator2.which_triggered() == "count", (
            "Count fired at offset 2s, timeout would fire at offset 10s. After restore, should still report 'count' not 'timeout'."
        )

    def test_timeout_wins_over_count_after_restore(self) -> None:
        """When timeout fired before count pre-crash, restore preserves ordering.

        Scenario:
        - Pre-crash at t=0: Accept 99 rows (count=100 not reached)
        - At t=1.5s: timeout fires (timeout_seconds=1.0, fired at t=1.0s)
        - At t=1.5s: Accept row 100 (count fires at t=1.5s)
        - Checkpoint: elapsed=1.5s, count_fire_offset=1.5s (but timeout offset=1.0s)
        - Resume: Should report timeout, not count
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=100, timeout_seconds=1.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept 99 rows at t=0
        for _ in range(99):
            evaluator.record_accept()

        # Time passes, timeout fires at t=1.0s
        clock.advance(1.5)  # Now at t=1.5s

        # Accept row 100 - count fires now
        evaluator.record_accept()

        # Both have fired, timeout was first
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "timeout"

        # Get checkpoint state
        elapsed_age = evaluator.get_age_seconds()
        count_fire_offset = evaluator.get_count_fire_offset()
        # Note: timeout fire time is computed, not stored

        # --- CRASH AND RESUME ---
        clock2 = MockClock(start=100.0)
        evaluator2 = TriggerEvaluator(config, clock=clock2)

        evaluator2.restore_from_checkpoint(
            batch_count=100,
            elapsed_age_seconds=elapsed_age,
            count_fire_offset=count_fire_offset,
            condition_fire_offset=None,
        )

        # Timeout should still win (it fired at offset 1.0s, count at 1.5s)
        assert evaluator2.should_trigger() is True
        assert evaluator2.which_triggered() == "timeout", (
            "Timeout fired at 1.0s offset, count at 1.5s offset. After restore, should still report 'timeout' not 'count'."
        )


class TestCheckpointRestoreStateFidelity:
    """Kill survivors in restore_from_checkpoint (lines 277-304).

    The existing TestTriggerCheckpointRestore tests verify fire-time ordering
    preservation, but don't assert on raw restored state (batch_count,
    batch_age_seconds) or test condition trigger restoration. These tests
    kill mutations in the assignment and arithmetic lines.
    """

    def test_batch_count_restored_correctly(self) -> None:
        """batch_count must equal the restored value, not zero or something else.

        Kills mutation: ``self._batch_count = batch_count`` → deleted or
        ``self._batch_count = 0``.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=100.0)
        config = TriggerConfig(count=50)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.restore_from_checkpoint(
            batch_count=42,
            elapsed_age_seconds=10.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        assert evaluator.batch_count == 42

    def test_batch_age_seconds_correct_after_restore(self) -> None:
        """batch_age_seconds must reflect elapsed time from restored first_accept.

        Kills line 293: ``current_time - elapsed_age_seconds`` → ``+``.
        With addition, _first_accept_time = 1000 + 30 = 1030 (in the future),
        so batch_age_seconds = 1000 - 1030 = -30.
        With subtraction, _first_accept_time = 1000 - 30 = 970,
        so batch_age_seconds = 1000 - 970 = 30.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=1000.0)
        config = TriggerConfig(timeout_seconds=60.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.restore_from_checkpoint(
            batch_count=10,
            elapsed_age_seconds=30.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        age = evaluator.batch_age_seconds
        # Correct: first_accept = 1000 - 30 = 970, age = 1000 - 970 = 30.0
        # Mutant:  first_accept = 1000 + 30 = 1030, age = 1000 - 1030 = -30.0
        assert 29.0 <= age <= 31.0, (
            f"batch_age_seconds should be ~30.0 after restore, got {age}. Sub→Add mutant on line 293 would produce -30.0."
        )

    def test_timeout_fires_correctly_after_restore(self) -> None:
        """Timeout must fire based on restored elapsed time, not reset to zero.

        If _first_accept_time is wrong (Sub→Add on line 293), timeout won't
        fire at the expected time.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=1000.0)
        config = TriggerConfig(timeout_seconds=60.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Restore with 50s already elapsed (10s remaining until timeout)
        evaluator.restore_from_checkpoint(
            batch_count=10,
            elapsed_age_seconds=50.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        # Should not trigger yet (50s of 60s elapsed)
        assert evaluator.should_trigger() is False

        # Advance 10s — now at 60s total, should trigger
        clock.advance(10.0)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "timeout"

    def test_condition_fire_offset_restored(self) -> None:
        """Condition fire time must be restored from offset, not lost.

        Kills line 302: ``_first_accept_time + condition_fire_offset`` → ``-``.
        With subtraction, condition fire time would be BEFORE first_accept,
        which changes "first to fire wins" ordering.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=500.0)
        # Condition with timeout — condition should win because it fired first
        config = TriggerConfig(
            timeout_seconds=10.0,
            condition="row['batch_count'] >= 5",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        # Restore: condition fired at offset 2s, timeout would fire at 10s
        # elapsed_age = 12s, so timeout has also fired
        evaluator.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=12.0,
            count_fire_offset=None,
            condition_fire_offset=2.0,
        )

        # Both condition (offset 2s) and timeout (offset 10s) have fired
        # Condition should win (fired first)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition", (
            "Condition fired at offset 2s, timeout at offset 10s. "
            "Condition should win. Sub mutant on line 302 would make "
            "condition fire time = first_accept - 2.0, which is before first_accept."
        )

    def test_count_fire_offset_restored(self) -> None:
        """Count fire time must be restored from offset correctly.

        Kills line 297: ``_first_accept_time + count_fire_offset`` → ``-``.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=500.0)
        # Count with timeout — count should win because it fired first
        config = TriggerConfig(count=5, timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Restore: count fired at offset 3s, timeout fires at 10s
        # elapsed_age = 12s, so timeout has also fired
        evaluator.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=12.0,
            count_fire_offset=3.0,
            condition_fire_offset=None,
        )

        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count", "Count fired at offset 3s, timeout at offset 10s. Count should win."

    def test_none_fire_offsets_leave_triggers_unfired(self) -> None:
        """When fire offsets are None, triggers must not spuriously report fired.

        Kills lines 299/304: the ``else: self._*_fire_time = None`` branches.
        If the None assignment is deleted, fire_time retains whatever value
        __init__ set (also None, but the mutation could be to skip the branch).
        More importantly, tests that count is NOT reported when offset is None
        and count < threshold.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=100.0)
        config = TriggerConfig(count=50, timeout_seconds=3600.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Restore with count below threshold and no fire offsets
        evaluator.restore_from_checkpoint(
            batch_count=10,
            elapsed_age_seconds=5.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        # Neither trigger should fire (count=10 < 50, elapsed=5s < 3600s)
        assert evaluator.should_trigger() is False
        assert evaluator.which_triggered() is None

    def test_restored_over_threshold_count_without_offset_triggers_immediately(self) -> None:
        """Old checkpoints without count_fire_offset still fire restored count batches."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=100.0)
        config = TriggerConfig(count=3, timeout_seconds=3600.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=1.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

    def test_zero_batch_count_valid(self) -> None:
        """batch_count=0 is valid (empty batch restore after flush).

        Kills line 277: ``batch_count < 0`` → ``<= 0``.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Should NOT raise — 0 is a valid batch count
        evaluator.restore_from_checkpoint(
            batch_count=0,
            elapsed_age_seconds=0.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )
        assert evaluator.batch_count == 0

    def test_zero_elapsed_age_valid(self) -> None:
        """elapsed_age_seconds=0.0 is valid (checkpoint taken immediately).

        Kills line 279: ``elapsed_age_seconds < 0`` → ``<= 0``.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=50.0)
        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Should NOT raise — 0.0 is valid
        evaluator.restore_from_checkpoint(
            batch_count=1,
            elapsed_age_seconds=0.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )
        # Age should be ~0 since we just restored with 0 elapsed
        assert evaluator.batch_age_seconds == 0.0

    def test_zero_fire_offset_valid(self) -> None:
        """Fire offset of 0.0 is valid (trigger fired at same instant as first accept).

        Kills lines 281/283: ``offset < 0`` → ``<= 0``.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=200.0)
        config = TriggerConfig(count=5, timeout_seconds=60.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Should NOT raise — 0.0 offset means fired at first_accept_time
        evaluator.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=10.0,
            count_fire_offset=0.0,
            condition_fire_offset=None,
        )

        # Count should be reported as triggered (fire offset 0.0 < timeout 60.0)
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

    def test_negative_batch_count_rejected(self) -> None:
        """Negative batch_count must raise ValueError."""
        import pytest

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10)
        evaluator = TriggerEvaluator(config, clock=clock)

        with pytest.raises(ValueError, match="non-negative"):
            evaluator.restore_from_checkpoint(
                batch_count=-1,
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
            )

    def test_negative_elapsed_age_rejected(self) -> None:
        """Negative elapsed_age_seconds must raise ValueError."""
        import pytest

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10)
        evaluator = TriggerEvaluator(config, clock=clock)

        with pytest.raises(ValueError, match="non-negative"):
            evaluator.restore_from_checkpoint(
                batch_count=1,
                elapsed_age_seconds=-5.0,
                count_fire_offset=None,
                condition_fire_offset=None,
            )

    def test_nan_elapsed_age_rejected(self) -> None:
        """NaN elapsed_age_seconds must raise ValueError."""
        import math

        import pytest

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10)
        evaluator = TriggerEvaluator(config, clock=clock)

        with pytest.raises(ValueError, match="finite"):
            evaluator.restore_from_checkpoint(
                batch_count=1,
                elapsed_age_seconds=math.nan,
                count_fire_offset=None,
                condition_fire_offset=None,
            )

    def test_negative_fire_offset_rejected(self) -> None:
        """Negative count_fire_offset must raise ValueError."""
        import pytest

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10)
        evaluator = TriggerEvaluator(config, clock=clock)

        with pytest.raises(ValueError, match="non-negative"):
            evaluator.restore_from_checkpoint(
                batch_count=5,
                elapsed_age_seconds=10.0,
                count_fire_offset=-1.0,
                condition_fire_offset=None,
            )

    def test_inf_condition_fire_offset_rejected(self) -> None:
        """Infinity condition_fire_offset must raise ValueError."""
        import math

        import pytest

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=10)
        evaluator = TriggerEvaluator(config, clock=clock)

        with pytest.raises(ValueError, match="finite"):
            evaluator.restore_from_checkpoint(
                batch_count=5,
                elapsed_age_seconds=10.0,
                count_fire_offset=None,
                condition_fire_offset=math.inf,
            )

    def test_full_round_trip_checkpoint_restore(self) -> None:
        """Full round-trip: accept rows, checkpoint, restore, verify behavior matches.

        This exercises the complete checkpoint API (get_count_fire_offset,
        get_condition_fire_offset, get_age_seconds) feeding into restore_from_checkpoint.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(
            count=5,
            timeout_seconds=30.0,
            condition="row['batch_count'] >= 3",
        )
        evaluator = TriggerEvaluator(config, clock=clock)

        # Accept 5 rows over 10 seconds
        evaluator.record_accept()  # t=0
        clock.advance(2.0)
        evaluator.record_accept()  # t=2
        clock.advance(2.0)
        evaluator.record_accept()  # t=4 — condition fires (count=3 >= 3)
        clock.advance(3.0)
        evaluator.record_accept()  # t=7
        evaluator.record_accept()  # t=7 — count fires (count=5 >= 5)

        # Verify pre-checkpoint state
        assert evaluator.should_trigger() is True
        # Condition fired at t=4, count at t=7 → condition wins
        assert evaluator.which_triggered() == "condition"

        # Capture checkpoint data
        elapsed = evaluator.get_age_seconds()
        count_offset = evaluator.get_count_fire_offset()
        condition_offset = evaluator.get_condition_fire_offset()
        batch_ct = evaluator.batch_count

        assert elapsed == 7.0
        assert count_offset == 7.0
        assert condition_offset == 4.0
        assert batch_ct == 5

        # --- CRASH AND RESUME ---
        clock2 = MockClock(start=5000.0)
        evaluator2 = TriggerEvaluator(config, clock=clock2)

        evaluator2.restore_from_checkpoint(
            batch_count=batch_ct,
            elapsed_age_seconds=elapsed,
            count_fire_offset=count_offset,
            condition_fire_offset=condition_offset,
        )

        # Verify restored state matches
        assert evaluator2.batch_count == 5
        assert 6.5 <= evaluator2.batch_age_seconds <= 7.5

        # Same trigger result: condition should still win
        assert evaluator2.should_trigger() is True
        assert evaluator2.which_triggered() == "condition", (
            "Condition fired at offset 4s, count at offset 7s, timeout at 30s. Condition should still win after restore."
        )


class TestBatchAgeSecondsSignFlip:
    """Kill mutant: ``clock.monotonic() - first_accept_time`` → ``+ first_accept_time``.

    With addition, batch_age_seconds returns the sum of two monotonic
    timestamps (e.g., 1002.5 instead of 2.5), causing every timeout
    trigger to fire immediately.
    """

    def test_batch_age_is_elapsed_not_sum(self) -> None:
        """batch_age_seconds must equal elapsed time, not sum of timestamps."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=1000.0)
        config = TriggerConfig(count=999)  # Won't fire on count
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # Records at t=1000.0
        clock.advance(2.5)  # Now at t=1002.5

        age = evaluator.batch_age_seconds
        # Correct: 1002.5 - 1000.0 = 2.5
        # Mutant:  1002.5 + 1000.0 = 2002.5
        assert 2.0 <= age <= 3.0, (
            f"batch_age_seconds should be ~2.5 (elapsed), got {age}. Sign-flip mutant would produce ~2002.5 (sum of timestamps)."
        )


class TestCountExceedsThreshold:
    """Kill mutant: ``self._batch_count >= self._config.count`` → ``==`` at line 116.

    Line 116 records the count fire TIME. If ``>=`` becomes ``==``, adding rows
    one-at-a-time still records the fire time at exactly the threshold. But if
    the count jumps past the threshold (e.g., via restore_from_checkpoint or
    bulk accepts where the threshold is crossed), the fire time would never be
    recorded, and which_triggered() could not report "count".
    """

    def test_which_triggered_reports_count_when_exceeded(self) -> None:
        """which_triggered() returns 'count' even when batch exceeds threshold.

        With ``>=``, fire time is recorded at the threshold row and persists.
        With ``==``, fire time is recorded only at exact threshold — still works
        for one-at-a-time accepts. So we verify via which_triggered() after
        exceeding, which requires the fire time to have been set.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=3)
        evaluator = TriggerEvaluator(config)

        for _ in range(5):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

    def test_fire_time_recorded_when_restore_exceeds_threshold(self) -> None:
        """Fire time must be set when batch count is restored past the threshold.

        restore_from_checkpoint sets _batch_count directly without calling
        record_accept(). After restore, a single record_accept() at count > threshold
        must still record the fire time (via ``>=``). With ``==``, the fire time
        is never set because batch_count is already past the threshold.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(count=3, timeout_seconds=3600.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Restore with count already past threshold but NO count_fire_offset
        # (simulates a checkpoint that didn't record fire time — e.g., old format)
        evaluator.restore_from_checkpoint(
            batch_count=5,
            elapsed_age_seconds=1.0,
            count_fire_offset=None,
            condition_fire_offset=None,
        )

        # Accept one more row — batch_count becomes 6, which is > threshold 3
        # With >=, fire time is recorded now. With ==, it's not (6 != 3).
        clock.advance(2.0)
        evaluator.record_accept()

        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count", (
            "Count fire time must be recorded when batch_count (6) >= threshold (3). The >= mutant to == would fail because 6 != 3."
        )


class TestTimeoutExactBoundary:
    """Kill mutant: ``current_time >= timeout_fire_time`` → ``>`` at line 159.

    At the exact boundary (elapsed == timeout_seconds), the trigger must fire.
    With ``>``, it would not fire until strictly past the boundary.
    """

    def test_timeout_fires_at_exact_boundary(self) -> None:
        """Timeout fires when elapsed time equals timeout_seconds exactly.

        With ``>=``, current_time == timeout_fire_time → True.
        With ``>``, current_time == timeout_fire_time → False.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=0.0)
        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # First accept at t=0.0
        clock.advance(10.0)  # Now at t=10.0, exactly at timeout boundary

        assert evaluator.should_trigger() is True, (
            "Timeout must fire at exact boundary (elapsed == timeout_seconds). The >= mutant to > would return False here."
        )
        assert evaluator.which_triggered() == "timeout"


class TestConditionTriggerCorrectAge:
    """Kill mutant: ``current_time - self._first_accept_time`` → ``+`` at lines 123, 176.

    batch_age_seconds in the condition context must be elapsed time (subtraction),
    not the sum of timestamps (addition). With a large start time, addition
    produces an enormous value that would satisfy any age threshold immediately.
    """

    def test_condition_age_is_subtraction_not_addition(self) -> None:
        """Condition using batch_age_seconds must reflect elapsed time only.

        With start=1000.0 and 5s elapsed:
        - Subtraction: age = 1005.0 - 1000.0 = 5.0 → 5.0 > 10 is False
        - Addition:    age = 1005.0 + 1000.0 = 2005.0 → 2005.0 > 10 is True

        Then after 11s total:
        - Subtraction: age = 1011.0 - 1000.0 = 11.0 → 11.0 > 10 is True
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=1000.0)
        config = TriggerConfig(condition="row['batch_age_seconds'] > 10")
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # First accept at t=1000.0

        # Advance 5s — age should be 5.0, condition (> 10) should be False
        clock.advance(5.0)  # Now at t=1005.0
        assert evaluator.should_trigger() is False, (
            "With 5s elapsed, batch_age_seconds should be 5.0 (not 2005.0). "
            "Condition 5.0 > 10 is False. Addition mutant would produce 2005.0 > 10 = True."
        )

        # Advance to 11s total — age should be 11.0, condition (> 10) should be True
        clock.advance(6.0)  # Now at t=1011.0
        assert evaluator.should_trigger() is True, "With 11s elapsed, batch_age_seconds should be 11.0. Condition 11.0 > 10 is True."
        assert evaluator.which_triggered() == "condition"


class TestAnchorRewindRecompute:
    """Backdated out-of-order adoption must not corrupt latched fire instants.

    ADR-030 §E.2 intake adopts rows ordered by (barrier_key, ingest_sequence,
    work_item_id) — NOT barrier_blocked_at — so a later record_accept can
    carry an EARLIER durable arrival than already-adopted members. The count
    and condition latches (and therefore the persisted checkpoint offsets)
    must be pure functions of the durable member set, invariant under
    adoption order (filigree elspeth-eed319ed3d). Timeout already anchors at
    the min durable arrival; these tests pin count/condition parity.
    """

    def test_count_latch_is_adoption_order_invariant(self) -> None:
        """count=2 over arrivals {5, 10, 12}: the durable count-fire instant
        is 10.0 (the 2nd-smallest arrival) regardless of adoption order."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=2)

        in_order = TriggerEvaluator(config, clock=MockClock(start=1000.0))
        for arrival in (5.0, 10.0, 12.0):
            in_order.record_accept(accept_time=arrival)

        backdated_last = TriggerEvaluator(config, clock=MockClock(start=1000.0))
        for arrival in (10.0, 12.0, 5.0):  # 5.0 rewinds the anchor AFTER count latched at 12.0
            backdated_last.record_accept(accept_time=arrival)

        assert in_order.get_count_fire_offset() == 5.0  # fire 10.0 - anchor 5.0
        assert backdated_last.get_count_fire_offset() == 5.0, (
            "The count latch must recompute as the N-th smallest durable arrival when a "
            "backdated adoption rewinds the anchor — not stay latched at the adoption-order instant."
        )

    def test_condition_latch_is_adoption_order_invariant(self) -> None:
        """A batch_count condition replays over durable order: first true at
        the 2nd-smallest arrival regardless of adoption order."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(condition="row['batch_count'] >= 2")

        in_order = TriggerEvaluator(config, clock=MockClock(start=1000.0))
        for arrival in (5.0, 10.0, 12.0):
            in_order.record_accept(accept_time=arrival)

        backdated_last = TriggerEvaluator(config, clock=MockClock(start=1000.0))
        for arrival in (10.0, 12.0, 5.0):
            backdated_last.record_accept(accept_time=arrival)

        assert in_order.get_condition_fire_offset() == 5.0
        assert backdated_last.get_condition_fire_offset() == 5.0, (
            "The condition latch must replay over the durable member order when a backdated adoption changes the prefix contexts."
        )

    def test_backdated_oldest_member_rewinds_anchor_and_grows_offsets(self) -> None:
        """The rewinding accept becomes the new oldest member: the timeout
        anchor moves to it and already-latched offsets grow accordingly
        (fire instants stay durable-absolute)."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=2, timeout_seconds=60.0)
        evaluator = TriggerEvaluator(config, clock=MockClock(start=1000.0))
        evaluator.record_accept(accept_time=10.0)
        evaluator.record_accept(accept_time=12.0)  # count latches at 12.0
        assert evaluator.get_count_fire_offset() == 2.0

        evaluator.record_accept(accept_time=5.0)  # new oldest member
        assert evaluator.batch_count == 3
        # Anchor rewound to 5.0; count fire recomputes to the 2nd-smallest (10.0).
        assert evaluator.get_count_fire_offset() == 5.0

    def test_restored_batch_keeps_restored_latches_on_backdated_accept(self) -> None:
        """Checkpoint-restored batches carry no member times; a subsequent
        backdated accept must not recompute (and corrupt) restored latches."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=2)
        clock = MockClock(start=1000.0)
        evaluator = TriggerEvaluator(config, clock=clock)
        evaluator.restore_from_checkpoint(
            batch_count=3,
            elapsed_age_seconds=100.0,
            count_fire_offset=1.5,
            condition_fire_offset=None,
        )
        restored_offset = evaluator.get_count_fire_offset()
        assert restored_offset == 1.5

        evaluator.record_accept(accept_time=850.0)  # backdated relative to the restored anchor (900.0)

        assert evaluator.batch_count == 4
        # The restored latch is preserved as an absolute instant; the anchor
        # rewind grows the offset, exactly as it did pre-recompute (the member
        # list is incomplete, so no recompute may run).
        count_fire_time = evaluator._count_fire_time
        assert count_fire_time == 901.5  # 900.0 restored anchor + 1.5 offset, unchanged

    def test_observation_latched_condition_survives_backdated_accept(self) -> None:
        """A condition latched at OBSERVATION time by should_trigger()
        (sampled-at-evaluation, elspeth-06df383e4a) is not owned by
        record_accept; a later backdated adoption must not rewrite it."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.clock import MockClock
        from elspeth.engine.triggers import TriggerEvaluator

        clock = MockClock(start=100.0)
        config = TriggerConfig(condition="row['batch_age_seconds'] >= 10")
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept(accept_time=100.0)
        clock.advance(15.0)  # t=115: condition first observed true
        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "condition"
        assert evaluator.get_condition_fire_offset() == 15.0

        evaluator.record_accept(accept_time=90.0)  # backdated adoption; anchor rewinds to 90.0
        # The observed latch instant (115.0) is untouched; only the anchor moved.
        assert evaluator.get_condition_fire_offset() == 25.0
