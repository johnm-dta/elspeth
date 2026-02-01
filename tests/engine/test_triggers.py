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
