"""Tests for TriggerEvaluator."""

import time


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
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(timeout_seconds=0.01)
        evaluator = TriggerEvaluator(config)

        evaluator.record_accept()
        time.sleep(0.02)
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
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(
            condition="row['batch_count'] >= 10 and row['batch_age_seconds'] > 0.01",
        )
        evaluator = TriggerEvaluator(config)

        for _ in range(15):
            evaluator.record_accept()

        time.sleep(0.02)
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
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=1000, timeout_seconds=0.01)
        evaluator = TriggerEvaluator(config)

        for _ in range(5):
            evaluator.record_accept()

        time.sleep(0.02)

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
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config)

        assert evaluator.batch_age_seconds == 0.0

        evaluator.record_accept()
        time.sleep(0.01)

        assert evaluator.batch_age_seconds > 0.0
