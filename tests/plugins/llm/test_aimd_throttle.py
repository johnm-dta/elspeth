# tests/plugins/llm/test_aimd_throttle.py
"""Tests for AIMD throttle state machine."""

from elspeth.plugins.llm.aimd_throttle import AIMDThrottle, ThrottleConfig


class TestAIMDThrottleInit:
    """Test throttle initialization and defaults."""

    def test_default_config_values(self) -> None:
        """Verify sensible defaults are applied."""
        throttle = AIMDThrottle()

        assert throttle.current_delay_ms == 0
        assert throttle.config.min_dispatch_delay_ms == 0
        assert throttle.config.max_dispatch_delay_ms == 5000
        assert throttle.config.backoff_multiplier == 2.0
        assert throttle.config.recovery_step_ms == 50

    def test_custom_config(self) -> None:
        """Verify custom config is applied."""
        config = ThrottleConfig(
            min_dispatch_delay_ms=10,
            max_dispatch_delay_ms=1000,
            backoff_multiplier=3.0,
            recovery_step_ms=25,
        )
        throttle = AIMDThrottle(config)

        assert throttle.config.min_dispatch_delay_ms == 10
        assert throttle.config.max_dispatch_delay_ms == 1000
        assert throttle.config.backoff_multiplier == 3.0
        assert throttle.config.recovery_step_ms == 25
