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


class TestAIMDThrottleBackoff:
    """Test multiplicative decrease on capacity errors."""

    def test_first_capacity_error_sets_initial_delay(self) -> None:
        """First error should set delay to initial backoff value."""
        throttle = AIMDThrottle()
        assert throttle.current_delay_ms == 0

        throttle.on_capacity_error()

        # First error with 0 delay should set to recovery_step (bootstrap)
        assert throttle.current_delay_ms == 50  # recovery_step default

    def test_subsequent_errors_multiply_delay(self) -> None:
        """Each error should multiply delay by backoff_multiplier."""
        config = ThrottleConfig(
            backoff_multiplier=2.0,
            recovery_step_ms=100,
        )
        throttle = AIMDThrottle(config)

        throttle.on_capacity_error()  # 0 -> 100
        assert throttle.current_delay_ms == 100

        throttle.on_capacity_error()  # 100 * 2 = 200
        assert throttle.current_delay_ms == 200

        throttle.on_capacity_error()  # 200 * 2 = 400
        assert throttle.current_delay_ms == 400

    def test_delay_capped_at_max(self) -> None:
        """Delay should not exceed max_dispatch_delay_ms."""
        config = ThrottleConfig(
            max_dispatch_delay_ms=500,
            backoff_multiplier=2.0,
            recovery_step_ms=100,
        )
        throttle = AIMDThrottle(config)

        # Drive delay up to cap
        for _ in range(10):
            throttle.on_capacity_error()

        assert throttle.current_delay_ms == 500


class TestAIMDThrottleRecovery:
    """Test additive increase on success (slow recovery)."""

    def test_success_subtracts_recovery_step(self) -> None:
        """Each success should subtract recovery_step_ms."""
        config = ThrottleConfig(recovery_step_ms=50)
        throttle = AIMDThrottle(config)

        # Set initial delay
        throttle.on_capacity_error()  # -> 50
        throttle.on_capacity_error()  # -> 100
        assert throttle.current_delay_ms == 100

        throttle.on_success()  # 100 - 50 = 50
        assert throttle.current_delay_ms == 50

        throttle.on_success()  # 50 - 50 = 0
        assert throttle.current_delay_ms == 0

    def test_delay_floored_at_min(self) -> None:
        """Delay should not go below min_dispatch_delay_ms."""
        config = ThrottleConfig(
            min_dispatch_delay_ms=10,
            recovery_step_ms=100,
        )
        throttle = AIMDThrottle(config)

        # Set initial delay
        throttle.on_capacity_error()  # -> 100

        # Multiple successes should stop at min
        for _ in range(5):
            throttle.on_success()

        assert throttle.current_delay_ms == 10

    def test_success_at_zero_stays_zero(self) -> None:
        """Success when already at zero should stay at zero."""
        throttle = AIMDThrottle()
        assert throttle.current_delay_ms == 0

        throttle.on_success()

        assert throttle.current_delay_ms == 0


class TestAIMDThrottleStats:
    """Test statistics tracking for audit."""

    def test_stats_track_capacity_retries(self) -> None:
        """Stats should count capacity retries."""
        throttle = AIMDThrottle()

        throttle.on_capacity_error()
        throttle.on_capacity_error()
        throttle.on_success()
        throttle.on_capacity_error()

        stats = throttle.get_stats()
        assert stats["capacity_retries"] == 3
        assert stats["successes"] == 1

    def test_stats_track_peak_delay(self) -> None:
        """Stats should track peak delay reached."""
        config = ThrottleConfig(
            max_dispatch_delay_ms=1000,
            backoff_multiplier=2.0,
            recovery_step_ms=50,
        )
        throttle = AIMDThrottle(config)

        throttle.on_capacity_error()  # 50
        throttle.on_capacity_error()  # 100
        throttle.on_capacity_error()  # 200
        throttle.on_success()  # 150
        throttle.on_success()  # 100

        stats = throttle.get_stats()
        assert stats["peak_delay_ms"] == 200
        assert stats["current_delay_ms"] == 100

    def test_stats_track_total_throttle_time(self) -> None:
        """Stats should track total time spent throttled."""
        throttle = AIMDThrottle()

        # Record some throttle time manually (simulating waits)
        throttle.record_throttle_wait(100.0)
        throttle.record_throttle_wait(50.0)

        stats = throttle.get_stats()
        assert stats["total_throttle_time_ms"] == 150.0

    def test_stats_reset(self) -> None:
        """Stats can be reset."""
        throttle = AIMDThrottle()

        throttle.on_capacity_error()
        throttle.on_success()

        throttle.reset_stats()

        stats = throttle.get_stats()
        assert stats["capacity_retries"] == 0
        assert stats["successes"] == 0
        # current_delay is NOT reset - only counters
        assert stats["current_delay_ms"] == 0  # Was recovered to 0
