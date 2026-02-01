# tests/testing/chaosllm/test_error_injector.py
"""Tests for ChaosLLM error injector."""

import random
import threading

import pytest

from elspeth.testing.chaosllm.config import BurstConfig, ErrorInjectionConfig
from elspeth.testing.chaosllm.error_injector import (
    CONNECTION_ERRORS,
    HTTP_ERRORS,
    MALFORMED_TYPES,
    ErrorCategory,
    ErrorDecision,
    ErrorInjector,
)


class FixedRandom(random.Random):
    """A Random instance that returns a fixed value for testing."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._fixed_value = value

    def random(self) -> float:
        return self._fixed_value


class TestErrorDecision:
    """Tests for ErrorDecision dataclass."""

    def test_success_decision(self) -> None:
        """Success decision has no error type."""
        decision = ErrorDecision.success()
        assert decision.error_type is None
        assert decision.status_code is None
        assert decision.should_inject is False
        assert decision.category is None

    def test_http_error_decision(self) -> None:
        """HTTP error decision has correct fields."""
        decision = ErrorDecision.http_error("rate_limit", 429, retry_after_sec=5)
        assert decision.error_type == "rate_limit"
        assert decision.status_code == 429
        assert decision.retry_after_sec == 5
        assert decision.category == ErrorCategory.HTTP
        assert decision.should_inject is True

    def test_http_error_without_retry_after(self) -> None:
        """HTTP error can omit retry_after_sec."""
        decision = ErrorDecision.http_error("internal_error", 500)
        assert decision.error_type == "internal_error"
        assert decision.status_code == 500
        assert decision.retry_after_sec is None
        assert decision.should_inject is True

    def test_connection_error_decision(self) -> None:
        """Connection error decision has correct fields."""
        decision = ErrorDecision.connection_error("timeout", delay_sec=30.5)
        assert decision.error_type == "timeout"
        assert decision.category == ErrorCategory.CONNECTION
        assert decision.delay_sec == 30.5
        assert decision.status_code is None
        assert decision.should_inject is True

    def test_connection_error_without_delay(self) -> None:
        """Connection error can omit delay_sec."""
        decision = ErrorDecision.connection_error("connection_reset")
        assert decision.error_type == "connection_reset"
        assert decision.delay_sec is None
        assert decision.should_inject is True

    def test_malformed_response_decision(self) -> None:
        """Malformed response decision has correct fields."""
        decision = ErrorDecision.malformed_response("invalid_json")
        assert decision.error_type == "malformed"
        assert decision.status_code == 200  # Malformed still returns 200
        assert decision.category == ErrorCategory.MALFORMED
        assert decision.malformed_type == "invalid_json"
        assert decision.should_inject is True

    def test_is_connection_level_property(self) -> None:
        """is_connection_level property returns True for CONNECTION errors."""
        # Connection error
        decision = ErrorDecision.connection_error("timeout", delay_sec=5.0)
        assert decision.is_connection_level is True

        # HTTP error
        decision = ErrorDecision.http_error("rate_limit", 429)
        assert decision.is_connection_level is False

        # Malformed response
        decision = ErrorDecision.malformed_response("invalid_json")
        assert decision.is_connection_level is False

        # Success
        decision = ErrorDecision.success()
        assert decision.is_connection_level is False

    def test_is_malformed_property(self) -> None:
        """is_malformed property returns True for MALFORMED errors."""
        # Malformed response
        decision = ErrorDecision.malformed_response("invalid_json")
        assert decision.is_malformed is True

        # HTTP error
        decision = ErrorDecision.http_error("rate_limit", 429)
        assert decision.is_malformed is False

        # Connection error
        decision = ErrorDecision.connection_error("timeout", delay_sec=5.0)
        assert decision.is_malformed is False

        # Success
        decision = ErrorDecision.success()
        assert decision.is_malformed is False

    def test_compatibility_properties_all_error_types(self) -> None:
        """Compatibility properties work for all error types."""
        # HTTP errors
        for error_type, status_code in [("rate_limit", 429), ("internal_error", 500)]:
            decision = ErrorDecision.http_error(error_type, status_code)
            assert decision.is_connection_level is False
            assert decision.is_malformed is False

        # Connection errors
        for error_type in ["timeout", "connection_failed", "connection_stall", "connection_reset", "slow_response"]:
            decision = ErrorDecision.connection_error(error_type)
            assert decision.is_connection_level is True
            assert decision.is_malformed is False

        # Malformed types
        for malformed_type in ["invalid_json", "truncated", "empty_body"]:
            decision = ErrorDecision.malformed_response(malformed_type)
            assert decision.is_connection_level is False
            assert decision.is_malformed is True


class TestErrorInjectorBasic:
    """Basic tests for ErrorInjector."""

    def test_no_errors_configured_always_succeeds(self) -> None:
        """With all error rates at 0, always succeeds."""
        config = ErrorInjectionConfig()  # All defaults are 0.0
        injector = ErrorInjector(config)

        for _ in range(100):
            decision = injector.decide()
            assert decision.should_inject is False

    def test_hundred_percent_always_triggers(self) -> None:
        """100% error rate always triggers."""
        config = ErrorInjectionConfig(rate_limit_pct=100.0)
        injector = ErrorInjector(config)

        for _ in range(100):
            decision = injector.decide()
            assert decision.should_inject is True
            assert decision.error_type == "rate_limit"
            assert decision.status_code == 429

    def test_retry_after_in_range(self) -> None:
        """Retry-After header is within configured range."""
        config = ErrorInjectionConfig(
            rate_limit_pct=100.0,
            retry_after_sec=(3, 7),
        )
        injector = ErrorInjector(config)

        retry_values = set()
        for _ in range(50):
            decision = injector.decide()
            assert decision.retry_after_sec is not None
            retry_values.add(decision.retry_after_sec)
            assert 3 <= decision.retry_after_sec <= 7

        # Should see some variation
        assert len(retry_values) > 1


class TestHTTPErrors:
    """Tests for HTTP-level error injection."""

    @pytest.mark.parametrize(
        "error_type,status_code,config_field",
        [
            ("rate_limit", 429, "rate_limit_pct"),
            ("capacity_529", 529, "capacity_529_pct"),
            ("service_unavailable", 503, "service_unavailable_pct"),
            ("bad_gateway", 502, "bad_gateway_pct"),
            ("gateway_timeout", 504, "gateway_timeout_pct"),
            ("internal_error", 500, "internal_error_pct"),
            ("forbidden", 403, "forbidden_pct"),
            ("not_found", 404, "not_found_pct"),
        ],
    )
    def test_http_error_types(
        self,
        error_type: str,
        status_code: int,
        config_field: str,
    ) -> None:
        """Each HTTP error type returns correct status code."""
        config = ErrorInjectionConfig(**{config_field: 100.0})
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == error_type
        assert decision.status_code == status_code
        assert decision.category == ErrorCategory.HTTP

    def test_rate_limit_has_retry_after(self) -> None:
        """429 rate limit includes Retry-After header."""
        config = ErrorInjectionConfig(rate_limit_pct=100.0)
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.retry_after_sec is not None

    def test_capacity_529_has_retry_after(self) -> None:
        """529 capacity error includes Retry-After header."""
        config = ErrorInjectionConfig(capacity_529_pct=100.0)
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.retry_after_sec is not None

    def test_other_http_errors_no_retry_after(self) -> None:
        """Non-429/529 errors don't have Retry-After."""
        config = ErrorInjectionConfig(internal_error_pct=100.0)
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.retry_after_sec is None


class TestConnectionErrors:
    """Tests for connection-level error injection."""

    def test_timeout_error(self) -> None:
        """Timeout error has delay and correct category."""
        config = ErrorInjectionConfig(
            timeout_pct=100.0,
            timeout_sec=(10, 20),
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "timeout"
        assert decision.category == ErrorCategory.CONNECTION
        assert decision.delay_sec is not None
        assert 10.0 <= decision.delay_sec <= 20.0

    def test_connection_failed_error(self) -> None:
        """Connection failed error has lead time and correct category."""
        config = ErrorInjectionConfig(
            connection_failed_pct=100.0,
            connection_failed_lead_sec=(2, 5),
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "connection_failed"
        assert decision.category == ErrorCategory.CONNECTION
        assert decision.start_delay_sec is not None
        assert 2.0 <= decision.start_delay_sec <= 5.0
        assert decision.delay_sec is None

    def test_connection_stall_error(self) -> None:
        """Connection stall error has start and stall delays."""
        config = ErrorInjectionConfig(
            connection_stall_pct=100.0,
            connection_stall_start_sec=(1, 2),
            connection_stall_sec=(10, 20),
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "connection_stall"
        assert decision.category == ErrorCategory.CONNECTION
        assert decision.start_delay_sec is not None
        assert decision.delay_sec is not None
        assert 1.0 <= decision.start_delay_sec <= 2.0
        assert 10.0 <= decision.delay_sec <= 20.0

    def test_connection_reset_error(self) -> None:
        """Connection reset error has correct type."""
        config = ErrorInjectionConfig(connection_reset_pct=100.0)
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "connection_reset"
        assert decision.category == ErrorCategory.CONNECTION
        assert decision.delay_sec is None  # No delay for reset

    def test_slow_response_error(self) -> None:
        """Slow response error has delay and correct category."""
        config = ErrorInjectionConfig(
            slow_response_pct=100.0,
            slow_response_sec=(5, 15),
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "slow_response"
        assert decision.category == ErrorCategory.CONNECTION
        assert decision.delay_sec is not None
        assert 5.0 <= decision.delay_sec <= 15.0

    def test_connection_errors_take_priority(self) -> None:
        """Connection errors have priority over HTTP errors."""
        config = ErrorInjectionConfig(
            timeout_pct=100.0,
            rate_limit_pct=100.0,
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "timeout"
        assert decision.category == ErrorCategory.CONNECTION


class TestMalformedResponses:
    """Tests for malformed response injection."""

    @pytest.mark.parametrize(
        "malformed_type,config_field",
        [
            ("invalid_json", "invalid_json_pct"),
            ("truncated", "truncated_pct"),
            ("empty_body", "empty_body_pct"),
            ("missing_fields", "missing_fields_pct"),
            ("wrong_content_type", "wrong_content_type_pct"),
        ],
    )
    def test_malformed_response_types(
        self,
        malformed_type: str,
        config_field: str,
    ) -> None:
        """Each malformed type is correctly identified."""
        config = ErrorInjectionConfig(**{config_field: 100.0})
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "malformed"
        assert decision.malformed_type == malformed_type
        assert decision.category == ErrorCategory.MALFORMED
        assert decision.status_code == 200  # Malformed returns 200

    def test_http_errors_take_priority_over_malformed(self) -> None:
        """HTTP errors have priority over malformed responses."""
        config = ErrorInjectionConfig(
            rate_limit_pct=100.0,
            invalid_json_pct=100.0,
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "rate_limit"
        assert decision.category == ErrorCategory.HTTP


class TestBurstStateMachine:
    """Tests for burst mode state machine."""

    def test_burst_disabled_by_default(self) -> None:
        """Burst mode is disabled by default."""
        config = ErrorInjectionConfig()
        injector = ErrorInjector(config)
        assert injector.is_in_burst() is False

    def test_burst_enabled_at_start(self) -> None:
        """When enabled, burst starts at time 0."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=30,
            duration_sec=5,
        )
        config = ErrorInjectionConfig(burst=burst)

        # Use fixed time starting at 0
        current_time = 0.0
        injector = ErrorInjector(config, time_func=lambda: current_time)

        assert injector.is_in_burst() is True

    def test_burst_cycle(self) -> None:
        """Burst cycles through on/off periods."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=10,
            duration_sec=3,
        )
        config = ErrorInjectionConfig(burst=burst)

        # Simulate time progression
        current_time = [0.0]

        def mock_time() -> float:
            return current_time[0]

        injector = ErrorInjector(config, time_func=mock_time)

        # t=0: start of burst
        assert injector.is_in_burst() is True

        # t=2: still in burst
        current_time[0] = 2.0
        assert injector.is_in_burst() is True

        # t=3: just exited burst
        current_time[0] = 3.0
        assert injector.is_in_burst() is False

        # t=9: still not in burst
        current_time[0] = 9.0
        assert injector.is_in_burst() is False

        # t=10: new burst started
        current_time[0] = 10.0
        assert injector.is_in_burst() is True

        # t=12: still in burst
        current_time[0] = 12.0
        assert injector.is_in_burst() is True

        # t=13: exited burst
        current_time[0] = 13.0
        assert injector.is_in_burst() is False

    def test_burst_elevates_rate_limit(self) -> None:
        """Burst mode uses elevated rate_limit_pct."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=10,
            duration_sec=5,
            rate_limit_pct=100.0,  # 100% during burst
        )
        config = ErrorInjectionConfig(
            rate_limit_pct=0.0,  # 0% normally
            burst=burst,
        )

        current_time = [0.0]
        injector = ErrorInjector(config, time_func=lambda: current_time[0])

        # During burst: should always trigger rate limit
        for _ in range(10):
            decision = injector.decide()
            assert decision.error_type == "rate_limit"

        # Outside burst: should never trigger
        current_time[0] = 6.0  # Past burst duration
        for _ in range(10):
            decision = injector.decide()
            assert decision.should_inject is False

    def test_burst_elevates_capacity_529(self) -> None:
        """Burst mode uses elevated capacity_pct."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=10,
            duration_sec=5,
            rate_limit_pct=0.0,  # Disable rate_limit during burst to test capacity
            capacity_pct=100.0,  # 100% during burst
        )
        config = ErrorInjectionConfig(
            rate_limit_pct=0.0,  # Disable rate_limit normally too
            capacity_529_pct=0.0,  # 0% normally
            burst=burst,
        )

        current_time = [0.0]
        injector = ErrorInjector(config, time_func=lambda: current_time[0])

        # During burst: should always trigger capacity error
        for _ in range(10):
            decision = injector.decide()
            assert decision.error_type == "capacity_529"

        # Outside burst: should never trigger
        current_time[0] = 6.0
        for _ in range(10):
            decision = injector.decide()
            assert decision.should_inject is False

    def test_burst_does_not_affect_other_errors(self) -> None:
        """Burst mode only affects rate_limit and capacity_529."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=10,
            duration_sec=5,
            rate_limit_pct=0.0,  # Disable burst rate_limit
            capacity_pct=0.0,  # Disable burst capacity
        )
        config = ErrorInjectionConfig(
            internal_error_pct=100.0,  # Not affected by burst
            burst=burst,
        )

        current_time = [0.0]
        injector = ErrorInjector(config, time_func=lambda: current_time[0])

        # During burst: internal error still fires (not rate limited)
        decision = injector.decide()
        assert decision.error_type == "internal_error"

        # Outside burst: same behavior
        current_time[0] = 6.0
        decision = injector.decide()
        assert decision.error_type == "internal_error"


class TestRandomDecisions:
    """Tests for random decision logic."""

    def test_fifty_percent_roughly_half(self) -> None:
        """50% error rate triggers roughly half the time."""
        config = ErrorInjectionConfig(rate_limit_pct=50.0)
        injector = ErrorInjector(config)

        errors = sum(1 for _ in range(1000) if injector.decide().should_inject)
        # Allow reasonable variance - should be between 40-60%
        assert 400 <= errors <= 600

    def test_deterministic_with_seeded_random(self) -> None:
        """Can use seeded random for deterministic testing."""
        config = ErrorInjectionConfig(rate_limit_pct=50.0)

        # FixedRandom that always returns 0.3 (30%)
        # 30% * 100 = 30 < 50% threshold, so should trigger
        injector = ErrorInjector(config, rng=FixedRandom(0.3))
        assert injector.decide().should_inject is True

        # FixedRandom that returns 0.6 (60%)
        # 60% * 100 = 60 >= 50% threshold, so should NOT trigger
        injector = ErrorInjector(config, rng=FixedRandom(0.6))
        assert injector.decide().should_inject is False

    def test_zero_percent_never_triggers(self) -> None:
        """0% error rate never triggers."""
        config = ErrorInjectionConfig(rate_limit_pct=0.0)
        injector = ErrorInjector(config)

        for _ in range(100):
            assert injector.decide().should_inject is False

    def test_seeded_random_deterministic_retry_after(self) -> None:
        """Retry-After values are deterministic with seeded random."""
        config = ErrorInjectionConfig(
            rate_limit_pct=100.0,
            retry_after_sec=(1, 10),
        )
        # Same seed should produce same sequence
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        injector1 = ErrorInjector(config, rng=rng1)
        injector2 = ErrorInjector(config, rng=rng2)

        for _ in range(10):
            d1 = injector1.decide()
            d2 = injector2.decide()
            assert d1.retry_after_sec == d2.retry_after_sec

    def test_seeded_random_deterministic_timeout_delay(self) -> None:
        """Timeout delay values are deterministic with seeded random."""
        config = ErrorInjectionConfig(
            timeout_pct=100.0,
            timeout_sec=(5.0, 30.0),
        )
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        injector1 = ErrorInjector(config, rng=rng1)
        injector2 = ErrorInjector(config, rng=rng2)

        for _ in range(10):
            d1 = injector1.decide()
            d2 = injector2.decide()
            assert d1.delay_sec == d2.delay_sec

    def test_seeded_random_deterministic_slow_response_delay(self) -> None:
        """Slow response delay values are deterministic with seeded random."""
        config = ErrorInjectionConfig(
            slow_response_pct=100.0,
            slow_response_sec=(1.0, 5.0),
        )
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        injector1 = ErrorInjector(config, rng=rng1)
        injector2 = ErrorInjector(config, rng=rng2)

        for _ in range(10):
            d1 = injector1.decide()
            d2 = injector2.decide()
            assert d1.delay_sec == d2.delay_sec


class TestErrorPriority:
    """Tests for error priority ordering."""

    def test_connection_before_http_before_malformed(self) -> None:
        """Priority: connection > HTTP > malformed."""
        config = ErrorInjectionConfig(
            timeout_pct=100.0,  # Connection
            rate_limit_pct=100.0,  # HTTP
            invalid_json_pct=100.0,  # Malformed
        )
        injector = ErrorInjector(config)

        # Should always get timeout (highest priority)
        for _ in range(10):
            decision = injector.decide()
            assert decision.error_type == "timeout"

    def test_http_error_order_within_category(self) -> None:
        """Within HTTP errors, rate_limit has priority."""
        config = ErrorInjectionConfig(
            rate_limit_pct=100.0,
            internal_error_pct=100.0,
        )
        injector = ErrorInjector(config)

        decision = injector.decide()
        assert decision.error_type == "rate_limit"


class TestWeightedSelection:
    """Tests for weighted error selection."""

    def test_weighted_mix_selects_multiple_types(self) -> None:
        """Weighted mode mixes error types instead of strict priority."""
        config = ErrorInjectionConfig(
            rate_limit_pct=50.0,
            capacity_529_pct=50.0,
            selection_mode="weighted",
        )
        injector = ErrorInjector(config, rng=random.Random(123))

        decisions = [injector.decide().error_type for _ in range(20)]
        assert "rate_limit" in decisions
        assert "capacity_529" in decisions


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_decide_calls(self) -> None:
        """Multiple threads can call decide() safely."""
        config = ErrorInjectionConfig(rate_limit_pct=50.0)
        injector = ErrorInjector(config)

        results = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(100):
                decision = injector.decide()
                with lock:
                    results.append(decision.should_inject)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads completed successfully
        assert len(results) == 1000
        # Should have some mix of True and False
        assert any(r for r in results)
        assert any(not r for r in results)

    def test_burst_state_thread_safe(self) -> None:
        """Burst state tracking is thread-safe."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=100,
            duration_sec=50,
        )
        config = ErrorInjectionConfig(burst=burst)
        injector = ErrorInjector(config)

        results = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(100):
                is_burst = injector.is_in_burst()
                with lock:
                    results.append(is_burst)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1000
        # All should be True (we're in burst at start)
        assert all(r for r in results)

    def test_reset_thread_safe(self) -> None:
        """Reset can be called from any thread."""
        config = ErrorInjectionConfig()
        injector = ErrorInjector(config)

        # Make some decisions to establish start time
        for _ in range(10):
            injector.decide()

        # Reset from multiple threads
        def worker() -> None:
            injector.reset()
            for _ in range(50):
                injector.decide()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors


class TestReset:
    """Tests for injector reset functionality."""

    def test_reset_clears_burst_timing(self) -> None:
        """Reset clears the start time for burst calculations."""
        burst = BurstConfig(
            enabled=True,
            interval_sec=10,
            duration_sec=3,
        )
        config = ErrorInjectionConfig(burst=burst)

        current_time = [0.0]
        injector = ErrorInjector(config, time_func=lambda: current_time[0])

        # Initialize at time 0 (in burst)
        injector.decide()
        assert injector.is_in_burst() is True

        # Move to outside burst period (elapsed = 5, past duration=3)
        current_time[0] = 5.0
        assert injector.is_in_burst() is False

        # Move to time 15 (would be elapsed=15, past one full cycle)
        current_time[0] = 15.0
        assert injector.is_in_burst() is False  # 15 % 10 = 5, outside burst

        # Reset at time 15 - this makes elapsed = 0 again (time 15 - start 15)
        injector.reset()
        # Now is_in_burst() will re-establish start_time as current_time
        # elapsed = 15 - 15 = 0, so we're back at start of burst
        assert injector.is_in_burst() is True


class TestConstants:
    """Tests for module constants."""

    def test_http_errors_mapping(self) -> None:
        """HTTP_ERRORS contains all expected error types."""
        expected = {
            "rate_limit": 429,
            "capacity_529": 529,
            "service_unavailable": 503,
            "bad_gateway": 502,
            "gateway_timeout": 504,
            "internal_error": 500,
            "forbidden": 403,
            "not_found": 404,
        }
        assert expected == HTTP_ERRORS

    def test_connection_errors_set(self) -> None:
        """CONNECTION_ERRORS contains all expected types."""
        expected = {
            "timeout",
            "connection_failed",
            "connection_stall",
            "connection_reset",
            "slow_response",
        }
        assert expected == CONNECTION_ERRORS

    def test_malformed_types_set(self) -> None:
        """MALFORMED_TYPES contains all expected types."""
        expected = {
            "invalid_json",
            "truncated",
            "empty_body",
            "missing_fields",
            "wrong_content_type",
        }
        assert expected == MALFORMED_TYPES
