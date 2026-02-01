# tests/property/engine/test_retry_properties.py
"""Property-based tests for retry configuration.

These tests verify that RuntimeRetryConfig correctly validates configurations
and factory methods produce valid configs.

Properties tested:
1. Valid configs are accepted
2. Invalid configs are rejected
3. Factory methods produce valid configs
4. Coercion at trust boundary (from_policy) handles bad data
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.engine.retry import MaxRetriesExceeded, RetryManager
from tests.property.conftest import valid_delays, valid_jitter, valid_max_attempts


class TestRuntimeRetryConfigValidationProperties:
    """Property tests for RuntimeRetryConfig validation."""

    @given(
        max_attempts=valid_max_attempts,
        base_delay=valid_delays,
        max_delay=valid_delays,
        jitter=valid_jitter,
    )
    @settings(max_examples=300)
    def test_valid_config_construction(
        self,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
    ) -> None:
        """Property: Valid configs construct without error."""
        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            exponential_base=2.0,
        )

        assert config.max_attempts == max_attempts
        assert config.base_delay == base_delay
        assert config.max_delay == max_delay
        assert config.jitter == jitter

    @given(max_attempts=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_invalid_max_attempts_rejected(self, max_attempts: int) -> None:
        """Property: max_attempts < 1 is rejected."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RuntimeRetryConfig(
                max_attempts=max_attempts,
                base_delay=1.0,
                max_delay=60.0,
                jitter=1.0,
                exponential_base=2.0,
            )

    @given(
        max_attempts=valid_max_attempts,
        base_delay=valid_delays,
        max_delay=valid_delays,
        jitter=valid_jitter,
    )
    @settings(max_examples=100)
    def test_config_fields_are_immutable_readable(
        self,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
    ) -> None:
        """Property: Config fields are readable after construction."""
        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            exponential_base=2.0,
        )

        # All fields must be readable
        _ = config.max_attempts
        _ = config.base_delay
        _ = config.max_delay
        _ = config.jitter


class TestRuntimeRetryConfigFactoryProperties:
    """Property tests for RuntimeRetryConfig factory methods."""

    def test_no_retry_is_single_attempt(self) -> None:
        """Property: no_retry() produces single-attempt config."""
        config = RuntimeRetryConfig.no_retry()

        assert config.max_attempts == 1, f"no_retry() should have max_attempts=1, got {config.max_attempts}"

    @given(
        max_attempts=st.integers(min_value=-100, max_value=100),
        base_delay=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        max_delay=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        jitter=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_from_policy_always_produces_valid_config(
        self,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
    ) -> None:
        """Property: from_policy() always produces valid config, even with bad input.

        This is a trust boundary - external policy config may have invalid values.
        The factory must coerce to valid values without crashing.
        """
        policy = {
            "max_attempts": max_attempts,
            "base_delay": base_delay,
            "max_delay": max_delay,
            "jitter": jitter,
        }

        # Should NOT raise, even with invalid inputs
        config = RuntimeRetryConfig.from_policy(policy)

        # Result must be valid
        assert config.max_attempts >= 1, "max_attempts must be >= 1 after coercion"
        assert config.base_delay >= 0.01, "base_delay must be >= 0.01 after coercion"
        assert config.max_delay >= 0.1, "max_delay must be >= 0.1 after coercion"
        assert config.jitter >= 0.0, "jitter must be >= 0.0 after coercion"

    def test_from_policy_none_returns_no_retry(self) -> None:
        """Property: from_policy(None) returns no-retry config."""
        config = RuntimeRetryConfig.from_policy(None)

        assert config.max_attempts == 1, f"from_policy(None) should return no_retry (max_attempts=1), got {config.max_attempts}"

    @given(policy=st.fixed_dictionaries({}))
    @settings(max_examples=10)
    def test_from_policy_empty_dict_uses_defaults(self, policy: dict) -> None:
        """Property: from_policy({}) uses sensible defaults."""
        config = RuntimeRetryConfig.from_policy(policy)

        # Should have default values
        assert config.max_attempts >= 1
        assert config.base_delay > 0
        assert config.max_delay > 0
        assert config.jitter >= 0

    @given(
        max_attempts=valid_max_attempts,
        base_delay=valid_delays,
        max_delay=valid_delays,
        jitter=valid_jitter,
    )
    @settings(max_examples=100)
    def test_from_policy_preserves_valid_values(
        self,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
    ) -> None:
        """Property: from_policy() preserves valid input values."""
        # Ensure values are above coercion thresholds
        assume(max_attempts >= 1)
        assume(base_delay >= 0.01)
        assume(max_delay >= 0.1)
        assume(jitter >= 0.0)

        policy = {
            "max_attempts": max_attempts,
            "base_delay": base_delay,
            "max_delay": max_delay,
            "jitter": jitter,
        }

        config = RuntimeRetryConfig.from_policy(policy)

        # Valid values should be preserved (not coerced)
        assert config.max_attempts == max_attempts
        assert config.base_delay == base_delay
        assert config.max_delay == max_delay
        assert config.jitter == jitter


class TestRuntimeRetryConfigCoercionProperties:
    """Property tests for trust boundary coercion in from_policy()."""

    @given(bad_max_attempts=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_negative_max_attempts_coerced_to_minimum(self, bad_max_attempts: int) -> None:
        """Property: Negative/zero max_attempts coerced to 1."""
        policy = {"max_attempts": bad_max_attempts}
        config = RuntimeRetryConfig.from_policy(policy)

        assert config.max_attempts >= 1, f"Bad max_attempts {bad_max_attempts} should coerce to >= 1, got {config.max_attempts}"

    @given(bad_base_delay=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_negative_base_delay_coerced_to_minimum(self, bad_base_delay: float) -> None:
        """Property: Negative/zero base_delay coerced to minimum."""
        policy = {"base_delay": bad_base_delay}
        config = RuntimeRetryConfig.from_policy(policy)

        assert config.base_delay >= 0.01, f"Bad base_delay {bad_base_delay} should coerce to >= 0.01, got {config.base_delay}"

    @given(bad_jitter=st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_negative_jitter_coerced_to_zero(self, bad_jitter: float) -> None:
        """Property: Negative jitter coerced to 0."""
        policy = {"jitter": bad_jitter}
        config = RuntimeRetryConfig.from_policy(policy)

        assert config.jitter >= 0.0, f"Bad jitter {bad_jitter} should coerce to >= 0.0, got {config.jitter}"


# =============================================================================
# RetryManager Execution Property Tests
# =============================================================================


class TestRetryManagerExecutionProperties:
    """Property tests for RetryManager.execute_with_retry() behavior."""

    @given(success_on_attempt=st.integers(min_value=1, max_value=5))
    @settings(max_examples=30)
    def test_callback_invoked_exactly_attempts_minus_one_times(self, success_on_attempt: int) -> None:
        """Property: on_retry called exactly (attempt - 1) times before success.

        If success happens on attempt N, on_retry is called for attempts 1..(N-1).
        """
        max_attempts = success_on_attempt + 2  # Ensure we have room to succeed

        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=0.001,  # Minimal delay for fast tests
            max_delay=0.01,
            jitter=0.0,
            exponential_base=2.0,
        )
        manager = RetryManager(config)

        callback_attempts: list[int] = []
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < success_on_attempt:
                raise ValueError(f"Fail on attempt {call_count}")
            return "success"

        def on_retry(attempt: int, error: BaseException) -> None:
            callback_attempts.append(attempt)

        result = manager.execute_with_retry(
            operation=operation,
            is_retryable=lambda e: isinstance(e, ValueError),
            on_retry=on_retry,
        )

        assert result == "success"
        assert len(callback_attempts) == success_on_attempt - 1, (
            f"Expected {success_on_attempt - 1} callbacks, got {len(callback_attempts)}"
        )

    @given(max_attempts=st.integers(min_value=1, max_value=5))
    @settings(max_examples=30)
    def test_non_retryable_error_fails_immediately(self, max_attempts: int) -> None:
        """Property: Non-retryable error causes immediate failure (no retries)."""
        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=0.001,
            max_delay=0.01,
            jitter=0.0,
            exponential_base=2.0,
        )
        manager = RetryManager(config)

        call_count = 0
        callback_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Non-retryable error")

        def on_retry(attempt: int, error: BaseException) -> None:
            nonlocal callback_count
            callback_count += 1

        with pytest.raises(RuntimeError, match="Non-retryable error"):
            manager.execute_with_retry(
                operation=operation,
                is_retryable=lambda e: isinstance(e, ValueError),  # Only ValueError is retryable
                on_retry=on_retry,
            )

        assert call_count == 1, "Non-retryable should fail on first attempt"
        assert callback_count == 0, "on_retry should not be called for non-retryable"

    @given(max_attempts=st.integers(min_value=2, max_value=5))
    @settings(max_examples=30)
    def test_max_attempts_respected(self, max_attempts: int) -> None:
        """Property: Exactly max_attempts tries before MaxRetriesExceeded."""
        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=0.001,
            max_delay=0.01,
            jitter=0.0,
            exponential_base=2.0,
        )
        manager = RetryManager(config)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Always fail - attempt {call_count}")

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            manager.execute_with_retry(
                operation=operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        assert call_count == max_attempts, f"Expected exactly {max_attempts} attempts, got {call_count}"
        assert exc_info.value.attempts == max_attempts

    def test_single_attempt_no_retry(self) -> None:
        """Property: max_attempts=1 means single attempt, no retries."""
        config = RuntimeRetryConfig.no_retry()  # max_attempts=1
        manager = RetryManager(config)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")

        with pytest.raises(MaxRetriesExceeded):
            manager.execute_with_retry(
                operation=operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        assert call_count == 1, "no_retry() should mean single attempt"

    @given(max_attempts=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_success_on_first_attempt_no_callbacks(self, max_attempts: int) -> None:
        """Property: Success on first attempt means no on_retry callbacks."""
        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=0.001,
            max_delay=0.01,
            jitter=0.0,
            exponential_base=2.0,
        )
        manager = RetryManager(config)

        callback_count = 0

        def on_retry(attempt: int, error: BaseException) -> None:
            nonlocal callback_count
            callback_count += 1

        result = manager.execute_with_retry(
            operation=lambda: "immediate success",
            is_retryable=lambda e: True,
            on_retry=on_retry,
        )

        assert result == "immediate success"
        assert callback_count == 0, "No callbacks for immediate success"


class TestRetryManagerErrorHandlingProperties:
    """Property tests for error handling in RetryManager."""

    @given(max_attempts=st.integers(min_value=2, max_value=4))
    @settings(max_examples=20)
    def test_last_error_preserved_in_max_retries_exceeded(self, max_attempts: int) -> None:
        """Property: MaxRetriesExceeded contains the last error."""
        config = RuntimeRetryConfig(
            max_attempts=max_attempts,
            base_delay=0.001,
            max_delay=0.01,
            jitter=0.0,
            exponential_base=2.0,
        )
        manager = RetryManager(config)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Error on attempt {call_count}")

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            manager.execute_with_retry(
                operation=operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        # Last error should be from the final attempt
        assert f"Error on attempt {max_attempts}" in str(exc_info.value.last_error)

    def test_none_callback_allowed(self) -> None:
        """Property: on_retry=None is allowed and works correctly."""
        config = RuntimeRetryConfig(
            max_attempts=3,
            base_delay=0.001,
            max_delay=0.01,
            jitter=0.0,
            exponential_base=2.0,
        )
        manager = RetryManager(config)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry me")
            return "success"

        # Should not raise even without callback
        result = manager.execute_with_retry(
            operation=operation,
            is_retryable=lambda e: isinstance(e, ValueError),
            on_retry=None,
        )

        assert result == "success"
        assert call_count == 2
