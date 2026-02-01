# tests/engine/test_retry.py
"""Tests for RetryManager."""

import pytest

from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.engine.retry import MaxRetriesExceeded, RetryManager


class TestRetryManager:
    """Retry logic with tenacity."""

    def test_retry_on_retryable_error(self) -> None:
        manager = RetryManager(RuntimeRetryConfig(max_attempts=3, base_delay=0.01, max_delay=60.0, jitter=1.0, exponential_base=2.0))

        call_count = 0

        def flaky_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = manager.execute_with_retry(
            flaky_operation,
            is_retryable=lambda e: isinstance(e, ValueError),
        )

        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retryable(self) -> None:
        manager = RetryManager(RuntimeRetryConfig(max_attempts=3, base_delay=0.01, max_delay=60.0, jitter=1.0, exponential_base=2.0))

        call_count = 0

        def failing_operation() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            manager.execute_with_retry(
                failing_operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        # Verify non-retryable error does NOT trigger retries - exactly 1 call
        assert call_count == 1

    def test_max_attempts_exceeded(self) -> None:
        manager = RetryManager(RuntimeRetryConfig(max_attempts=2, base_delay=0.01, max_delay=60.0, jitter=1.0, exponential_base=2.0))

        def always_fails() -> None:
            raise ValueError("Always fails")

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            manager.execute_with_retry(
                always_fails,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        assert exc_info.value.attempts == 2

    def test_on_retry_uses_zero_based_attempts(self) -> None:
        """on_retry receives 0-based attempt numbers matching audit convention.

        Landscape audit system uses 0-based attempts (first attempt = 0).
        The on_retry callback must use the same convention for consistency.
        """
        manager = RetryManager(RuntimeRetryConfig(max_attempts=3, base_delay=0.01, max_delay=60.0, jitter=1.0, exponential_base=2.0))
        attempts: list[tuple[int, str]] = []

        call_count = 0

        def flaky_with_tracking() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "ok"

        result = manager.execute_with_retry(
            flaky_with_tracking,
            is_retryable=lambda e: isinstance(e, ValueError),
            on_retry=lambda attempt, error: attempts.append((attempt, str(error))),
        )

        assert result == "ok"
        assert len(attempts) == 1
        # First attempt is 0, so callback receives 0 (not 1)
        assert attempts[0][0] == 0, "on_retry should use 0-based attempt numbering"

    def test_on_retry_not_called_on_final_attempt(self) -> None:
        """on_retry should NOT fire when no retry will occur.

        With max_attempts=1, there's only one attempt and no retries possible.
        The callback should never be invoked because no retry is scheduled.
        """
        manager = RetryManager(RuntimeRetryConfig(max_attempts=1, base_delay=0.01, max_delay=60.0, jitter=1.0, exponential_base=2.0))
        attempts: list[tuple[int, BaseException]] = []

        with pytest.raises(MaxRetriesExceeded):
            manager.execute_with_retry(
                lambda: (_ for _ in ()).throw(ValueError("Fail")),
                is_retryable=lambda e: isinstance(e, ValueError),
                on_retry=lambda attempt, error: attempts.append((attempt, error)),
            )

        assert len(attempts) == 0, "on_retry should not fire with max_attempts=1"

    def test_on_retry_not_called_on_exhausted_retries(self) -> None:
        """on_retry should NOT fire for the final failing attempt.

        With max_attempts=3 and all attempts failing, on_retry should be
        called twice (after attempts 0 and 1), but NOT after attempt 2
        because no retry will follow.
        """
        manager = RetryManager(RuntimeRetryConfig(max_attempts=3, base_delay=0.01, max_delay=60.0, jitter=1.0, exponential_base=2.0))
        attempts: list[int] = []

        with pytest.raises(MaxRetriesExceeded):
            manager.execute_with_retry(
                lambda: (_ for _ in ()).throw(ValueError("Always fails")),
                is_retryable=lambda e: isinstance(e, ValueError),
                on_retry=lambda attempt, error: attempts.append(attempt),
            )

        # With 3 max attempts, on_retry fires after attempts 0 and 1 (before retries 2 and 3)
        # It should NOT fire after attempt 2 (the final attempt)
        assert attempts == [0, 1], f"Expected [0, 1], got {attempts}"

    def test_from_policy_none_returns_no_retry(self) -> None:
        """Missing policy defaults to no-retry for safety."""
        config = RuntimeRetryConfig.from_policy(None)

        assert config.max_attempts == 1

    def test_from_policy_handles_malformed(self) -> None:
        """Malformed policy values are clamped to safe minimums."""
        config = RuntimeRetryConfig.from_policy(
            {
                "max_attempts": -5,  # Invalid, should clamp to 1
                "base_delay": -1,  # Invalid, should clamp to 0.01
            }
        )

        assert config.max_attempts == 1
        assert config.base_delay >= 0.01


class TestRuntimeRetryConfig:
    """RuntimeRetryConfig validation and factories."""

    def test_from_settings_creates_config(self) -> None:
        """RetrySettings maps to RuntimeRetryConfig."""
        from elspeth.core.config import RetrySettings

        settings = RetrySettings(
            max_attempts=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=120.0,
            exponential_base=3.0,
        )

        config = RuntimeRetryConfig.from_settings(settings)

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        # jitter defaults to 1.0 when not specified in settings
        assert config.jitter == 1.0
        # exponential_base must be preserved (P2-2026-01-21 bug fix)
        assert config.exponential_base == 3.0

    def test_default_values(self) -> None:
        config = RuntimeRetryConfig.default()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter == 1.0
        assert config.exponential_base == 2.0

    def test_invalid_max_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RuntimeRetryConfig(max_attempts=0, base_delay=1.0, max_delay=60.0, jitter=1.0, exponential_base=2.0)

        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RuntimeRetryConfig(max_attempts=-1, base_delay=1.0, max_delay=60.0, jitter=1.0, exponential_base=2.0)

    def test_no_retry_factory(self) -> None:
        config = RuntimeRetryConfig.no_retry()

        assert config.max_attempts == 1

    def test_from_policy_with_valid_values(self) -> None:
        config = RuntimeRetryConfig.from_policy(
            {
                "max_attempts": 5,
                "base_delay": 2.0,
                "max_delay": 120.0,
                "jitter": 0.5,
                "exponential_base": 4.0,
            }
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter == 0.5
        assert config.exponential_base == 4.0


class TestMaxRetriesExceeded:
    """MaxRetriesExceeded exception."""

    def test_preserves_attempt_count(self) -> None:
        original = ValueError("original error")
        exc = MaxRetriesExceeded(attempts=3, last_error=original)

        assert exc.attempts == 3
        assert exc.last_error is original

    def test_message_format(self) -> None:
        original = ValueError("original error")
        exc = MaxRetriesExceeded(attempts=5, last_error=original)

        # Assert exact message format, not just substring presence
        assert str(exc) == "Max retries (5) exceeded: original error"
