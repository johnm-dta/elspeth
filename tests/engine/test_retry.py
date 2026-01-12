# tests/engine/test_retry.py
"""Tests for RetryManager."""

import pytest


class TestRetryManager:
    """Retry logic with tenacity."""

    def test_retry_on_retryable_error(self) -> None:
        from elspeth.engine.retry import RetryConfig, RetryManager

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))

        call_count = 0

        def flaky_operation():
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
        from elspeth.engine.retry import RetryConfig, RetryManager

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))

        def failing_operation():
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            manager.execute_with_retry(
                failing_operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

    def test_max_attempts_exceeded(self) -> None:
        from elspeth.engine.retry import MaxRetriesExceeded, RetryConfig, RetryManager

        manager = RetryManager(RetryConfig(max_attempts=2, base_delay=0.01))

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            manager.execute_with_retry(
                always_fails,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        assert exc_info.value.attempts == 2

    def test_records_attempts(self) -> None:
        from elspeth.engine.retry import RetryConfig, RetryManager

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))
        attempts: list[tuple[int, str]] = []

        call_count = 0

        def flaky_with_tracking():
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
        assert attempts[0][0] == 1

    def test_from_policy_none_returns_no_retry(self) -> None:
        """Missing policy defaults to no-retry for safety."""
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig.from_policy(None)

        assert config.max_attempts == 1

    def test_from_policy_handles_malformed(self) -> None:
        """Malformed policy values are clamped to safe minimums."""
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig.from_policy(
            {
                "max_attempts": -5,  # Invalid, should clamp to 1
                "base_delay": -1,  # Invalid, should clamp to 0.01
            }
        )

        assert config.max_attempts == 1
        assert config.base_delay >= 0.01


class TestRetryConfig:
    """RetryConfig validation and factories."""

    def test_default_values(self) -> None:
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter == 1.0

    def test_invalid_max_attempts_raises(self) -> None:
        from elspeth.engine.retry import RetryConfig

        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryConfig(max_attempts=0)

        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryConfig(max_attempts=-1)

    def test_no_retry_factory(self) -> None:
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig.no_retry()

        assert config.max_attempts == 1

    def test_from_policy_with_valid_values(self) -> None:
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig.from_policy(
            {
                "max_attempts": 5,
                "base_delay": 2.0,
                "max_delay": 120.0,
                "jitter": 0.5,
            }
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter == 0.5


class TestMaxRetriesExceeded:
    """MaxRetriesExceeded exception."""

    def test_preserves_attempt_count(self) -> None:
        from elspeth.engine.retry import MaxRetriesExceeded

        original = ValueError("original error")
        exc = MaxRetriesExceeded(attempts=3, last_error=original)

        assert exc.attempts == 3
        assert exc.last_error is original

    def test_message_format(self) -> None:
        from elspeth.engine.retry import MaxRetriesExceeded

        original = ValueError("original error")
        exc = MaxRetriesExceeded(attempts=5, last_error=original)

        assert "5" in str(exc)
        assert "original error" in str(exc)
