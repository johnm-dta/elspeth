# src/elspeth/engine/retry.py
"""RetryManager: Retry logic with tenacity integration.

Provides configurable retry behavior for transform execution:
- Exponential backoff with jitter
- Configurable max attempts
- Retryable error filtering
- Attempt tracking for Landscape
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from elspeth.contracts import RetryPolicy

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Raised when max retry attempts are exceeded."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Max retries ({attempts}) exceeded: {last_error}")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    max_attempts is the TOTAL number of tries, not the number of retries.
    So max_attempts=3 means: try, retry, retry (3 total).
    """

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: float = 1.0  # seconds

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        """Factory for no-retry configuration (single attempt)."""
        return cls(max_attempts=1)

    @classmethod
    def from_policy(cls, policy: RetryPolicy | None) -> "RetryConfig":
        """Factory from plugin policy dict with safe defaults.

        Handles missing/malformed policy gracefully.
        This is a trust boundary - external config may have invalid values.
        """
        if policy is None:
            return cls.no_retry()

        return cls(
            max_attempts=max(1, policy.get("max_attempts", 3)),
            base_delay=max(0.01, policy.get("base_delay", 1.0)),
            max_delay=max(0.1, policy.get("max_delay", 60.0)),
            jitter=max(0.0, policy.get("jitter", 1.0)),
        )


class RetryManager:
    """Manages retry logic for transform execution.

    Uses tenacity for exponential backoff with jitter.
    Integrates with Landscape for attempt tracking.

    Example:
        manager = RetryManager(RetryConfig(max_attempts=3))

        result = manager.execute_with_retry(
            operation=lambda: transform.process(row, ctx),
            is_retryable=lambda e: e.retryable,
            on_retry=lambda attempt, error: recorder.record_attempt(attempt, error),
        )
    """

    def __init__(self, config: RetryConfig) -> None:
        """Initialize with config.

        Args:
            config: Retry configuration
        """
        self._config = config

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        *,
        is_retryable: Callable[[BaseException], bool],
        on_retry: Callable[[int, BaseException], None] | None = None,
    ) -> T:
        """Execute operation with retry logic.

        Args:
            operation: Operation to execute
            is_retryable: Function to check if error is retryable
            on_retry: Optional callback on retry (attempt, error)

        Returns:
            Result of operation

        Raises:
            MaxRetriesExceeded: If max attempts exceeded
            Exception: If non-retryable error occurs
        """
        attempt = 0
        last_error: BaseException | None = None

        try:
            for attempt_state in Retrying(
                stop=stop_after_attempt(self._config.max_attempts),
                wait=wait_exponential_jitter(
                    initial=self._config.base_delay,
                    max=self._config.max_delay,
                    jitter=self._config.jitter,
                ),
                retry=retry_if_exception(is_retryable),
                reraise=False,  # We catch RetryError and convert to MaxRetriesExceeded
            ):
                with attempt_state:
                    attempt = attempt_state.retry_state.attempt_number
                    try:
                        return operation()
                    except Exception as e:
                        last_error = e
                        # Only call on_retry for retryable errors that will be retried
                        if is_retryable(e) and on_retry:
                            on_retry(attempt, e)
                        raise

        except RetryError as e:
            # Retries exhausted - wrap in MaxRetriesExceeded
            # last_error is always set because RetryError means at least one attempt failed
            final_error = last_error or e.last_attempt.exception()
            assert final_error is not None, "RetryError without exception is impossible"
            raise MaxRetriesExceeded(attempt, final_error) from e

        # Should not reach here - Retrying always returns or raises
        raise RuntimeError("Unexpected state in retry loop")  # pragma: no cover
