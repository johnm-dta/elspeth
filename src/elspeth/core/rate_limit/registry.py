"""Registry for managing rate limiters."""

from __future__ import annotations

import threading
from types import TracebackType
from typing import TYPE_CHECKING

from elspeth.core.rate_limit.limiter import RateLimiter

if TYPE_CHECKING:
    from elspeth.contracts.config.runtime import RuntimeRateLimitConfig


class NoOpLimiter:
    """No-op limiter when rate limiting is disabled.

    Provides the same interface as RateLimiter but does nothing.
    All operations succeed instantly without any rate limiting.
    """

    def acquire(self, weight: int = 1, timeout: float | None = None) -> None:
        """No-op acquire (always succeeds instantly).

        Args:
            weight: Number of tokens to acquire (ignored)
            timeout: Maximum wait time in seconds (ignored - always instant)
        """

    def try_acquire(self, weight: int = 1) -> bool:
        """No-op try_acquire (always succeeds)."""
        return True

    def close(self) -> None:
        """No-op close (nothing to clean up)."""

    def __enter__(self) -> NoOpLimiter:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()


class RateLimitRegistry:
    """Registry that manages rate limiters per service.

    Creates limiters on demand based on configuration.
    Reuses limiter instances for the same service.
    Thread-safe for concurrent access.

    Example:
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig

        config = RuntimeRateLimitConfig.from_settings(settings.rate_limit)
        registry = RateLimitRegistry(config)

        # In external call code:
        limiter = registry.get_limiter("openai")
        limiter.acquire()
        response = call_openai()

        # Clean up when done
        registry.close()
    """

    def __init__(self, config: RuntimeRateLimitConfig) -> None:
        """Initialize registry with rate limit configuration.

        Args:
            config: Runtime rate limit configuration (from RuntimeRateLimitConfig.from_settings())
        """
        self._config = config
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
        self._noop_limiter = NoOpLimiter()

    def get_limiter(self, service_name: str) -> RateLimiter | NoOpLimiter:
        """Get or create a rate limiter for a service.

        Thread-safe: multiple threads can call this concurrently.

        Args:
            service_name: Name of the external service

        Returns:
            RateLimiter (or NoOpLimiter if disabled)
        """
        if not self._config.enabled:
            return self._noop_limiter

        with self._lock:
            if service_name not in self._limiters:
                service_config = self._config.get_service_config(service_name)
                self._limiters[service_name] = RateLimiter(
                    name=service_name,
                    requests_per_minute=service_config.requests_per_minute,
                    persistence_path=self._config.persistence_path,
                )

            return self._limiters[service_name]

    def reset_all(self) -> None:
        """Reset all limiters (for testing).

        Closes all existing limiters and clears the registry.
        """
        with self._lock:
            for limiter in self._limiters.values():
                limiter.close()
            self._limiters.clear()

    def close(self) -> None:
        """Close all limiters and release resources.

        Should be called when the registry is no longer needed.
        """
        with self._lock:
            for limiter in self._limiters.values():
                limiter.close()
            self._limiters.clear()
