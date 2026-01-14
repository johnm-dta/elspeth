"""Registry for managing rate limiters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.core.rate_limit.limiter import RateLimiter

if TYPE_CHECKING:
    from elspeth.core.config import RateLimitSettings


class NoOpLimiter:
    """No-op limiter when rate limiting is disabled."""

    def acquire(self, weight: int = 1) -> None:
        """No-op acquire (always succeeds instantly)."""

    def try_acquire(self, weight: int = 1) -> bool:
        """No-op try_acquire (always succeeds)."""
        return True


class RateLimitRegistry:
    """Registry that manages rate limiters per service.

    Creates limiters on demand based on configuration.
    Reuses limiter instances for the same service.

    Example:
        settings = RateLimitSettings(...)
        registry = RateLimitRegistry(settings)

        # In external call code:
        limiter = registry.get_limiter("openai")
        limiter.acquire()
        response = call_openai()
    """

    def __init__(self, settings: RateLimitSettings) -> None:
        """Initialize registry with rate limit settings.

        Args:
            settings: Rate limit configuration
        """
        self._settings = settings
        self._limiters: dict[str, RateLimiter | NoOpLimiter] = {}

    def get_limiter(self, service_name: str) -> RateLimiter | NoOpLimiter:
        """Get or create a rate limiter for a service.

        Args:
            service_name: Name of the external service

        Returns:
            RateLimiter (or NoOpLimiter if disabled)
        """
        if not self._settings.enabled:
            return NoOpLimiter()

        if service_name not in self._limiters:
            config = self._settings.get_service_config(service_name)
            self._limiters[service_name] = RateLimiter(
                name=service_name,
                requests_per_second=config.requests_per_second,
                requests_per_minute=config.requests_per_minute,
                persistence_path=self._settings.persistence_path,
            )

        return self._limiters[service_name]

    def reset_all(self) -> None:
        """Reset all limiters (for testing)."""
        self._limiters.clear()
