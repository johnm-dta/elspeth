"""Registry for managing rate limiters."""

from __future__ import annotations

import hashlib
import re
import threading
from types import TracebackType
from typing import TYPE_CHECKING

from elspeth.core.rate_limit.limiter import RateLimiter

# Mirror of RateLimiter's accepted-name grammar (limiter.py _VALID_NAME_PATTERN):
# a name matching this is already a valid bucket key and is returned unchanged.
_VALID_LIMITER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _sanitize_limiter_name(service_name: str) -> str:
    """Convert an arbitrary service name to a valid, INJECTIVE RateLimiter bucket name.

    RateLimiter requires names matching ^[A-Za-z][A-Za-z0-9_]*$ and uses the name as
    its persistent SQLite bucket/table key (``ratelimit_<name>``). The previous
    sanitizer replaced every non-alphanumeric with ``_`` with no discriminator, so
    distinct services like ``api.example`` and ``api_example`` both became
    ``api_example`` and silently shared one persistent quota bucket (elspeth-2af4e98ee2).

    This is now injective AND edge-only:
    - A name that ALREADY matches the grammar is returned byte-identical, so the
      common case keeps its existing bucket (no migration).
    - A name that must be rewritten gets a short hash discriminator derived from the
      ORIGINAL name appended, so two raw names cannot sanitize to the same bucket.
      The only buckets whose persisted name changes are exactly the special-char
      names that were sharing the buggy collided bucket.
    """
    if _VALID_LIMITER_NAME.match(service_name):
        return service_name

    # Rewrite: replace non-alphanumerics, ensure a leading letter, then append a
    # discriminator from the ORIGINAL name to restore injectivity within the
    # limiter-name charset. Hex keeps the suffix in-grammar.
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", service_name)
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "svc_" + sanitized
    suffix = hashlib.sha256(service_name.encode()).hexdigest()[:8]
    return f"{sanitized}_{suffix}"


if TYPE_CHECKING:
    from elspeth.contracts.config.protocols import RuntimeRateLimitProtocol


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

    def __init__(self, config: RuntimeRateLimitProtocol) -> None:
        """Initialize registry with rate limit configuration.

        Args:
            config: Runtime rate limit configuration
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
                    name=_sanitize_limiter_name(service_name),
                    requests_per_minute=service_config.requests_per_minute,
                    persistence_path=self._config.persistence_path,
                )

            return self._limiters[service_name]

    def close(self) -> None:
        """Close all limiters and release resources.

        Should be called when the registry is no longer needed.
        """
        with self._lock:
            for limiter in self._limiters.values():
                limiter.close()
            self._limiters.clear()
