"""Rate limiting for external calls.

Uses pyrate-limiter with SQLite persistence.
"""

from elspeth.core.rate_limit.limiter import RateLimiter

__all__ = ["RateLimiter"]
