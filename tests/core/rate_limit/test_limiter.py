"""Tests for rate limiter."""

from __future__ import annotations

import time
from pathlib import Path


class TestRateLimiter:
    """Tests for rate limiting wrapper."""

    def test_create_limiter(self) -> None:
        """Can create a rate limiter."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(
            name="test_api",
            requests_per_second=10,
        ) as limiter:
            assert limiter.name == "test_api"

    def test_acquire_within_limit(self) -> None:
        """acquire() succeeds when under limit."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="test", requests_per_second=100) as limiter:
            # Should not raise or block significantly
            start = time.monotonic()
            limiter.acquire()
            elapsed = time.monotonic() - start

            assert elapsed < 0.1  # Should be near-instant

    def test_acquire_blocks_when_exceeded(self) -> None:
        """acquire() blocks when rate exceeded."""
        from elspeth.core.rate_limit import RateLimiter

        # Very restrictive: 1 request per second
        with RateLimiter(name="test", requests_per_second=1) as limiter:
            # First request: instant
            limiter.acquire()

            # Second request: should block ~1 second
            start = time.monotonic()
            limiter.acquire()
            elapsed = time.monotonic() - start

            assert elapsed >= 0.9  # Should have waited ~1s

    def test_try_acquire_returns_false_when_exceeded(self) -> None:
        """try_acquire() returns False instead of blocking."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="test", requests_per_second=1) as limiter:
            # First: succeeds
            assert limiter.try_acquire() is True

            # Second (immediate): should fail without blocking
            assert limiter.try_acquire() is False

    def test_limiter_with_sqlite_persistence(self, tmp_path: Path) -> None:
        """Rate limits persist across limiter instances."""
        from elspeth.core.rate_limit import RateLimiter

        db_path = tmp_path / "limits.db"

        # First limiter uses up the quota
        limiter1 = RateLimiter(
            name="persistent",
            requests_per_second=1,
            persistence_path=str(db_path),
        )
        limiter1.acquire()
        limiter1.close()  # Clean up first limiter

        # Second limiter (same name, same db) should see used quota
        limiter2 = RateLimiter(
            name="persistent",
            requests_per_second=1,
            persistence_path=str(db_path),
        )

        # Should fail because quota already used
        assert limiter2.try_acquire() is False
        limiter2.close()

    def test_limiter_context_manager(self) -> None:
        """RateLimiter can be used as context manager."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="ctx_test", requests_per_second=10) as limiter:
            limiter.acquire()
            assert limiter.try_acquire() is True

    def test_weight_parameter(self) -> None:
        """acquire() respects weight parameter."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="weighted", requests_per_second=5) as limiter:
            # Use up all 5 tokens at once
            limiter.acquire(weight=5)

            # Should fail - all tokens used
            assert limiter.try_acquire(weight=1) is False

    def test_requests_per_minute_limit(self) -> None:
        """Supports per-minute rate limits.

        Note: pyrate-limiter sorts rates by interval and uses an optimization
        that may skip checking longer-interval rates when the bucket is under
        the shorter-interval limit. Our implementation uses separate limiters
        for each rate interval to ensure both limits are properly enforced.
        """
        from elspeth.core.rate_limit import RateLimiter

        # Allow many per second but only 3 per minute total
        with RateLimiter(
            name="minute_limit",
            requests_per_second=100,  # Very permissive per-second
            requests_per_minute=3,    # But only 3 per minute total
        ) as limiter:
            # First three should work (under minute limit)
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True

            # Fourth should fail (hit minute limit)
            assert limiter.try_acquire() is False
