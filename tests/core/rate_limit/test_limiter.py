"""Tests for rate limiter."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


class TestRateLimiterValidation:
    """Tests for rate limiter input validation."""

    def test_rejects_empty_name(self) -> None:
        """Empty name is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="", requests_per_second=10)

    def test_rejects_name_starting_with_number(self) -> None:
        """Name starting with number is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="123api", requests_per_second=10)

    def test_rejects_name_with_special_characters(self) -> None:
        """Name with special characters is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="my-api", requests_per_second=10)

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="my.api", requests_per_second=10)

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="my api", requests_per_second=10)

    def test_rejects_sql_injection_attempt(self) -> None:
        """SQL injection attempts in name are rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="api; DROP TABLE users;--", requests_per_second=10)

    def test_accepts_valid_names(self) -> None:
        """Valid names are accepted."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="api", requests_per_second=10) as limiter:
            assert limiter.name == "api"

        with RateLimiter(name="my_api", requests_per_second=10) as limiter:
            assert limiter.name == "my_api"

        with RateLimiter(name="MyApi123", requests_per_second=10) as limiter:
            assert limiter.name == "MyApi123"

        with RateLimiter(name="API_v2_production", requests_per_second=10) as limiter:
            assert limiter.name == "API_v2_production"

    def test_rejects_zero_requests_per_second(self) -> None:
        """Zero requests_per_second is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="requests_per_second must be positive"):
            RateLimiter(name="test", requests_per_second=0)

    def test_rejects_negative_requests_per_second(self) -> None:
        """Negative requests_per_second is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="requests_per_second must be positive"):
            RateLimiter(name="test", requests_per_second=-5)

    def test_rejects_zero_requests_per_minute(self) -> None:
        """Zero requests_per_minute is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(name="test", requests_per_second=10, requests_per_minute=0)

    def test_rejects_negative_requests_per_minute(self) -> None:
        """Negative requests_per_minute is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(name="test", requests_per_second=10, requests_per_minute=-1)

    def test_accepts_none_requests_per_minute(self) -> None:
        """None requests_per_minute is accepted (no per-minute limit)."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(
            name="test", requests_per_second=10, requests_per_minute=None
        ) as limiter:
            assert limiter._requests_per_minute is None


class TestTryAcquireAtomicity:
    """Tests for try_acquire() atomic behavior across multiple rate limits."""

    def test_try_acquire_does_not_consume_on_partial_failure(self) -> None:
        """When per-minute limit blocks, per-second tokens are not consumed.

        This test verifies the fix for the partial consumption bug where
        try_acquire() would consume tokens from the per-second bucket even
        when the per-minute bucket would reject.
        """
        from elspeth.core.rate_limit import RateLimiter

        # Allow 100/second but only 2/minute
        with RateLimiter(
            name="atomic_test",
            requests_per_second=100,
            requests_per_minute=2,
        ) as limiter:
            # Use up the minute quota
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True

            # This should fail due to per-minute limit
            assert limiter.try_acquire() is False

            # The per-second bucket should NOT have been consumed
            # by the failed attempt. Check by examining bucket counts.
            # We should have exactly 2 items in each bucket.
            assert limiter._buckets[0].count() == 2  # per-second bucket
            assert limiter._buckets[1].count() == 2  # per-minute bucket


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
            requests_per_minute=3,  # But only 3 per minute total
        ) as limiter:
            # First three should work (under minute limit)
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True

            # Fourth should fail (hit minute limit)
            assert limiter.try_acquire() is False


class TestRateLimitRegistry:
    """Tests for rate limiter registry."""

    def test_get_or_create_limiter(self) -> None:
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_second=10)
        registry = RateLimitRegistry(settings)

        limiter1 = registry.get_limiter("api_a")
        limiter2 = registry.get_limiter("api_a")

        # Same instance returned
        assert limiter1 is limiter2

    def test_different_services_different_limiters(self) -> None:
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_second=10)
        registry = RateLimitRegistry(settings)

        limiter_a = registry.get_limiter("api_a")
        limiter_b = registry.get_limiter("api_b")

        assert limiter_a is not limiter_b

    def test_registry_respects_service_config(self) -> None:
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit
        from elspeth.core.rate_limit import RateLimiter, RateLimitRegistry

        settings = RateLimitSettings(
            default_requests_per_second=10,
            services={
                "slow_api": ServiceRateLimit(requests_per_second=1),
            },
        )
        registry = RateLimitRegistry(settings)

        default_limiter = registry.get_limiter("fast_api")
        slow_limiter = registry.get_limiter("slow_api")

        # Type narrowing: enabled registry returns RateLimiter, not NoOpLimiter
        assert isinstance(default_limiter, RateLimiter)
        assert isinstance(slow_limiter, RateLimiter)
        assert default_limiter._requests_per_second == 10
        assert slow_limiter._requests_per_second == 1

    def test_registry_disabled(self) -> None:
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import NoOpLimiter, RateLimitRegistry

        settings = RateLimitSettings(enabled=False)
        registry = RateLimitRegistry(settings)

        limiter = registry.get_limiter("any_api")

        # Should return no-op limiter
        assert isinstance(limiter, NoOpLimiter)

    def test_registry_disabled_returns_same_noop_instance(self) -> None:
        """When disabled, registry returns the same NoOpLimiter instance."""
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import NoOpLimiter, RateLimitRegistry

        settings = RateLimitSettings(enabled=False)
        registry = RateLimitRegistry(settings)

        limiter1 = registry.get_limiter("api_a")
        limiter2 = registry.get_limiter("api_b")

        # Should return the same cached instance
        assert isinstance(limiter1, NoOpLimiter)
        assert limiter1 is limiter2

    def test_reset_all_clears_registry(self) -> None:
        """reset_all() clears all limiters from the registry."""
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_second=10)
        registry = RateLimitRegistry(settings)

        # Create some limiters
        limiter_a = registry.get_limiter("api_a")
        limiter_b = registry.get_limiter("api_b")

        # Verify they exist
        assert limiter_a is registry.get_limiter("api_a")
        assert limiter_b is registry.get_limiter("api_b")

        # Reset all
        registry.reset_all()

        # New calls should create new instances
        new_limiter_a = registry.get_limiter("api_a")
        new_limiter_b = registry.get_limiter("api_b")

        assert new_limiter_a is not limiter_a
        assert new_limiter_b is not limiter_b

    def test_registry_close(self) -> None:
        """close() cleans up all limiters."""
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_second=10)
        registry = RateLimitRegistry(settings)

        # Create some limiters
        registry.get_limiter("api_a")
        registry.get_limiter("api_b")

        # Close should not raise
        registry.close()

        # After close, registry should be empty (new calls create new instances)
        # This verifies close() cleared internal state
        assert len(registry._limiters) == 0


class TestNoOpLimiter:
    """Tests for NoOpLimiter."""

    def test_noop_limiter_acquire(self) -> None:
        """NoOpLimiter.acquire() always succeeds instantly."""
        from elspeth.core.rate_limit import NoOpLimiter

        limiter = NoOpLimiter()

        # Should not block or raise
        limiter.acquire()
        limiter.acquire(weight=100)

    def test_noop_limiter_try_acquire(self) -> None:
        """NoOpLimiter.try_acquire() always returns True."""
        from elspeth.core.rate_limit import NoOpLimiter

        limiter = NoOpLimiter()

        assert limiter.try_acquire() is True
        assert limiter.try_acquire(weight=100) is True

    def test_noop_limiter_context_manager(self) -> None:
        """NoOpLimiter can be used as a context manager."""
        from elspeth.core.rate_limit import NoOpLimiter

        with NoOpLimiter() as limiter:
            limiter.acquire()
            assert limiter.try_acquire() is True

    def test_noop_limiter_close(self) -> None:
        """NoOpLimiter.close() does nothing but doesn't raise."""
        from elspeth.core.rate_limit import NoOpLimiter

        limiter = NoOpLimiter()
        limiter.close()  # Should not raise
