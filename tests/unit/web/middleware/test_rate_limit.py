"""Tests for ComposerRateLimiter -- in-memory sliding window rate limiter."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi import HTTPException

from elspeth.web.middleware.rate_limit import ComposerRateLimiter


class TestRateLimiterConstruction:
    """Tests that the rate limiter can be constructed in sync context."""

    def test_construction_without_running_event_loop(self) -> None:
        """ComposerRateLimiter must be constructable in synchronous code.

        Regression test for elspeth-7760c5f5c6: asyncio.Lock() in __init__
        crashes on Python 3.12+ when no event loop is running. The fix uses
        lazy initialization — _locks_lock is None until first async use.
        """
        # This is a plain def (not async def), so no event loop is running.
        limiter = ComposerRateLimiter(limit=10)
        assert limiter._locks_lock is None

    @pytest.mark.asyncio
    async def test_lazy_lock_created_on_first_use(self) -> None:
        """_locks_lock is created on the first call to check()."""
        limiter = ComposerRateLimiter(limit=10)
        assert limiter._locks_lock is None
        await limiter.check("user_1")
        assert limiter._locks_lock is not None


class TestRateLimiterAllow:
    """Tests that requests within the limit are allowed."""

    @pytest.mark.asyncio
    async def test_allows_requests_within_limit(self) -> None:
        limiter = ComposerRateLimiter(limit=5)
        for _ in range(5):
            await limiter.check("user_1")
        # Each successful check must record a bucket entry; otherwise the limiter is a no-op.
        assert len(limiter._buckets["user_1"]) == 5

    @pytest.mark.asyncio
    async def test_first_request_always_allowed(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        assert len(limiter._buckets["user_1"]) == 1


class TestRateLimiterDeny:
    """Tests that requests exceeding the limit are denied."""

    @pytest.mark.asyncio
    async def test_denies_request_exceeding_limit(self) -> None:
        limiter = ComposerRateLimiter(limit=2)
        await limiter.check("user_1")
        await limiter.check("user_1")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("user_1")
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers is not None
        assert "Retry-After" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_429_response_body_shape(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("user_1")
        detail: dict[str, Any] = exc_info.value.detail  # type: ignore[assignment]
        assert detail["error_type"] == "rate_limited"
        assert "retry_after" in detail
        assert isinstance(detail["retry_after"], int)


class TestRateLimiterWindowReset:
    """Tests that the sliding window resets correctly."""

    @pytest.mark.asyncio
    async def test_window_resets_after_60_seconds(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        # Manually inject an old timestamp to simulate window expiry. Inject the
        # paired lock too: production always creates the _user_locks entry
        # (via _get_user_lock) before the _buckets entry, so the sweep's bare
        # ``del self._user_locks[uid]`` relies on that invariant — a bucket
        # without a lock is a state production never produces.
        limiter._buckets["user_1"] = [time.monotonic() - 61.0]
        limiter._user_locks["user_1"] = asyncio.Lock()
        # Should pass -- the old request is outside the window
        await limiter.check("user_1")
        # The expired timestamp must be evicted; only the fresh call remains.
        assert len(limiter._buckets["user_1"]) == 1
        assert limiter._buckets["user_1"][0] > time.monotonic() - 1.0


class TestRateLimiterConcurrentUsers:
    """Tests that per-user limits are independent."""

    @pytest.mark.asyncio
    async def test_independent_per_user_limits(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        # user_2 should still be allowed -- separate bucket
        await limiter.check("user_2")
        # Both users must hold their own bucket — a single shared bucket would conflate them.
        assert len(limiter._buckets["user_1"]) == 1
        assert len(limiter._buckets["user_2"]) == 1
        assert limiter._buckets["user_1"] is not limiter._buckets["user_2"]

    @pytest.mark.asyncio
    async def test_user1_exhausted_user2_unaffected(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        with pytest.raises(HTTPException):
            await limiter.check("user_1")
        # user_2 is not affected
        await limiter.check("user_2")


class TestRateLimiterSweep:
    """Tests that stale entries are evicted by the periodic sweep."""

    @pytest.mark.asyncio
    async def test_sweep_removes_stale_entries(self) -> None:
        """Users with expired timestamps are evicted after sweep interval."""
        limiter = ComposerRateLimiter(limit=5)
        # Simulate old activity from an inactive user
        old_ts = time.monotonic() - 120.0  # 2 minutes ago (past 60s window)
        limiter._buckets["stale_user"] = [old_ts]
        limiter._user_locks["stale_user"] = asyncio.Lock()
        # Force sweep by setting last_sweep far in the past
        limiter._last_sweep = 0.0
        # Trigger sweep via a check for a different user
        await limiter.check("active_user")
        # Stale user should be evicted
        assert "stale_user" not in limiter._buckets
        assert "stale_user" not in limiter._user_locks
        # Active user should still be present
        assert "active_user" in limiter._buckets

    @pytest.mark.asyncio
    async def test_sweep_removes_empty_buckets(self) -> None:
        """Users with empty bucket lists are evicted."""
        limiter = ComposerRateLimiter(limit=5)
        limiter._buckets["ghost_user"] = []
        limiter._user_locks["ghost_user"] = asyncio.Lock()
        limiter._last_sweep = 0.0
        await limiter.check("active_user")
        assert "ghost_user" not in limiter._buckets
        assert "ghost_user" not in limiter._user_locks

    @pytest.mark.asyncio
    async def test_sweep_preserves_active_entries(self) -> None:
        """Users with recent timestamps survive sweep."""
        limiter = ComposerRateLimiter(limit=5)
        await limiter.check("active_user")
        limiter._last_sweep = 0.0  # Force sweep on next check
        await limiter.check("active_user")
        assert "active_user" in limiter._buckets

    @pytest.mark.asyncio
    async def test_sweep_skips_when_interval_not_elapsed(self) -> None:
        """Sweep does not run when interval hasn't elapsed."""
        limiter = ComposerRateLimiter(limit=5)
        old_ts = time.monotonic() - 120.0
        limiter._buckets["stale_user"] = [old_ts]
        limiter._user_locks["stale_user"] = asyncio.Lock()
        limiter._last_sweep = time.monotonic()  # Just swept
        await limiter.check("other_user")
        # Stale user should NOT be evicted (sweep interval not elapsed)
        assert "stale_user" in limiter._buckets

    @pytest.mark.asyncio
    async def test_sweep_boundary_exact_cutoff_evicts(self) -> None:
        """A bucket whose last entry is exactly at the cutoff is evicted."""
        limiter = ComposerRateLimiter(limit=5)
        # Place timestamp exactly at the window boundary
        boundary_ts = time.monotonic() - ComposerRateLimiter._WINDOW_SECONDS
        limiter._buckets["boundary_user"] = [boundary_ts]
        limiter._user_locks["boundary_user"] = asyncio.Lock()
        limiter._last_sweep = 0.0
        await limiter.check("trigger_user")
        # <= means exactly-at-cutoff IS evicted
        assert "boundary_user" not in limiter._buckets

    @pytest.mark.asyncio
    async def test_sweep_preserves_bucket_with_recent_last_entry(self) -> None:
        """A bucket with old + recent entries survives (last entry is recent)."""
        limiter = ComposerRateLimiter(limit=5)
        old_ts = time.monotonic() - 120.0
        recent_ts = time.monotonic()
        limiter._buckets["mixed_user"] = [old_ts, recent_ts]
        limiter._user_locks["mixed_user"] = asyncio.Lock()
        limiter._last_sweep = 0.0
        await limiter.check("trigger_user")
        # Last entry is recent — user survives sweep
        assert "mixed_user" in limiter._buckets

    @pytest.mark.asyncio
    async def test_swept_user_re_acquires_lock_on_next_check(self) -> None:
        """After being swept, a user's next check() recreates lock and bucket."""
        limiter = ComposerRateLimiter(limit=5)
        old_ts = time.monotonic() - 120.0
        limiter._buckets["returning_user"] = [old_ts]
        limiter._user_locks["returning_user"] = asyncio.Lock()
        limiter._last_sweep = 0.0
        # First check triggers sweep, evicting returning_user
        await limiter.check("other_user")
        assert "returning_user" not in limiter._buckets
        # Now returning_user makes a new request — should work normally
        limiter._last_sweep = time.monotonic()  # Prevent re-sweep
        await limiter.check("returning_user")
        assert "returning_user" in limiter._buckets
        assert "returning_user" in limiter._user_locks


class TestRateLimiterPerUserLocks:
    """Tests that per-user locks prevent interleaving."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_user_serialized(self) -> None:
        """Concurrent requests from the same user are serialized by the
        per-user lock, preventing race conditions in the
        prune-check-append sequence."""
        limiter = ComposerRateLimiter(limit=3)
        # Fire 3 concurrent requests from same user
        results = await asyncio.gather(
            *[limiter.check("user_1") for _ in range(3)],
            return_exceptions=True,
        )
        # All 3 should succeed (within limit)
        assert all(r is None for r in results)

        # Now fire 2 more -- should both fail
        results = await asyncio.gather(
            *[limiter.check("user_1") for _ in range(2)],
            return_exceptions=True,
        )
        assert all(isinstance(r, HTTPException) for r in results)
