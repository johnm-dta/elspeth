"""In-memory sliding window rate limiter for composer messages.

Per-user rate limiting via FastAPI Depends(). Not thread-safe across
multiple uvicorn workers -- each worker has its own counter. Multi-worker
deployments need Redis or equivalent shared store for accurate
cross-process rate limiting.

Layer: L3 (application).
"""

from __future__ import annotations

import asyncio
import time

from fastapi import HTTPException, Request


class ComposerRateLimiter:
    """In-memory sliding window rate limiter for composer messages.

    Tracks message timestamps per user_id. On each request, prunes
    timestamps older than 60 seconds, then checks count against limit.
    Returns 429 if exceeded.

    Uses per-user asyncio.Lock instances to avoid contention between
    unrelated users. asyncio.Lock guards coroutine suspension points
    (e.g., between the prune-check-append sequence where another
    coroutine could interleave), not thread safety. A top-level
    _locks_lock (held for microseconds -- dict lookup only) serializes
    creation/fetch of per-user locks.

    Rate limiting is per-process. Deployments with multiple uvicorn
    workers have an effective rate limit of N * limit across the
    cluster. Multi-worker deployments require Redis or an equivalent
    shared store for accurate cross-process rate limiting.
    """

    _WINDOW_SECONDS: float = 60.0
    _SWEEP_INTERVAL: float = 300.0  # Evict stale entries at most every 5 minutes

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buckets: dict[str, list[float]] = {}
        self._user_locks: dict[str, asyncio.Lock] = {}
        # Lazily initialized on first async use — asyncio.Lock() requires a
        # running event loop in Python 3.12+, but __init__ may be called from
        # synchronous create_app().  In asyncio, code between await points
        # cannot be preempted, so the None check + assignment is race-free.
        self._locks_lock: asyncio.Lock | None = None
        # Zero ensures the first check() triggers a sweep (no-op on empty dicts).
        self._last_sweep: float = 0.0

    def _ensure_locks_lock(self) -> asyncio.Lock:
        """Return the top-level lock, creating it on first call.

        Safe to call without await-guarding: in asyncio only one coroutine
        runs between suspension points, so two coroutines cannot both see
        None simultaneously.
        """
        if self._locks_lock is None:
            self._locks_lock = asyncio.Lock()
        return self._locks_lock

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for the given user."""
        async with self._ensure_locks_lock():
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    async def _sweep_stale_entries(self) -> None:
        """Evict buckets and locks for users with no active timestamps.

        Time-gated: runs at most once per _SWEEP_INTERVAL seconds.
        Called at the start of check() so cleanup piggybacks on normal
        traffic. The fast-path (interval not elapsed) is a single
        monotonic clock read with no lock acquisition.
        """
        now = time.monotonic()
        if now - self._last_sweep < self._SWEEP_INTERVAL:
            return
        async with self._ensure_locks_lock():
            # Re-check under lock with a fresh timestamp — another coroutine
            # may have swept between our fast-path check and acquiring the lock.
            now = time.monotonic()
            if now - self._last_sweep < self._SWEEP_INTERVAL:
                return
            self._last_sweep = now
            cutoff = now - self._WINDOW_SECONDS
            stale_users = [uid for uid, bucket in self._buckets.items() if not bucket or bucket[-1] <= cutoff]
            for uid in stale_users:
                del self._buckets[uid]
                self._user_locks.pop(uid, None)

    async def check(self, user_id: str) -> None:
        """Check rate limit for the given user.

        Raises HTTPException(429) with Retry-After header if the
        per-user rate limit is exceeded.
        """
        await self._sweep_stale_entries()
        lock = await self._get_user_lock(user_id)
        async with lock:
            now = time.monotonic()
            cutoff = now - self._WINDOW_SECONDS

            # Get or create bucket
            if user_id not in self._buckets:
                self._buckets[user_id] = []

            bucket = self._buckets[user_id]

            # Prune timestamps outside the window
            bucket[:] = [ts for ts in bucket if ts > cutoff]

            # Check limit
            if len(bucket) >= self._limit:
                # Earliest timestamp determines retry delay
                earliest = bucket[0]
                retry_after = int(earliest - cutoff) + 1
                if retry_after < 1:
                    retry_after = 1
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error_type": "rate_limited",
                        "detail": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            # Record this request
            bucket.append(now)


async def get_rate_limiter(request: Request) -> ComposerRateLimiter:
    """FastAPI dependency that extracts the rate limiter from app state."""
    limiter: ComposerRateLimiter = request.app.state.rate_limiter
    return limiter


async def check_auth_rate_limit(request: Request) -> None:
    """Rate-limit auth endpoints by client IP.

    Side-effect dependency: raises HTTPException(429) if the per-IP
    rate limit is exceeded. The route does not need the return value.

    Extracts the client IP from request.client.host. Behind a reverse
    proxy with --proxy-headers enabled, Starlette populates this from
    X-Forwarded-For automatically.
    """
    limiter: ComposerRateLimiter = request.app.state.auth_rate_limiter
    client_ip = request.client.host if request.client else "unknown"
    await limiter.check(client_ip)
