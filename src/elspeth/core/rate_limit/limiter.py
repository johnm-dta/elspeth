"""Rate limiter wrapper around pyrate-limiter."""

from __future__ import annotations

import sqlite3
import threading
import time as time_module
from typing import TYPE_CHECKING

from pyrate_limiter import (  # type: ignore[attr-defined]
    BucketFullException,
    Duration,
    InMemoryBucket,
    Limiter,
    Rate,
    SQLiteBucket,
    SQLiteQueries,
)

if TYPE_CHECKING:
    from types import TracebackType


class RateLimiter:
    """Rate limiter for external API calls.

    Wraps pyrate-limiter with sensible defaults and optional
    SQLite persistence for cross-process rate limiting.

    Example:
        limiter = RateLimiter("openai", requests_per_second=10)

        # Blocking acquire (waits if needed)
        limiter.acquire()
        call_openai_api()

        # Non-blocking check
        if limiter.try_acquire():
            call_openai_api()
        else:
            handle_rate_limit()

        # Context manager usage
        with RateLimiter("api", requests_per_second=10) as limiter:
            limiter.acquire()
            call_api()
    """

    def __init__(
        self,
        name: str,
        requests_per_second: int,
        requests_per_minute: int | None = None,
        persistence_path: str | None = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            name: Identifier for this rate limiter (used as bucket key)
            requests_per_second: Maximum requests allowed per second
            requests_per_minute: Optional maximum requests per minute
            persistence_path: Optional SQLite database path for persistence
        """
        self.name = name
        self._requests_per_second = requests_per_second
        self._requests_per_minute = requests_per_minute
        self._persistence_path = persistence_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

        # Due to pyrate-limiter's internal optimization that can skip checking
        # longer-interval rates when under shorter-interval limits, we use
        # separate limiters for per-second and per-minute rates.
        self._limiters: list[Limiter] = []
        self._buckets: list[InMemoryBucket | SQLiteBucket] = []

        # Per-second rate limiter
        second_rates = [Rate(requests_per_second, Duration.SECOND)]
        if persistence_path:
            self._conn = sqlite3.connect(persistence_path, check_same_thread=False)
            table_name = f"ratelimit_{name}_second"
            self._conn.execute(SQLiteQueries.CREATE_BUCKET_TABLE.format(table=table_name))
            self._conn.commit()
            second_bucket: InMemoryBucket | SQLiteBucket = SQLiteBucket(
                rates=second_rates,
                conn=self._conn,
                table=table_name,
            )
        else:
            second_bucket = InMemoryBucket(rates=second_rates)

        self._buckets.append(second_bucket)
        self._limiters.append(
            Limiter(second_bucket, max_delay=Duration.MINUTE, raise_when_fail=True)
        )

        # Per-minute rate limiter (if specified)
        if requests_per_minute is not None:
            minute_rates = [Rate(requests_per_minute, Duration.MINUTE)]
            if persistence_path and self._conn is not None:
                table_name = f"ratelimit_{name}_minute"
                self._conn.execute(SQLiteQueries.CREATE_BUCKET_TABLE.format(table=table_name))
                self._conn.commit()
                minute_bucket: InMemoryBucket | SQLiteBucket = SQLiteBucket(
                    rates=minute_rates,
                    conn=self._conn,
                    table=table_name,
                )
            else:
                minute_bucket = InMemoryBucket(rates=minute_rates)

            self._buckets.append(minute_bucket)
            self._limiters.append(
                Limiter(minute_bucket, max_delay=Duration.MINUTE, raise_when_fail=True)
            )

    def acquire(self, weight: int = 1) -> None:
        """Acquire rate limit tokens, blocking if necessary.

        Args:
            weight: Number of tokens to acquire (default 1)
        """
        # Acquire from all limiters (blocks on each if needed)
        for limiter in self._limiters:
            limiter.try_acquire(self.name, weight=weight)

    def try_acquire(self, weight: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            weight: Number of tokens to acquire (default 1)

        Returns:
            True if acquired, False if rate limited
        """
        with self._lock:
            # First check if ALL limiters would allow the acquire
            # We need to temporarily disable blocking for each check
            for limiter in self._limiters:
                original_max_delay = limiter.max_delay
                limiter.max_delay = None
                try:
                    limiter.try_acquire(self.name, weight=weight)
                except BucketFullException:
                    # Restore and return failure
                    limiter.max_delay = original_max_delay
                    return False
                finally:
                    limiter.max_delay = original_max_delay

            return True

    def close(self) -> None:
        """Close the rate limiter and release resources."""
        # Dispose all buckets from their limiters
        for limiter, bucket in zip(self._limiters, self._buckets, strict=True):
            limiter.dispose(bucket)

        if self._conn is not None:
            # Allow leaker threads to process disposal before closing connection
            time_module.sleep(0.05)
            self._conn.close()
            self._conn = None

    def __enter__(self) -> RateLimiter:
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
