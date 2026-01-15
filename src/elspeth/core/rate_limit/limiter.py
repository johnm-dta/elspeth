"""Rate limiter wrapper around pyrate-limiter."""

from __future__ import annotations

import re
import sqlite3
import threading
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

# Pattern for valid rate limiter names (used in SQL table names)
_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

# Track original thread excepthook
_original_excepthook = threading.excepthook

# Count of suppressions pending for each thread name
# We use a count because multiple threads may have the same name
_suppressed_threads: dict[str, int] = {}
_suppressed_lock = threading.Lock()


def _custom_excepthook(args: threading.ExceptHookArgs) -> None:
    """Custom thread excepthook that suppresses expected cleanup exceptions.

    pyrate-limiter's Leaker thread has a race condition where it can raise
    AssertionError when all buckets are disposed. This is benign (the thread
    is exiting anyway) but produces noisy warnings in tests.
    """
    thread_name = args.thread.name if args.thread else None
    with _suppressed_lock:
        if thread_name and thread_name in _suppressed_threads:
            # Decrement the suppression count
            _suppressed_threads[thread_name] -= 1
            if _suppressed_threads[thread_name] <= 0:
                del _suppressed_threads[thread_name]
            return

    # Not a suppressed thread, use original handler
    _original_excepthook(args)


# Install custom excepthook
threading.excepthook = _custom_excepthook


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
            name: Identifier for this rate limiter (used as bucket key).
                Must start with a letter and contain only alphanumeric
                characters and underscores.
            requests_per_second: Maximum requests allowed per second.
                Must be greater than 0.
            requests_per_minute: Optional maximum requests per minute.
                Must be greater than 0 if provided.
            persistence_path: Optional SQLite database path for persistence

        Raises:
            ValueError: If name is invalid or rate limits are not positive.
        """
        # Validate name - used in SQL table names, so must be safe
        if not _VALID_NAME_PATTERN.match(name):
            msg = (
                f"Invalid rate limiter name: {name!r}. "
                "Name must start with a letter and contain only "
                "alphanumeric characters and underscores."
            )
            raise ValueError(msg)

        # Validate rate limits
        if requests_per_second <= 0:
            msg = f"requests_per_second must be positive, got {requests_per_second}"
            raise ValueError(msg)

        if requests_per_minute is not None and requests_per_minute <= 0:
            msg = f"requests_per_minute must be positive, got {requests_per_minute}"
            raise ValueError(msg)

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
            self._conn.execute(
                SQLiteQueries.CREATE_BUCKET_TABLE.format(table=table_name)
            )
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
                self._conn.execute(
                    SQLiteQueries.CREATE_BUCKET_TABLE.format(table=table_name)
                )
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

    def _would_all_buckets_accept(self, weight: int) -> bool:
        """Check if all buckets would accept without consuming tokens.

        This performs a peek-style check by examining bucket count and rate limits
        without actually putting items into the buckets.

        Args:
            weight: Number of tokens to check

        Returns:
            True if all buckets would accept, False otherwise
        """
        for bucket in self._buckets:
            current_count = bucket.count()
            for rate in bucket.rates:
                if current_count + weight > rate.limit:
                    return False
        return True

    def try_acquire(self, weight: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            weight: Number of tokens to acquire (default 1)

        Returns:
            True if acquired, False if rate limited
        """
        with self._lock:
            # First check if ALL buckets would accept (peek without consuming)
            if not self._would_all_buckets_accept(weight):
                return False

            # All buckets would accept, now actually acquire from all limiters
            # Since we checked capacity, these should all succeed
            for limiter in self._limiters:
                original_max_delay = limiter.max_delay
                limiter.max_delay = None
                try:
                    limiter.try_acquire(self.name, weight=weight)
                except BucketFullException:
                    # This should not happen since we pre-checked capacity
                    # but if it does due to a race, restore and return failure
                    limiter.max_delay = original_max_delay
                    return False
                finally:
                    limiter.max_delay = original_max_delay

            return True

    def close(self) -> None:
        """Close the rate limiter and release resources."""
        # Get references to the leaker threads before disposing
        # Each limiter's bucket_factory has a _leaker attribute
        leakers = []
        for limiter in self._limiters:
            leaker = limiter.bucket_factory._leaker
            if leaker is not None and leaker.is_alive():
                leakers.append(leaker)
                # Register thread for exception suppression
                # pyrate-limiter has a race condition that causes AssertionError
                # during cleanup - this is benign but noisy
                with _suppressed_lock:
                    # Initialize to 0 if not present, then increment
                    if leaker.name not in _suppressed_threads:
                        _suppressed_threads[leaker.name] = 0
                    _suppressed_threads[leaker.name] += 1

        # Dispose all buckets from their limiters
        # This deregisters them from the leaker thread
        for limiter, bucket in zip(self._limiters, self._buckets, strict=True):
            limiter.dispose(bucket)

        # Wait for leaker threads to exit
        # The pyrate-limiter Leaker thread has a race condition where it can fail
        # with an assertion error if we close too quickly. We suppress that
        # exception via the custom excepthook above.
        for leaker in leakers:
            # Wait up to 50ms for thread to exit
            leaker.join(timeout=0.05)

        if self._conn is not None:
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
