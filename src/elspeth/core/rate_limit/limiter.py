"""Rate limiter wrapper around pyrate-limiter."""

from __future__ import annotations

import re
import sqlite3
import threading
import time
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

# Thread idents registered for exception suppression during cleanup.
# We track by thread ident (not name) to avoid accidental suppression of
# unrelated threads that happen to share a name.
_suppressed_thread_idents: set[int] = set()
_suppressed_lock = threading.Lock()


def _custom_excepthook(args: threading.ExceptHookArgs) -> None:
    """Custom thread excepthook that suppresses expected cleanup exceptions.

    pyrate-limiter's Leaker thread has a race condition where it can raise
    AssertionError when all buckets are disposed. This is benign (the thread
    is exiting anyway) but produces noisy warnings in tests.

    Suppression is narrowly scoped:
    - Only for threads registered by RateLimiter.close()
    - Only for AssertionError (the known benign exception from pyrate-limiter)
    - Logs when suppression occurs for observability
    """
    import structlog

    logger = structlog.get_logger()

    thread_ident = args.thread.ident if args.thread else None

    # Only suppress if:
    # 1. Thread is registered for suppression
    # 2. Exception is AssertionError (the known benign cleanup race)
    with _suppressed_lock:
        if thread_ident is not None and thread_ident in _suppressed_thread_idents and args.exc_type is AssertionError:
            # Remove from suppression set (one-time suppression per thread)
            _suppressed_thread_idents.discard(thread_ident)
            logger.debug(
                "Suppressed expected pyrate-limiter cleanup exception",
                thread_ident=thread_ident,
                thread_name=args.thread.name if args.thread else None,
                exc_type=args.exc_type.__name__ if args.exc_type else None,
            )
            return

    # Not a suppressed scenario, use original handler
    _original_excepthook(args)


# Install custom excepthook
threading.excepthook = _custom_excepthook


class RateLimiter:
    """Rate limiter for external API calls.

    Wraps pyrate-limiter with sensible defaults and optional
    SQLite persistence for cross-process rate limiting.

    Example:
        limiter = RateLimiter("openai", requests_per_minute=60)

        # Blocking acquire (waits if needed)
        limiter.acquire()
        call_openai_api()

        # Non-blocking check
        if limiter.try_acquire():
            call_openai_api()
        else:
            handle_rate_limit()

        # Context manager usage
        with RateLimiter("api", requests_per_minute=60) as limiter:
            limiter.acquire()
            call_api()
    """

    def __init__(
        self,
        name: str,
        requests_per_minute: int,
        persistence_path: str | None = None,
        window_ms: int | Duration | None = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            name: Identifier for this rate limiter (used as bucket key).
                Must start with a letter and contain only alphanumeric
                characters and underscores.
            requests_per_minute: Maximum requests allowed per window.
                Defaults to per-minute behavior. Must be greater than 0.
            persistence_path: Optional SQLite database path for persistence
            window_ms: Optional window override in milliseconds.
                Defaults to Duration.MINUTE.

        Raises:
            ValueError: If name is invalid or rate limit is not positive.
        """
        # Validate name - used in SQL table names, so must be safe
        if not _VALID_NAME_PATTERN.match(name):
            msg = (
                f"Invalid rate limiter name: {name!r}. "
                "Name must start with a letter and contain only "
                "alphanumeric characters and underscores."
            )
            raise ValueError(msg)

        # Validate rate limit
        if requests_per_minute <= 0:
            msg = f"requests_per_minute must be positive, got {requests_per_minute}"
            raise ValueError(msg)

        self.name = name
        self._requests_per_minute = requests_per_minute
        self._persistence_path = persistence_path
        self._window_ms = int(Duration.MINUTE if window_ms is None else window_ms)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

        # Single rate - sliding window
        if self._window_ms <= 0:
            msg = f"window_ms must be positive, got {self._window_ms}"
            raise ValueError(msg)

        rates: list[Rate] = [Rate(requests_per_minute, self._window_ms)]

        # Create bucket (persistent or in-memory)
        if persistence_path:
            self._conn = sqlite3.connect(persistence_path, check_same_thread=False)
            table_name = f"ratelimit_{name}"
            self._conn.execute(SQLiteQueries.CREATE_BUCKET_TABLE.format(table=table_name))
            self._conn.commit()
            self._bucket: InMemoryBucket | SQLiteBucket = SQLiteBucket(
                rates=rates,
                conn=self._conn,
                table=table_name,
            )
        else:
            self._bucket = InMemoryBucket(rates=rates)

        # Single limiter with per-minute rate
        self._limiter = Limiter(self._bucket, max_delay=self._window_ms, raise_when_fail=True)

    def acquire(self, weight: int = 1, timeout: float | None = None) -> None:
        """Acquire rate limit tokens, blocking if necessary.

        Thread-safe. Blocks by polling try_acquire() until successful or
        timeout expires.

        Args:
            weight: Number of tokens to acquire (default 1)
            timeout: Maximum time to wait in seconds (None = wait forever)

        Raises:
            TimeoutError: If timeout expires before tokens are acquired
        """
        deadline = None if timeout is None else (time.monotonic() + timeout)

        while True:
            if self.try_acquire(weight):
                return

            # Check timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Failed to acquire {weight} tokens within {timeout}s timeout")
                # Sleep for shorter of: 10ms or remaining time
                time.sleep(min(0.01, remaining))
            else:
                # No timeout - sleep 10ms and retry
                time.sleep(0.01)

    def try_acquire(self, weight: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            weight: Number of tokens to acquire (default 1)

        Returns:
            True if acquired, False if rate limited
        """
        with self._lock:
            # Temporarily disable max_delay to get immediate response
            original_max_delay = self._limiter.max_delay
            self._limiter.max_delay = None
            try:
                self._limiter.try_acquire(self.name, weight=weight)
                return True
            except BucketFullException:
                return False
            finally:
                self._limiter.max_delay = original_max_delay

    def close(self) -> None:
        """Close the rate limiter and release resources."""
        # Get reference to the leaker thread before disposing.
        # The limiter's bucket_factory has a _leaker attribute.
        # We capture (thread, ident) pair because ident may become None after thread exits.
        leaker = self._limiter.bucket_factory._leaker
        leaker_ident: int | None = None
        if leaker is not None and leaker.is_alive() and leaker.ident is not None:
            leaker_ident = leaker.ident  # Capture before it can become None
            # Register thread ident for exception suppression.
            # pyrate-limiter has a race condition that causes AssertionError
            # during cleanup - this is benign but noisy. We register by
            # ident (not name) to avoid accidentally suppressing unrelated
            # threads that share a name.
            with _suppressed_lock:
                _suppressed_thread_idents.add(leaker_ident)

        # Dispose bucket from limiter
        # This deregisters it from the leaker thread
        self._limiter.dispose(self._bucket)

        # Wait for leaker thread to exit
        # The pyrate-limiter Leaker thread has a race condition where it can fail
        # with an assertion error if we close too quickly. We suppress that
        # exception via the custom excepthook above.
        if leaker is not None and leaker_ident is not None:
            # Wait up to 50ms for thread to exit
            leaker.join(timeout=0.05)
            # Clean up suppression registration after join completes.
            # If the thread raised AssertionError, the hook already removed it (discard is safe).
            # If the thread exited cleanly, we remove it here to prevent stale idents.
            with _suppressed_lock:
                _suppressed_thread_idents.discard(leaker_ident)

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
