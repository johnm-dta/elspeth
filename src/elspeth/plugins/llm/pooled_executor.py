# src/elspeth/plugins/llm/pooled_executor.py
"""Pooled executor for parallel LLM API calls with AIMD throttling.

Manages concurrent requests while:
- Respecting pool size limits via semaphore
- Applying AIMD throttle delays between dispatches
- Reordering results to match submission order
- Tracking statistics for audit trail
- Enforcing max retry timeout for capacity errors
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Semaphore
from typing import Any

from elspeth.contracts import TransformResult
from elspeth.plugins.llm.aimd_throttle import AIMDThrottle
from elspeth.plugins.llm.base import PoolConfig
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.reorder_buffer import ReorderBuffer


@dataclass
class RowContext:
    """Context for processing a single row in the pool.

    Attributes:
        row: The row data to process
        state_id: State ID for audit trail. Can be unique per row OR shared
            across batch rows (when used with aggregation). When shared,
            call_index in PluginContext provides uniqueness for external_calls.
        row_index: Original index for result ordering
    """

    row: dict[str, Any]
    state_id: str
    row_index: int


class PooledExecutor:
    """Executor for parallel LLM API calls with strict ordering.

    Manages a pool of concurrent requests with:
    - Semaphore-controlled dispatch (max pool_size in flight)
    - AIMD throttle for adaptive rate limiting
    - Reorder buffer for strict submission order output
    - Max retry timeout for capacity errors

    The executor is synchronous from the caller's perspective -
    execute_batch() blocks until all results are ready in order.

    Usage:
        executor = PooledExecutor(pool_config)

        # Prepare row contexts with per-row state IDs
        contexts = [
            RowContext(row=row, state_id=state_id, row_index=i)
            for i, (row, state_id) in enumerate(zip(rows, state_ids))
        ]

        # Process batch
        results = executor.execute_batch(
            contexts=contexts,
            process_fn=lambda row, state_id: transform.process_single(row, state_id),
        )

        # Results are in submission order
        assert len(results) == len(contexts)

        # Get stats for audit
        stats = executor.get_stats()
    """

    def __init__(self, config: PoolConfig) -> None:
        """Initialize executor with pool configuration.

        Args:
            config: Pool configuration with size and AIMD settings
        """
        self._config = config
        self._pool_size = config.pool_size
        self._max_capacity_retry_seconds = config.max_capacity_retry_seconds

        # Thread pool for concurrent execution
        self._thread_pool = ThreadPoolExecutor(max_workers=config.pool_size)

        # Semaphore limits concurrent in-flight requests
        self._semaphore = Semaphore(config.pool_size)

        # AIMD throttle for adaptive rate control
        self._throttle = AIMDThrottle(config.to_throttle_config())

        # Reorder buffer for strict output ordering
        self._buffer: ReorderBuffer[TransformResult] = ReorderBuffer()

        self._shutdown = False

    @property
    def pool_size(self) -> int:
        """Maximum concurrent requests."""
        return self._pool_size

    @property
    def pending_count(self) -> int:
        """Number of requests in flight or buffered."""
        return self._buffer.pending_count

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending requests to complete
        """
        self._shutdown = True
        self._thread_pool.shutdown(wait=wait)

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics for audit trail.

        Returns:
            Dict with pool_size, throttle stats, etc.
        """
        throttle_stats = self._throttle.get_stats()
        return {
            "pool_config": {
                "pool_size": self._pool_size,
                "max_capacity_retry_seconds": self._max_capacity_retry_seconds,
            },
            "pool_stats": {
                "capacity_retries": throttle_stats["capacity_retries"],
                "successes": throttle_stats["successes"],
                "peak_delay_ms": throttle_stats["peak_delay_ms"],
                "current_delay_ms": throttle_stats["current_delay_ms"],
                "total_throttle_time_ms": throttle_stats["total_throttle_time_ms"],
            },
        }

    def execute_batch(
        self,
        contexts: list[RowContext],
        process_fn: Callable[[dict[str, Any], str], TransformResult],
    ) -> list[TransformResult]:
        """Execute batch of rows with parallel processing.

        Dispatches rows to the thread pool with semaphore control,
        applies AIMD throttle delays, and returns results in
        submission order.

        Each row is processed with its own state_id for audit trail.

        Args:
            contexts: List of RowContext with row data and state_ids
            process_fn: Function that processes a single row with state_id

        Returns:
            List of TransformResults in same order as input contexts
        """
        if not contexts:
            return []

        # Track futures by their buffer index
        futures: dict[Future[tuple[int, TransformResult]], int] = {}

        # Submit all rows
        for ctx in contexts:
            # Reserve slot in reorder buffer
            buffer_idx = self._buffer.submit()

            # Submit to thread pool
            # NOTE: Semaphore is acquired INSIDE the worker, not here.
            # This prevents deadlock when capacity errors cause workers to
            # release-then-reacquire: if we acquired here, the main thread
            # could steal permits for queued tasks that can't run because
            # worker threads are blocked waiting to reacquire.
            future = self._thread_pool.submit(
                self._execute_single,
                buffer_idx,
                ctx.row,
                ctx.state_id,
                process_fn,
            )
            futures[future] = buffer_idx

        # Wait for all futures and collect results
        results: list[TransformResult] = []

        for future in as_completed(futures):
            buffer_idx, result = future.result()

            # Complete in buffer (may be out of order)
            self._buffer.complete(buffer_idx, result)

            # Collect any ready results
            ready = self._buffer.get_ready_results()
            for entry in ready:
                results.append(entry.result)

        # CRITICAL: Final drain - collect any remaining results not yet emitted
        # (the last completed future may not have been at the head of the queue)
        while self._buffer.pending_count > 0:
            ready = self._buffer.get_ready_results()
            if not ready:
                break  # Safety: shouldn't happen if all futures completed
            for entry in ready:
                results.append(entry.result)

        return results

    def _execute_single(
        self,
        buffer_idx: int,
        row: dict[str, Any],
        state_id: str,
        process_fn: Callable[[dict[str, Any], str], TransformResult],
    ) -> tuple[int, TransformResult]:
        """Execute single row with capacity error retry and timeout.

        Capacity errors trigger AIMD throttle and are retried until
        max_capacity_retry_seconds is exceeded. Normal errors/results
        are returned as-is.

        Semaphore is acquired at the start of this method (not in execute_batch)
        to prevent deadlock: if acquire happened in execute_batch, the main thread
        could acquire permits for queued tasks while workers are blocked trying
        to re-acquire after capacity errors.

        Uses holding_semaphore flag for defensive tracking - ensures we
        only release what we hold, even in edge cases.

        Args:
            buffer_idx: Index in reorder buffer
            row: Row to process
            state_id: State ID for audit trail
            process_fn: Processing function

        Returns:
            Tuple of (buffer_idx, result)
        """
        start_time = time.monotonic()
        max_time = start_time + self._max_capacity_retry_seconds

        # Acquire semaphore at start of worker (not in execute_batch)
        # This prevents deadlock when capacity errors cause release-then-reacquire
        self._semaphore.acquire()
        holding_semaphore = True

        # Track if we just retried - skip pre-dispatch delay after capacity retry
        # to avoid double-sleeping (we already slept during the retry backoff)
        just_retried = False

        try:
            while True:
                # Apply throttle delay INSIDE worker (after semaphore acquired)
                # Skip if we just retried - we already slept during retry backoff
                if not just_retried:
                    delay_ms = self._throttle.current_delay_ms
                    if delay_ms > 0:
                        time.sleep(delay_ms / 1000)
                        self._throttle.record_throttle_wait(delay_ms)
                just_retried = False  # Reset for next iteration

                try:
                    result = process_fn(row, state_id)
                    self._throttle.on_success()
                    return (buffer_idx, result)
                except CapacityError as e:
                    # Check if we've exceeded max retry time
                    if time.monotonic() >= max_time:
                        elapsed = time.monotonic() - start_time
                        return (
                            buffer_idx,
                            TransformResult.error(
                                {
                                    "reason": "capacity_retry_timeout",
                                    "error": str(e),
                                    "status_code": e.status_code,
                                    "elapsed_seconds": elapsed,
                                    "max_seconds": self._max_capacity_retry_seconds,
                                },
                                retryable=False,
                            ),
                        )

                    # Trigger throttle backoff
                    self._throttle.on_capacity_error()

                    # CRITICAL: Release semaphore BEFORE sleeping
                    # This allows other workers to make progress while we wait
                    self._semaphore.release()
                    holding_semaphore = False

                    # Wait throttle delay before retry
                    retry_delay_ms = self._throttle.current_delay_ms
                    if retry_delay_ms > 0:
                        time.sleep(retry_delay_ms / 1000)
                        self._throttle.record_throttle_wait(retry_delay_ms)

                    # Re-acquire semaphore for retry
                    self._semaphore.acquire()
                    holding_semaphore = True

                    # Mark that we just retried - skip pre-dispatch delay on next iteration
                    just_retried = True

                    # Retry (continue to top of loop)
        finally:
            # Release semaphore only if we're holding it
            # This defensive check ensures correctness even if an unexpected
            # exception occurs between release and re-acquire
            if holding_semaphore:
                self._semaphore.release()
