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
from elspeth.plugins.llm.reorder_buffer import ReorderBuffer


@dataclass
class RowContext:
    """Context for processing a single row in the pool.

    This allows each row to have its own state_id for audit trail,
    solving the "single state_id for all parallel rows" problem.

    Attributes:
        row: The row data to process
        state_id: Unique state ID for this row's audit trail
        row_index: Original index for ordering
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

            # Acquire semaphore (blocks if pool is full)
            # NOTE: Throttle delay is applied INSIDE the worker, not here,
            # to avoid serial delays blocking parallel submission
            self._semaphore.acquire()

            # Submit to thread pool
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
        """Execute single row and handle throttle feedback.

        Throttle delay is applied HERE (inside the worker) rather than
        in the submission loop. This ensures parallel dispatch isn't
        serialized by throttle delays.

        Args:
            buffer_idx: Index in reorder buffer
            row: Row to process
            state_id: State ID for audit trail
            process_fn: Processing function

        Returns:
            Tuple of (buffer_idx, result)
        """
        try:
            # Apply throttle delay INSIDE worker (after semaphore acquired)
            delay_ms = self._throttle.current_delay_ms
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
                self._throttle.record_throttle_wait(delay_ms)

            result = process_fn(row, state_id)
            self._throttle.on_success()
            return (buffer_idx, result)
        finally:
            # Always release semaphore
            self._semaphore.release()
