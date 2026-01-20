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

from concurrent.futures import ThreadPoolExecutor
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
