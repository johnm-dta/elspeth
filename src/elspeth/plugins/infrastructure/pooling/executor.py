"""Pooled executor for parallel API calls with AIMD throttling.

Manages concurrent requests while:
- Respecting pool size limits via semaphore
- Applying AIMD throttle delays between dispatches
- Reordering results to match submission order
- Tracking statistics for audit trail
- Enforcing max retry timeout for capacity errors
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from threading import Event, Lock, Semaphore
from typing import Any

from elspeth.contracts import TransformErrorReason, TransformResult
from elspeth.contracts.engine import BufferEntry
from elspeth.contracts.errors import CapacityError, PluginRetryableError
from elspeth.contracts.freeze import freeze_fields
from elspeth.plugins.infrastructure.pooling.config import PoolConfig
from elspeth.plugins.infrastructure.pooling.reorder_buffer import ReorderBuffer
from elspeth.plugins.infrastructure.pooling.throttle import AIMDThrottle


@dataclass(frozen=True, slots=True)
class RowContext:
    """Context for processing a single row in the pool.

    Frozen: row contexts cross thread boundaries — immutability prevents
    data races between the dispatch thread and worker threads.

    ``row`` is deep-frozen in ``__post_init__`` (dict → MappingProxyType),
    isolating the context from external mutation of the source dict.
    Consumers that need to hand the row off to ``PipelineRow`` (which
    enforces ``type(data) is dict`` as a Tier 1 guard) must materialise a
    fresh dict at that boundary, e.g. ``PipelineRow(dict(ctx.row), ...)``.

    Attributes:
        row: The row data to process (stored as MappingProxyType)
        state_id: State ID for audit trail. Can be unique per row OR shared
            across batch rows (when used with aggregation). When shared,
            call_index in PluginContext provides uniqueness for external_calls.
        row_index: Original index for result ordering
    """

    # ``Mapping`` (not ``dict``): ``freeze_fields`` below replaces the
    # bound value with ``MappingProxyType`` in ``__post_init__``. The
    # runtime type is a read-only view; the annotation must describe
    # what callers actually observe, not the pre-freeze construction
    # type. Callers wanting a writable dict must ``dict(ctx.row)``.
    row: Mapping[str, Any]
    state_id: str
    row_index: int

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("RowContext.state_id must not be empty")
        if self.row_index < 0:
            raise ValueError(f"RowContext.row_index must be non-negative, got {self.row_index}")
        freeze_fields(self, "row")


class PooledExecutor:
    """Executor for parallel API calls with strict ordering.

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

        # Process batch - returns BufferEntry with ordering metadata
        entries = executor.execute_batch(
            contexts=contexts,
            process_fn=lambda row, state_id: transform.process_single(row, state_id),
        )

        # Entries are in submission order with full metadata
        if len(entries) != len(contexts):
            raise RuntimeError(f"ReorderBuffer returned {len(entries)} entries for {len(contexts)} contexts")

        # Extract results when needed
        results = [entry.result for entry in entries]

        # Access ordering metadata for audit trail
        for entry in entries:
            print(f"Row {entry.submit_index} completed {entry.complete_index}th")
            print(f"Buffer wait: {entry.buffer_wait_ms}ms")

        # Get pool stats for audit
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

        # Lock to serialize execute_batch calls (single-flight)
        # This prevents concurrent batches from mixing results in the shared buffer
        self._batch_lock = Lock()

        # Global dispatch gate - ensures minimum time between dispatches
        # This implements the design spec: "Dispatcher waits current_delay between dispatches"
        self._last_dispatch_time: float = 0.0  # time.monotonic() of last dispatch
        self._dispatch_gate_lock = Lock()  # Serializes dispatch timing coordination

        # Concurrency tracking for audit trail
        self._stats_lock = Lock()
        self._active_workers: int = 0
        self._max_concurrent: int = 0
        self._dispatch_delay_at_completion_ms: float = 0.0

        self._shutdown_event = Event()

    @property
    def pool_size(self) -> int:
        """Maximum concurrent requests."""
        return self._pool_size

    def _increment_active_workers(self) -> None:
        """Increment active worker count and update max_concurrent (thread-safe)."""
        with self._stats_lock:
            self._active_workers += 1
            if self._active_workers > self._max_concurrent:
                self._max_concurrent = self._active_workers

    def _decrement_active_workers(self) -> None:
        """Decrement active worker count (thread-safe)."""
        with self._stats_lock:
            self._active_workers -= 1

    def _reset_batch_stats(self) -> None:
        """Reset per-batch statistics at start of new batch."""
        with self._stats_lock:
            self._max_concurrent = 0
            self._dispatch_delay_at_completion_ms = 0.0
        self._throttle.reset_stats()

    def _capture_completion_stats(self) -> None:
        """Capture statistics at batch completion."""
        with self._stats_lock:
            self._dispatch_delay_at_completion_ms = self._throttle.current_delay_ms

    @property
    def pending_count(self) -> int:
        """Number of requests in flight or buffered."""
        return self._buffer.pending_count

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending requests to complete
        """
        self._shutdown_event.set()
        self._thread_pool.shutdown(wait=wait)

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics for audit trail.

        Returns:
            Dict with pool_config and pool_stats including:
            - pool_config: pool_size, max_capacity_retry_seconds, dispatch_delay_at_completion_ms
            - pool_stats: capacity_retries, successes, peak_delay_ms, current_delay_ms,
                          total_throttle_time_ms, max_concurrent_reached
        """
        throttle_stats = self._throttle.get_stats()
        with self._stats_lock:
            max_concurrent = self._max_concurrent
            dispatch_delay_at_completion = self._dispatch_delay_at_completion_ms
        return {
            "pool_config": {
                "pool_size": self._pool_size,
                "max_capacity_retry_seconds": self._max_capacity_retry_seconds,
                "dispatch_delay_at_completion_ms": dispatch_delay_at_completion,
            },
            "pool_stats": {
                "capacity_retries": throttle_stats["capacity_retries"],
                "successes": throttle_stats["successes"],
                "peak_delay_ms": throttle_stats["peak_delay_ms"],
                "current_delay_ms": throttle_stats["current_delay_ms"],
                "total_throttle_time_ms": throttle_stats["total_throttle_time_ms"],
                "max_concurrent_reached": max_concurrent,
            },
        }

    def execute_batch(
        self,
        contexts: list[RowContext],
        process_fn: Callable[[Mapping[str, Any], str], TransformResult],
        shutdown_event: Event | None = None,
    ) -> list[BufferEntry[TransformResult]]:
        """Execute batch of rows with parallel processing.

        Dispatches rows to the thread pool with semaphore control,
        applies AIMD throttle delays, and returns results in
        submission order with full ordering metadata.

        Each row is processed with its own state_id for audit trail.

        Note: This method is serialized - only one batch can execute at a time.
        Concurrent calls will block until the previous batch completes.

        Args:
            contexts: List of RowContext with row data and state_ids
            process_fn: Function that processes a single row with state_id
            shutdown_event: Optional run-cancellation signal. When set, workers
                stop before dispatch and during retry sleeps with
                shutdown_requested results.

        Returns:
            List of BufferEntry in submission order, each containing:
            - result: The TransformResult from process_fn
            - submit_index: Order in which row was submitted (0-indexed)
            - complete_index: Order in which row completed (may differ)
            - submit_timestamp: time.perf_counter() when submitted
            - complete_timestamp: time.perf_counter() when completed
            - buffer_wait_ms: Time spent waiting in buffer after completion
        """
        if not contexts:
            return []

        # Serialize batch execution to prevent result mixing
        # The ReorderBuffer uses sequential indices, so concurrent batches
        # would interleave indices and cause results to be returned to the wrong caller
        with self._batch_lock:
            self._buffer = ReorderBuffer()
            try:
                return self._execute_batch_locked(contexts, process_fn, shutdown_event)
            except Exception:
                self._buffer = ReorderBuffer()
                raise

    def _execute_batch_locked(
        self,
        contexts: list[RowContext],
        process_fn: Callable[[Mapping[str, Any], str], TransformResult],
        shutdown_event: Event | None = None,
    ) -> list[BufferEntry[TransformResult]]:
        """Internal batch execution (must be called while holding _batch_lock).

        Args:
            contexts: List of RowContext with row data and state_ids
            process_fn: Function that processes a single row with state_id

        Returns:
            List of BufferEntry in same order as input contexts, preserving
            full ordering metadata for audit trail.
        """
        # Reset per-batch statistics
        self._reset_batch_stats()

        # Fatal exceptions from one worker must stop sibling retry loops promptly.
        batch_cancel_event = Event()

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
            try:
                future = self._thread_pool.submit(
                    self._execute_single,
                    buffer_idx,
                    ctx.row,
                    ctx.state_id,
                    process_fn,
                    batch_cancel_event,
                    shutdown_event,
                )
            except RuntimeError:
                # Thread pool is shut down (concurrent shutdown() call).
                # Complete the reserved buffer slot with a deterministic error
                # so the buffer stays consistent and the row gets a proper
                # error result instead of being silently dropped.
                shutdown_result = TransformResult.error(
                    {"reason": "shutdown_requested", "error": "thread pool shut down during submission"},
                    retryable=False,
                )
                self._buffer.complete(buffer_idx, shutdown_result)
                # Mark remaining contexts as shutdown errors too
                for _ in contexts[contexts.index(ctx) + 1 :]:
                    remaining_idx = self._buffer.submit()
                    self._buffer.complete(
                        remaining_idx,
                        TransformResult.error(
                            {"reason": "shutdown_requested", "error": "thread pool shut down during submission"},
                            retryable=False,
                        ),
                    )
                break
            else:
                futures[future] = buffer_idx

        # Wait for all futures and collect results with full metadata
        entries: list[BufferEntry[TransformResult]] = []

        for future in as_completed(futures):
            buffer_idx = futures[future]
            try:
                _returned_idx, result = future.result()
            except Exception:
                self._drain_failed_batch_futures(futures, batch_cancel_event)
                raise

            # Complete in buffer (may be out of order)
            self._buffer.complete(buffer_idx, result)

            # Collect any ready entries (preserving full BufferEntry, not just result)
            ready = self._buffer.get_ready_results()
            entries.extend(ready)

        # CRITICAL: Final drain - collect any remaining entries not yet emitted
        # (the last completed future may not have been at the head of the queue)
        while self._buffer.pending_count > 0:
            ready = self._buffer.get_ready_results()
            if not ready:
                break  # Safety: shouldn't happen if all futures completed
            entries.extend(ready)

        if len(entries) != len(contexts):
            raise RuntimeError(f"Pool returned {len(entries)} entries for {len(contexts)} contexts")

        # Capture statistics at batch completion
        self._capture_completion_stats()

        return entries

    def _drain_failed_batch_futures(
        self,
        futures: Mapping[Future[tuple[int, TransformResult]], int],
        batch_cancel_event: Event,
    ) -> None:
        """Drain/cancel outstanding futures before propagating a batch-crashing exception."""
        batch_cancel_event.set()
        for future in futures:
            if future.done():
                continue
            future.cancel()
        wait(futures)

    def _is_interrupted(self, batch_cancel_event: Event | None = None, shutdown_event: Event | None = None) -> bool:
        """Return True when executor-wide or batch-local shutdown has been requested."""
        return (
            self._shutdown_event.is_set()
            or (batch_cancel_event is not None and batch_cancel_event.is_set())
            or (shutdown_event is not None and shutdown_event.is_set())
        )

    def _wait_interruptibly(
        self,
        timeout_seconds: float,
        batch_cancel_event: Event | None = None,
        shutdown_event: Event | None = None,
    ) -> tuple[bool, float]:
        """Wait up to timeout_seconds, returning (interrupted, actual_wait_ms)."""
        if timeout_seconds <= 0:
            return (self._is_interrupted(batch_cancel_event, shutdown_event), 0.0)

        start = time.monotonic()
        deadline = start + timeout_seconds

        while True:
            if self._is_interrupted(batch_cancel_event, shutdown_event):
                return (True, (time.monotonic() - start) * 1000)

            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return (False, (time.monotonic() - start) * 1000)

            wait_slice_s = min(remaining_s, 0.05)
            if self._shutdown_event.wait(timeout=wait_slice_s):
                return (True, (time.monotonic() - start) * 1000)

    def _acquire_worker_slot(self, batch_cancel_event: Event | None = None, shutdown_event: Event | None = None) -> bool:
        """Acquire a worker permit while remaining responsive to cancellation."""
        while True:
            if self._is_interrupted(batch_cancel_event, shutdown_event):
                return False
            if self._semaphore.acquire(timeout=0.05):
                self._increment_active_workers()
                return True

    def _build_retry_timeout_result(
        self,
        retryable_error: PluginRetryableError | CapacityError,
        start_time: float,
    ) -> TransformResult:
        """Build a retry_timeout result for the retryable error that exhausted the budget."""
        elapsed = time.monotonic() - start_time
        error_data: TransformErrorReason = {
            "reason": "retry_timeout",
            "error": str(retryable_error),
            "error_type": type(retryable_error).__name__,
            "elapsed_seconds": elapsed,
            "max_seconds": self._max_capacity_retry_seconds,
        }
        # Both union arms expose ``status_code`` (CapacityError always carries
        # an int; PluginRetryableError carries int | None). Record the value
        # when present — absence stays absent (no fabricated default), so a
        # NetworkError-style error with no HTTP status omits the field while a
        # rate-limit error surfaces its code.
        status_code = retryable_error.status_code
        if status_code is not None:
            error_data["status_code"] = status_code
        return TransformResult.error(error_data, retryable=False)

    def _build_interrupted_result(
        self,
        retryable_error: PluginRetryableError | CapacityError | None,
        batch_cancel_event: Event | None = None,
        shutdown_event: Event | None = None,
    ) -> TransformResult:
        """Build a deterministic result when work stops before redispatch."""
        if self._shutdown_event.is_set() or (shutdown_event is not None and shutdown_event.is_set()):
            return TransformResult.error(
                {
                    "reason": "shutdown_requested",
                    "error": "Shutdown requested before dispatch" if retryable_error is None else str(retryable_error),
                },
                retryable=False,
            )

        if batch_cancel_event is not None and batch_cancel_event.is_set():
            error_message = "Cancelled because another worker in the batch failed unexpectedly"
            if retryable_error is not None:
                error_message = f"{error_message}: {retryable_error}"
            return TransformResult.error(
                {"reason": "unexpected_pool_error", "error": error_message},
                retryable=False,
            )

        raise RuntimeError("Interrupted result requested without shutdown or batch cancellation")

    def _wait_for_dispatch_gate(self, batch_cancel_event: Event | None = None, shutdown_event: Event | None = None) -> bool:
        """Wait until we're allowed to dispatch, ensuring global pacing.

        Enforces min_dispatch_delay_ms between consecutive dispatches across
        ALL workers. This prevents burst-hammering the API when many workers
        are ready simultaneously.

        NOTE: This gate intentionally uses only the static min_dispatch_delay_ms,
        NOT the AIMD delay. Congestion backoff is enforced per-worker in the
        retry sleep (_execute_single). Feeding AIMD into the gate would
        double-penalize workers (once in retry sleep, again at the gate) and
        serialize all retries behind a single bottleneck, violating
        max_capacity_retry_seconds guarantees.

        The lock is only held during check-and-update, not during sleep,
        allowing other workers to make progress.
        """
        delay_ms = self._config.min_dispatch_delay_ms
        if delay_ms <= 0:
            return not self._is_interrupted(batch_cancel_event, shutdown_event)  # No pacing needed

        delay_s = delay_ms / 1000
        total_wait_ms = 0.0

        while True:
            with self._dispatch_gate_lock:
                now = time.monotonic()

                # Handle first dispatch: let it through immediately
                # _last_dispatch_time == 0 means no dispatch yet (since monotonic() > 0)
                if self._last_dispatch_time == 0.0:
                    self._last_dispatch_time = now
                    break

                time_since_last = now - self._last_dispatch_time

                if time_since_last >= delay_s:
                    # Gate is open - we can dispatch
                    self._last_dispatch_time = now
                    break

                # Calculate remaining wait time
                remaining_s = delay_s - time_since_last

            # Bail out if shutdown or batch cancellation was requested.
            if self._is_interrupted(batch_cancel_event, shutdown_event):
                break

            # Sleep OUTSIDE the lock to allow other workers to check
            interrupted, waited_ms = self._wait_interruptibly(remaining_s, batch_cancel_event, shutdown_event)
            total_wait_ms += waited_ms
            if interrupted:
                break

        # Record accumulated wait time for audit trail
        if total_wait_ms > 0:
            self._throttle.record_throttle_wait(total_wait_ms)

        return not self._is_interrupted(batch_cancel_event, shutdown_event)

    def _execute_single(
        self,
        buffer_idx: int,
        row: Mapping[str, Any],
        state_id: str,
        process_fn: Callable[[Mapping[str, Any], str], TransformResult],
        batch_cancel_event: Event,
        shutdown_event: Event | None = None,
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
            batch_cancel_event: Per-batch signal used to stop sibling retries after a fatal failure

        Returns:
            Tuple of (buffer_idx, result)
        """
        start_time = time.monotonic()
        max_time = start_time + self._max_capacity_retry_seconds
        retryable_error: PluginRetryableError | CapacityError | None = None

        # Acquire semaphore at start of worker (not in execute_batch)
        # This prevents deadlock when capacity errors cause release-then-reacquire
        if not self._acquire_worker_slot(batch_cancel_event, shutdown_event):
            return (buffer_idx, self._build_interrupted_result(None, batch_cancel_event, shutdown_event))
        holding_semaphore = True

        try:
            while True:
                if retryable_error is not None and time.monotonic() >= max_time:
                    return (buffer_idx, self._build_retry_timeout_result(retryable_error, start_time))

                # Wait for global dispatch gate (ensures pacing between ALL dispatches)
                # CRITICAL: Always check the gate, even after retries. The retry backoff
                # sleep is for THIS worker's cooldown, but OTHER workers may have
                # dispatched while we slept. We must check the global gate to maintain
                # min_dispatch_delay_ms between ALL dispatches across ALL workers.
                if not self._wait_for_dispatch_gate(batch_cancel_event, shutdown_event):
                    return (buffer_idx, self._build_interrupted_result(retryable_error, batch_cancel_event, shutdown_event))

                if retryable_error is not None and time.monotonic() >= max_time:
                    return (buffer_idx, self._build_retry_timeout_result(retryable_error, start_time))

                # Check shutdown before dispatching to external service
                if self._is_interrupted(batch_cancel_event, shutdown_event):
                    return (buffer_idx, self._build_interrupted_result(retryable_error, batch_cancel_event, shutdown_event))

                try:
                    result = process_fn(row, state_id)
                    self._throttle.on_success()
                    return (buffer_idx, result)
                except PluginRetryableError as e:
                    if not e.retryable:
                        return (
                            buffer_idx,
                            TransformResult.error(
                                {
                                    "reason": "permanent_error",
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                retryable=False,
                            ),
                        )

                    retryable_error = e
                except CapacityError as e:
                    retryable_error = e

                if time.monotonic() >= max_time:
                    return (buffer_idx, self._build_retry_timeout_result(retryable_error, start_time))

                if self._is_interrupted(batch_cancel_event, shutdown_event):
                    return (buffer_idx, self._build_interrupted_result(retryable_error, batch_cancel_event, shutdown_event))

                self._throttle.on_capacity_error()

                # CRITICAL: Release semaphore BEFORE sleeping
                # This allows other workers to make progress while we wait
                self._semaphore.release()
                self._decrement_active_workers()
                holding_semaphore = False

                # Wait throttle delay before retry
                retry_delay_ms = self._throttle.current_delay_ms
                if retry_delay_ms > 0:
                    remaining_retry_budget_s = max(0.0, max_time - time.monotonic())
                    retry_sleep_s = min(retry_delay_ms / 1000, remaining_retry_budget_s)
                    interrupted, waited_ms = self._wait_interruptibly(retry_sleep_s, batch_cancel_event, shutdown_event)
                    self._throttle.record_throttle_wait(waited_ms)
                    if interrupted:
                        return (buffer_idx, self._build_interrupted_result(retryable_error, batch_cancel_event, shutdown_event))

                if time.monotonic() >= max_time:
                    return (buffer_idx, self._build_retry_timeout_result(retryable_error, start_time))

                # Re-acquire semaphore for retry
                if not self._acquire_worker_slot(batch_cancel_event, shutdown_event):
                    return (buffer_idx, self._build_interrupted_result(retryable_error, batch_cancel_event, shutdown_event))
                holding_semaphore = True

                # Continue to top of loop for retry
                # Note: We DO NOT skip the dispatch gate check after retry.
                # The retry backoff is personal cooldown; the gate ensures global pacing.
        finally:
            # Release semaphore only if we're holding it
            # This defensive check ensures correctness even if an unexpected
            # exception occurs between release and re-acquire
            if holding_semaphore:
                self._semaphore.release()
                self._decrement_active_workers()
