# src/elspeth/plugins/batching/mixin.py
"""Mixin for transforms that process rows concurrently with FIFO output ordering.

Transforms using this mixin can process multiple rows concurrently while
guaranteeing FIFO output order. The orchestrator sees simple accept() calls;
concurrency is hidden inside the plugin.

Key principle: The transform doesn't know what's downstream. It just emits
to its output port. Could be a sink, could be another transform.
"""

from __future__ import annotations

import contextlib
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from elspeth.plugins.batching.ports import OutputPort
from elspeth.plugins.batching.row_reorder_buffer import (
    RowReorderBuffer,
    RowTicket,
    ShutdownError,
)

if TYPE_CHECKING:
    from typing import Any

    from elspeth.contracts import ExceptionResult, TransformResult
    from elspeth.contracts.identity import TokenInfo
    from elspeth.plugins.context import PluginContext


class BatchTransformMixin:
    """Mixin that adds concurrent sub-task processing to any transform.

    This mixin enables processing work items concurrently within a single row
    while guaranteeing FIFO output order. From the orchestrator's perspective,
    accept() takes work and may block on backpressure. Results flow through
    the output port in submission order.

    Concurrency Model (IMPORTANT):
        - WITHIN-ROW concurrency: Worker threads process sub-tasks concurrently
          (e.g., multi-query transforms make 10+ LLM calls per row in parallel)
        - ACROSS-ROW sequencing: The TransformExecutor blocks on each row's
          completion before processing the next row (see executors.py for rationale)
        - Buffer capacity: The RowReorderBuffer can queue work when multiple
          submissions happen before the orchestrator waits

    Architecture:
        Orchestrator → accept() → [RowReorderBuffer] → [Worker Pool] → emit() → Output

    The transform doesn't know if its output goes to a sink or another transform.
    It just emits to the output port.

    Thread Model:
        - Orchestrator thread: calls accept(), may block on backpressure
        - Worker threads: execute processing, call complete() when done
        - Release thread: calls wait_for_next_release(), emits to output port

    Usage:
        class MyLLMTransform(BaseTransform, BatchTransformMixin):
            def __init__(self, config: dict, output: OutputPort):
                super().__init__(config)
                self.init_batch_processing(
                    max_pending=30,
                    output=output,
                    name="my-llm-transform",
                )

            def accept(self, row: dict, ctx: PluginContext) -> None:
                self.accept_row(row, ctx, self._do_processing)

            def _do_processing(
                self, row: dict, ctx: PluginContext
            ) -> TransformResult:
                # Actual processing here (runs in worker thread)
                result = self._call_llm(row)
                return TransformResult.success(result, success_reason={"action": "processed"})

            def close(self) -> None:
                self.shutdown_batch_processing()
                super().close()
    """

    # These are set by init_batch_processing()
    # Buffer stores (token, result_or_exception, state_id) tuples
    # Result can be TransformResult (normal) or ExceptionResult (plugin bug)
    _batch_buffer: RowReorderBuffer[tuple[TokenInfo, TransformResult | ExceptionResult, str | None]]
    _batch_executor: ThreadPoolExecutor
    _batch_output: OutputPort
    _batch_release_thread: threading.Thread
    _batch_shutdown: threading.Event
    _batch_name: str
    _batch_submissions: dict[tuple[str, str], RowTicket]
    _batch_submissions_lock: threading.Lock
    _batch_wait_timeout: float  # Timeout for waiter.wait() in executor

    def init_batch_processing(
        self,
        max_pending: int,
        output: OutputPort,
        name: str | None = None,
        max_workers: int | None = None,
        batch_wait_timeout: float = 3600.0,
    ) -> None:
        """Initialize batch processing infrastructure.

        Call this in __init__ after super().__init__().

        IMPORTANT: max_workers should equal max_pending to avoid starvation.
        Each pending row needs a worker thread to process it.

        Args:
            max_pending: Max rows in flight (backpressure threshold)
            output: Output port to emit results to (sink or next transform)
            name: Name for logging/metrics (default: class name)
            max_workers: Worker threads (default: max_pending)
            batch_wait_timeout: Max seconds for executor to wait for row result.
                Should match max_capacity_retry_seconds from pool config (default 3600).
        """
        self._batch_name = name or self.__class__.__name__
        self._batch_output = output
        self._batch_shutdown = threading.Event()
        self._batch_wait_timeout = batch_wait_timeout

        # Track submissions by (token_id, state_id) for eviction on timeout
        self._batch_submissions = {}
        self._batch_submissions_lock = threading.Lock()

        # Row reorder buffer with backpressure
        self._batch_buffer = RowReorderBuffer(
            max_pending=max_pending,
            name=self._batch_name,
        )

        # Worker pool - size matches max_pending to avoid starvation
        self._batch_executor = ThreadPoolExecutor(
            max_workers=max_workers or max_pending,
            thread_name_prefix=f"{self._batch_name}-worker",
        )

        # Release thread - emits results in FIFO order
        self._batch_release_thread = threading.Thread(
            target=self._release_loop,
            name=f"{self._batch_name}-release",
            daemon=False,  # Non-daemon: ensure clean shutdown
        )
        self._batch_release_thread.start()

    def accept_row(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> None:
        """Accept a row for processing.

        Blocks only on backpressure (buffer full). Processing happens
        asynchronously in worker threads. Results are emitted in FIFO
        order via the output port.

        Args:
            row: The row data
            ctx: Plugin context (must have ctx.token set)
            processor: Function that does actual processing

        Raises:
            ValueError: If ctx.token is None
            ShutdownError: If batch processing is shut down
        """
        # No defensive fallback - ctx.token is required (CLAUDE.md compliance)
        if ctx.token is None:
            raise ValueError("BatchTransformMixin requires ctx.token to be set. This is a bug in the calling code.")

        token = ctx.token
        row_id = token.token_id
        state_id = ctx.state_id

        # Submit to buffer (blocks on backpressure)
        ticket = self._batch_buffer.submit(row_id)

        # Track submission for potential eviction on timeout
        if state_id is not None:
            with self._batch_submissions_lock:
                self._batch_submissions[(token.token_id, state_id)] = ticket

        # Submit to worker pool
        self._batch_executor.submit(
            self._process_and_complete,
            ticket,
            token,
            row,
            ctx,
            processor,
        )

    def _process_and_complete(
        self,
        ticket: RowTicket,
        token: TokenInfo,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> None:
        """Worker thread: process row and mark complete.

        Called by worker threads. Executes the processor function and
        marks the ticket complete with the result.

        Exception Handling (CLAUDE.md compliance):
            - TransformResult.error() from processor: expected row processing failure
            - Uncaught exceptions from processor: plugin bugs, should crash

        Plugin bugs are wrapped in ExceptionResult and re-raised in the
        orchestrator thread when the waiter retrieves the result. This ensures
        bugs crash the pipeline rather than being silently converted to errors.

        Retry Safety:
            The state_id is captured from ctx and stored with the result to ensure
            results are routed to the correct waiter even during retry scenarios.
        """
        # Capture state_id for retry-safe result routing
        state_id = ctx.state_id

        try:
            result = processor(row, ctx)
        except Exception as e:
            # Plugin bug - wrap exception for propagation to orchestrator
            # The waiter will re-raise this exception in the main thread
            tb = traceback.format_exc()

            # Import here to avoid circular dependency at module load time
            from elspeth.contracts import ExceptionResult

            exception_result = ExceptionResult(exception=e, traceback=tb)
            # KeyError means ticket was evicted due to timeout - discard late result.
            # This is expected when a waiter times out and retry proceeds
            # while the original worker was still processing.
            with contextlib.suppress(KeyError):
                self._batch_buffer.complete(ticket, (token, exception_result, state_id))
            return

        # Mark complete - result will be released in FIFO order
        # Include state_id for retry-safe waiter matching
        # KeyError means ticket was evicted due to timeout - discard late result.
        with contextlib.suppress(KeyError):
            self._batch_buffer.complete(ticket, (token, result, state_id))

    def _release_loop(self) -> None:
        """Release thread: emit results in FIFO order to output port.

        Runs in a dedicated thread. Blocks on wait_for_next_release() until
        the next FIFO-ordered result is ready, then emits to output port.

        Retry Safety:
            Each result includes state_id to ensure correct waiter matching.
            This prevents stale results from being delivered to retry attempts.
        """
        while not self._batch_shutdown.is_set():
            try:
                # Block until next result is ready (FIFO order)
                entry = self._batch_buffer.wait_for_next_release(timeout=1.0)

                # Unpack token, result, and state_id for retry-safe routing
                token, result, state_id = entry.result

                # Clean up submission tracking (before emit, in case emit fails)
                if state_id is not None:
                    with self._batch_submissions_lock:
                        self._batch_submissions.pop((token.token_id, state_id), None)

                # Emit to output port with state_id for correct waiter matching
                # The port may block if downstream is applying backpressure
                self._batch_output.emit(token, result, state_id)

            except TimeoutError:
                # Normal during low load - just check shutdown and continue
                continue

            except ShutdownError:
                # Buffer was shut down - exit cleanly
                break

            except Exception as e:
                # Output port failure - this is a bug (CLAUDE.md compliance)
                # Wrap exception and emit it so the waiter can propagate, rather than hanging
                tb = traceback.format_exc()

                from elspeth.contracts import ExceptionResult

                exception_result = ExceptionResult(exception=e, traceback=tb)

                try:
                    # Emit exception result so waiter can re-raise in orchestrator thread
                    # state_id is from the current entry (captured above before exception)
                    self._batch_output.emit(token, exception_result, state_id)
                except Exception as emit_err:
                    # Port is completely broken - log critical and continue
                    # Other results might still be deliverable if port is intermittently failing
                    import logging

                    logging.getLogger(__name__).critical(
                        f"Cannot deliver exception to waiter for {self._batch_name}. "
                        f"Waiter will hang until timeout. "
                        f"Original error: {e}\nEmit error: {emit_err}\n{tb}"
                    )

    def flush_batch_processing(self, timeout: float = 300.0) -> None:
        """Wait for all pending rows to complete and emit.

        Call this at end of pipeline to ensure all work is done.

        Args:
            timeout: Maximum seconds to wait

        Raises:
            TimeoutError: If not all rows complete in time
        """
        deadline = threading.Event()
        start = threading.Event()

        def check_empty() -> None:
            start.wait()
            while self._batch_buffer.pending_count > 0:
                if deadline.is_set():
                    return
                threading.Event().wait(0.1)  # Poll every 100ms

        checker = threading.Thread(target=check_empty)
        checker.start()
        start.set()
        checker.join(timeout=timeout)

        if self._batch_buffer.pending_count > 0:
            # Signal checker thread to exit before raising
            deadline.set()
            checker.join(timeout=1.0)  # Give it a moment to exit cleanly
            raise TimeoutError(f"Flush timeout: {self._batch_buffer.pending_count} rows still pending")

    def evict_submission(self, token_id: str, state_id: str) -> bool:
        """Evict a submission from the buffer.

        Called by the executor when a waiter times out. This removes the
        buffer entry so that retries can proceed without FIFO blocking.

        The eviction flow:
        1. First attempt times out at waiter.wait()
        2. Executor calls evict_submission() to remove buffer entry
        3. Retry attempt gets new sequence number and can proceed
        4. Original worker may still complete, but result is discarded

        Args:
            token_id: Token ID of the submission to evict
            state_id: State ID of the submission to evict

        Returns:
            True if entry was evicted, False if not found or already completed
        """
        key = (token_id, state_id)
        with self._batch_submissions_lock:
            ticket = self._batch_submissions.pop(key, None)

        if ticket is None:
            return False

        return self._batch_buffer.evict(ticket)

    def shutdown_batch_processing(self, timeout: float = 30.0) -> None:
        """Shutdown batch processing gracefully.

        Call this in close() or cleanup.

        Order:
        1. Signal shutdown
        2. Wait for workers to finish current tasks
        3. Shutdown buffer (wakes release thread)
        4. Wait for release thread to finish
        """
        # 1. Signal shutdown
        self._batch_shutdown.set()

        # 2. Shutdown worker pool (finish current tasks)
        self._batch_executor.shutdown(wait=True)

        # 3. Shutdown buffer (wakes release thread with ShutdownError)
        self._batch_buffer.shutdown()

        # 4. Wait for release thread
        self._batch_release_thread.join(timeout=timeout)
        if self._batch_release_thread.is_alive():
            import logging

            logging.getLogger(__name__).warning(f"Release thread for {self._batch_name} did not stop cleanly")

    # --- Observability ---

    def get_batch_metrics(self) -> dict[str, Any]:
        """Get batch processing metrics."""
        return self._batch_buffer.get_metrics()

    @property
    def batch_pending_count(self) -> int:
        """Number of rows currently in flight."""
        return self._batch_buffer.pending_count
