"""Mixin for transforms that process rows concurrently with FIFO output ordering.

Transforms using this mixin can process multiple rows concurrently while
guaranteeing FIFO output order. The orchestrator sees simple accept() calls;
concurrency is hidden inside the plugin.

Key principle: The transform doesn't know what's downstream. It just emits
to its output port. Could be a sink, could be another transform.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts.errors import FrameworkBugError
from elspeth.plugins.infrastructure.batching.ports import OutputPort
from elspeth.plugins.infrastructure.batching.row_reorder_buffer import (
    RowReorderBuffer,
    RowTicket,
    ShutdownError,
)

if TYPE_CHECKING:
    from typing import Any

    from elspeth.contracts import ExceptionResult, TransformResult
    from elspeth.contracts.contexts import TransformContext
    from elspeth.contracts.identity import TokenInfo
    from elspeth.contracts.schema_contract import PipelineRow

_logger = structlog.get_logger(__name__)


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

            def accept(self, row: dict, ctx: TransformContext) -> None:
                self.accept_row(row, ctx, self._do_processing)

            def _do_processing(
                self, row: dict, ctx: TransformContext
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
    _pool_size: int = 30  # Max concurrent rows; used by executor to cap adapter max_pending

    @property
    def batch_runtime_enabled(self) -> bool:
        """Runtime marker for the engine's row-pipelined batch path."""
        return True

    @property
    def batch_pool_size(self) -> int:
        """Preferred max pending submissions for the engine output adapter."""
        return self._pool_size

    @property
    def batch_wait_timeout(self) -> float:
        """Max seconds the engine should wait for one row result."""
        return self._batch_wait_timeout

    def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
        """Accept a row for concurrent processing.

        Concrete classes must override this to call accept_row() with their
        processing function.  The executor dispatches to this method after
        detecting the mixin via isinstance.
        """
        raise NotImplementedError("Subclasses must implement accept()")

    def connect_output(self, output: OutputPort, max_pending: int = 30) -> None:
        """Connect the output port and initialize batch infrastructure.

        Concrete classes may override to add pre/post logic, but must call
        init_batch_processing() to wire up the buffer and worker pool.
        """
        raise NotImplementedError("Subclasses must implement connect_output()")

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
        self._pool_size = max_pending

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
        row: PipelineRow,
        ctx: TransformContext,
        processor: Callable[[PipelineRow, TransformContext], TransformResult],
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
        # Guard: reject rows after shutdown signal (before touching buffer)
        if self._batch_shutdown.is_set():
            raise ShutdownError("Batch processing has been shut down")

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
        try:
            self._batch_executor.submit(
                self._process_and_complete,
                ticket,
                token,
                row,
                ctx,
                processor,
            )
        except RuntimeError:
            # The row has already been admitted to the FIFO buffer. Convert a
            # shutdown-time submit race into an explicit row failure so the
            # waiter/output path is satisfied and no ticket is stranded.
            if state_id is not None:
                with self._batch_submissions_lock:
                    submission_key = (token.token_id, state_id)
                    if submission_key in self._batch_submissions:
                        self._batch_submissions.pop(submission_key)

            from elspeth.contracts import TransformResult

            shutdown_result = TransformResult.error(
                {
                    "reason": "shutdown_requested",
                    "error": "thread pool shut down during submission",
                },
                retryable=False,
            )
            self._complete_ticket(ticket, token, shutdown_result, state_id)

    def _process_and_complete(
        self,
        ticket: RowTicket,
        token: TokenInfo,
        row: PipelineRow,
        ctx: TransformContext,
        processor: Callable[[PipelineRow, TransformContext], TransformResult],
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
            # Worker threads cannot crash the orchestrator directly. Transport
            # both plugin bugs and Tier 1 exceptions through the waiter path so
            # the orchestrator thread re-raises the original exception instead
            # of timing out waiting for a result.
            tb = traceback.format_exc()
            from elspeth.contracts import ExceptionResult

            exception_result = ExceptionResult(exception=e, traceback=tb)
            self._complete_ticket(ticket, token, exception_result, state_id)
            return

        # Mark complete — result will be released in FIFO order
        # Include state_id for retry-safe waiter matching
        self._complete_ticket(ticket, token, result, state_id)

    def _complete_ticket(
        self,
        ticket: RowTicket,
        token: TokenInfo,
        result: TransformResult | ExceptionResult,
        state_id: str | None,
    ) -> None:
        """Complete a ticket, discarding late results after timeout eviction."""
        try:
            self._batch_buffer.complete(ticket, (token, result, state_id))
        except (KeyError, ShutdownError):
            _logger.debug(
                "late_result_discarded",
                token_id=token.token_id,
                state_id=state_id,
                reason="timeout_evicted_or_shutdown",
            )

    def _release_loop(self) -> None:
        """Release thread: emit results in FIFO order to output port.

        Runs in a dedicated thread. Blocks on wait_for_next_release() until
        the next FIFO-ordered result is ready, then emits to output port.

        Exit condition: ShutdownError from buffer (set during shutdown_batch_processing).
        The loop does NOT check _batch_shutdown -- it keeps running until the buffer
        is explicitly shut down, ensuring all completed results are drained first.

        Retry Safety:
            Each result includes state_id to ensure correct waiter matching.
            This prevents stale results from being delivered to retry attempts.
        """
        while True:
            # Reset per-iteration to detect pre-unpack failures (Bug 2 fix).
            # If an exception occurs before entry.result is unpacked, token/state_id
            # remain None, preventing the handler from using stale values from a
            # previous iteration.
            token: TokenInfo | None = None
            state_id: str | None = None
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
                # Normal during low load - just loop and try again
                continue

            except ShutdownError:
                # Buffer was shut down - exit cleanly.
                # This is the ONLY exit path: shutdown_batch_processing() waits for
                # workers to finish, then calls buffer.shutdown(), so all completed
                # results have been drained by this point.
                break

            except Exception as e:
                if token is None:
                    # Exception occurred BEFORE entry.result was unpacked.
                    # This is an internal invariant violation (e.g., buffer corruption).
                    # Re-raise immediately -- this is our bug, not a row-level error.
                    raise
                # Output port failure - this is a bug (CLAUDE.md compliance)
                # Wrap exception and emit it so the waiter can propagate, rather than hanging
                tb = traceback.format_exc()

                from elspeth.contracts import ExceptionResult

                exception_result = ExceptionResult(exception=e, traceback=tb)

                try:
                    # Emit exception result so waiter can re-raise in orchestrator thread
                    # state_id is from the current entry (captured above before exception)
                    self._batch_output.emit(token, exception_result, state_id)
                except contract_errors.TIER_1_ERRORS:
                    raise  # Tier 1 errors must crash immediately
                except Exception as emit_err:
                    # Port is completely broken — crash the release thread.
                    # A broken output port is a system bug: silently continuing
                    # would lose this token's result (waiter hangs until timeout)
                    # and potentially lose subsequent results too.
                    raise FrameworkBugError(
                        f"Output port for {self._batch_name} is broken: "
                        f"cannot deliver exception result to waiter. "
                        f"Original error: {e!r}. Emit error: {emit_err!r}"
                    ) from emit_err

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

    def shutdown_batch_processing(self, timeout: float = 30.0, *, wait_for_workers: bool = True) -> None:
        """Shutdown batch processing gracefully.

        Call this in close() or cleanup.

        Order (drain-first -- ensures no in-flight rows are dropped):
        1. Signal shutdown (prevents new submissions via accept_row)
        2. Wait for worker pool to finish current tasks
        3. Shutdown buffer (wakes release thread with ShutdownError)
        4. Wait for release thread to finish draining

        The release loop does NOT check _batch_shutdown; it exits only when
        the buffer raises ShutdownError. This ensures the loop keeps emitting
        completed results until the buffer is explicitly shut down AFTER
        workers have finished.

        When wait_for_workers is False, shutdown is cancellation-oriented:
        queued work is cancelled and completed late results are discarded
        rather than blocking close() behind provider calls that the run has
        already abandoned.
        """
        # 1. Signal shutdown (accept_row will raise ShutdownError on new submits)
        self._batch_shutdown.set()

        # 2. Wait for worker pool to finish current tasks.
        #    After this returns, all workers have called complete() on their
        #    tickets, so the release loop can drain every remaining entry.
        self._batch_executor.shutdown(wait=wait_for_workers, cancel_futures=not wait_for_workers)

        # 3. Shutdown buffer -- this wakes the release thread with ShutdownError
        #    so it exits after draining all completed entries.
        self._batch_buffer.shutdown()

        # 4. Wait for release thread to finish
        self._batch_release_thread.join(timeout=timeout)
        if self._batch_release_thread.is_alive():
            raise FrameworkBugError(
                f"Release thread for {self._batch_name} did not stop within {timeout}s. "
                f"In-flight results may not have been drained — pipeline cannot report success."
            )

    # --- Observability ---

    def get_batch_metrics(self) -> dict[str, Any]:
        """Get batch processing metrics."""
        return self._batch_buffer.get_metrics()

    @property
    def batch_pending_count(self) -> int:
        """Number of rows currently in flight."""
        return self._batch_buffer.pending_count
