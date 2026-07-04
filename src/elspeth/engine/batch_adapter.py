"""Adapter for batch transform integration with TransformExecutor.

Allows TransformExecutor to call accept() and wait for results while
maintaining concurrency across multiple rows.

Architecture:
    Orchestrator → TransformExecutor → [BatchTransformMixin] → Worker Pool
                                             ↓
                                    SharedBatchAdapter (this file)
                                             ↓
                                    Multiple RowWaiters registered by (token_id, state_id)
                                             ↓
                                    emit() routes results to correct waiter
                                             ↓
                                    Each waiter.wait() returns its result

The key insight is that ONE adapter is connected to the transform's output port,
but MULTIPLE waiters can be registered (one per in-flight row). The emit() method
routes each result to the correct waiter based on (token_id, state_id).

Retry Safety:
    Waiters are keyed by (token_id, state_id) to prevent stale results from being
    delivered to retry attempts. If a timeout occurs and a retry happens, the new
    attempt gets a new state_id, so the old worker's result goes to the orphaned
    waiter (which will be garbage collected), not to the retry's waiter.

Exception Propagation:
    Plugin bugs should crash the orchestrator (CLAUDE.md compliance). Worker threads
    wrap uncaught exceptions in ExceptionResult so they propagate through the async
    pattern. RowWaiter.wait() detects these and re-raises the original exception.
"""

from __future__ import annotations

__all__ = ["ExceptionResult", "RowWaiter", "SharedBatchAdapter", "WaiterKey", "_WaiterEntry"]

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from elspeth.contracts import ExceptionResult
from elspeth.contracts.errors import OrchestrationInvariantError

if TYPE_CHECKING:
    from elspeth.contracts import TransformResult
    from elspeth.contracts.identity import TokenInfo

# Type alias for waiter keys: (token_id, state_id)
WaiterKey = tuple[str, str]


@dataclass
class _WaiterEntry:
    """Consolidated state for a single registered waiter.

    Replaces two parallel dicts (_waiters + _results) that were
    both keyed by WaiterKey. The event is set when a result arrives;
    result starts as None and is populated by deliver().

    The deliver() method atomically sets result and signals the event,
    enforcing the invariant that result is always set before the event
    is signaled.
    """

    event: threading.Event = field(default_factory=threading.Event)
    result: TransformResult | ExceptionResult | None = None

    def deliver(self, result: TransformResult | ExceptionResult) -> None:
        """Atomically set result and signal the event.

        Enforces the invariant: result MUST be set before event.set().
        This replaces the two-step pattern (entry.result = x; entry.event.set())
        that relied on callers to maintain ordering.
        """
        self.result = result
        self.event.set()


class RowWaiter:
    """Waiter for a specific row's result.

    Created by SharedBatchAdapter.register() and returned to the caller.
    The caller blocks on wait() until emit() delivers the matching result.
    """

    def __init__(
        self,
        key: WaiterKey,
        entry: _WaiterEntry,
        entries: dict[WaiterKey, _WaiterEntry],
        lock: threading.Lock,
    ):
        """Initialize waiter for a specific token and attempt.

        Args:
            key: (token_id, state_id) tuple identifying this waiter
            entry: The waiter entry containing event and result slot
            entries: Shared entries dict (owned by SharedBatchAdapter)
            lock: Shared lock for thread-safe access
        """
        self._key = key
        self._event = entry.event
        self._entries = entries
        self._lock = lock

    def wait(self, timeout: float = 300.0, shutdown_event: threading.Event | None = None) -> TransformResult:
        """Block until this row's result arrives.

        Args:
            timeout: Maximum seconds to wait
            shutdown_event: Optional run-cancellation signal. When set, the
                waiter stops waiting immediately instead of sleeping out the
                full transform timeout.

        Returns:
            TransformResult for this row

        Raises:
            TimeoutError: If result not received within timeout
            InterruptedError: Run cancellation observed while waiting. The
                waiter is a delivery primitive — it signals cancellation with
                the same typed exception the sync path uses (retry.py) and the
                processor translates it into the row-scoped
                ``shutdown_requested`` error result
                (``_convert_retryable_to_error_result``), so shutdown policy
                lives in one layer (filigree elspeth-14571961a6).
            Exception: Re-raised from worker thread if plugin bug occurred
        """
        token_id, state_id = self._key
        deadline = time.monotonic() + timeout
        while not self._event.is_set():
            if shutdown_event is not None and shutdown_event.is_set():
                with self._lock:
                    if self._key in self._entries:
                        del self._entries[self._key]
                raise InterruptedError(f"Shutdown requested while waiting for token {token_id} (state {state_id})")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Clean up entry on timeout to prevent memory leak.
                # Guarded delete: clear() can race with timeout during shutdown — the
                # only path that removes entries we did not pop ourselves. Explicit
                # membership check expresses the cleanup intent without R9's pop-default.
                with self._lock:
                    if self._key in self._entries:
                        del self._entries[self._key]
                raise TimeoutError(
                    f"No result received for token {token_id} (state {state_id}) within {timeout}s. "
                    f"This may indicate a hung transform, rate limit exhaustion, or "
                    f"insufficient timeout."
                )

            self._event.wait(timeout=min(remaining, 0.05))

        with self._lock:
            entry = self._entries.pop(self._key)

            # Check for wrapped exception from worker thread
            # Plugin bugs should crash - re-raise the original exception
            if isinstance(entry.result, ExceptionResult):
                raise entry.result.exception

            # result is guaranteed non-None here: emit() sets it before signaling event
            if entry.result is None:
                raise OrchestrationInvariantError(
                    "BatchAdapter waiter signaled but result is None — emit() must set result before event.set()"
                )
            return entry.result


class SharedBatchAdapter:
    """Shared output port adapter for batch transforms.

    Allows multiple rows to be in flight concurrently while routing
    results back to the correct waiter based on (token_id, state_id).

    This solves the deadlock in the previous BlockingResultAdapter design,
    where one adapter per row was created but only the first was connected
    to the transform's output port.

    Retry Safety:
        Waiters are keyed by (token_id, state_id) to prevent stale results
        from being delivered to retry attempts. Each attempt gets a unique
        state_id from begin_node_state(), so:
        - First attempt: state_id=abc → waiter keyed by (token, abc)
        - Timeout → waiter times out
        - Retry: state_id=def → NEW waiter keyed by (token, def)
        - First worker finishes → emit with state_id=abc → no matching waiter
        - Retry worker finishes → emit with state_id=def → correct waiter

    Thread Safety:
        - register() called by orchestrator thread (sequential)
        - emit() called by BatchTransformMixin's release thread
        - wait() called by orchestrator thread (blocks until result)
        - All shared state protected by self._lock

    Usage:
        adapter = SharedBatchAdapter()
        transform.connect_output(adapter, max_pending=30)

        # For each row (sequential in orchestrator):
        waiter = adapter.register(token.token_id, ctx.state_id)
        transform.accept(row, ctx)
        result = waiter.wait()  # Blocks until THIS attempt's result arrives
    """

    def __init__(self) -> None:
        """Initialize adapter."""
        self._entries: dict[WaiterKey, _WaiterEntry] = {}
        self._lock = threading.Lock()

    def register(self, token_id: str, state_id: str) -> RowWaiter:
        """Register a waiter for a specific token and attempt.

        Must be called BEFORE accept() for the corresponding row.
        The returned RowWaiter can then be used to wait() for the result.

        Args:
            token_id: Token ID to wait for
            state_id: State ID for this attempt (unique per retry)

        Returns:
            RowWaiter that can wait() for the result
        """
        key: WaiterKey = (token_id, state_id)
        with self._lock:
            entry = _WaiterEntry()
            self._entries[key] = entry
            return RowWaiter(key, entry, self._entries, self._lock)

    def emit(self, token: TokenInfo, result: TransformResult | ExceptionResult, state_id: str | None) -> None:
        """Receive result from batch transform's release thread.

        Routes result to the correct waiter based on (token_id, state_id).
        Called by BatchTransformMixin._release_loop() via output port.

        If no entry exists for this (token_id, state_id), the result is
        discarded. This happens when:
        - A timeout occurred and a retry is in progress (correct behavior)
        - A bug caused mismatched state_ids (should not happen)

        Args:
            token: Token for this row
            result: Transform result or wrapped exception
            state_id: State ID for the attempt that produced this result.

        Raises:
            OrchestrationInvariantError: If state_id is None AND no waiter can
                be found by token_id fallback — indicates an executor bug that
                cannot be recovered.
        """
        if state_id is None:
            # state_id=None is an executor bug (state_id should be set by
            # begin_node_state before calling the batch transform). Instead of
            # raising here (which the release loop would catch, retry with the
            # same None, and ultimately leave the waiter hanging until timeout),
            # find the waiter by token_id and deliver the error directly.
            error = OrchestrationInvariantError(
                f"SharedBatchAdapter.emit() called with state_id=None for token "
                f"{token.token_id}. This indicates an executor bug: state_id must "
                f"be set by begin_node_state() before calling the batch transform."
            )
            self._signal_waiters_by_token_id(token.token_id, error)
            return

        key: WaiterKey = (token.token_id, state_id)
        with self._lock:
            if key in self._entries and not self._entries[key].event.is_set():
                # First delivery: store result and wake the waiter.
                # Entry stays until wait() pops it to retrieve the result.
                self._entries[key].deliver(result)
            # If no entry exists or already signaled, result is discarded.
            # No entry: stale result from timed-out attempt.
            # Already signaled: duplicate emit — preserve first-result-wins.

    def clear(self) -> None:
        """Clear all pending entries.

        For testing and cleanup. In production, normal flow ensures
        all waiters receive results before shutdown.
        """
        with self._lock:
            self._entries.clear()

    def _signal_waiters_by_token_id(self, token_id: str, error: Exception) -> None:
        """Signal all waiters for a token_id with an error when state_id is unknown.

        This handles the case where emit() is called with state_id=None (an
        executor bug). Without state_id we can't construct the exact WaiterKey,
        so we scan all registered entries for matching token_id and deliver
        the error as an ExceptionResult. This ensures waiters fail fast instead
        of hanging until batch_wait_timeout.

        If no matching entry is found, the error is raised (no recovery possible).
        """
        import traceback as tb_mod

        exception_result = ExceptionResult(
            exception=error,
            traceback="".join(tb_mod.format_exception(error)),
        )
        with self._lock:
            matched_keys = [k for k in self._entries if k[0] == token_id]
            if not matched_keys:
                raise error
            for key in matched_keys:
                self._entries[key].deliver(exception_result)
                # Entry stays until wait() pops it
