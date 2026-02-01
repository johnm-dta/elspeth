# src/elspeth/engine/batch_adapter.py
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

import threading
from typing import TYPE_CHECKING

from elspeth.contracts import ExceptionResult

if TYPE_CHECKING:
    from elspeth.contracts import TransformResult
    from elspeth.contracts.identity import TokenInfo

# Type alias for waiter keys: (token_id, state_id)
WaiterKey = tuple[str, str]


class RowWaiter:
    """Waiter for a specific row's result.

    Created by SharedBatchAdapter.register() and returned to the caller.
    The caller blocks on wait() until emit() delivers the matching result.
    """

    def __init__(
        self,
        key: WaiterKey,
        event: threading.Event,
        results: dict[WaiterKey, TransformResult | ExceptionResult],
        waiters: dict[WaiterKey, threading.Event],
        lock: threading.Lock,
    ):
        """Initialize waiter for a specific token and attempt.

        Args:
            key: (token_id, state_id) tuple identifying this waiter
            event: Event that will be signaled when result arrives
            results: Shared results dict (owned by SharedBatchAdapter)
            waiters: Shared waiters dict (owned by SharedBatchAdapter)
            lock: Shared lock for thread-safe access
        """
        self._key = key
        self._event = event
        self._results = results
        self._waiters = waiters
        self._lock = lock

    def wait(self, timeout: float = 300.0) -> TransformResult:
        """Block until this row's result arrives.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            TransformResult for this row

        Raises:
            TimeoutError: If result not received within timeout
            Exception: Re-raised from worker thread if plugin bug occurred
        """
        token_id, state_id = self._key
        if not self._event.wait(timeout=timeout):
            # Clean up waiter AND any late result on timeout to prevent memory leak.
            #
            # Race condition: emit() can execute between event.wait() timeout and
            # this lock acquisition, storing a result that no one will retrieve:
            #
            #   wait(): event.wait() → False
            #   emit(): acquires lock, stores _results[key], deletes _waiters[key]
            #   wait(): acquires lock, removes from _waiters (gone), raises
            #   Result: _results[key] leaked forever
            #
            # Fix: also clean up _results in the timeout path.
            with self._lock:
                self._waiters.pop(self._key, None)
                self._results.pop(self._key, None)  # Clean up any late result from race
            raise TimeoutError(
                f"No result received for token {token_id} (state {state_id}) within {timeout}s. "
                f"This may indicate a hung transform, rate limit exhaustion, or "
                f"insufficient timeout."
            )

        with self._lock:
            result = self._results.pop(self._key)

            # Check for wrapped exception from worker thread
            # Plugin bugs should crash - re-raise the original exception
            if isinstance(result, ExceptionResult):
                raise result.exception from None

            return result


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
        self._waiters: dict[WaiterKey, threading.Event] = {}
        self._results: dict[WaiterKey, TransformResult | ExceptionResult] = {}
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
            event = threading.Event()
            self._waiters[key] = event
            return RowWaiter(key, event, self._results, self._waiters, self._lock)

    def emit(self, token: TokenInfo, result: TransformResult | ExceptionResult, state_id: str | None) -> None:
        """Receive result from batch transform's release thread.

        Routes result to the correct waiter based on (token_id, state_id).
        Called by BatchTransformMixin._release_loop() via output port.

        If no waiter exists for this (token_id, state_id), the result is
        discarded. This happens when:
        - A timeout occurred and a retry is in progress (correct behavior)
        - A bug caused mismatched state_ids (should not happen)

        Args:
            token: Token for this row
            result: Transform result or wrapped exception
            state_id: State ID for the attempt that produced this result.
                     If None, result is discarded (no waiter can be matched).
        """
        if state_id is None:
            # Cannot match a waiter without state_id - discard result
            return
        key: WaiterKey = (token.token_id, state_id)
        with self._lock:
            if key in self._waiters:
                # Store result and wake the waiter
                self._results[key] = result
                self._waiters[key].set()
                # Clean up waiter entry (result stays until wait() retrieves it)
                del self._waiters[key]
            # If no waiter exists, result is discarded (stale result from timed-out attempt)
            # Note: We don't store the result - that would be a memory leak

    def clear(self) -> None:
        """Clear all pending waiters and results.

        For testing and cleanup. In production, normal flow ensures
        all waiters receive results before shutdown.
        """
        with self._lock:
            self._waiters.clear()
            self._results.clear()
