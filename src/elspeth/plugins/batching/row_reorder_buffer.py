# src/elspeth/plugins/batching/row_reorder_buffer.py
"""Row-level reorder buffer with backpressure support.

Extends the ReorderBuffer pattern to row-level pipelining:
- Accepts rows out-of-order, releases in submission order (FIFO)
- Blocks on submission when max_pending reached (backpressure)
- Provides blocking wait_for_next_release() for release loop

This is the row-level equivalent of ReorderBuffer (used for queries within a row).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Any


class ShutdownError(RuntimeError):
    """Raised when operations are attempted on a shutdown buffer."""

    pass


@dataclass(frozen=True)
class RowTicket:
    """Handle for a row submitted to the buffer.

    Immutable ticket returned by submit(). Pass to complete() when processing
    finishes. The sequence number determines FIFO release order.
    """

    sequence: int
    row_id: str
    submitted_at: float


@dataclass
class RowBufferEntry[T]:
    """Entry emitted from the buffer with timing metadata."""

    sequence: int
    row_id: str
    result: T
    submitted_at: float
    completed_at: float
    buffer_wait_ms: float  # Time between completion and release


@dataclass
class _PendingEntry[T]:
    """Internal entry tracking a pending row."""

    sequence: int
    row_id: str
    submitted_at: float
    completed_at: float | None = None
    result: T | None = None
    is_complete: bool = False


class RowReorderBuffer[T]:
    """Thread-safe buffer for row-level pipelining with FIFO ordering.

    Accepts rows out-of-order (as workers complete), releases in strict
    submission order. Provides backpressure when max_pending is reached.

    Thread Safety Model:
        - submit(): Called by orchestrator thread, blocks on backpressure
        - complete(): Called by worker threads, wakes release waiter
        - wait_for_next_release(): Called by release thread, blocks until FIFO-ready

    Key difference from ReorderBuffer:
        - ReorderBuffer.get_ready_results() returns a list (polling)
        - RowReorderBuffer.wait_for_next_release() blocks until ready (event-driven)

    Invariants:
        - next_release_seq <= next_submit_seq
        - len(pending) <= max_pending
        - Results are released in exact submission order

    Usage:
        buffer = RowReorderBuffer[TransformResult](max_pending=30)

        # Orchestrator thread submits
        ticket = buffer.submit("row-123")  # May block on backpressure

        # Worker thread completes (may be out of order)
        buffer.complete(ticket, result)

        # Release thread emits in order
        entry = buffer.wait_for_next_release()  # Blocks until FIFO-ready
        output_port.emit(entry.result)
    """

    def __init__(
        self,
        max_pending: int = 100,
        name: str = "row-reorder",
    ) -> None:
        """Initialize buffer with backpressure limit.

        Args:
            max_pending: Maximum rows in flight before submit() blocks
            name: Name for logging/metrics
        """
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}")

        self._name = name
        self._max_pending = max_pending

        # Single lock protects all state
        self._lock = Lock()

        # Two conditions for different wait scenarios
        # submit_condition: wait for space (backpressure relief)
        # release_condition: wait for next FIFO result
        self._submit_condition = Condition(self._lock)
        self._release_condition = Condition(self._lock)

        # Sequence tracking
        self._next_submit_seq = 0
        self._next_release_seq = 0

        # Pending entries: sequence -> entry
        self._pending: dict[int, _PendingEntry[T]] = {}

        # Shutdown flag
        self._shutdown = False

        # Metrics
        self._total_submitted = 0
        self._total_released = 0
        self._max_observed_pending = 0
        self._total_wait_time_ms = 0.0

    # --- Core Operations ---

    def submit(self, row_id: str, timeout: float | None = None) -> RowTicket:
        """Submit a row for processing. Returns ticket to complete later.

        Blocks if max_pending rows are already in flight (backpressure).

        Args:
            row_id: Unique identifier for the row (e.g., token_id)
            timeout: Maximum seconds to wait for space (None = forever)

        Returns:
            RowTicket to pass to complete() when processing finishes

        Raises:
            ShutdownError: If buffer is shut down
            TimeoutError: If timeout exceeded waiting for space
        """
        deadline = time.monotonic() + timeout if timeout else None

        with self._submit_condition:
            # Wait for space (backpressure)
            while len(self._pending) >= self._max_pending:
                if self._shutdown:
                    raise ShutdownError(f"Buffer '{self._name}' is shut down")

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timeout waiting for buffer space (pending={len(self._pending)}, max={self._max_pending})")
                    self._submit_condition.wait(timeout=remaining)
                else:
                    self._submit_condition.wait()

            # Check shutdown again after wait
            if self._shutdown:
                raise ShutdownError(f"Buffer '{self._name}' is shut down")

            # Reserve slot
            seq = self._next_submit_seq
            self._next_submit_seq += 1
            now = time.perf_counter()

            self._pending[seq] = _PendingEntry(
                sequence=seq,
                row_id=row_id,
                submitted_at=now,
            )

            self._total_submitted += 1
            self._max_observed_pending = max(self._max_observed_pending, len(self._pending))

            return RowTicket(sequence=seq, row_id=row_id, submitted_at=now)

    def complete(self, ticket: RowTicket, result: T) -> None:
        """Mark a row as complete with its result.

        Called by worker threads when processing finishes. The result will
        be released when all predecessors have been released (FIFO).

        Args:
            ticket: The ticket from submit()
            result: The processing result

        Raises:
            KeyError: If ticket was never submitted
            ValueError: If ticket was already completed
        """
        with self._lock:
            if ticket.sequence not in self._pending:
                raise KeyError(f"Ticket {ticket.sequence} (row_id={ticket.row_id}) was never submitted")

            entry = self._pending[ticket.sequence]
            if entry.is_complete:
                raise ValueError(f"Ticket {ticket.sequence} (row_id={ticket.row_id}) already completed")

            entry.result = result
            entry.completed_at = time.perf_counter()
            entry.is_complete = True

            # Wake release waiter - use notify() not notify_all() to avoid thundering herd
            # Only one waiter can be next in sequence
            self._release_condition.notify()

    def wait_for_next_release(self, timeout: float | None = None) -> RowBufferEntry[T]:
        """Block until the next FIFO-ordered result is ready.

        Called by the release loop thread. Blocks until:
        1. The next sequence number is completed, AND
        2. We're ready to release it (all predecessors released)

        Args:
            timeout: Maximum seconds to wait (None = forever)

        Returns:
            RowBufferEntry with result and timing metadata

        Raises:
            ShutdownError: If buffer is shut down
            TimeoutError: If timeout exceeded
        """
        deadline = time.monotonic() + timeout if timeout else None

        with self._release_condition:
            while True:
                # Check shutdown
                if self._shutdown:
                    raise ShutdownError(f"Buffer '{self._name}' is shut down")

                # Check if next sequence is ready
                if self._next_release_seq in self._pending:
                    entry = self._pending[self._next_release_seq]
                    if entry.is_complete:
                        # Ready to release!
                        # Invariants: is_complete implies result and completed_at are set
                        if entry.result is None:
                            raise RuntimeError("Invariant violation: is_complete=True but result is None")
                        if entry.completed_at is None:
                            raise RuntimeError("Invariant violation: is_complete=True but completed_at is None")

                        now = time.perf_counter()
                        buffer_wait_ms = (now - entry.completed_at) * 1000

                        result_entry = RowBufferEntry(
                            sequence=entry.sequence,
                            row_id=entry.row_id,
                            result=entry.result,
                            submitted_at=entry.submitted_at,
                            completed_at=entry.completed_at,
                            buffer_wait_ms=buffer_wait_ms,
                        )

                        # Remove from pending
                        del self._pending[self._next_release_seq]
                        self._next_release_seq += 1
                        self._total_released += 1
                        self._total_wait_time_ms += buffer_wait_ms

                        # Wake one submitter waiting for space
                        self._submit_condition.notify()

                        return result_entry

                # Not ready - wait
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timeout waiting for sequence {self._next_release_seq}")
                    self._release_condition.wait(timeout=remaining)
                else:
                    self._release_condition.wait()

    def evict(self, ticket: RowTicket) -> bool:
        """Evict an entry that will never complete.

        Called when the caller gives up waiting for this entry (e.g., timeout
        occurred and retry is in progress). The release loop will skip this
        sequence number.

        This is essential for retry scenarios:
        1. Original attempt times out at waiter level
        2. Retry gets a new sequence number
        3. Without eviction, release loop blocks on original sequence forever
        4. With eviction, original sequence is skipped and retry can proceed

        Args:
            ticket: The ticket from submit() to evict

        Returns:
            True if entry was evicted, False if not found or already completed
        """
        with self._lock:
            if ticket.sequence not in self._pending:
                return False  # Already released or never submitted

            entry = self._pending[ticket.sequence]
            if entry.is_complete:
                return False  # Already complete, will be released soon

            # Remove from pending
            del self._pending[ticket.sequence]

            # Advance past any gaps (this sequence and any other evicted ones)
            while self._next_release_seq not in self._pending and self._next_release_seq < self._next_submit_seq:
                self._next_release_seq += 1

            # Wake release waiter in case next sequence is now ready
            self._release_condition.notify()

            # Wake submit waiter in case we freed up a slot
            self._submit_condition.notify()

            return True

    def shutdown(self) -> None:
        """Signal shutdown. Wakes all waiters with ShutdownError."""
        with self._lock:
            self._shutdown = True
            # Wake ALL waiters for shutdown (exception to notify() rule)
            self._submit_condition.notify_all()
            self._release_condition.notify_all()

    # --- Properties & Metrics ---

    @property
    def pending_count(self) -> int:
        """Number of rows currently in flight."""
        with self._lock:
            return len(self._pending)

    @property
    def is_shutdown(self) -> bool:
        """Whether buffer is shut down."""
        with self._lock:
            return self._shutdown

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics snapshot for observability."""
        with self._lock:
            completed_waiting = sum(1 for e in self._pending.values() if e.is_complete)
            return {
                "name": self._name,
                "max_pending": self._max_pending,
                "current_pending": len(self._pending),
                "completed_waiting": completed_waiting,
                "next_release_seq": self._next_release_seq,
                "total_submitted": self._total_submitted,
                "total_released": self._total_released,
                "max_observed_pending": self._max_observed_pending,
                "avg_buffer_wait_ms": (self._total_wait_time_ms / self._total_released if self._total_released > 0 else 0.0),
            }
