# src/elspeth/plugins/llm/reorder_buffer.py
"""Reorder buffer for maintaining strict submission order with timing.

Results may complete out of order (due to varying API latencies),
but are emitted in the exact order they were submitted. Timing
metadata is captured for audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class BufferEntry(Generic[T]):
    """Entry emitted from the reorder buffer with timing metadata.

    Attributes:
        submit_index: Order in which item was submitted (0-indexed)
        complete_index: Order in which item completed (may differ from submit)
        result: The actual result value
        submit_timestamp: time.perf_counter() when submitted
        complete_timestamp: time.perf_counter() when completed
        buffer_wait_ms: Time spent waiting in buffer after completion
    """

    submit_index: int
    complete_index: int
    result: T
    submit_timestamp: float
    complete_timestamp: float
    buffer_wait_ms: float


@dataclass
class _InternalEntry(Generic[T]):
    """Internal entry in the reorder buffer."""

    submit_index: int
    submit_timestamp: float
    complete_index: int | None = None
    complete_timestamp: float | None = None
    result: T | None = None
    is_complete: bool = False


class ReorderBuffer(Generic[T]):
    """Thread-safe buffer that reorders results to match submission order.

    Captures timing metadata for each entry to support audit trail
    requirements.

    Usage:
        buffer = ReorderBuffer[TransformResult]()

        # Submit work (returns index)
        idx = buffer.submit()

        # ... do async work ...

        # Complete with result (may be out of order)
        buffer.complete(idx, result)

        # Get results in submission order with timing
        ready = buffer.get_ready_results()
        for entry in ready:
            print(f"Result: {entry.result}, waited {entry.buffer_wait_ms}ms")
    """

    def __init__(self) -> None:
        """Initialize empty buffer."""
        self._entries: dict[int, _InternalEntry[T]] = {}
        self._next_submit: int = 0
        self._next_emit: int = 0
        self._complete_counter: int = 0
        self._lock = Lock()

    @property
    def pending_count(self) -> int:
        """Number of submitted but not-yet-emitted items (thread-safe)."""
        with self._lock:
            return self._next_submit - self._next_emit

    def submit(self) -> int:
        """Reserve a slot and return its index (thread-safe).

        Returns:
            Index to use when completing this item
        """
        with self._lock:
            idx = self._next_submit
            self._entries[idx] = _InternalEntry(
                submit_index=idx,
                submit_timestamp=time.perf_counter(),
            )
            self._next_submit += 1
            return idx

    def complete(self, index: int, result: T) -> None:
        """Mark an item as complete with its result (thread-safe).

        Args:
            index: Index returned from submit()
            result: The result for this item

        Raises:
            KeyError: If index was never submitted
            ValueError: If index was already completed
        """
        with self._lock:
            if index not in self._entries:
                raise KeyError(f"Index {index} was never submitted")

            entry = self._entries[index]
            if entry.is_complete:
                raise ValueError(f"Index {index} was already completed")

            entry.result = result
            entry.complete_index = self._complete_counter
            entry.complete_timestamp = time.perf_counter()
            entry.is_complete = True
            self._complete_counter += 1

    def get_ready_results(self) -> list[BufferEntry[T]]:
        """Get all results that are ready to emit in order (thread-safe).

        Returns results that are:
        1. Complete (result received)
        2. All previous indices are also complete

        Returns:
            List of BufferEntry in submission order (may be empty)
        """
        with self._lock:
            ready: list[BufferEntry[T]] = []
            now = time.perf_counter()

            while self._next_emit in self._entries:
                entry = self._entries[self._next_emit]
                if not entry.is_complete:
                    break

                # Entry is complete and all previous are emitted
                # Calculate buffer wait time (time between completion and emission)
                buffer_wait_ms = (now - entry.complete_timestamp) * 1000  # type: ignore[operator]

                ready.append(
                    BufferEntry(
                        submit_index=entry.submit_index,
                        complete_index=entry.complete_index,  # type: ignore[arg-type]
                        result=entry.result,  # type: ignore[arg-type]
                        submit_timestamp=entry.submit_timestamp,
                        complete_timestamp=entry.complete_timestamp,  # type: ignore[arg-type]
                        buffer_wait_ms=buffer_wait_ms,
                    )
                )
                del self._entries[self._next_emit]
                self._next_emit += 1

            return ready
