# src/elspeth/telemetry/buffer.py
"""Bounded buffer for telemetry event batching.

Provides a ring buffer implementation that drops oldest events on overflow.
Used by telemetry exporters to batch events before sending to external
observability platforms.

Key design decisions:
- Ring buffer via deque(maxlen=N): Automatic oldest-first eviction
- Correct overflow counting: Check was_full BEFORE append (deque evicts during)
- Aggregate logging: Log every 100 drops to prevent Warning Fatigue
"""

from collections import deque

import structlog

from elspeth.contracts.events import TelemetryEvent

logger = structlog.get_logger(__name__)


class BoundedBuffer:
    """Ring buffer that drops oldest events on overflow.

    NOTE: Aggregate logging per Systems Thinking review - logs every 100 drops
    instead of per-event to avoid Warning Fatigue.

    Thread Safety:
        NOT thread-safe. External synchronization required if used from
        multiple threads. The TelemetryManager is responsible for
        serializing access when using background export threads.

    Attributes:
        dropped_count: Total number of events dropped due to buffer overflow.

    Example:
        buffer = BoundedBuffer(max_size=1000)
        buffer.append(event)
        batch = buffer.pop_batch(max_count=100)
    """

    # Log aggregate metrics every N drops to avoid Warning Fatigue
    _LOG_INTERVAL = 100

    def __init__(self, max_size: int = 10_000) -> None:
        """Initialize the bounded buffer.

        Args:
            max_size: Maximum number of events to buffer. When full, oldest
                events are automatically evicted on append. Defaults to 10,000.

        Raises:
            ValueError: If max_size < 1.
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._buffer: deque[TelemetryEvent] = deque(maxlen=max_size)
        self._dropped_count: int = 0
        self._last_logged_drop_count: int = 0

    def append(self, event: TelemetryEvent) -> None:
        """Append event to buffer, tracking drops correctly.

        Logging strategy (Warning Fatigue prevention):
        - Logs every 100 drops, not every single drop
        - Includes drop rate trend information

        Args:
            event: The telemetry event to buffer.
        """
        was_full = len(self._buffer) == self._buffer.maxlen
        self._buffer.append(event)
        if was_full:
            # deque auto-dropped the oldest item
            self._dropped_count += 1

            # Aggregate logging: log every _LOG_INTERVAL drops
            if self._dropped_count - self._last_logged_drop_count >= self._LOG_INTERVAL:
                logger.warning(
                    "Telemetry buffer overflow - events dropped",
                    dropped_since_last_log=self._LOG_INTERVAL,
                    dropped_total=self._dropped_count,
                    buffer_size=self._buffer.maxlen,
                    hint="Consider increasing buffer size or reducing granularity",
                )
                self._last_logged_drop_count = self._dropped_count

    def pop_batch(self, max_count: int) -> list[TelemetryEvent]:
        """Pop up to max_count events from the buffer.

        Events are returned in FIFO order (oldest first).

        Args:
            max_count: Maximum number of events to retrieve.

        Returns:
            List of events, up to max_count. May be empty if buffer is empty.
        """
        batch = []
        for _ in range(min(max_count, len(self._buffer))):
            batch.append(self._buffer.popleft())
        return batch

    @property
    def dropped_count(self) -> int:
        """Number of events dropped due to buffer overflow."""
        return self._dropped_count

    def __len__(self) -> int:
        """Return the current number of events in the buffer."""
        return len(self._buffer)
