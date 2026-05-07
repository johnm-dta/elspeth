"""Per-exporter circuit breaker for failure isolation.

Implements the Circuit Breaker pattern to prevent calling exporters that are
persistently failing. This saves resources (network calls, CPU) and allows
failed exporters to recover without causing cascading issues.

States:
- CLOSED: Normal operation. Calls proceed, failures are counted.
- OPEN: Exporter is failing. Calls are skipped until reset_timeout expires.
- HALF_OPEN: Testing recovery. One call is allowed through:
  - Success -> CLOSED (recovered)
  - Failure -> OPEN (still failing)

Thread Safety:
    CircuitBreaker is designed for single-threaded access within
    TelemetryManager's export thread. State transitions and counters
    are not locked — the export thread is the only writer.
"""

from __future__ import annotations

import time
from enum import Enum, auto


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, skipping calls
    HALF_OPEN = auto()  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for telemetry exporter failure isolation.

    Tracks failures for a single exporter and trips open when the failure
    threshold is reached. After reset_timeout_seconds, allows one test call
    through (HALF_OPEN) to check if the exporter has recovered.

    Attributes:
        name: Exporter name (for logging/metrics)
        failure_threshold: Number of failures before tripping open (default: 5)
        reset_timeout_seconds: Seconds to wait before testing recovery (default: 60)

    Example:
        >>> breaker = CircuitBreaker("otlp", failure_threshold=3, reset_timeout_seconds=30)
        >>> if breaker.is_open():
        ...     return  # Skip this exporter
        >>> try:
        ...     exporter.export(event)
        ...     breaker.record_success()
        >>> except TransportError:
        ...     breaker.record_failure()
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize circuit breaker for an exporter.

        Args:
            name: Exporter name (for logging/metrics)
            failure_threshold: Number of consecutive failures before tripping
            reset_timeout_seconds: Seconds to wait before testing recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._success_count = 0  # Total successes (for metrics)
        self._total_failures = 0  # Total failures (for metrics)
        self._trip_count = 0  # Times breaker has tripped (for metrics)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Consecutive failures in current CLOSED period."""
        return self._failure_count

    @property
    def metrics(self) -> dict[str, int | str]:
        """Return circuit breaker metrics for health reporting.

        Returns:
            Dict with state, counts, and trip history
        """
        return {
            "state": self._state.name,
            "consecutive_failures": self._failure_count,
            "total_successes": self._success_count,
            "total_failures": self._total_failures,
            "trip_count": self._trip_count,
        }

    def is_open(self) -> bool:
        """Check if circuit is open (should skip exporter).

        If OPEN and timeout has elapsed, transitions to HALF_OPEN to
        allow a test call through.

        Returns:
            True if calls should be skipped (OPEN state)
            False if calls should proceed (CLOSED or HALF_OPEN state)
        """
        if self._state == CircuitState.CLOSED:
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow one call through to test recovery
            return False

        # State is OPEN — check if timeout has elapsed
        if self._last_failure_time is None:
            # Should not happen, but be defensive
            self._state = CircuitState.CLOSED
            return False

        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self.reset_timeout_seconds:
            # Transition to HALF_OPEN to test recovery
            self._state = CircuitState.HALF_OPEN
            return False

        # Still in OPEN timeout period
        return True

    def record_success(self) -> None:
        """Record a successful export.

        Resets failure count and closes circuit if it was HALF_OPEN.
        """
        self._success_count += 1
        self._failure_count = 0

        if self._state == CircuitState.HALF_OPEN:
            # Recovery confirmed — close the circuit
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed export.

        Increments failure count. If threshold reached while CLOSED,
        trips the circuit OPEN. If HALF_OPEN, reopens immediately.
        """
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Recovery test failed — reopen circuit
            self._state = CircuitState.OPEN
            self._trip_count += 1
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                # Threshold reached — trip the circuit
                self._state = CircuitState.OPEN
                self._trip_count += 1

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state.

        Use for administrative reset (e.g., after config change).
        Does not reset metrics counters.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None


__all__ = [
    "CircuitBreaker",
    "CircuitState",
]
