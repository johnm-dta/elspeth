# src/elspeth/engine/clock.py
"""Clock abstraction for testable timeout logic.

This module provides a Clock protocol that abstracts time access,
enabling deterministic testing of timeout-dependent code paths
like aggregation triggers and coalesce timeouts.

Production code uses SystemClock (the default).
Tests inject MockClock to control time advancement.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Abstract clock for timeout-based operations.

    Enables testability and control over time in aggregations and coalesce.

    Implementations:
    - SystemClock: Uses time.monotonic() (production)
    - MockClock: Returns controllable times (testing)
    """

    def monotonic(self) -> float:
        """Return monotonic time in seconds.

        Must be monotonic (never goes backwards), suitable for elapsed
        time calculations and timeouts. Corresponds to time.monotonic().

        Returns:
            Current monotonic time in seconds.
        """
        ...


class SystemClock:
    """Production clock using time.monotonic().

    This is the default clock used when no clock is explicitly provided.
    It delegates to the system's monotonic clock, which is:
    - Monotonically increasing (immune to NTP/system time changes)
    - Suitable for elapsed time and timeout calculations
    """

    def monotonic(self) -> float:
        """Return system monotonic time."""
        return time.monotonic()


class MockClock:
    """Controllable clock for deterministic testing.

    Allows tests to advance time programmatically without sleep().

    Example:
        clock = MockClock(start=0.0)
        evaluator = TriggerEvaluator(config, clock=clock)

        evaluator.record_accept()  # Records at t=0
        clock.advance(0.5)  # Advance 500ms
        assert evaluator.batch_age_seconds == 0.5

        clock.advance(0.6)  # Total elapsed = 1.1s
        assert evaluator.should_trigger()  # Triggers if timeout=1.0s
    """

    def __init__(self, start: float = 0.0) -> None:
        """Initialize mock clock at a given time.

        Args:
            start: Initial monotonic time value (default 0.0).
        """
        self._current = start

    def monotonic(self) -> float:
        """Return current mock time."""
        return self._current

    def advance(self, seconds: float) -> None:
        """Advance mock time by specified seconds.

        Args:
            seconds: Amount to advance (must be non-negative).

        Raises:
            ValueError: If seconds is negative.
        """
        if seconds < 0:
            raise ValueError(f"Cannot advance time by negative amount: {seconds}")
        self._current += seconds

    def set(self, value: float) -> None:
        """Set mock time to an absolute value.

        Args:
            value: New monotonic time value.

        Note:
            Unlike advance(), this can set time to any value including
            earlier times. Use with caution - monotonic clocks shouldn't
            go backwards in production.
        """
        self._current = value


# Default clock for production use
DEFAULT_CLOCK: Clock = SystemClock()
