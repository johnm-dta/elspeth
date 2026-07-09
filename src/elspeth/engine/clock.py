"""Clock implementations for testable timeout logic.

The shared clock protocols live in ``elspeth.core.clock``. This engine module
keeps compatibility re-exports and provides the default production/test
implementations used by engine scheduling and timeout code.

Production code uses SystemClock (the default).
Tests inject MockClock to control time advancement.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime

from elspeth.core.clock import Clock, MonotonicClock, UtcClock

__all__ = ["DEFAULT_CLOCK", "Clock", "MockClock", "MonotonicClock", "SystemClock", "UtcClock"]


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

    def now_utc(self) -> datetime:
        """Return current UTC wall-clock time."""
        return datetime.now(UTC)


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

        Raises:
            ValueError: If start is NaN or Infinity.
        """
        if not math.isfinite(start):
            raise ValueError(f"MockClock start must be finite, got {start}")
        self._current = float(start)

    def monotonic(self) -> float:
        """Return current mock time."""
        return self._current

    def now_utc(self) -> datetime:
        """Return current mock time as a UTC datetime."""
        return datetime.fromtimestamp(self._current, UTC)

    def advance(self, seconds: float) -> None:
        """Advance mock time by specified seconds.

        Args:
            seconds: Amount to advance (must be non-negative and finite).

        Raises:
            ValueError: If seconds is negative, NaN, or Infinity.
        """
        if not math.isfinite(seconds):
            raise ValueError(f"Cannot advance time by non-finite amount: {seconds}")
        if seconds < 0:
            raise ValueError(f"Cannot advance time by negative amount: {seconds}")
        self._current += float(seconds)

    def set(self, value: float) -> None:
        """Set mock time to an absolute value.

        Args:
            value: New monotonic time value (must be finite and >= current time).

        Raises:
            ValueError: If value is NaN, Infinity, or less than current time
                (non-monotonic).
        """
        if not math.isfinite(value):
            raise ValueError(f"MockClock.set() requires a finite value, got {value}")
        if value < self._current:
            raise ValueError(f"MockClock.set() requires monotonic time: value={value} < current={self._current}")
        self._current = float(value)


# Default clock for production use
DEFAULT_CLOCK: Clock = SystemClock()
