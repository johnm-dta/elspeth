"""Neutral clock protocols shared across core and engine layers."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class MonotonicClock(Protocol):
    """Clock contract for elapsed-time calculations and timeouts."""

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...


class UtcClock(Protocol):
    """Clock contract for persisted UTC wall-clock timestamps."""

    def now_utc(self) -> datetime:
        """Return the current UTC wall-clock time."""
        ...


class Clock(MonotonicClock, UtcClock, Protocol):
    """Combined clock contract for components that need both time domains."""
