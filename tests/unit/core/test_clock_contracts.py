"""Tests for neutral clock contracts shared across layers."""

from __future__ import annotations


def test_core_clock_defines_narrow_protocol_contracts() -> None:
    from elspeth.core.clock import Clock, MonotonicClock, UtcClock

    assert MonotonicClock.__module__ == "elspeth.core.clock"
    assert UtcClock.__module__ == "elspeth.core.clock"
    assert Clock.__module__ == "elspeth.core.clock"
    assert "monotonic" in MonotonicClock.__dict__
    assert "now_utc" in UtcClock.__dict__


def test_engine_clock_reexports_core_contracts_for_compatibility() -> None:
    from elspeth.core.clock import Clock as CoreClock
    from elspeth.core.clock import MonotonicClock as CoreMonotonicClock
    from elspeth.core.clock import UtcClock as CoreUtcClock
    from elspeth.engine.clock import Clock as EngineClock
    from elspeth.engine.clock import MonotonicClock as EngineMonotonicClock
    from elspeth.engine.clock import UtcClock as EngineUtcClock

    assert EngineClock is CoreClock
    assert EngineMonotonicClock is CoreMonotonicClock
    assert EngineUtcClock is CoreUtcClock
