# tests/unit/engine/orchestrator/test_leader_follower_drain.py
"""Characterization net for LeaderFollowerDrain (filigree elspeth-6630fb3e31).

Pins the bounded peer-lease wait and follower pending-sink drain loops extracted
from Orchestrator._execute_run. Drives the coordinator with a fake processor and
fake monotonic/sleep so the poll/latch/reap/timeout/sleep mechanics are asserted
in isolation. Must-preserve invariants covered:

  - per-iteration order shutdown -> latch -> (drain) -> reap -> recheck -> timeout -> sleep;
  - timeout BREAKS (never raises) in BOTH loops;
  - a productive drain iteration continues WITHOUT sleeping;
  - shutdown raises make_shutdown_error()'s live-counter result in both methods;
  - a coordination-latch RunWorkerEvictedError propagates;
  - the slog.warning event text + payloads are preserved.

The end-to-end behaviour through _execute_run stays covered by
test_adr030_follower_teardown and the e2e recovery multi-worker/follower suites.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
from elspeth.contracts.errors import RunWorkerEvictedError
from elspeth.engine.orchestrator.leader_follower_drain import LeaderFollowerDrain


class _FakeProcessor:
    """State-driven fake of the narrow processor surface the drain polls."""

    def __init__(self, *, peer_active: bool = False, scheduled: bool = False, wait_budget: float = 5.0) -> None:
        self.peer_active = peer_active
        self.scheduled = scheduled
        self.wait_budget = wait_budget
        self.owners = ["peer-a"]
        self.reap_count = 0
        self.on_reap: Callable[[_FakeProcessor], None] | None = None

    def peer_lease_wait_budget_seconds(self) -> float:
        return self.wait_budget

    def has_peer_active_leases(self) -> bool:
        return self.peer_active

    def has_scheduled_work(self) -> bool:
        return self.scheduled

    def reap_expired_peer_leases(self) -> int:
        self.reap_count += 1
        if self.on_reap is not None:
            self.on_reap(self)
        return self.reap_count

    def peer_active_lease_owners(self) -> list[str]:
        return list(self.owners)


class _FakeClock:
    """Callable returning a scripted monotonic sequence (holds the final value)."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


class _ShutdownSentinel(Exception):
    """Stand-in for the GracefulShutdownError that make_shutdown_error() builds."""


def _make(
    processor: _FakeProcessor,
    *,
    monotonic: _FakeClock,
    sleeps: list[float],
    shutdown_set: bool | None = None,
    latch: Callable[[], None] | None = None,
) -> LeaderFollowerDrain:
    shutdown_event = None if shutdown_set is None else SimpleNamespace(is_set=lambda: shutdown_set)
    return LeaderFollowerDrain(
        processor=processor,
        run_id="run-x",
        shutdown_event=shutdown_event,  # type: ignore[arg-type]
        check_coordination_latch=latch,
        make_shutdown_error=lambda: _ShutdownSentinel(),  # type: ignore[arg-type, return-value]
        monotonic=monotonic,
        sleep=lambda s: sleeps.append(s),
        poll_interval=0.5,
    )


# =============================================================================
# wait_for_peer_leases
# =============================================================================


def test_wait_returns_immediately_when_no_peer_leases() -> None:
    proc = _FakeProcessor(peer_active=False)
    sleeps: list[float] = []
    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps)

    drain.wait_for_peer_leases()

    assert sleeps == []
    assert proc.reap_count == 0


def test_wait_times_out_warns_and_returns_with_half_second_cadence() -> None:
    proc = _FakeProcessor(peer_active=True, wait_budget=5.0)  # never clears
    sleeps: list[float] = []
    # deadline base=0 -> deadline=5; iter1 check 1.0 (<5) -> sleep; iter2 check 999 (>=5) -> timeout.
    drain = _make(proc, monotonic=_FakeClock([0.0, 1.0, 999.0]), sleeps=sleeps)

    with capture_logs() as logs:
        drain.wait_for_peer_leases()

    assert sleeps == [0.5]
    events = [e["event"] for e in logs]
    assert any(str(e).startswith("Bounded peer-lease wait timed out") for e in events)


def test_wait_breaks_when_reap_clears_the_dead_peer() -> None:
    proc = _FakeProcessor(peer_active=True)

    def _clear(p: _FakeProcessor) -> None:
        p.peer_active = False

    proc.on_reap = _clear
    sleeps: list[float] = []
    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps)

    drain.wait_for_peer_leases()

    assert proc.reap_count == 1
    assert sleeps == []  # broke at the post-reap recheck, never reached sleep


def test_wait_raises_make_shutdown_error_on_shutdown() -> None:
    proc = _FakeProcessor(peer_active=True)
    sleeps: list[float] = []
    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps, shutdown_set=True)

    with pytest.raises(_ShutdownSentinel):
        drain.wait_for_peer_leases()


def test_wait_propagates_latch_eviction() -> None:
    proc = _FakeProcessor(peer_active=True)
    sleeps: list[float] = []

    def _evict() -> None:
        raise RunWorkerEvictedError(worker_id="w1", run_id="run-x")

    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps, latch=_evict)

    with pytest.raises(RunWorkerEvictedError):
        drain.wait_for_peer_leases()


# =============================================================================
# drain_pending_sink_work
# =============================================================================


def test_drain_redrains_late_pending_sink_until_quiescent() -> None:
    proc = _FakeProcessor(peer_active=False, scheduled=True)
    sleeps: list[float] = []
    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps)
    calls = {"n": 0}

    def _drain_and_flush() -> bool:
        calls["n"] += 1
        proc.scheduled = False  # this pass consumed the last PENDING_SINK row
        return True

    drain.drain_pending_sink_work(_drain_and_flush)

    assert calls["n"] == 1
    assert sleeps == []  # productive iteration continued without sleeping


def test_drain_productive_iteration_does_not_sleep_before_quiescing() -> None:
    proc = _FakeProcessor(peer_active=False, scheduled=True)
    sleeps: list[float] = []
    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps)
    calls = {"n": 0}

    def _drain_and_flush() -> bool:
        calls["n"] += 1
        if calls["n"] >= 2:
            proc.scheduled = False
        return True  # progress every pass

    drain.drain_pending_sink_work(_drain_and_flush)

    assert calls["n"] == 2
    assert sleeps == []  # every iteration was productive -> continue, never slept


def test_drain_times_out_warns_and_returns() -> None:
    proc = _FakeProcessor(peer_active=True, scheduled=False)  # peer never clears, no drainable work
    sleeps: list[float] = []
    horizon = 3.0 * DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
    # deadline base=0 -> deadline=240; iter1 check 1.0 (<240) -> sleep; iter2 check huge -> timeout.
    drain = _make(proc, monotonic=_FakeClock([0.0, 1.0, horizon + 1.0]), sleeps=sleeps)

    with capture_logs() as logs:
        drain.drain_pending_sink_work(lambda: False)

    assert sleeps == [0.5]
    events = [e["event"] for e in logs]
    assert any(str(e).startswith("Bounded follower-drain wait timed out") for e in events)


def test_drain_raises_make_shutdown_error_on_shutdown() -> None:
    proc = _FakeProcessor(scheduled=True)
    sleeps: list[float] = []
    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps, shutdown_set=True)

    with pytest.raises(_ShutdownSentinel):
        drain.drain_pending_sink_work(lambda: True)


def test_drain_propagates_latch_eviction() -> None:
    proc = _FakeProcessor(scheduled=True)
    sleeps: list[float] = []

    def _evict() -> None:
        raise RunWorkerEvictedError(worker_id="w1", run_id="run-x")

    drain = _make(proc, monotonic=_FakeClock([0.0]), sleeps=sleeps, latch=_evict)

    with pytest.raises(RunWorkerEvictedError):
        drain.drain_pending_sink_work(lambda: True)
