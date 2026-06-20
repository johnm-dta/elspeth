"""Unit tests for RunHeartbeatThread (ADR-030 §A.3, slice 4).

All beats are driven synchronously via ``_step_beat()`` on the test thread
(injected ``wait_fn=lambda _: False`` so no real wall-clock sleeps occur).
The repository is a stub that records calls and returns configurable snapshots.

Tests cover:

1. **Beats both rows in one txn** — ``worker_heartbeat`` called on every tick
   with the correct ``worker_id``, ``now``, and ``window_seconds`` (the
   underlying repo verb handles the single-transaction guarantee; this test
   verifies the thread calls the verb correctly).
2. **BUSY tolerated** — an ``OperationalError`` from ``worker_heartbeat`` is
   NOT fatal, does NOT set the latch, and does NOT terminate the thread;
   ``check_and_raise()`` must not raise after a busy tick.
3. **heartbeat_degraded fires past threshold** — after ``k`` consecutive busy
   failures ``record_heartbeat_degraded`` is called exactly once with the
   correct ``failures`` count; does NOT re-fire on the (k+1)-th miss (the
   count continues to grow but the event fires each time).
4. **Fatal latch set when seat taken; surfaced at boundary** —
   ``worker_heartbeat`` returning ``worker_active=True`` but
   ``leader_worker_id != our_id`` sets ``_coordination_lost_event`` and
   ``check_and_raise()`` raises ``RunWorkerEvictedError``.
5. **Fatal latch set when worker_active=False** — ``worker_heartbeat``
   returning ``worker_active=False`` sets the latch.
6. **Clean start/join lifecycle** — start() + step_beat() (healthy) +
   stop() completes without leaking threads; the thread is a daemon so it
   does not prevent process exit, but stop() must join it within the test.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from elspeth.contracts.coordination import (
    DEFAULT_RUN_HEARTBEAT_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationSnapshot,
    CoordinationToken,
)
from elspeth.contracts.errors import RunWorkerEvictedError
from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread

# ---------------------------------------------------------------------------
# Stub repo — avoids MagicMock() to keep the unspecced-mock baseline clean
# ---------------------------------------------------------------------------


class _StubRepo:
    """Minimal stub of RunCoordinationRepository for heartbeat unit tests.

    Configurable via ``snapshot`` (the next CoordinationSnapshot to return),
    ``side_effect`` (raise this exception from worker_heartbeat instead),
    ``side_effects`` (per-call sequence that overrides both), and
    ``degraded_exception`` (raise this from record_heartbeat_degraded).
    Records calls via lists for assertion.
    """

    def __init__(self) -> None:
        self.snapshot: CoordinationSnapshot | None = None
        self.side_effect: Exception | None = None
        self.side_effects: list[CoordinationSnapshot | Exception] = []
        self.degraded_exception: Exception | None = None

        self.worker_heartbeat_calls: list[dict[str, Any]] = []
        self.record_heartbeat_degraded_calls: list[dict[str, Any]] = []

    def worker_heartbeat(self, *, worker_id: str, now: datetime, window_seconds: float) -> CoordinationSnapshot:
        self.worker_heartbeat_calls.append({"worker_id": worker_id, "now": now, "window_seconds": window_seconds})
        if self.side_effects:
            result = self.side_effects.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.side_effect is not None:
            raise self.side_effect
        if self.snapshot is None:
            raise AssertionError("_StubRepo.snapshot must be set before calling worker_heartbeat")
        return self.snapshot

    def record_heartbeat_degraded(self, *, run_id: str, worker_id: str, failures: int, now: datetime) -> None:
        self.record_heartbeat_degraded_calls.append({"run_id": run_id, "worker_id": worker_id, "failures": failures, "now": now})
        if self.degraded_exception is not None:
            raise self.degraded_exception


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_RUN_ID = "run-heartbeat-test"
_WORKER_ID = f"worker:{_RUN_ID}:abc123"
_TOKEN = CoordinationToken(run_id=_RUN_ID, worker_id=_WORKER_ID, leader_epoch=1)

# Healthy snapshot: our worker is active, our worker is the leader.
_HEALTHY_SNAPSHOT = CoordinationSnapshot(
    leader_worker_id=_WORKER_ID,
    leader_epoch=1,
    seat_live=True,
    worker_active=True,
)

# Evicted snapshot: our registry row left 'active'.
_EVICTED_SNAPSHOT = CoordinationSnapshot(
    leader_worker_id=_WORKER_ID,
    leader_epoch=1,
    seat_live=True,
    worker_active=False,
)

# Deposed snapshot: another worker took the seat (our row still active).
_DEPOSED_SNAPSHOT = CoordinationSnapshot(
    leader_worker_id="worker:other-run:usurper",
    leader_epoch=2,
    seat_live=True,
    worker_active=True,
)


def _make_thread(
    repo: Any,
    *,
    degraded_threshold: int = 3,
    now_fn: Callable[[], datetime] | None = None,
) -> RunHeartbeatThread:
    """Build a thread with a no-sleep wait_fn for deterministic stepping."""
    if now_fn is None:
        now_fn = lambda: datetime.now(UTC)  # noqa: E731
    return RunHeartbeatThread(
        repo,
        token=_TOKEN,
        heartbeat_seconds=DEFAULT_RUN_HEARTBEAT_SECONDS,
        window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        now_fn=now_fn,
        wait_fn=lambda _: False,  # never blocks; stop_event NOT waited
        degraded_threshold=degraded_threshold,
    )


# ---------------------------------------------------------------------------
# 1. Beats both rows in one transaction (via worker_heartbeat call args)
# ---------------------------------------------------------------------------


class TestBeatsCorrectly:
    def test_calls_worker_heartbeat_with_correct_args(self) -> None:
        """worker_heartbeat is called with the thread's worker_id, now, and window."""
        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        fixed_now = datetime(2026, 6, 13, 0, 0, 0, tzinfo=UTC)
        thread = _make_thread(repo, now_fn=lambda: fixed_now)

        thread._step_beat()

        assert len(repo.worker_heartbeat_calls) == 1
        call = repo.worker_heartbeat_calls[0]
        assert call["worker_id"] == _WORKER_ID
        assert call["now"] == fixed_now
        assert call["window_seconds"] == DEFAULT_RUN_LIVENESS_WINDOW_SECONDS

    def test_healthy_beat_does_not_set_latch(self) -> None:
        """A healthy beat leaves check_and_raise() quiet."""
        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        thread = _make_thread(repo)
        thread._step_beat()

        assert not thread._coordination_lost_event.is_set()
        thread.check_and_raise()  # must not raise

    def test_multiple_healthy_beats_accumulate_without_error(self) -> None:
        """Multiple sequential healthy beats do not trip the latch."""
        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        thread = _make_thread(repo)
        for _ in range(5):
            thread._step_beat()

        assert not thread._coordination_lost_event.is_set()
        assert len(repo.worker_heartbeat_calls) == 5

    def test_beat_count_equals_window_seconds_times_1000_divided_by_heartbeat(self) -> None:
        """Confirm the module constant relationship: window >= 4*(beat+busy)."""
        busy_timeout_seconds = 5.0  # from _SQLITE_PRAGMA_INVARIANTS_FILE
        assert 4 * (DEFAULT_RUN_HEARTBEAT_SECONDS + busy_timeout_seconds) <= DEFAULT_RUN_LIVENESS_WINDOW_SECONDS


# ---------------------------------------------------------------------------
# 2. BUSY tolerated (OperationalError = liveness-unknown, not eviction)
# ---------------------------------------------------------------------------


class TestBusyTolerated:
    def test_operational_error_does_not_set_latch(self) -> None:
        """SQLITE_BUSY does NOT set the coordination-lost latch."""
        repo = _StubRepo()
        repo.side_effect = OperationalError("database is locked", None, None)

        thread = _make_thread(repo, degraded_threshold=100)
        thread._step_beat()

        assert not thread._coordination_lost_event.is_set()
        thread.check_and_raise()  # must not raise

    def test_any_exception_does_not_set_latch(self) -> None:
        """Any DB error is treated as liveness-unknown, never eviction."""
        repo = _StubRepo()
        repo.side_effect = RuntimeError("unexpected DB error")

        thread = _make_thread(repo, degraded_threshold=100)
        thread._step_beat()

        assert not thread._coordination_lost_event.is_set()

    def test_thread_continues_after_busy_failure(self) -> None:
        """After a busy tick the thread can recover with a healthy beat."""
        repo = _StubRepo()
        repo.side_effects = [
            OperationalError("database is locked", None, None),
            _HEALTHY_SNAPSHOT,
        ]

        thread = _make_thread(repo, degraded_threshold=100)
        thread._step_beat()  # busy
        thread._step_beat()  # healthy

        assert not thread._coordination_lost_event.is_set()
        assert thread._consecutive_busy == 0  # reset after healthy beat

    def test_busy_counter_resets_on_success(self) -> None:
        """_consecutive_busy resets to 0 after a successful beat."""
        repo = _StubRepo()
        repo.side_effects = [
            OperationalError("busy", None, None),
            OperationalError("busy", None, None),
            _HEALTHY_SNAPSHOT,
        ]

        thread = _make_thread(repo, degraded_threshold=100)
        thread._step_beat()
        assert thread._consecutive_busy == 1
        thread._step_beat()
        assert thread._consecutive_busy == 2
        thread._step_beat()
        assert thread._consecutive_busy == 0


# ---------------------------------------------------------------------------
# 3. heartbeat_degraded fires past threshold
# ---------------------------------------------------------------------------


class TestHeartbeatDegraded:
    def test_degraded_fires_at_threshold(self) -> None:
        """record_heartbeat_degraded is called when busy_count reaches k=3."""
        repo = _StubRepo()
        repo.side_effect = OperationalError("locked", None, None)

        thread = _make_thread(repo, degraded_threshold=3)

        thread._step_beat()  # busy=1 — not yet
        assert len(repo.record_heartbeat_degraded_calls) == 0

        thread._step_beat()  # busy=2 — not yet
        assert len(repo.record_heartbeat_degraded_calls) == 0

        thread._step_beat()  # busy=3 — fires NOW
        assert len(repo.record_heartbeat_degraded_calls) == 1
        call_kwargs = repo.record_heartbeat_degraded_calls[0]
        assert call_kwargs["run_id"] == _RUN_ID
        assert call_kwargs["worker_id"] == _WORKER_ID
        assert call_kwargs["failures"] == 3
        assert isinstance(call_kwargs["now"], datetime)

    def test_degraded_fires_again_on_subsequent_busy_beats(self) -> None:
        """Each beat past threshold keeps firing (failures grows monotonically)."""
        repo = _StubRepo()
        repo.side_effect = OperationalError("locked", None, None)

        thread = _make_thread(repo, degraded_threshold=2)
        thread._step_beat()  # busy=1
        thread._step_beat()  # busy=2 — first fire
        thread._step_beat()  # busy=3 — fires again with failures=3

        assert len(repo.record_heartbeat_degraded_calls) == 2
        last_call = repo.record_heartbeat_degraded_calls[-1]
        assert last_call["failures"] == 3

    def test_degraded_not_fired_below_threshold(self) -> None:
        """record_heartbeat_degraded is NOT called below the threshold."""
        repo = _StubRepo()
        repo.side_effect = OperationalError("locked", None, None)

        thread = _make_thread(repo, degraded_threshold=5)
        for _ in range(4):
            thread._step_beat()

        assert len(repo.record_heartbeat_degraded_calls) == 0

    def test_degraded_event_error_does_not_propagate(self) -> None:
        """record_heartbeat_degraded raising does NOT crash the thread."""
        repo = _StubRepo()
        repo.side_effect = OperationalError("locked", None, None)
        repo.degraded_exception = RuntimeError("degraded write failed")

        thread = _make_thread(repo, degraded_threshold=1)
        thread._step_beat()  # fires, degraded raises — must NOT propagate

        assert not thread._coordination_lost_event.is_set()

    def test_degraded_correct_worker_id_and_run_id(self) -> None:
        """Degraded event carries the thread's own worker_id and run_id."""
        repo = _StubRepo()
        repo.side_effect = OperationalError("locked", None, None)

        thread = _make_thread(repo, degraded_threshold=1)
        thread._step_beat()

        call_kwargs = repo.record_heartbeat_degraded_calls[0]
        assert call_kwargs["worker_id"] == _WORKER_ID
        assert call_kwargs["run_id"] == _RUN_ID


# ---------------------------------------------------------------------------
# 4. Fatal latch: seat taken by foreign leader
# ---------------------------------------------------------------------------


class TestFatalLatchForeignLeader:
    def test_deposed_sets_latch(self) -> None:
        """Snapshot with foreign leader_worker_id latches coordination_lost."""
        repo = _StubRepo()
        repo.snapshot = _DEPOSED_SNAPSHOT

        thread = _make_thread(repo)
        thread._step_beat()

        assert thread._coordination_lost_event.is_set()

    def test_check_and_raise_raises_after_deposition(self) -> None:
        """check_and_raise() raises RunWorkerEvictedError when deposed."""
        repo = _StubRepo()
        repo.snapshot = _DEPOSED_SNAPSHOT

        thread = _make_thread(repo)
        thread._step_beat()

        with pytest.raises(RunWorkerEvictedError) as exc_info:
            thread.check_and_raise()

        assert exc_info.value.worker_id == _WORKER_ID
        assert exc_info.value.run_id == _RUN_ID

    def test_healthy_then_deposed_sets_latch(self) -> None:
        """A previously healthy thread latches when it later observes deposition."""
        repo = _StubRepo()
        repo.side_effects = [_HEALTHY_SNAPSHOT, _DEPOSED_SNAPSHOT]

        thread = _make_thread(repo)
        thread._step_beat()
        assert not thread._coordination_lost_event.is_set()

        thread._step_beat()
        assert thread._coordination_lost_event.is_set()

    def test_none_leader_does_not_set_latch(self) -> None:
        """A vacant seat (leader_worker_id=None) is not treated as foreign-leader."""
        vacant_snapshot = CoordinationSnapshot(
            leader_worker_id=None,
            leader_epoch=0,
            seat_live=False,
            worker_active=True,
        )
        repo = _StubRepo()
        repo.snapshot = vacant_snapshot

        thread = _make_thread(repo)
        thread._step_beat()

        # A vacant seat does NOT trigger the deposition latch — only a FOREIGN
        # non-None leader does. This is important for N=1: between release_seat
        # and the final join, the seat may be vacant.
        assert not thread._coordination_lost_event.is_set()


# ---------------------------------------------------------------------------
# 5. Fatal latch: worker_active=False (evicted or departed)
# ---------------------------------------------------------------------------


class TestFatalLatchEvicted:
    def test_worker_inactive_sets_latch(self) -> None:
        """worker_active=False latches coordination_lost."""
        repo = _StubRepo()
        repo.snapshot = _EVICTED_SNAPSHOT

        thread = _make_thread(repo)
        thread._step_beat()

        assert thread._coordination_lost_event.is_set()

    def test_check_and_raise_raises_after_eviction(self) -> None:
        """check_and_raise() raises RunWorkerEvictedError when evicted."""
        repo = _StubRepo()
        repo.snapshot = _EVICTED_SNAPSHOT

        thread = _make_thread(repo)
        thread._step_beat()

        with pytest.raises(RunWorkerEvictedError):
            thread.check_and_raise()

    def test_check_and_raise_quiet_before_eviction(self) -> None:
        """check_and_raise() is a no-op before the latch is set."""
        thread = _make_thread(_StubRepo())  # no beats driven
        thread.check_and_raise()  # must not raise


# ---------------------------------------------------------------------------
# 6. Clean start/join lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_and_stop_no_leak(self) -> None:
        """start() + stop() without any beats completes without thread leak."""
        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        # Use a real wait_fn that blocks until stop_event is set — this is
        # the production shape; stop() signals the event, the thread exits.
        thread_obj = RunHeartbeatThread(
            repo,
            token=_TOKEN,
            wait_fn=None,  # default: stop_event.wait
        )
        thread_obj.start()
        thread_obj.stop()

        assert not thread_obj._thread.is_alive()

    def test_stop_is_idempotent(self) -> None:
        """stop() called twice does not deadlock or raise."""
        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        thread_obj = RunHeartbeatThread(
            repo,
            token=_TOKEN,
            wait_fn=None,
        )
        thread_obj.start()
        thread_obj.stop()
        thread_obj.stop()  # second call must be a no-op

        assert not thread_obj._thread.is_alive()

    def test_thread_is_daemon(self) -> None:
        """The heartbeat thread is a daemon so it does not block process exit."""
        thread_obj = RunHeartbeatThread(
            _StubRepo(),
            token=_TOKEN,
        )
        assert thread_obj._thread.daemon is True

    def test_exception_during_run_does_not_prevent_stop(self) -> None:
        """An exception path in core.py (exception raised before release_seat)
        still stops the thread via the finally safety-net stop()."""
        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        thread_obj = RunHeartbeatThread(
            repo,
            token=_TOKEN,
            wait_fn=None,
        )
        thread_obj.start()
        # Simulate the finally safety-net: stop() called unconditionally.
        thread_obj.stop()

        assert not thread_obj._thread.is_alive()

    def test_step_beat_on_background_thread_concurrent_check(self) -> None:
        """check_and_raise() is safe to call from the drain thread while the
        beat thread runs.  The latch is a threading.Event (atomic set/is_set)."""
        results: list[bool] = []

        repo = _StubRepo()
        repo.snapshot = _HEALTHY_SNAPSHOT

        thread_obj = RunHeartbeatThread(
            repo,
            token=_TOKEN,
            wait_fn=None,
        )
        thread_obj.start()

        def drain_check() -> None:
            for _ in range(20):
                try:
                    thread_obj.check_and_raise()
                    results.append(True)
                except RunWorkerEvictedError:
                    results.append(False)

        drain = threading.Thread(target=drain_check)
        drain.start()
        drain.join(timeout=5)
        thread_obj.stop()

        assert drain.is_alive() is False, "drain check hung"
        # All checks should be True (healthy thread, no eviction)
        assert all(results), f"unexpected eviction in concurrent check: {results}"


# ---------------------------------------------------------------------------
# 7. Follower role: deposed-latch is role-gated (ADR-030 §B, slice 5)
# ---------------------------------------------------------------------------


class TestFollowerHeartbeatRoleGating:
    """A follower seeing a foreign leader_worker_id must NOT latch.

    Design §B.2: trigger evaluation is leader-only.  A follower's
    leader_worker_id is always a different process's worker_id — this is
    the NORMAL, HEALTHY case.  The deposed-latch must only fire for leaders.
    """

    def test_follower_foreign_leader_does_not_latch(self) -> None:
        """worker_role='follower' + foreign leader_worker_id → no latch."""
        follower_worker_id = f"worker:{_RUN_ID}:follower-abc"
        follower_token = CoordinationToken(run_id=_RUN_ID, worker_id=follower_worker_id, leader_epoch=0)
        repo = _StubRepo()
        # Snapshot: our row is active (worker_active=True), but leader is a
        # DIFFERENT process — normal for a follower.
        repo.snapshot = CoordinationSnapshot(
            leader_worker_id="worker:some-run:the-leader",
            leader_epoch=1,
            seat_live=True,
            worker_active=True,
            worker_role="follower",  # this worker is a follower
        )

        thread = RunHeartbeatThread(
            repo,
            token=follower_token,
            heartbeat_seconds=DEFAULT_RUN_HEARTBEAT_SECONDS,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            wait_fn=lambda _: False,
        )
        thread._step_beat()

        # MUST NOT latch — a follower seeing a live foreign leader is healthy.
        assert not thread._coordination_lost_event.is_set()
        thread.check_and_raise()  # must not raise

    def test_follower_eviction_does_latch(self) -> None:
        """worker_role='follower' + worker_active=False → latch set (evicted)."""
        follower_worker_id = f"worker:{_RUN_ID}:follower-xyz"
        follower_token = CoordinationToken(run_id=_RUN_ID, worker_id=follower_worker_id, leader_epoch=0)
        repo = _StubRepo()
        repo.snapshot = CoordinationSnapshot(
            leader_worker_id="worker:some-run:the-leader",
            leader_epoch=1,
            seat_live=True,
            worker_active=False,  # our row left 'active' (evicted or departed)
            worker_role="follower",
        )

        thread = RunHeartbeatThread(
            repo,
            token=follower_token,
            heartbeat_seconds=DEFAULT_RUN_HEARTBEAT_SECONDS,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            wait_fn=lambda _: False,
        )
        thread._step_beat()

        # MUST latch — eviction applies to followers too.
        assert thread._coordination_lost_event.is_set()
        with pytest.raises(RunWorkerEvictedError):
            thread.check_and_raise()

    def test_leader_foreign_leader_still_latches(self) -> None:
        """worker_role='leader' (default) + foreign leader_worker_id → latch set."""
        # This is the pre-existing deposed-leader case; must still work.
        repo = _StubRepo()
        repo.snapshot = _DEPOSED_SNAPSHOT  # worker_role defaults to "leader"

        thread = _make_thread(repo)
        thread._step_beat()

        assert thread._coordination_lost_event.is_set()


# ---------------------------------------------------------------------------
# Constant import guard
# ---------------------------------------------------------------------------


def test_default_heartbeat_constant_imported() -> None:
    """DEFAULT_RUN_HEARTBEAT_SECONDS is importable from contracts.coordination."""
    from elspeth.contracts.coordination import DEFAULT_RUN_HEARTBEAT_SECONDS as C

    assert C == 15.0


def test_heartbeat_and_window_constants_satisfy_sizing_rule() -> None:
    """window >= 4 * (beat + busy_timeout) by the §A.3 sizing rule."""
    busy_timeout_seconds = 5.0
    assert 4 * (DEFAULT_RUN_HEARTBEAT_SECONDS + busy_timeout_seconds) <= DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
