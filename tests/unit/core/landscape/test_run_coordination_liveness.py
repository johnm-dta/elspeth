"""Slice-4 liveness tests: heartbeat thread, snapshot latch, eviction (ADR-030 §A).

Focused on the INTERACTION between the repository verbs and the heartbeat
thread; the exhaustive thread-only unit coverage lives in
``tests/unit/engine/orchestrator/test_run_heartbeat_thread.py`` and the
repository-only coverage lives in
``tests/unit/core/landscape/test_run_coordination_repository.py``.

Tests:

2a. ``test_leader_heartbeat_beats_seat_and_worker_row_in_one_transaction`` —
    a leader beat advances BOTH rows (verify against the actual DB rows) and
    returns a snapshot with ``seat_live=True``, ``worker_active=True``,
    correct epoch.  Companion: follower beat advances only its own row.

2b. ``test_heartbeat_busy_is_liveness_unknown_never_evicts`` — the heartbeat
    thread receives a busy OperationalError; the latch stays False; the
    thread is still alive; check_and_raise does NOT raise.

2c. ``test_heartbeat_degraded_event_after_k_consecutive_busy_failures`` —
    after k=3 consecutive OperationalError ticks the thread calls
    ``record_heartbeat_degraded`` with consecutive_busy_failures=3 written
    to a FRESH connection into the real DB; the event is best-effort (an
    unwritable degraded event never raises).

2d. ``test_leader_observes_foreign_seat_latches_fatal`` — a snapshot whose
    ``leader_worker_id`` differs from our worker_id (seat taken) sets the
    coordination-lost latch independently of the per-verb fences;
    ``check_and_raise`` raises ``RunWorkerEvictedError`` with correct
    worker_id + run_id.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert, select
from sqlalchemy.exc import OperationalError

from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationSnapshot,
    CoordinationToken,
    mint_worker_id,
)
from elspeth.contracts.errors import RunWorkerEvictedError
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.schema import (
    metadata,
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
)
from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "run-liveness-unit"
NOW = datetime(2026, 6, 13, 10, 0, 0, tzinfo=UTC)
WINDOW = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS  # 80 s


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def _make_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _seed_run(engine: Tier1Engine, *, run_id: str = RUN_ID, status: str = "running") -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                started_at=NOW,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status=status,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )


def _seat_row(engine: Tier1Engine, run_id: str = RUN_ID) -> dict[str, object]:
    with engine.connect() as conn:
        row = conn.execute(select(run_coordination_table).where(run_coordination_table.c.run_id == run_id)).mappings().one()
    return dict(row)


def _worker_row(engine: Tier1Engine, worker_id: str) -> dict[str, object]:
    with engine.connect() as conn:
        row = conn.execute(select(run_workers_table).where(run_workers_table.c.worker_id == worker_id)).mappings().one()
    return dict(row)


def _coordination_events(engine: Tier1Engine, event_type: str, run_id: str = RUN_ID) -> list[dict[str, object]]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == run_id)
                .where(run_coordination_events_table.c.event_type == event_type)
                .order_by(run_coordination_events_table.c.seq)
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def _make_thread(
    repo: RunCoordinationRepository,
    *,
    token: CoordinationToken,
    now: datetime = NOW,
) -> RunHeartbeatThread:
    """Stub-free helper: real repo, deterministic (no sleeps, no real wall-clock)."""
    return RunHeartbeatThread(
        repo,
        token=token,
        heartbeat_seconds=15.0,
        window_seconds=WINDOW,
        now_fn=lambda: now,
        wait_fn=lambda _: False,  # never actually waits
    )


# ---------------------------------------------------------------------------
# Stub repo for thread tests that need controlled side-effects
# ---------------------------------------------------------------------------


class _StubRepo:
    """Minimal typed stub — avoids unspecced MagicMock to keep mock baseline clean."""

    def __init__(self) -> None:
        self.side_effects: list[CoordinationSnapshot | Exception] = []
        self.worker_heartbeat_calls: list[dict[str, object]] = []
        self.degraded_calls: list[dict[str, object]] = []
        self.degraded_raise: Exception | None = None

    def worker_heartbeat(self, *, worker_id: str, now: datetime, window_seconds: float) -> CoordinationSnapshot:
        self.worker_heartbeat_calls.append({"worker_id": worker_id, "now": now, "window_seconds": window_seconds})
        if not self.side_effects:
            raise AssertionError("_StubRepo: no more side_effects configured")
        result = self.side_effects.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def record_heartbeat_degraded(self, *, run_id: str, worker_id: str, failures: int, now: datetime) -> None:
        self.degraded_calls.append({"run_id": run_id, "worker_id": worker_id, "failures": failures})
        if self.degraded_raise is not None:
            raise self.degraded_raise


# ---------------------------------------------------------------------------
# 2a — repository: leader beats BOTH rows; follower beats only its own
# ---------------------------------------------------------------------------


class TestLeaderHeartbeatBeatsBothRows:
    """§A.3 :126 — worker-fresher-than-seat impossibility pinned at the DB layer."""

    def test_leader_heartbeat_beats_seat_and_worker_row_in_one_transaction(self) -> None:
        """A leader beat advances both the run_coordination seat expiry AND the
        run_workers row expiry, and the returned snapshot carries seat_live=True,
        worker_active=True, correct epoch.

        Pins run_coordination_repository.py :739-747 (the leader-role BOTH-rows
        branch) and design §A.3:126.
        """
        engine = _make_engine()
        repo = RunCoordinationRepository(engine)
        _seed_run(engine)

        leader_id = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=leader_id, now=NOW, window_seconds=WINDOW)

        seat_before = _seat_row(engine)
        worker_before = _worker_row(engine, leader_id)

        # Simulate the heartbeat thread's next tick: 15 s later.
        beat_at = NOW + timedelta(seconds=15)
        snapshot = repo.worker_heartbeat(worker_id=leader_id, now=beat_at, window_seconds=WINDOW)

        expected_expiry = beat_at + timedelta(seconds=WINDOW)

        # Snapshot fields.
        assert snapshot.worker_active is True
        assert snapshot.seat_live is True
        assert snapshot.leader_worker_id == leader_id
        assert snapshot.leader_epoch == 1

        # Both rows advanced — neither can be fresher than the other.
        seat_after = _seat_row(engine)
        worker_after = _worker_row(engine, leader_id)

        # Strip timezone for comparison (stored as naive UTC).
        expected_naive = expected_expiry.replace(tzinfo=None)
        assert seat_after["leader_heartbeat_expires_at"] == expected_naive, "seat expiry must advance on a leader beat"
        assert worker_after["heartbeat_expires_at"] == expected_naive, "worker row expiry must advance on a leader beat"
        # Both advanced by the same amount — worker-fresher-than-seat is impossible.
        assert seat_after["leader_heartbeat_expires_at"] == worker_after["heartbeat_expires_at"], (
            "seat and worker expiries are identical: no skew possible"
        )
        # Sanity: both actually moved from before values.
        assert seat_after["leader_heartbeat_expires_at"] != seat_before["leader_heartbeat_expires_at"]
        assert worker_after["heartbeat_expires_at"] != worker_before["heartbeat_expires_at"]

    def test_follower_heartbeat_beats_only_its_own_row(self) -> None:
        """A follower beat advances run_workers but NOT the seat row.

        Design §A.3: only the leader beats the seat.  A follower's beat
        is purely membership-maintenance.
        """
        engine = _make_engine()
        repo = RunCoordinationRepository(engine)
        _seed_run(engine, status="running")

        leader_id = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=leader_id, now=NOW, window_seconds=WINDOW)

        follower_id = mint_worker_id(RUN_ID)
        # Seed follower via raw INSERT (no admit_follower — that needs a live seat).
        with engine.begin() as conn:
            conn.execute(
                insert(run_workers_table).values(
                    worker_id=follower_id,
                    run_id=RUN_ID,
                    role="follower",
                    status="active",
                    registered_at=NOW,
                    heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
                )
            )

        seat_before = _seat_row(engine)
        beat_at = NOW + timedelta(seconds=15)
        snapshot = repo.worker_heartbeat(worker_id=follower_id, now=beat_at, window_seconds=WINDOW)

        # Snapshot reports the seat as viewed — leader_worker_id matches but
        # it's not this worker.
        assert snapshot.worker_active is True

        # The seat row is UNCHANGED by a follower beat.
        seat_after = _seat_row(engine)
        assert seat_after["leader_heartbeat_expires_at"] == seat_before["leader_heartbeat_expires_at"], (
            "follower beat must not advance the seat expiry"
        )

        # The follower's own row advanced.
        follower_row = _worker_row(engine, follower_id)
        expected_naive = (beat_at + timedelta(seconds=WINDOW)).replace(tzinfo=None)
        assert follower_row["heartbeat_expires_at"] == expected_naive


# ---------------------------------------------------------------------------
# 2b — thread: BUSY = liveness-unknown, never sets latch
# ---------------------------------------------------------------------------


class TestBusyToleranceNeverEvicts:
    """§A.3 :128-129 — OperationalError from worker_heartbeat is liveness-UNKNOWN."""

    def test_heartbeat_busy_is_liveness_unknown_never_evicts(self) -> None:
        """The thread catches OperationalError (SQLITE_BUSY) at the tick
        boundary and does NOT set the coordination-lost latch.

        Contrast arm: a rowcount-0 beat (worker_active=False) DOES set the latch.
        """
        token = CoordinationToken(run_id=RUN_ID, worker_id="worker-busy", leader_epoch=1)
        repo = _StubRepo()

        # k-1 busy ticks then a success.
        busy = OperationalError("statement", {}, Exception("database is locked"))
        healthy = CoordinationSnapshot(leader_worker_id="worker-busy", leader_epoch=1, seat_live=True, worker_active=True)
        repo.side_effects = [busy, busy, healthy]

        thread = RunHeartbeatThread(
            repo,
            token=token,
            heartbeat_seconds=15.0,
            window_seconds=WINDOW,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
            degraded_threshold=3,
        )

        # Drive two busy ticks — latch must NOT be set.
        thread._step_beat()
        assert not thread._coordination_lost_event.is_set(), "busy tick 1: latch must stay False"
        thread.check_and_raise()  # must not raise

        thread._step_beat()
        assert not thread._coordination_lost_event.is_set(), "busy tick 2: latch must stay False"
        thread.check_and_raise()  # must not raise

        # Third tick succeeds — latch still False.
        thread._step_beat()
        assert not thread._coordination_lost_event.is_set(), "success after busy: latch must stay False"
        thread.check_and_raise()  # must not raise

        # Contrast arm: a worker_active=False snapshot DOES set the latch.
        evicted_snapshot = CoordinationSnapshot(leader_worker_id=None, leader_epoch=1, seat_live=False, worker_active=False)
        repo2 = _StubRepo()
        repo2.side_effects = [evicted_snapshot]
        token2 = CoordinationToken(run_id=RUN_ID, worker_id="worker-evicted", leader_epoch=1)
        thread2 = RunHeartbeatThread(
            repo2,
            token=token2,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
        )
        thread2._step_beat()
        assert thread2._coordination_lost_event.is_set(), "worker_active=False must set latch"
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            thread2.check_and_raise()
        assert exc_info.value.worker_id == "worker-evicted"


# ---------------------------------------------------------------------------
# 2c — thread + repo: heartbeat_degraded event after k busy failures
# ---------------------------------------------------------------------------


class TestHeartbeatDegradedEvent:
    """§A.3 :130 + record_heartbeat_degraded (run_coordination_repository.py :684-706)."""

    def test_heartbeat_degraded_event_after_k_consecutive_busy_failures(self) -> None:
        """After k=3 consecutive busy failures the thread calls
        ``record_heartbeat_degraded`` with the correct failures count.

        Uses a real in-memory DB to verify the event is actually written to
        the ``run_coordination_events`` table on a FRESH connection.
        """
        engine = _make_engine()
        repo = RunCoordinationRepository(engine)
        _seed_run(engine)
        leader_id = mint_worker_id(RUN_ID)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=leader_id, now=NOW, window_seconds=WINDOW)

        # Replace the repo's worker_heartbeat with a stubbed busy side-effect
        # while letting record_heartbeat_degraded write to the real DB.
        class _BusyStubRepo:
            def worker_heartbeat(self, *, worker_id: str, now: datetime, window_seconds: float) -> CoordinationSnapshot:
                raise OperationalError("stmt", {}, Exception("database is locked"))

            def record_heartbeat_degraded(self, *, run_id: str, worker_id: str, failures: int, now: datetime) -> None:
                repo.record_heartbeat_degraded(run_id=run_id, worker_id=worker_id, failures=failures, now=now)

        k = 3
        thread = RunHeartbeatThread(
            _BusyStubRepo(),
            token=token,
            heartbeat_seconds=15.0,
            window_seconds=WINDOW,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
            degraded_threshold=k,
        )

        # Drive k-1 ticks: event must NOT be written yet.
        for _ in range(k - 1):
            thread._step_beat()
        events_before = _coordination_events(engine, "heartbeat_degraded")
        assert events_before == [], "heartbeat_degraded must not fire before k failures"
        assert not thread._coordination_lost_event.is_set(), "k-1 busy ticks: latch must stay False"

        # k-th tick: fires the degraded event exactly once.
        thread._step_beat()
        events_after = _coordination_events(engine, "heartbeat_degraded")
        assert len(events_after) == 1, "exactly one heartbeat_degraded event at k consecutive failures"
        evt = events_after[0]
        ctx = json.loads(str(evt["context_json"]))
        assert ctx["consecutive_busy_failures"] == k
        # leader_epoch is None for degraded events (best-effort, no fence).
        assert evt["leader_epoch"] is None
        assert not thread._coordination_lost_event.is_set(), "k busy ticks: latch must still be False"

    def test_degraded_event_error_does_not_propagate(self) -> None:
        """An unwritable degraded event (record_heartbeat_degraded raises) must
        NOT propagate out of the thread — degraded eventing is best-effort.
        """
        token = CoordinationToken(run_id=RUN_ID, worker_id="worker-busy-2", leader_epoch=1)
        repo = _StubRepo()
        busy = OperationalError("stmt", {}, Exception("database is locked"))
        repo.side_effects = [busy, busy, busy]
        repo.degraded_raise = RuntimeError("degraded event write failed")

        thread = RunHeartbeatThread(
            repo,
            token=token,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
            degraded_threshold=3,
        )
        # k ticks with degraded write failing: must not raise.
        for _ in range(3):
            thread._step_beat()  # no exception expected
        # Thread latch still not set (DB error, not eviction).
        assert not thread._coordination_lost_event.is_set()


# ---------------------------------------------------------------------------
# 2d — thread: foreign-leader snapshot sets FATAL latch
# ---------------------------------------------------------------------------


class TestForeignLeaderFatalLatch:
    """§A.3 :130 — foreign leader_worker_id is a SECOND independent latch.

    Design: even if A's own run_workers row is still 'active', observing a
    different leader_worker_id in the seat snapshot is fatal.  This latch is
    independent of the per-verb epoch fences.
    """

    def test_leader_observes_foreign_seat_latches_fatal(self) -> None:
        """worker_heartbeat returns a snapshot with leader_worker_id != our
        worker_id; the thread latches _coordination_lost_event; check_and_raise
        raises RunWorkerEvictedError naming our worker_id and run_id.

        The worker_active=True condition demonstrates the latch is based on the
        SEAT snapshot, not the row state — independent of the eviction fence.
        """
        our_id = "worker-A"
        token = CoordinationToken(run_id=RUN_ID, worker_id=our_id, leader_epoch=1)
        repo = _StubRepo()

        # Snapshot: our worker row is still active, BUT the seat shows worker-B.
        foreign_snapshot = CoordinationSnapshot(
            leader_worker_id="worker-B",
            leader_epoch=2,
            seat_live=True,
            worker_active=True,  # our row is active — yet the seat is foreign
        )
        repo.side_effects = [foreign_snapshot]

        thread = RunHeartbeatThread(
            repo,
            token=token,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
        )
        thread._step_beat()

        assert thread._coordination_lost_event.is_set(), (
            "foreign leader_worker_id must set coordination-lost latch even when worker_active=True"
        )
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            thread.check_and_raise()
        assert exc_info.value.worker_id == our_id
        assert exc_info.value.run_id == RUN_ID

    def test_foreign_seat_latch_uses_snapshot_not_row_state(self) -> None:
        """Even when the snapshot says the seat is foreign, the latch is set on
        the FIRST affected tick — subsequent ticks with the same snapshot do NOT
        re-emit (the flag is already set, re-setting an Event is idempotent).
        """
        our_id = "worker-A2"
        token = CoordinationToken(run_id=RUN_ID, worker_id=our_id, leader_epoch=1)
        repo = _StubRepo()

        foreign = CoordinationSnapshot(leader_worker_id="worker-C", leader_epoch=3, seat_live=True, worker_active=True)
        repo.side_effects = [foreign, foreign]

        thread = RunHeartbeatThread(
            repo,
            token=token,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
        )
        thread._step_beat()
        assert thread._coordination_lost_event.is_set()

        # Second tick with same foreign snapshot — still latched, no double-error.
        thread._step_beat()
        assert thread._coordination_lost_event.is_set()
        with pytest.raises(RunWorkerEvictedError):
            thread.check_and_raise()

    def test_none_leader_does_not_set_latch(self) -> None:
        """A snapshot with leader_worker_id=None (vacant seat) does NOT set the
        latch — the vacant seat means the run finalized gracefully, and the
        heartbeat thread will be stopped by core.py's finally block before this
        matters.  The latch is only for a FOREIGN takeover.
        """
        our_id = "worker-A3"
        token = CoordinationToken(run_id=RUN_ID, worker_id=our_id, leader_epoch=1)
        repo = _StubRepo()
        # Vacant seat (our row still active).
        vacant = CoordinationSnapshot(leader_worker_id=None, leader_epoch=1, seat_live=False, worker_active=True)
        repo.side_effects = [vacant]

        thread = RunHeartbeatThread(
            repo,
            token=token,
            now_fn=lambda: NOW,
            wait_fn=lambda _: False,
        )
        thread._step_beat()
        # Vacant seat is not foreign: latch must NOT be set.
        assert not thread._coordination_lost_event.is_set()
        thread.check_and_raise()  # must not raise
