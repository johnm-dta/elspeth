"""RunCoordinationRepository unit tests (epoch 21, ADR-030 slice 2).

Dedicated tests for the seat lifecycle and the shared leader epoch fence
(design §B.4/§C.4/§G — one definition, one dedicated unit test each):

- seat mint (``register_run_leader``: epoch 1, leader registered, evented);
- the takeover CAS — winner bumps the epoch and flips run status in ONE
  transaction; the loser is refused with ``NonResumableRunError`` and ZERO
  durable mutation (the pinned refusal-before-mutation discipline);
- identity-eviction: the deposed leader is evicted BY IDENTITY even when its
  worker-row heartbeat is FRESH; followers are never bulk-evicted;
- ``verify_and_extend_leader_fence`` hit (extends the seat as a side effect;
  identity+epoch only — NEVER expiry) and miss (``RunLeadershipLostError``);
- ``fence_refusal`` eventing on a FRESH connection even though the refused
  payload transaction rolled back (and best-effort: never raises);
- BUSY-vs-CAS-loss discrimination: a held write lock surfaces as the
  operator-actionable ``WriteLockHeldError`` naming registered pids — never
  as "leadership held".

Real file-backed Tier-1 SQLite (two engine handles where a second
lock-holding connection is needed — the races-test pattern). All clocks are
injected fixed datetimes.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.coordination import CoordinationToken, mint_worker_id
from elspeth.contracts.errors import (
    AuditIntegrityError,
    JoinRefusedError,
    RunLeadershipLostError,
    WriteLockHeldError,
)
from elspeth.core.checkpoint.recovery import NonResumableRunError
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import (
    RunCoordinationRepository,
    fenced_leader_transaction,
    verify_and_extend_leader_fence,
)
from elspeth.core.landscape.schema import (
    metadata,
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
)

RUN_ID = "run-coord-1"
NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)
WINDOW = 80.0
# Past the 80s liveness window: the seat registered at NOW is expired here.
AFTER_EXPIRY = NOW + timedelta(seconds=200)


@pytest.fixture
def engines(tmp_path: Path) -> Iterator[tuple[Tier1Engine, Tier1Engine]]:
    """Two independent Tier-1 engine handles onto ONE file-backed SQLite DB.

    The WriteLockHeldError test needs a second real connection holding
    ``BEGIN IMMEDIATE``; ``:memory:`` databases are per-engine, so the
    canonical in-memory helper cannot be shared. Both handles go through
    ``LandscapeDB._configure_sqlite`` + ``_verify_sqlite_pragmas`` so they
    satisfy the same Tier-1 PRAGMA invariants as production engines.
    """
    url = f"sqlite:///{tmp_path / 'landscape.db'}"
    raw_engines: list[Engine] = []
    for _ in range(2):
        raw = create_engine(url, echo=False)
        LandscapeDB._configure_sqlite(raw)
        LandscapeDB._verify_sqlite_pragmas(raw, url)
        raw_engines.append(raw)
    metadata.create_all(raw_engines[0])
    yield Tier1Engine(raw_engines[0]), Tier1Engine(raw_engines[1])
    for raw in raw_engines:
        raw.dispose()


@pytest.fixture
def engine(engines: tuple[Tier1Engine, Tier1Engine]) -> Tier1Engine:
    return engines[0]


@pytest.fixture
def repo(engine: Tier1Engine) -> RunCoordinationRepository:
    return RunCoordinationRepository(engine)


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


def _events(engine: Tier1Engine, run_id: str = RUN_ID) -> list[dict[str, object]]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == run_id)
                .order_by(run_coordination_events_table.c.seq)
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def _expire_seat(engine: Tier1Engine, *, run_id: str = RUN_ID) -> None:
    """Drive the seat clock into the past WITHOUT touching worker-row heartbeats."""
    with engine.begin() as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == run_id)
            .values(leader_heartbeat_expires_at=NOW - timedelta(seconds=1))
        )


class TestWorkerIdentity:
    def test_mint_worker_id_shape_and_uniqueness(self) -> None:
        wid = mint_worker_id(RUN_ID)
        assert wid.startswith(f"worker:{RUN_ID}:")
        assert len(wid.rsplit(":", 1)[1]) == 32  # uuid4().hex
        assert mint_worker_id(RUN_ID) != mint_worker_id(RUN_ID)  # single-use mint


class TestSeatMint:
    """register_run_leader: begin_run's uniformity-rule substrate (epoch 1)."""

    def test_register_mints_epoch_one_token(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        wid = mint_worker_id(RUN_ID)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=wid, now=NOW, window_seconds=WINDOW)
        assert token == CoordinationToken(run_id=RUN_ID, worker_id=wid, leader_epoch=1)

        seat = _seat_row(engine)
        assert seat["leader_worker_id"] == wid
        assert seat["leader_epoch"] == 1
        assert seat["leader_heartbeat_expires_at"] == (NOW + timedelta(seconds=WINDOW)).replace(tzinfo=None)

    def test_register_creates_leader_registry_row_with_forensics(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        wid = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=wid, now=NOW, window_seconds=WINDOW)

        worker = _worker_row(engine, wid)
        assert worker["role"] == "leader"
        assert worker["status"] == "active"
        assert worker["entry_point"] == "run"
        assert worker["pid"] == os.getpid()
        assert worker["hostname"] == socket.gethostname()
        assert worker["evicted_at"] is None

    def test_register_events_worker_register_then_leader_acquire(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        wid = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=wid, now=NOW, window_seconds=WINDOW)

        events = _events(engine)
        assert [e["event_type"] for e in events] == ["worker_register", "leader_acquire"]
        assert all(e["worker_id"] == wid for e in events)
        assert all(e["leader_epoch"] == 1 for e in events)


class TestAcquireRunLeadershipCAS:
    """§B.4: exactly one of two racers wins; the loser is side-effect-free."""

    def test_takeover_of_expired_seat_bumps_epoch_and_flips_run_status(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="failed")
        leader_a = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=leader_a, now=NOW, window_seconds=WINDOW)

        leader_b = mint_worker_id(RUN_ID)
        token = repo.acquire_run_leadership(run_id=RUN_ID, worker_id=leader_b, now=AFTER_EXPIRY, window_seconds=WINDOW)

        assert token == CoordinationToken(run_id=RUN_ID, worker_id=leader_b, leader_epoch=2)
        seat = _seat_row(engine)
        assert seat["leader_worker_id"] == leader_b
        assert seat["leader_epoch"] == 2
        # The winner's run-status flip rides the SAME transaction (the old
        # resume.py:591 first-durable-write, subsumed).
        with engine.connect() as conn:
            run = conn.execute(select(runs_table.c.status, runs_table.c.completed_at).where(runs_table.c.run_id == RUN_ID)).one()
        assert run.status == "running"
        assert run.completed_at is None

    def test_dead_leader_running_takeover_arm_skips_status_flip(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """RUNNING + expired seat: admissible takeover; the flip predicate skips."""
        _seed_run(engine, status="running")
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        token = repo.acquire_run_leadership(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=AFTER_EXPIRY, window_seconds=WINDOW)
        assert token.leader_epoch == 2
        with engine.connect() as conn:
            assert conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == RUN_ID)).scalar_one() == "running"

    def test_loser_against_live_seat_is_refused_with_zero_mutation(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """Two racers: B wins the expired seat; C's CAS rowcount-0 is side-effect-free."""
        _seed_run(engine, status="failed")
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        leader_b = mint_worker_id(RUN_ID)
        repo.acquire_run_leadership(run_id=RUN_ID, worker_id=leader_b, now=AFTER_EXPIRY, window_seconds=WINDOW)
        events_before = _events(engine)

        loser_c = mint_worker_id(RUN_ID)
        with pytest.raises(NonResumableRunError) as excinfo:
            repo.acquire_run_leadership(run_id=RUN_ID, worker_id=loser_c, now=AFTER_EXPIRY, window_seconds=WINDOW)

        assert "run leadership is held by" in str(excinfo.value)
        assert leader_b in str(excinfo.value)
        # Zero mutation: seat, registry, ledger, and runs row all untouched.
        seat = _seat_row(engine)
        assert seat["leader_worker_id"] == leader_b
        assert seat["leader_epoch"] == 2
        with engine.connect() as conn:
            assert conn.execute(select(run_workers_table.c.worker_id).where(run_workers_table.c.worker_id == loser_c)).one_or_none() is None
            assert conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == RUN_ID)).scalar_one() == "running"
        assert _events(engine) == events_before

    def test_acquire_on_run_without_seat_row_is_audit_corruption(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """Epoch-21 invariant: begin_run mints the seat; a missing row is Tier-1."""
        _seed_run(engine, status="failed")
        with pytest.raises(AuditIntegrityError, match="no run_coordination seat row"):
            repo.acquire_run_leadership(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

    @pytest.mark.parametrize("terminal_status", ["completed", "completed_with_failures", "empty"])
    def test_immutable_success_backstop_refuses_takeover_with_zero_mutation(
        self, engine: Tier1Engine, repo: RunCoordinationRepository, terminal_status: str
    ) -> None:
        """§B.4 durable backstop: a terminally-successful run is never a takeover target.

        The takeover CAS subsumed update_run_status(RUNNING) — whose
        conditional UPDATE was the pinned loser-after-winner refusal — so the
        immutability refusal now lives INSIDE the arbiter transaction: even a
        VACANT seat on a COMPLETED run is refused (AuditIntegrityError, same
        "Successful terminal runs are immutable" register), with zero
        mutation. This is what test #1's loser-after-COMPLETED arm pins e2e
        (tests/e2e/recovery/test_concurrent_resume.py).
        """
        _seed_run(engine, status=terminal_status)
        leader_a = mint_worker_id(RUN_ID)
        token_a = repo.register_run_leader(run_id=RUN_ID, worker_id=leader_a, now=NOW, window_seconds=WINDOW)
        # The completed winner released its seat on the way out — the seat is
        # VACANT, so without the backstop the CAS would happily seize it.
        repo.release_seat(token=token_a, now=NOW)
        events_before = _events(engine)

        loser = mint_worker_id(RUN_ID)
        with pytest.raises(AuditIntegrityError, match=r"from COMPLETED .*immutable|Successful terminal runs are immutable"):
            repo.acquire_run_leadership(run_id=RUN_ID, worker_id=loser, now=AFTER_EXPIRY, window_seconds=WINDOW)

        # Zero mutation: seat still vacant at epoch 1, run status untouched,
        # no loser registry row, no new ledger rows.
        seat = _seat_row(engine)
        assert seat["leader_worker_id"] is None
        assert seat["leader_epoch"] == 1
        with engine.connect() as conn:
            assert conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == RUN_ID)).scalar_one() == terminal_status
            assert conn.execute(select(run_workers_table.c.worker_id).where(run_workers_table.c.worker_id == loser)).one_or_none() is None
        assert _events(engine) == events_before


class TestIdentityEviction:
    """§B.4 correction 1: the incumbent is evicted by identity, never by liveness."""

    def test_deposed_leader_evicted_even_with_fresh_worker_heartbeat(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="failed")
        leader_a = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=leader_a, now=NOW, window_seconds=WINDOW)
        # The dangerous skew: seat clock expired, worker-row clock FRESH.
        _expire_seat(engine)
        with engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == leader_a)
                .values(heartbeat_expires_at=NOW + timedelta(seconds=3600))
            )

        leader_b = mint_worker_id(RUN_ID)
        repo.acquire_run_leadership(run_id=RUN_ID, worker_id=leader_b, now=NOW, window_seconds=WINDOW)

        deposed = _worker_row(engine, leader_a)
        assert deposed["status"] == "evicted"
        assert deposed["evicted_at"] == NOW.replace(tzinfo=None)
        assert deposed["evicted_by_worker_id"] == leader_b
        evict_events = [e for e in _events(engine) if e["event_type"] == "worker_evict"]
        assert len(evict_events) == 1
        assert evict_events[0]["worker_id"] == leader_a

    def test_takeover_never_bulk_evicts_followers(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """§B.4 correction 2: a stale-heartbeat follower survives the takeover."""
        _seed_run(engine, status="failed")
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        stale_follower = mint_worker_id(RUN_ID)
        with engine.begin() as conn:
            conn.execute(
                insert(run_workers_table).values(
                    worker_id=stale_follower,
                    run_id=RUN_ID,
                    role="follower",
                    status="active",
                    registered_at=NOW,
                    heartbeat_expires_at=NOW - timedelta(seconds=3600),  # long dead
                )
            )

        repo.acquire_run_leadership(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=AFTER_EXPIRY, window_seconds=WINDOW)

        follower = _worker_row(engine, stale_follower)
        assert follower["status"] == "active"  # housekeeping eviction is slice 4, under the full §C.2 predicate


class TestVerifyAndExtendLeaderFence:
    """§C.4: the verify-and-extend UPDATE CAS — identity+epoch only, never expiry."""

    def test_fence_hit_extends_seat_as_side_effect(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        later = NOW + timedelta(seconds=30)
        with begin_write(engine) as conn:
            verify_and_extend_leader_fence(conn, token=token, now=later, window_seconds=WINDOW, verb="test_verb")

        seat = _seat_row(engine)
        assert seat["leader_heartbeat_expires_at"] == (later + timedelta(seconds=WINDOW)).replace(tzinfo=None)
        assert seat["updated_at"] == later.replace(tzinfo=None)

    def test_fence_passes_for_lapsed_but_uncontested_seat(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """An idle N=1 leader whose seat lapsed mid-run still passes its own fence.

        The predicate is identity+epoch only — there is no heartbeat thread
        until slice 4, so expiry must NOT be part of the fence; every fenced
        verb doubles as the seat heartbeat instead.
        """
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        _expire_seat(engine)

        with begin_write(engine) as conn:
            verify_and_extend_leader_fence(conn, token=token, now=AFTER_EXPIRY, window_seconds=WINDOW, verb="test_verb")

        seat = _seat_row(engine)  # revived: the fenced verb re-heartbeated the seat
        assert seat["leader_heartbeat_expires_at"] == (AFTER_EXPIRY + timedelta(seconds=WINDOW)).replace(tzinfo=None)

    def test_fence_miss_on_stale_epoch_raises_leadership_lost(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="failed")
        stale = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        # The in-DB image of a takeover: bump the epoch directly (§H doctrine).
        with engine.begin() as conn:
            conn.execute(
                update(run_coordination_table).where(run_coordination_table.c.run_id == RUN_ID).values(leader_epoch=stale.leader_epoch + 1)
            )

        with begin_write(engine) as conn, pytest.raises(RunLeadershipLostError) as excinfo:
            verify_and_extend_leader_fence(conn, token=stale, now=NOW, window_seconds=WINDOW, verb="complete_run")
        err = excinfo.value
        assert err.run_id == RUN_ID
        assert err.worker_id == stale.worker_id
        assert err.leader_epoch == stale.leader_epoch
        assert err.verb == "complete_run"

    def test_fence_miss_on_foreign_worker_identity(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        foreign = CoordinationToken(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), leader_epoch=1)

        with begin_write(engine) as conn, pytest.raises(RunLeadershipLostError):
            verify_and_extend_leader_fence(conn, token=foreign, now=NOW, window_seconds=WINDOW, verb="save_checkpoint")


class TestFencedLeaderTransaction:
    """fenced_leader_transaction: rollback + fresh-connection fence_refusal event."""

    def test_fence_hit_commits_payload(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        probe = mint_worker_id(RUN_ID)
        with fenced_leader_transaction(engine, token=token, now=NOW, window_seconds=WINDOW, verb="test_verb") as conn:
            conn.execute(
                insert(run_workers_table).values(
                    worker_id=probe,
                    run_id=RUN_ID,
                    role="follower",
                    status="active",
                    registered_at=NOW,
                    heartbeat_expires_at=NOW,
                )
            )
        assert _worker_row(engine, probe)["status"] == "active"

    def test_fence_miss_rolls_back_and_events_refusal_on_fresh_connection(
        self, engine: Tier1Engine, repo: RunCoordinationRepository
    ) -> None:
        """The refused payload txn rolls back; the fence_refusal event survives it."""
        _seed_run(engine)
        stale = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        with engine.begin() as conn:
            conn.execute(update(run_coordination_table).where(run_coordination_table.c.run_id == RUN_ID).values(leader_epoch=99))
        probe = mint_worker_id(RUN_ID)

        with (
            pytest.raises(RunLeadershipLostError),
            fenced_leader_transaction(engine, token=stale, now=NOW, window_seconds=WINDOW, verb="ingest_source_row") as conn,
        ):
            conn.execute(  # pragma: no cover — the fence refuses before the body runs
                insert(run_workers_table).values(
                    worker_id=probe,
                    run_id=RUN_ID,
                    role="follower",
                    status="active",
                    registered_at=NOW,
                    heartbeat_expires_at=NOW,
                )
            )

        # Zero payload mutation...
        with engine.connect() as conn:
            assert conn.execute(select(run_workers_table.c.worker_id).where(run_workers_table.c.worker_id == probe)).one_or_none() is None
        # ...but the refusal IS attributed, durably, despite the rollback.
        refusals = [e for e in _events(engine) if e["event_type"] == "fence_refusal"]
        assert len(refusals) == 1
        assert refusals[0]["worker_id"] == stale.worker_id
        assert refusals[0]["leader_epoch"] == stale.leader_epoch
        assert "ingest_source_row" in str(refusals[0]["context_json"])
        # The seat itself is untouched by the refused fence.
        assert _seat_row(engine)["leader_epoch"] == 99

    def test_record_fence_refusal_is_best_effort_never_raises(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """A failing ledger insert (FK: run does not exist) is swallowed and logged."""
        repo.record_fence_refusal(run_id="run-that-does-not-exist", worker_id="worker:x:0", leader_epoch=1, verb="v", now=NOW)
        with engine.connect() as conn:
            count = conn.execute(select(run_coordination_events_table.c.seq)).all()
        assert count == []


class TestWriteLockHeld:
    """§B.4 BUSY discrimination: a held write lock is NOT 'leadership held'."""

    def test_acquire_raises_write_lock_held_with_registered_pids(
        self, engines: tuple[Tier1Engine, Tier1Engine], repo: RunCoordinationRepository
    ) -> None:
        engine, holder_engine = engines
        _seed_run(engine, status="failed")
        leader_a = mint_worker_id(RUN_ID)
        repo.register_run_leader(run_id=RUN_ID, worker_id=leader_a, now=NOW, window_seconds=WINDOW)

        # Shrink busy_timeout for the repo engine's FUTURE connections so the
        # BEGIN IMMEDIATE poll fails fast instead of taking the full 5000 ms.
        @event.listens_for(engine, "connect")
        def _shrink_busy_timeout(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA busy_timeout=200")
            cursor.close()

        engine.dispose()  # recycle pooled connections so the listener applies
        try:
            # A live-or-frozen peer holds the WAL write lock (BEGIN IMMEDIATE
            # taken at BEGIN by the slice-1 write-intent discipline).
            with begin_write(holder_engine), pytest.raises(WriteLockHeldError) as excinfo:
                repo.acquire_run_leadership(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=AFTER_EXPIRY, window_seconds=WINDOW)
        finally:
            event.remove(engine, "connect", _shrink_busy_timeout)

        err = excinfo.value
        assert err.run_id == RUN_ID
        roster = {w.worker_id: w for w in err.workers}
        assert leader_a in roster
        assert roster[leader_a].pid == os.getpid()
        assert roster[leader_a].role == "leader"
        assert str(os.getpid()) in str(err)
        # Distinct failure mode: NOT NonResumableRunError, and zero mutation.
        assert _seat_row(engine)["leader_epoch"] == 1


class TestReleaseSeatAndLiveLeader:
    def test_release_vacates_seat_departs_worker_and_events(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        repo.release_seat(token=token, now=NOW)

        seat = _seat_row(engine)
        assert seat["leader_worker_id"] is None
        assert seat["leader_heartbeat_expires_at"] is None
        worker = _worker_row(engine, token.worker_id)
        assert worker["status"] == "departed"
        assert worker["departed_at"] == NOW.replace(tzinfo=None)
        assert [e["event_type"] for e in _events(engine)][-1] == "leader_release"

    def test_release_is_idempotent(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        repo.release_seat(token=token, now=NOW)
        events_after_first = _events(engine)

        repo.release_seat(token=token, now=NOW)  # no error, no second event

        assert _events(engine) == events_after_first

    def test_vacant_seat_is_acquirable_at_next_epoch(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """The CAS's vacant-seat arm: release then re-acquire without waiting for expiry."""
        _seed_run(engine, status="failed")
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        repo.release_seat(token=token, now=NOW)

        new_token = repo.acquire_run_leadership(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        assert new_token.leader_epoch == 2
        # No incumbent to evict on the vacant-seat arm.
        assert [e for e in _events(engine) if e["event_type"] == "worker_evict"] == []

    def test_live_leader_views(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        assert repo.live_leader(run_id=RUN_ID, now=NOW) is None  # no seat row yet

        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        live = repo.live_leader(run_id=RUN_ID, now=NOW)
        assert live is not None
        assert live.leader_worker_id == token.worker_id
        assert live.leader_epoch == 1
        assert live.seat_live is True

        lapsed = repo.live_leader(run_id=RUN_ID, now=AFTER_EXPIRY)
        assert lapsed is not None
        assert lapsed.seat_live is False  # dead seat: the slice-4 admissible-takeover signal

        repo.release_seat(token=token, now=AFTER_EXPIRY)
        assert repo.live_leader(run_id=RUN_ID, now=AFTER_EXPIRY) is None  # vacant


class TestRegistryVerbs:
    """Minimal correctness for the slice-4/5-consumed registry surfaces."""

    def test_worker_heartbeat_leader_beats_both_rows(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        later = NOW + timedelta(seconds=30)
        snapshot = repo.worker_heartbeat(worker_id=token.worker_id, now=later, window_seconds=WINDOW)

        assert snapshot.worker_active is True
        assert snapshot.leader_worker_id == token.worker_id
        assert snapshot.leader_epoch == 1
        assert snapshot.seat_live is True
        assert _worker_row(engine, token.worker_id)["heartbeat_expires_at"] == (later + timedelta(seconds=WINDOW)).replace(tzinfo=None)
        assert _seat_row(engine)["leader_heartbeat_expires_at"] == (later + timedelta(seconds=WINDOW)).replace(tzinfo=None)

    def test_worker_heartbeat_cas_miss_after_eviction(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="failed")
        deposed = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        usurper = mint_worker_id(RUN_ID)
        repo.acquire_run_leadership(run_id=RUN_ID, worker_id=usurper, now=AFTER_EXPIRY, window_seconds=WINDOW)

        snapshot = repo.worker_heartbeat(worker_id=deposed.worker_id, now=AFTER_EXPIRY, window_seconds=WINDOW)

        assert snapshot.worker_active is False  # the coordination-lost latch signal
        assert snapshot.leader_worker_id == usurper  # foreign leader: fatal for a leader-mode process

    def test_admit_follower_happy_path_and_refusals(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="running")
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        follower = mint_worker_id(RUN_ID)
        repo.admit_follower(run_id=RUN_ID, worker_id=follower, config_hash="config", now=NOW, window_seconds=WINDOW)
        worker = _worker_row(engine, follower)
        assert worker["role"] == "follower"
        assert worker["status"] == "active"
        assert worker["entry_point"] == "join"

        with pytest.raises(JoinRefusedError, match="does not match"):
            repo.admit_follower(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), config_hash="other", now=NOW, window_seconds=WINDOW)
        with pytest.raises(JoinRefusedError, match="no live leader"):
            repo.admit_follower(
                run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), config_hash="config", now=AFTER_EXPIRY, window_seconds=WINDOW
            )
        _seed_run(engine, run_id="run-terminal", status="completed")
        with pytest.raises(JoinRefusedError, match="terminal"):
            repo.admit_follower(
                run_id="run-terminal", worker_id=mint_worker_id("run-terminal"), config_hash="config", now=NOW, window_seconds=WINDOW
            )

    def test_depart_worker_idempotent_cas(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="running")
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        follower = mint_worker_id(RUN_ID)
        repo.admit_follower(run_id=RUN_ID, worker_id=follower, config_hash="config", now=NOW, window_seconds=WINDOW)

        repo.depart_worker(worker_id=follower, now=NOW)
        assert _worker_row(engine, follower)["status"] == "departed"
        departs = [e for e in _events(engine) if e["event_type"] == "worker_depart"]
        assert len(departs) == 1

        repo.depart_worker(worker_id=follower, now=NOW)  # no-op
        repo.depart_worker(worker_id="worker:ghost:0", now=NOW)  # unregistered: no-op
        assert len([e for e in _events(engine) if e["event_type"] == "worker_depart"]) == 1

    def test_evict_worker_grace_predicate_and_fence(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine, status="running")
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        fresh = mint_worker_id(RUN_ID)
        stale = mint_worker_id(RUN_ID)
        with engine.begin() as conn:
            for wid, expires in ((fresh, NOW + timedelta(seconds=3600)), (stale, NOW - timedelta(seconds=3600))):
                conn.execute(
                    insert(run_workers_table).values(
                        worker_id=wid,
                        run_id=RUN_ID,
                        role="follower",
                        status="active",
                        registered_at=NOW,
                        heartbeat_expires_at=expires,
                    )
                )

        assert repo.evict_worker(token=token, target_worker_id=fresh, now=NOW, grace_seconds=10, window_seconds=WINDOW) is False
        assert _worker_row(engine, fresh)["status"] == "active"

        assert repo.evict_worker(token=token, target_worker_id=stale, now=NOW, grace_seconds=10, window_seconds=WINDOW) is True
        evicted = _worker_row(engine, stale)
        assert evicted["status"] == "evicted"
        assert evicted["evicted_by_worker_id"] == token.worker_id

        # A deposed leader's housekeeping is fence-refused (stale epoch).
        stale_token = CoordinationToken(run_id=RUN_ID, worker_id=token.worker_id, leader_epoch=token.leader_epoch - 1)
        with pytest.raises(RunLeadershipLostError):
            repo.evict_worker(token=stale_token, target_worker_id=fresh, now=NOW, grace_seconds=10, window_seconds=WINDOW)


class TestPragmaProbe:
    def test_constructor_refuses_unprobed_engine(self, tmp_path: Path) -> None:
        bare = create_engine(f"sqlite:///{tmp_path / 'bare.db'}")  # no PRAGMA configuration
        metadata.create_all(bare)
        with pytest.raises(AuditIntegrityError, match="not opened through LandscapeDB"):
            RunCoordinationRepository(Tier1Engine(bare))
        bare.dispose()


class TestEventLedgerDiscipline:
    def test_same_transaction_eventing_rolls_back_with_state_change(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        """Loser CAS leaves NO leader_acquire/worker_register ghost events."""
        _seed_run(engine, status="failed")
        repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        before = [e["event_id"] for e in _events(engine)]

        with pytest.raises(NonResumableRunError):
            repo.acquire_run_leadership(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)

        assert [e["event_id"] for e in _events(engine)] == before

    def test_event_id_dedup_is_enforced_by_unique_index(self, engine: Tier1Engine, repo: RunCoordinationRepository) -> None:
        _seed_run(engine)
        token = repo.register_run_leader(run_id=RUN_ID, worker_id=mint_worker_id(RUN_ID), now=NOW, window_seconds=WINDOW)
        existing = _events(engine)[0]
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                insert(run_coordination_events_table).values(
                    event_id=existing["event_id"],
                    run_id=RUN_ID,
                    event_type="worker_register",
                    worker_id=token.worker_id,
                    leader_epoch=1,
                    recorded_at=NOW,
                    context_json="{}",
                )
            )
