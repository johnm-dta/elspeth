"""Run-coordination repository: the epoch-21 leader/follower substrate (ADR-030).

Owns the ``run_coordination`` seat row, the ``run_workers`` registry, and the
``run_coordination_events`` ledger (design
notes/option-c-multi-worker-coordination-design-2026-06-11.md §A.2/§B.4/§C/§G).
Two shared fence constructs live here and in the schema module — one
definition, one dedicated unit test each (design §G):

- :func:`verify_and_extend_leader_fence` (this module) — the leader epoch
  fence, emitted as the FIRST statement of every leader-fenced transaction;
- ``active_worker_fence_clause`` (schema module) — the membership EXISTS
  predicate slice 4 compiles into the claim/enqueue verbs.

Every coordination state transition writes its event row in the SAME
transaction as the state change (the scheduler_events discipline). The one
exception is ``fence_refusal``: the payload transaction that tripped the
fence rolls back, so the refusal event is written on a FRESH connection
immediately after rollback — best-effort attribution, never a durability
guarantee (§A.2).

All write transactions use the slice-1 write-intent discipline
(:func:`~elspeth.core.landscape.database.begin_write` — ``BEGIN IMMEDIATE``,
WAL write lock at BEGIN, ``OperationalError("database is locked")`` after the
5000 ms ``busy_timeout`` poll).

Slice ownership of the §G verb surface (this module is slice 2):

==========================  =======================================================
verb                        consumer
==========================  =======================================================
register_run_leader         slice 2/3: ``begin_run`` (uniformity rule, epoch 1)
acquire_run_leadership      slice 2/3: ``resume()``'s first durable act (§B.4)
release_seat                slice 2/3: run/resume teardown + ceremony arms
live_leader                 implemented now; WIRED in slice 4 (entry-guard precision)
record_fence_refusal        slice 2: every fenced verb's refusal path
verify_and_extend_fence     slice 2: finalize/run-status/checkpoint/barrier/ingest
worker_heartbeat            slice 4: the dedicated heartbeat thread (§A.3)
record_heartbeat_degraded   slice 4: heartbeat thread after k busy failures (§A.3)
evict_worker                slice 4: leader housekeeping sweep (§C.2 path 1)
depart_worker               slice 5: follower clean exit (§B.1 step 5)
admit_follower              slice 5: ``elspeth join`` atomic admission (§B.1 step 2)
==========================  =======================================================
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError

from elspeth.contracts.coordination import (
    CoordinationSnapshot,
    CoordinationToken,
    LeaderInfo,
    RegisteredWorker,
)
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import (
    AuditIntegrityError,
    JoinRefusedError,
    RunLeadershipLostError,
    WriteLockHeldError,
)
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.database import Tier1Engine, begin_write
from elspeth.core.landscape.schema import (
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
)

__all__ = [
    "RunCoordinationRepository",
    "fenced_leader_transaction",
    "record_coordination_event",
    "verify_and_extend_leader_fence",
]

logger = logging.getLogger(__name__)

# Run statuses the takeover CAS flips back to 'running' (§B.4). The
# dead-leader RUNNING takeover arm is skipped by this predicate by
# construction; terminal-success statuses are refused by the
# immutable-success backstop below before the seat CAS runs.
_TAKEOVER_FLIPPABLE_RUN_STATUSES = (RunStatus.FAILED.value, RunStatus.INTERRUPTED.value)

# Immutable-success run statuses (§B.4 closing line: "the immutability guards
# retained beneath as the durable backstop"). Historically the resume path's
# first durable write was ``update_run_status(RUNNING)``, whose conditional
# UPDATE refused these durably; the takeover CAS subsumed that write, so the
# durable backstop moves INTO the arbiter transaction: a takeover of a
# terminally-successful run is refused with zero mutation BEFORE the seat CAS.
# Mirrors ``_IMMUTABLE_SUCCESS_RUN_STATUSES`` in run_lifecycle_repository (the
# update_run_status guard remains for every other caller of that verb).
_IMMUTABLE_SUCCESS_RUN_STATUSES = (
    RunStatus.COMPLETED.value,
    RunStatus.COMPLETED_WITH_FAILURES.value,
    RunStatus.EMPTY.value,
)


def _utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime read back from SQLite (storage is UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _is_database_locked(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def record_coordination_event(
    conn: Connection,
    *,
    run_id: str,
    event_type: str,
    worker_id: str,
    leader_epoch: int | None,
    recorded_at: datetime,
    context: Mapping[str, object] | None = None,
) -> None:
    """Append one ledger row in the caller's transaction (same-txn discipline).

    ``event_id`` is ``sha256(canonical_json(identity))`` — the scheduler-events
    dedup recipe — enforced by the ``uq_run_coordination_events_event_id``
    unique index. ``seq`` (AUTOINCREMENT) is the authoritative replay order;
    ``recorded_at`` is forensic wall-clock only.
    """
    context_json = canonical_json({} if context is None else dict(context))
    identity = canonical_json(
        {
            "context_json": context_json,
            "event_type": event_type,
            "leader_epoch": leader_epoch,
            "recorded_at": recorded_at.isoformat(),
            "run_id": run_id,
            "worker_id": worker_id,
        }
    )
    conn.execute(
        insert(run_coordination_events_table).values(
            event_id=hashlib.sha256(identity.encode()).hexdigest(),
            run_id=run_id,
            event_type=event_type,
            worker_id=worker_id,
            leader_epoch=leader_epoch,
            recorded_at=recorded_at,
            context_json=context_json,
        )
    )


def _record_best_effort_event(
    engine: Tier1Engine,
    *,
    run_id: str,
    event_type: str,
    worker_id: str,
    leader_epoch: int | None,
    recorded_at: datetime,
    context: Mapping[str, object] | None = None,
) -> None:
    """Write a ledger row on a FRESH connection; catch-all, log, never raise.

    For events whose triggering transaction rolled back (``fence_refusal``) or
    whose writer must never crash (``heartbeat_degraded``, §A.3). Best-effort
    attribution by design: a crash between the rollback and this write loses
    the event, which is benign because the refused transaction left no durable
    state needing explanation (§A.2).
    """
    try:
        with begin_write(engine) as conn:
            record_coordination_event(
                conn,
                run_id=run_id,
                event_type=event_type,
                worker_id=worker_id,
                leader_epoch=leader_epoch,
                recorded_at=recorded_at,
                context=context,
            )
    except Exception:
        logger.warning(
            "best-effort coordination event %r for run %r (worker %r) could not be recorded",
            event_type,
            run_id,
            worker_id,
            exc_info=True,
        )


def verify_and_extend_leader_fence(
    conn: Connection,
    *,
    token: CoordinationToken,
    now: datetime,
    window_seconds: float,
    verb: str,
) -> None:
    """The leader epoch fence (§C.4): verify-and-extend UPDATE CAS.

    MUST be the first statement of the caller's ``BEGIN IMMEDIATE``
    transaction: under IMMEDIATE the verify and the payload are one atomic
    unit, and rowcount-0 raising here unwinds the whole transaction before
    any payload write. Every fenced verb thereby doubles as the seat
    heartbeat (the predicate is identity+epoch only — NEVER expiry: an idle
    N=1 leader whose seat lapsed mid-run must still pass its own fence).

    On rowcount 0 raises :class:`RunLeadershipLostError`. The caller (or
    :func:`fenced_leader_transaction`) records the ``fence_refusal`` event on
    a fresh connection AFTER its rollback completes — writing it here on a
    second connection would deadlock against the caller's own write lock.
    """
    result = conn.execute(
        update(run_coordination_table)
        .where(
            run_coordination_table.c.run_id == token.run_id,
            run_coordination_table.c.leader_worker_id == token.worker_id,
            run_coordination_table.c.leader_epoch == token.leader_epoch,
        )
        .values(
            leader_heartbeat_expires_at=now + timedelta(seconds=window_seconds),
            updated_at=now,
        )
    )
    if result.rowcount != 1:
        raise RunLeadershipLostError(
            run_id=token.run_id,
            worker_id=token.worker_id,
            leader_epoch=token.leader_epoch,
            verb=verb,
        )


@contextmanager
def fenced_leader_transaction(
    engine: Tier1Engine,
    *,
    token: CoordinationToken,
    now: datetime,
    window_seconds: float,
    verb: str,
) -> Iterator[Connection]:
    """One leader-fenced ``BEGIN IMMEDIATE`` transaction, refusal-evented.

    Composes :func:`~elspeth.core.landscape.database.begin_write` with
    :func:`verify_and_extend_leader_fence` as the first statement, and — on a
    fence miss — records the ``fence_refusal`` event on a fresh connection
    AFTER the payload transaction has rolled back, then re-raises
    :class:`RunLeadershipLostError`. Fenced verbs in other repositories
    (finalize / run-status / checkpoint / complete_barrier / ingest / repair
    sweep) wrap their existing transaction bodies in this.
    """
    try:
        with begin_write(engine) as conn:
            verify_and_extend_leader_fence(conn, token=token, now=now, window_seconds=window_seconds, verb=verb)
            yield conn
    except RunLeadershipLostError:
        # The begin_write context has exited (rolled back) by the time we get
        # here, so the fresh-connection write cannot deadlock on our own lock.
        _record_best_effort_event(
            engine,
            run_id=token.run_id,
            event_type="fence_refusal",
            worker_id=token.worker_id,
            leader_epoch=token.leader_epoch,
            recorded_at=now,
            context={"verb": verb},
        )
        raise


class RunCoordinationRepository:
    """Persistence boundary for the run-coordination substrate (ADR-030)."""

    def __init__(self, engine: Tier1Engine) -> None:
        # Runtime PRAGMA probe — defence in depth against a caller that slips
        # past the type checker (e.g. a ``cast()`` in test code or a mypy
        # suppression).  Tier-1 doctrine: the coordination substrate arbitrates
        # writes to the audit DB; we must refuse to proceed if the engine's
        # SQLite guarantees are unmet.
        #
        # The probe mirrors :meth:`LandscapeDB._verify_sqlite_pragmas` and is
        # replicated verbatim from :class:`TokenSchedulerRepository.__init__`.
        # We check only ``foreign_keys`` and ``journal_mode`` here — they are
        # the invariants most likely to be missing on a bare
        # ``create_engine()`` call that bypassed
        # ``LandscapeDB._configure_sqlite``.
        with engine.connect() as conn:
            fk_result = conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one_or_none()
            jm_result = conn.exec_driver_sql("PRAGMA journal_mode").scalar_one_or_none()

        foreign_keys = "" if fk_result is None else str(fk_result).lower()
        journal_mode = "" if jm_result is None else str(jm_result).lower()

        violations: list[str] = []
        if foreign_keys != "1":
            violations.append(f"PRAGMA foreign_keys: expected '1', observed {foreign_keys!r}")
        if journal_mode not in ("wal", "memory"):
            violations.append(f"PRAGMA journal_mode: expected 'wal' (or 'memory' for :memory: DBs), observed {journal_mode!r}")

        if violations:
            raise AuditIntegrityError(
                "RunCoordinationRepository received an engine that does not meet Tier-1 audit-integrity "
                "requirements; the engine was not opened through LandscapeDB. " + "; ".join(violations)
            )

        self._engine = engine

    # ── seat lifecycle ───────────────────────────────────────────────────

    def register_run_leader(
        self,
        *,
        run_id: str,
        worker_id: str,
        now: datetime,
        window_seconds: float,
        entry_point: str = "run",
    ) -> CoordinationToken:
        """Mint the run's seat at epoch 1 (uniformity rule: N=1 = leader-of-its-own-run).

        Standalone-transaction form for repository-level callers and test
        fixtures; ``begin_run`` composes :meth:`_register_run_leader_on` into
        ITS transaction instead so the runs INSERT and the seat mint commit
        atomically (design §B.4 closing line).
        """
        with begin_write(self._engine) as conn:
            return self._register_run_leader_on(
                conn,
                run_id=run_id,
                worker_id=worker_id,
                now=now,
                window_seconds=window_seconds,
                entry_point=entry_point,
            )

    def _register_run_leader_on(
        self,
        conn: Connection,
        *,
        run_id: str,
        worker_id: str,
        now: datetime,
        window_seconds: float,
        entry_point: str = "run",
    ) -> CoordinationToken:
        """Connection-accepting seat mint: composes into the caller's transaction.

        INSERTs the ``run_coordination`` seat row (epoch 1), the leader's
        ``run_workers`` row (with the §A.1 pid/hostname/entry_point
        forensics), and the ``leader_acquire`` + ``worker_register`` events —
        all on ``conn``. The ``runs`` row must already exist in this
        transaction (FK).
        """
        expires = now + timedelta(seconds=window_seconds)
        conn.execute(
            insert(run_coordination_table).values(
                run_id=run_id,
                leader_worker_id=worker_id,
                leader_epoch=1,
                leader_heartbeat_expires_at=expires,
                updated_at=now,
            )
        )
        self._insert_worker_row(
            conn,
            run_id=run_id,
            worker_id=worker_id,
            role="leader",
            now=now,
            heartbeat_expires_at=expires,
            entry_point=entry_point,
        )
        record_coordination_event(
            conn,
            run_id=run_id,
            event_type="worker_register",
            worker_id=worker_id,
            leader_epoch=1,
            recorded_at=now,
            context={"role": "leader", "entry_point": entry_point},
        )
        record_coordination_event(
            conn,
            run_id=run_id,
            event_type="leader_acquire",
            worker_id=worker_id,
            leader_epoch=1,
            recorded_at=now,
            context={"entry_point": entry_point},
        )
        return CoordinationToken(run_id=run_id, worker_id=worker_id, leader_epoch=1)

    def acquire_run_leadership(
        self,
        *,
        run_id: str,
        worker_id: str,
        now: datetime,
        window_seconds: float,
        entry_point: str = "resume",
    ) -> CoordinationToken:
        """The §B.4 seat-takeover CAS — resume()'s first durable act (TOCTOU closure).

        One ``BEGIN IMMEDIATE`` transaction:

        1. read the incumbent seat (``:prior``, may be vacant) and refuse a
           terminally-successful run with ``AuditIntegrityError`` (the
           immutable-success durable backstop — formerly update_run_status's
           conditional UPDATE, subsumed by this verb);
        2. seat CAS — bump ``leader_epoch``, claim the seat — admissible only
           when the seat is vacant or expired; rowcount 0 ⇒ ROLLBACK with zero
           mutation and ``NonResumableRunError("run leadership is held by
           …")`` (the pinned refusal-before-mutation discipline);
        3. the run-status flip ``failed/interrupted → running`` (subsumes the
           old ``update_run_status(RUNNING)`` first-durable-write at
           resume.py:591; skipped by predicate on the dead-leader RUNNING
           takeover arm);
        4. identity-eviction of the deposed leader, unconditional — by
           identity, no heartbeat predicate (the expired seat IS the proof of
           lost custody); NO bulk follower eviction (§C.2 housekeeping is
           slice 4);
        5. new leader ``run_workers`` row + ``worker_register`` /
           ``leader_acquire`` (+ ``worker_evict``) events.

        BUSY-vs-CAS-loss discrimination (§B.4): a busy timeout at BEGIN (or
        anywhere inside) is NOT "leadership held" — it means a live-or-frozen
        process holds the WAL write lock; raised as the operator-actionable
        :class:`WriteLockHeldError` naming the registered workers (pids read
        on a plain read connection — WAL readers don't block on the writer).
        """
        try:
            with begin_write(self._engine) as conn:
                return self._acquire_run_leadership_on(
                    conn,
                    run_id=run_id,
                    worker_id=worker_id,
                    now=now,
                    window_seconds=window_seconds,
                    entry_point=entry_point,
                )
        except OperationalError as exc:
            if not _is_database_locked(exc):
                raise
            raise WriteLockHeldError(run_id=run_id, workers=self._read_registered_workers(run_id)) from exc

    def _acquire_run_leadership_on(
        self,
        conn: Connection,
        *,
        run_id: str,
        worker_id: str,
        now: datetime,
        window_seconds: float,
        entry_point: str,
    ) -> CoordinationToken:
        seat = conn.execute(
            select(
                run_coordination_table.c.leader_worker_id,
                run_coordination_table.c.leader_epoch,
                run_coordination_table.c.leader_heartbeat_expires_at,
            ).where(run_coordination_table.c.run_id == run_id)
        ).one_or_none()
        if seat is None:
            # Epoch-21 invariant: begin_run mints the seat in the same
            # transaction as the runs row, so a run without a seat row is
            # audit corruption, not a coordination outcome.
            raise AuditIntegrityError(
                f"Run {run_id!r} has no run_coordination seat row; at schema epoch 21 "
                "begin_run creates it atomically with the run. The audit DB is corrupt "
                "or was written by incompatible code."
            )
        prior_worker: str | None = seat.leader_worker_id
        expires = now + timedelta(seconds=window_seconds)

        # Immutable-success durable backstop (§B.4 closing line). The takeover
        # CAS subsumed the resume path's update_run_status(RUNNING) first
        # durable write, whose conditional UPDATE was the pinned
        # loser-after-winner refusal (test_concurrent_resume.py): a resume
        # racing a winner that already COMPLETED must be refused durably, in
        # the arbiter transaction, with zero mutation — a vacant seat on a
        # terminally-successful run is NOT an admissible takeover target.
        run_status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).scalar_one_or_none()
        if run_status in _IMMUTABLE_SUCCESS_RUN_STATUSES:
            status_enum = RunStatus(run_status)
            raise AuditIntegrityError(
                f"Cannot acquire run leadership: cannot transition run {run_id} from "
                f"{status_enum.name} ({status_enum.value!r}) to 'running'. "
                f"Successful terminal runs are immutable. "
                f"FAILED/INTERRUPTED runs can be resumed via seat takeover."
            )

        cas = conn.execute(
            update(run_coordination_table)
            .where(
                run_coordination_table.c.run_id == run_id,
                (run_coordination_table.c.leader_worker_id.is_(None)) | (run_coordination_table.c.leader_heartbeat_expires_at < now),
            )
            .values(
                leader_worker_id=worker_id,
                leader_epoch=run_coordination_table.c.leader_epoch + 1,
                leader_heartbeat_expires_at=expires,
                updated_at=now,
            )
        )
        if cas.rowcount != 1:
            # Clean CAS loss to a live seat: raising inside the transaction
            # rolls everything back — the loser is side-effect-free (the
            # pinned refusal-before-mutation discipline, crash walk T3).
            #
            # Local import: recovery.py imports the landscape factory at module
            # scope, and the factory imports this module — a module-level
            # import here would close that cycle.
            from elspeth.core.checkpoint.recovery import NonResumableRunError

            held_expiry = seat.leader_heartbeat_expires_at
            expiry_text = "unknown" if held_expiry is None else _utc(held_expiry).isoformat()
            raise NonResumableRunError(
                run_id,
                f"run leadership is held by {prior_worker!r} (seat expires {expiry_text})",
            )
        new_epoch = int(seat.leader_epoch) + 1

        # The winner's run-status flip rides the same transaction (§B.4).
        # Predicate-skipped on the dead-leader RUNNING takeover arm; terminal
        # SUCCESS statuses were refused above by the immutable-success
        # backstop before the seat CAS ran.
        conn.execute(
            update(runs_table)
            .where(runs_table.c.run_id == run_id, runs_table.c.status.in_(_TAKEOVER_FLIPPABLE_RUN_STATUSES))
            .values(status=RunStatus.RUNNING.value, completed_at=None)
        )

        if prior_worker is not None and prior_worker != worker_id:
            evicted = conn.execute(
                update(run_workers_table)
                .where(
                    run_workers_table.c.worker_id == prior_worker,
                    run_workers_table.c.status == "active",
                )
                .values(status="evicted", evicted_at=now, evicted_by_worker_id=worker_id)
            )
            if evicted.rowcount == 1:
                record_coordination_event(
                    conn,
                    run_id=run_id,
                    event_type="worker_evict",
                    worker_id=prior_worker,
                    leader_epoch=new_epoch,
                    recorded_at=now,
                    context={"evicted_by_worker_id": worker_id, "reason": "deposed_leader_takeover"},
                )

        self._insert_worker_row(
            conn,
            run_id=run_id,
            worker_id=worker_id,
            role="leader",
            now=now,
            heartbeat_expires_at=expires,
            entry_point=entry_point,
        )
        record_coordination_event(
            conn,
            run_id=run_id,
            event_type="worker_register",
            worker_id=worker_id,
            leader_epoch=new_epoch,
            recorded_at=now,
            context={"role": "leader", "entry_point": entry_point},
        )
        record_coordination_event(
            conn,
            run_id=run_id,
            event_type="leader_acquire",
            worker_id=worker_id,
            leader_epoch=new_epoch,
            recorded_at=now,
            context={"entry_point": entry_point, "deposed_leader_worker_id": prior_worker},
        )
        return CoordinationToken(run_id=run_id, worker_id=worker_id, leader_epoch=new_epoch)

    def release_seat(self, *, token: CoordinationToken, now: datetime) -> None:
        """Graceful leader shutdown: CAS the seat vacant + depart own row. Idempotent.

        Rowcount 0 on the seat CAS (already released, or deposed) is a
        silent no-op — release is best-effort hygiene on teardown/ceremony
        paths, never a fence.
        """
        with begin_write(self._engine) as conn:
            released = conn.execute(
                update(run_coordination_table)
                .where(
                    run_coordination_table.c.run_id == token.run_id,
                    run_coordination_table.c.leader_worker_id == token.worker_id,
                    run_coordination_table.c.leader_epoch == token.leader_epoch,
                )
                .values(leader_worker_id=None, leader_heartbeat_expires_at=None, updated_at=now)
            )
            departed = conn.execute(
                update(run_workers_table)
                .where(
                    run_workers_table.c.worker_id == token.worker_id,
                    run_workers_table.c.status == "active",
                )
                .values(status="departed", departed_at=now)
            )
            if released.rowcount == 1:
                record_coordination_event(
                    conn,
                    run_id=token.run_id,
                    event_type="leader_release",
                    worker_id=token.worker_id,
                    leader_epoch=token.leader_epoch,
                    recorded_at=now,
                    context={"worker_row_departed": bool(departed.rowcount)},
                )

    def live_leader(self, *, run_id: str, now: datetime) -> LeaderInfo | None:
        """Read-only seat read (§B.3). Implemented now; WIRED into the entry guard in slice 4.

        Returns None when the run has no seat row or the seat is vacant;
        otherwise the incumbent with ``seat_live`` evaluated at ``now``.
        Check-then-act at the caller is acceptable because the leadership CAS
        is the arbiter.
        """
        with self._engine.connect() as conn:
            seat = conn.execute(
                select(
                    run_coordination_table.c.leader_worker_id,
                    run_coordination_table.c.leader_epoch,
                    run_coordination_table.c.leader_heartbeat_expires_at,
                ).where(run_coordination_table.c.run_id == run_id)
            ).one_or_none()
        if seat is None or seat.leader_worker_id is None:
            return None
        expires = _utc(seat.leader_heartbeat_expires_at)
        return LeaderInfo(
            run_id=run_id,
            leader_worker_id=seat.leader_worker_id,
            leader_epoch=int(seat.leader_epoch),
            leader_heartbeat_expires_at=expires,
            seat_live=expires >= now,
        )

    # ── eventing ─────────────────────────────────────────────────────────

    def record_fence_refusal(
        self,
        *,
        run_id: str,
        worker_id: str,
        leader_epoch: int,
        verb: str,
        now: datetime,
    ) -> None:
        """Record a ``fence_refusal`` event on a FRESH connection; best-effort, never raises.

        Called AFTER the refused transaction's rollback completes (§A.2);
        :func:`fenced_leader_transaction` does this automatically.
        """
        _record_best_effort_event(
            self._engine,
            run_id=run_id,
            event_type="fence_refusal",
            worker_id=worker_id,
            leader_epoch=leader_epoch,
            recorded_at=now,
            context={"verb": verb},
        )

    def record_heartbeat_degraded(
        self,
        *,
        run_id: str,
        worker_id: str,
        failures: int,
        now: datetime,
    ) -> None:
        """Record a ``heartbeat_degraded`` event; fresh connection, best-effort (§A.3).

        Consumed by the slice-4 heartbeat thread after ``k`` consecutive busy
        failures, so a later eviction is diagnosable post-hoc as "could not
        reach the DB" rather than "process died". Never raises.
        """
        _record_best_effort_event(
            self._engine,
            run_id=run_id,
            event_type="heartbeat_degraded",
            worker_id=worker_id,
            leader_epoch=None,
            recorded_at=now,
            context={"consecutive_busy_failures": failures},
        )

    # ── registry membership (slice-4/5 consumers) ────────────────────────

    def worker_heartbeat(self, *, worker_id: str, now: datetime, window_seconds: float) -> CoordinationSnapshot:
        """Worker-row liveness CAS + seat snapshot, one transaction (§A.3).

        SLICE-4 CONSUMER: the dedicated heartbeat thread. Semantics pinned by
        the design: rowcount 0 ⇒ ``worker_active=False`` (this worker is no
        longer active — the thread latches its coordination-lost flag on
        THAT, never on a DB error). A leader beats BOTH rows in one
        transaction (identity CAS on the seat — no epoch parameter here; the
        fenced verbs are the epoch arbiters), so the two liveness clocks can
        never skew in the dangerous worker-fresher-than-seat direction.
        """
        with begin_write(self._engine) as conn:
            member = conn.execute(
                select(run_workers_table.c.run_id, run_workers_table.c.role).where(run_workers_table.c.worker_id == worker_id)
            ).one_or_none()
            if member is None:
                raise AuditIntegrityError(
                    f"worker_heartbeat for unregistered worker_id={worker_id!r}; "
                    "registration precedes the heartbeat thread by construction."
                )
            beat = conn.execute(
                update(run_workers_table)
                .where(
                    run_workers_table.c.worker_id == worker_id,
                    run_workers_table.c.status == "active",
                )
                .values(heartbeat_expires_at=now + timedelta(seconds=window_seconds))
            )
            worker_active = beat.rowcount == 1
            if worker_active and member.role == "leader":
                conn.execute(
                    update(run_coordination_table)
                    .where(
                        run_coordination_table.c.run_id == member.run_id,
                        run_coordination_table.c.leader_worker_id == worker_id,
                    )
                    .values(leader_heartbeat_expires_at=now + timedelta(seconds=window_seconds), updated_at=now)
                )
            seat = conn.execute(
                select(
                    run_coordination_table.c.leader_worker_id,
                    run_coordination_table.c.leader_epoch,
                    run_coordination_table.c.leader_heartbeat_expires_at,
                ).where(run_coordination_table.c.run_id == member.run_id)
            ).one_or_none()
        if seat is None:
            seat_live = False
            leader_worker_id = None
            leader_epoch = 0
        else:
            leader_worker_id = seat.leader_worker_id
            leader_epoch = int(seat.leader_epoch)
            expires = seat.leader_heartbeat_expires_at
            seat_live = leader_worker_id is not None and expires is not None and _utc(expires) >= now
        return CoordinationSnapshot(
            leader_worker_id=leader_worker_id,
            leader_epoch=leader_epoch,
            seat_live=seat_live,
            worker_active=worker_active,
        )

    def admit_follower(
        self,
        *,
        run_id: str,
        worker_id: str,
        config_hash: str,
        now: datetime,
        window_seconds: float,
    ) -> None:
        """§B.1 step 2: atomic IMMEDIATE follower admission.

        SLICE-5 CONSUMER: ``elspeth join`` (which performs the filesystem
        preflight BEFORE calling this). Refuses with
        :class:`JoinRefusedError` when the run is not RUNNING, the joiner's
        config hash disagrees, or the leader seat is not live (a follower
        must never be the first process on an abandoned run).
        """
        with begin_write(self._engine) as conn:
            run = conn.execute(select(runs_table.c.status, runs_table.c.config_hash).where(runs_table.c.run_id == run_id)).one_or_none()
            if run is None:
                raise JoinRefusedError(run_id, "run not found")
            if run.status != RunStatus.RUNNING.value:
                raise JoinRefusedError(
                    run_id,
                    f"run status is {run.status!r} — "
                    + ("a terminal run cannot be joined" if run.status == RunStatus.COMPLETED.value else "use `elspeth resume`"),
                )
            if run.config_hash != config_hash:
                raise JoinRefusedError(
                    run_id,
                    f"resolved settings hash {config_hash!r} does not match the run's "
                    f"config_hash {run.config_hash!r}; a joiner must run the identical pipeline",
                )
            seat = conn.execute(
                select(
                    run_coordination_table.c.leader_worker_id,
                    run_coordination_table.c.leader_heartbeat_expires_at,
                ).where(run_coordination_table.c.run_id == run_id)
            ).one_or_none()
            seat_live = (
                seat is not None
                and seat.leader_worker_id is not None
                and seat.leader_heartbeat_expires_at is not None
                and _utc(seat.leader_heartbeat_expires_at) >= now
            )
            if not seat_live:
                raise JoinRefusedError(run_id, "no live leader — use `elspeth resume` to take the seat")
            self._insert_worker_row(
                conn,
                run_id=run_id,
                worker_id=worker_id,
                role="follower",
                now=now,
                heartbeat_expires_at=now + timedelta(seconds=window_seconds),
                entry_point="join",
            )
            record_coordination_event(
                conn,
                run_id=run_id,
                event_type="worker_register",
                worker_id=worker_id,
                leader_epoch=None,
                recorded_at=now,
                context={"role": "follower", "entry_point": "join"},
            )

    def depart_worker(self, *, worker_id: str, now: datetime) -> None:
        """CAS ``active → departed`` + ``worker_depart`` event. Idempotent.

        SLICE-5 CONSUMER: follower clean exit (§B.1 step 5). No-op when the
        row already left ``active`` (e.g. finalize's leftover-member hygiene
        departed it first).
        """
        with begin_write(self._engine) as conn:
            member = conn.execute(select(run_workers_table.c.run_id).where(run_workers_table.c.worker_id == worker_id)).one_or_none()
            if member is None:
                return
            departed = conn.execute(
                update(run_workers_table)
                .where(
                    run_workers_table.c.worker_id == worker_id,
                    run_workers_table.c.status == "active",
                )
                .values(status="departed", departed_at=now)
            )
            if departed.rowcount == 1:
                record_coordination_event(
                    conn,
                    run_id=member.run_id,
                    event_type="worker_depart",
                    worker_id=worker_id,
                    leader_epoch=None,
                    recorded_at=now,
                    context={},
                )

    def evict_worker(
        self,
        *,
        token: CoordinationToken,
        target_worker_id: str,
        now: datetime,
        grace_seconds: float,
        window_seconds: float,
    ) -> bool:
        """§C.2 path 1: leader evicts a dead follower. Returns True iff evicted.

        SLICE-4 CONSUMER: the leader's housekeeping sweep. One leader-fenced
        IMMEDIATE transaction: (1) verify-and-extend epoch fence; (2) the
        belt-and-braces no-unexpired-leases precondition (registry eviction
        must never outrun a lease the item layer still considers possibly
        alive); (3) CAS eviction gated on ``heartbeat_expires_at < now -
        grace``. Rowcount 0 anywhere ⇒ benign skip (the worker heartbeated,
        or still holds live leases). A fence miss raises
        :class:`RunLeadershipLostError` via :func:`fenced_leader_transaction`
        (refusal evented on a fresh connection).
        """
        # Local import: scheduler schema names live in the same module tree;
        # token_work_items is only needed by this slice-4 surface.
        from elspeth.core.landscape.schema import token_work_items_table

        with fenced_leader_transaction(self._engine, token=token, now=now, window_seconds=window_seconds, verb="evict_worker") as conn:
            live_lease = conn.execute(
                select(token_work_items_table.c.work_item_id)
                .where(
                    token_work_items_table.c.run_id == token.run_id,
                    token_work_items_table.c.status == TokenWorkStatus.LEASED.value,
                    token_work_items_table.c.lease_owner == target_worker_id,
                    token_work_items_table.c.lease_expires_at >= now,
                )
                .limit(1)
            ).one_or_none()
            if live_lease is not None:
                return False
            evicted = conn.execute(
                update(run_workers_table)
                .where(
                    run_workers_table.c.worker_id == target_worker_id,
                    run_workers_table.c.run_id == token.run_id,
                    run_workers_table.c.status == "active",
                    run_workers_table.c.heartbeat_expires_at < now - timedelta(seconds=grace_seconds),
                )
                .values(status="evicted", evicted_at=now, evicted_by_worker_id=token.worker_id)
            )
            if evicted.rowcount != 1:
                return False
            record_coordination_event(
                conn,
                run_id=token.run_id,
                event_type="worker_evict",
                worker_id=target_worker_id,
                leader_epoch=token.leader_epoch,
                recorded_at=now,
                context={"evicted_by_worker_id": token.worker_id, "reason": "liveness_expired"},
            )
            return True

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _insert_worker_row(
        conn: Connection,
        *,
        run_id: str,
        worker_id: str,
        role: str,
        now: datetime,
        heartbeat_expires_at: datetime,
        entry_point: str,
    ) -> None:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=run_id,
                role=role,
                status="active",
                registered_at=now,
                heartbeat_expires_at=heartbeat_expires_at,
                pid=os.getpid(),
                hostname=socket.gethostname(),
                entry_point=entry_point,
            )
        )

    def dead_non_leader_workers(
        self,
        *,
        run_id: str,
        leader_worker_id: str,
        now: datetime,
        grace_seconds: float,
    ) -> tuple[str, ...]:
        """Return worker_ids of ACTIVE non-leader members whose heartbeat has expired.

        Read-only (plain connection, no write lock). Used by the leader
        housekeeping sweep (§C.2 path 1, slice 4) to enumerate candidates for
        individual ``evict_worker`` calls. Only ``status='active'`` rows are
        returned — departed/evicted rows are already done. The grace_seconds
        MUST equal the value passed to ``evict_worker`` so the liveness
        definition is consistent across the read (who to attempt to evict)
        and the write (the CAS guard inside evict_worker).

        Returns a tuple of worker_ids (deterministic order by registered_at).
        The caller then calls ``evict_worker`` for each, which is idempotent
        (benign skip if the worker heartbeated or holds a live lease).
        """
        grace_threshold = now - timedelta(seconds=grace_seconds)
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(run_workers_table.c.worker_id)
                .where(
                    run_workers_table.c.run_id == run_id,
                    run_workers_table.c.status == "active",
                    run_workers_table.c.worker_id != leader_worker_id,
                    run_workers_table.c.heartbeat_expires_at < grace_threshold,
                )
                .order_by(run_workers_table.c.registered_at)
            ).scalars()
        return tuple(rows)

    def _read_registered_workers(self, run_id: str) -> tuple[RegisteredWorker, ...]:
        """Forensic registry read for the BUSY-takeover diagnostic (§B.4).

        Plain read connection by necessity: the write lock is held by the
        very process we are diagnosing (WAL readers don't block on the
        writer; a write-intent connection here would re-deadlock). Best
        effort — an unreadable registry yields an empty roster, never masks
        the WriteLockHeldError.
        """
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(
                        run_workers_table.c.worker_id,
                        run_workers_table.c.role,
                        run_workers_table.c.status,
                        run_workers_table.c.pid,
                        run_workers_table.c.hostname,
                    )
                    .where(run_workers_table.c.run_id == run_id)
                    .order_by(run_workers_table.c.registered_at)
                ).all()
        except Exception:
            logger.warning("could not read run_workers roster for run %r while diagnosing a held write lock", run_id, exc_info=True)
            return ()
        return tuple(
            RegisteredWorker(
                worker_id=row.worker_id,
                role=row.role,
                status=row.status,
                pid=None if row.pid is None else int(row.pid),
                hostname=row.hostname,
            )
            for row in rows
        )
