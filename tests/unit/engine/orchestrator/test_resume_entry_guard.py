"""Unit tests for the resume() entry guard (elspeth-2f23292372, option b).

``RecoveryManager.can_resume()`` is ADVISORY — callers may skip it — so
``ResumeCoordinator.resume()`` re-checks the run status at entry through the
SAME shared implementation (``check_run_status_resumable``) and refuses
non-resumable statuses with ``NonResumableRunError`` BEFORE any mutation.

What only a unit test can pin (the e2e proof in
tests/e2e/recovery/test_concurrent_resume.py asserts durable surfaces, which
cannot see in-memory coordinator state):

- the refusal fires BEFORE ``rebase_sequence`` — the first mutation on the
  resume path — via a recording stub CheckpointCoordinator;
- the shared check itself: one implementation, same reasons as can_resume
  (including the §B.3 live-seat enrichment, which lives in the shared
  ``check_run_status_resumable`` so advisory and enforcing surfaces never
  drift);
- the §H test-#1 flip (epoch 21, ADR-030): immutable-success statuses
  (COMPLETED / COMPLETED_WITH_FAILURES / EMPTY) are now REFUSED at the entry
  guard with a "Run is terminal" NonResumableRunError. The durable
  run-immutability guards stay beneath as the backstop (independently pinned
  in tests/unit/core/landscape/test_run_lifecycle_repository.py and
  test_run_coordination_repository.py).

The old TOCTOU residual (two resumes both observing FAILED at the guard) is
CLOSED at epoch 21: resume()'s first durable act is the seat-acquisition CAS
``acquire_run_leadership`` (ADR-030 §B.4) — the guard now only closes the
caller-convention gap, and check-then-act here is acceptable because the
leadership CAS is the arbiter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from elspeth.contracts import RunStatus
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.checkpoint.manager import CheckpointCorruptionError
from elspeth.core.checkpoint.recovery import NonResumableRunError, check_run_status_resumable
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import runs_table
from elspeth.engine.orchestrator.resume import ResumeCoordinator
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.stores import MockPayloadStore


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


def _insert_run(db: LandscapeDB, run_id: str, *, status: RunStatus | str) -> None:
    with db.connection() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=datetime.now(UTC),
                config_hash="cfg",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=status,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )


def _run_status_of(db: LandscapeDB, run_id: str) -> str:
    from sqlalchemy import select

    with db.engine.connect() as conn:
        return str(conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).scalar_one())


class _RecordingCheckpoints:
    """Stub CheckpointCoordinator recording the first resume-path mutation."""

    def __init__(self) -> None:
        self.rebase_calls: list[int] = []

    def rebase_sequence(self, sequence_number: int) -> None:
        self.rebase_calls.append(sequence_number)


def _coordinator(db: LandscapeDB) -> tuple[ResumeCoordinator, _RecordingCheckpoints]:
    checkpoints = _RecordingCheckpoints()
    coordinator = ResumeCoordinator(
        db=db,
        events=cast(Any, object()),
        ceremony=cast(Any, object()),
        checkpoints=cast(Any, checkpoints),
        run_core=cast(Any, object()),
        # None is the post-guard tripwire: an ADMITTED resume raises
        # OrchestrationInvariantError("CheckpointManager is required...")
        # inside reconstruct_resume_state, proving it got past the guard.
        checkpoint_manager=None,
    )
    return coordinator, checkpoints


def _resume_point_for(run_id: str, *, sequence_number: int = 7) -> Any:
    """Opaque resume-point stub: resume() entry reads only checkpoint.run_id
    and (post-guard) sequence_number."""
    return SimpleNamespace(checkpoint=SimpleNamespace(run_id=run_id), sequence_number=sequence_number)


class TestCheckRunStatusResumable:
    """The shared existence + status implementation (can_resume parity)."""

    def test_missing_run_refused(self, db: LandscapeDB) -> None:
        run_status, check = check_run_status_resumable(db, "missing")
        assert run_status is None
        assert check.can_resume is False
        assert check.reason == "Run missing not found"

    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (RunStatus.COMPLETED, "Run already completed successfully"),
            (RunStatus.RUNNING, "Run is still in progress"),
            (RunStatus.COMPLETED_WITH_FAILURES, "Run status 'completed_with_failures' is not resumable"),
            (RunStatus.EMPTY, "Run status 'empty' is not resumable"),
        ],
    )
    def test_non_resumable_statuses_refused_with_can_resume_reasons(self, db: LandscapeDB, status: RunStatus, reason: str) -> None:
        _insert_run(db, "run-x", status=status)
        run_status, check = check_run_status_resumable(db, "run-x")
        assert run_status is status
        assert check.can_resume is False
        assert check.reason == reason

    @pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.INTERRUPTED])
    def test_resumable_statuses_pass(self, db: LandscapeDB, status: RunStatus) -> None:
        _insert_run(db, "run-x", status=status)
        run_status, check = check_run_status_resumable(db, "run-x")
        assert run_status is status
        assert check.can_resume is True
        assert check.reason is None

    def test_invalid_persisted_status_is_corruption_not_refuse(self, db: LandscapeDB) -> None:
        _insert_run(db, "run-x", status="bogus")
        with pytest.raises(CheckpointCorruptionError, match="invalid status 'bogus'"):
            check_run_status_resumable(db, "run-x")

    def test_running_with_live_seat_reason_is_shared_with_the_advisory_surface(self, db: LandscapeDB) -> None:
        """§B.3 parity pin: the live-seat enrichment lives in the SHARED check.

        ``can_resume`` (advisory) and ``resume()`` (enforcing) both consume
        ``check_run_status_resumable``, so the join-flavoured reason must be
        produced HERE — the elspeth-2f23292372 "never drift" contract.
        """
        from elspeth.contracts.coordination import mint_worker_id
        from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository

        _insert_run(db, "run-seat-parity", status=RunStatus.RUNNING)
        leader_id = mint_worker_id("run-seat-parity")
        RunCoordinationRepository(db.engine).register_run_leader(
            run_id="run-seat-parity",
            worker_id=leader_id,
            now=datetime.now(UTC),
            window_seconds=80.0,
        )
        run_status, check = check_run_status_resumable(db, "run-seat-parity")
        assert run_status is RunStatus.RUNNING
        assert check.can_resume is False
        assert check.reason is not None
        assert leader_id in check.reason
        assert "seat expires" in check.reason
        assert "elspeth join" in check.reason


class TestResumeEntryGuard:
    """resume() refusal semantics, pinned ahead of the first mutation."""

    def test_refuses_running_run_before_rebase_sequence(self, db: LandscapeDB) -> None:
        _insert_run(db, "run-running", status=RunStatus.RUNNING)
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(NonResumableRunError, match=r"Cannot resume run 'run-running': Run is still in progress") as exc_info:
            coordinator.resume(
                _resume_point_for("run-running"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == "run-running"
        assert exc_info.value.reason == "Run is still in progress"
        assert checkpoints.rebase_calls == [], "entry guard must fire BEFORE rebase_sequence (the first mutation)"
        assert _run_status_of(db, "run-running") == RunStatus.RUNNING.value

    def test_running_with_live_seat_names_leader_and_points_at_join(self, db: LandscapeDB) -> None:
        """Epoch 21 (ADR-030 §B.3, live-seat half shipped in slice 2).

        RUNNING under a LIVE leader seat is refused naming the incumbent
        worker_id, the seat expiry, and the `elspeth join` direction (the
        join verb itself is slice 5). Polarity unchanged: still a refusal,
        still before any mutation.
        """
        from elspeth.contracts.coordination import mint_worker_id
        from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository

        _insert_run(db, "run-live-leader", status=RunStatus.RUNNING)
        leader_id = mint_worker_id("run-live-leader")
        RunCoordinationRepository(db.engine).register_run_leader(
            run_id="run-live-leader",
            worker_id=leader_id,
            now=datetime.now(UTC),
            window_seconds=80.0,
        )
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(NonResumableRunError, match=r"in progress under live leader") as exc_info:
            coordinator.resume(
                _resume_point_for("run-live-leader"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert leader_id in exc_info.value.reason
        assert "seat expires" in exc_info.value.reason
        assert "elspeth join" in exc_info.value.reason
        assert checkpoints.rebase_calls == []
        assert _run_status_of(db, "run-live-leader") == RunStatus.RUNNING.value

    def test_running_with_expired_seat_keeps_flat_refusal_in_slice_2(self, db: LandscapeDB) -> None:
        """RUNNING + expired seat keeps today's flat refusal until slice 4.

        The dead-leader takeover admission ("entry guard learns seat
        liveness", test #2 contract (c)) is deliberately a slice-4 flip;
        slice 2 only ships the live-seat precision arm above.
        """
        from datetime import timedelta

        from sqlalchemy import update as sa_update

        from elspeth.contracts.coordination import mint_worker_id
        from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
        from elspeth.core.landscape.schema import run_coordination_table

        _insert_run(db, "run-dead-leader", status=RunStatus.RUNNING)
        RunCoordinationRepository(db.engine).register_run_leader(
            run_id="run-dead-leader",
            worker_id=mint_worker_id("run-dead-leader"),
            now=datetime.now(UTC),
            window_seconds=80.0,
        )
        with db.connection() as conn:
            conn.execute(
                sa_update(run_coordination_table)
                .where(run_coordination_table.c.run_id == "run-dead-leader")
                .values(leader_heartbeat_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(NonResumableRunError) as exc_info:
            coordinator.resume(
                _resume_point_for("run-dead-leader"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.reason == "Run is still in progress"
        assert checkpoints.rebase_calls == []

    def test_refuses_missing_run_before_rebase_sequence(self, db: LandscapeDB) -> None:
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(NonResumableRunError, match=r"Run run-gone not found") as exc_info:
            coordinator.resume(
                _resume_point_for("run-gone"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == "run-gone"
        assert checkpoints.rebase_calls == []

    @pytest.mark.parametrize(
        "status",
        [RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY],
    )
    def test_immutable_success_statuses_refused_as_terminal_at_entry_guard(self, db: LandscapeDB, status: RunStatus) -> None:
        """The §H test-#1 FLIP: terminal runs are refused AT the entry guard.

        The designed loser-after-winner contract (ADR-030 §H item 1):
        a resume of an immutable-success run is a clean operator-facing
        ``NonResumableRunError`` ("Run is terminal …") raised BEFORE
        ``rebase_sequence``, not an ``AuditIntegrityError`` surfacing from
        the durable backstop mid-reconstruction. The backstops themselves
        (the immutable-success arm inside ``acquire_run_leadership`` and the
        ``update_run_status``/``complete_run`` conditional UPDATEs) remain
        and are independently pinned in tests/unit/core/landscape/.
        """
        run_id = f"run-{status.value}"
        _insert_run(db, run_id, status=status)
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(NonResumableRunError, match=r"Run is terminal") as exc_info:
            coordinator.resume(
                _resume_point_for(run_id),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == run_id
        assert status.value in exc_info.value.reason
        assert "immutable" in exc_info.value.reason
        assert checkpoints.rebase_calls == [], "terminal refusal must fire BEFORE rebase_sequence (the first mutation)"
        assert _run_status_of(db, run_id) == status.value

    @pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.INTERRUPTED])
    def test_resumable_statuses_are_admitted(self, db: LandscapeDB, status: RunStatus) -> None:
        run_id = f"run-{status.value}"
        _insert_run(db, run_id, status=status)
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(OrchestrationInvariantError, match="CheckpointManager is required"):
            coordinator.resume(
                _resume_point_for(run_id),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert checkpoints.rebase_calls == [7]
