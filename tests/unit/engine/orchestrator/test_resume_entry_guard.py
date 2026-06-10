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
- the shared check itself: one implementation, same reasons as can_resume;
- the deliberate carve-out: immutable-success statuses (COMPLETED /
  COMPLETED_WITH_FAILURES / EMPTY) are ADMITTED past the entry guard so the
  durable run-immutability guard in update_run_status() keeps owning their
  refusal (the pinned loser-after-winner contract).

KNOWN RESIDUAL (deliberately NOT closed — operator option c, cross-process
coordination, post-F1): two resumes can both observe FAILED at the guard
before either flips the run to RUNNING (TOCTOU); the guard closes the
caller-convention gap only.
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
    def test_immutable_success_statuses_are_deferred_to_the_immutability_guard(self, db: LandscapeDB, status: RunStatus) -> None:
        """The deliberate carve-out: immutable-success runs pass the ENTRY guard.

        Their refusal stays owned by the durable run-immutability guard
        (AuditIntegrityError, "Successful terminal runs are immutable") —
        the pinned loser-after-winner e2e contract. Here the post-guard
        tripwire (checkpoint_manager=None) raising AFTER rebase_sequence
        proves admission past the entry guard.
        """
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

        assert checkpoints.rebase_calls == [7], "immutable-success status must be ADMITTED past the entry guard"

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
