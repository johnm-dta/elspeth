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


class _StubCheckpointManager:
    """Stub CheckpointManager serving a canned latest checkpoint.

    The checkpoint-currency entry guard reads ONLY get_latest_checkpoint —
    a read-only surface — so a canned return is a faithful double.
    """

    def __init__(self, latest: Any) -> None:
        self._latest = latest

    def get_latest_checkpoint(self, run_id: str) -> Any:
        return self._latest


def _coordinator(db: LandscapeDB) -> tuple[ResumeCoordinator, _RecordingCheckpoints]:
    checkpoints = _RecordingCheckpoints()
    coordinator = ResumeCoordinator(
        db=db,
        events=cast(Any, object()),
        ceremony=cast(Any, object()),
        checkpoints=cast(Any, checkpoints),
        context_factory=cast(Any, object()),
        sink_flush=cast(Any, object()),
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


def _full_resume_point(
    run_id: str,
    *,
    checkpoint_id: str = "cp-1",
    sequence_number: int = 7,
    topology_hash: str = "hash-1",
) -> Any:
    """Resume-point stub with the fields the checkpoint-currency guard reads."""
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            sequence_number=sequence_number,
            upstream_topology_hash=topology_hash,
            full_topology_hash=topology_hash,
        ),
        sequence_number=sequence_number,
    )


def _latest_checkpoint(
    run_id: str,
    *,
    checkpoint_id: str = "cp-1",
    sequence_number: int = 7,
    topology_hash: str = "hash-1",
) -> Any:
    """Canned latest-checkpoint stub for _StubCheckpointManager."""
    return SimpleNamespace(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        sequence_number=sequence_number,
        upstream_topology_hash=topology_hash,
        full_topology_hash=topology_hash,
    )


def _currency_coordinator(db: LandscapeDB, latest: Any) -> tuple[ResumeCoordinator, _RecordingCheckpoints]:
    checkpoints = _RecordingCheckpoints()
    coordinator = ResumeCoordinator(
        db=db,
        events=cast(Any, object()),
        ceremony=cast(Any, object()),
        checkpoints=cast(Any, checkpoints),
        context_factory=cast(Any, object()),
        sink_flush=cast(Any, object()),
        checkpoint_manager=cast(Any, _StubCheckpointManager(latest)),
    )
    return coordinator, checkpoints


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
            # NOTE: RunStatus.RUNNING is NOT in this list — RUNNING is now
            # seat-dependent: absent/expired seat → resumable (slice-4 flip,
            # §H test #2(c)); live seat → refused with join reason.  The
            # RUNNING+absent-seat case is covered by
            # test_running_with_absent_seat_is_resumable_in_slice_4 below.
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

    def test_running_with_absent_seat_is_resumable_in_slice_4(self, db: LandscapeDB) -> None:
        """RUNNING + absent seat row → resumable (§H test #2(c) slice-4 flip).

        A run whose seat was never minted (or was vacated) has no live leader;
        the dead-leader takeover path admits it.
        """
        _insert_run(db, "run-x", status=RunStatus.RUNNING)
        # NOTE: no register_run_leader call — seat row is absent entirely.
        run_status, check = check_run_status_resumable(db, "run-x")
        assert run_status is RunStatus.RUNNING
        assert check.can_resume is True  # slice-4 flip: absent seat → resumable

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

    def test_refuses_running_run_with_live_seat_before_rebase_sequence(self, db: LandscapeDB) -> None:
        """RUNNING + LIVE seat is refused at the entry guard before any mutation.

        Slice-4 flip: RUNNING without a live seat is now RESUMABLE (dead-leader
        takeover); this test verifies the refusal fires for the live-seat arm
        and that no mutation occurs before the guard.
        """
        from elspeth.contracts.coordination import mint_worker_id
        from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository

        _insert_run(db, "run-running", status=RunStatus.RUNNING)
        leader_id = mint_worker_id("run-running")
        RunCoordinationRepository(db.engine).register_run_leader(
            run_id="run-running",
            worker_id=leader_id,
            now=datetime.now(UTC),
            window_seconds=80.0,
        )
        coordinator, checkpoints = _coordinator(db)

        with pytest.raises(NonResumableRunError, match=r"in progress under live leader") as exc_info:
            coordinator.resume(
                _resume_point_for("run-running"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == "run-running"
        assert "in progress" in exc_info.value.reason
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

    def test_running_with_expired_seat_is_resumable_in_slice_4(self, db: LandscapeDB) -> None:
        """RUNNING + expired seat is resumable in slice 4 (ADR-030 §H test #2 contract (c)).

        The dead-leader takeover admission ("entry guard learns seat liveness",
        test #2 contract (c)) flips here in slice 4.  The previously-wedged
        state — RUNNING with an expired/dead leader seat — is now admitted past
        the entry guard so that ``acquire_run_leadership`` (the first durable
        act of resume()) can perform the takeover CAS.

        The post-guard tripwire (``checkpoint_manager=None``) raises
        ``OrchestrationInvariantError("CheckpointManager is required")``
        AFTER the status guard passes, proving admission without exercising
        the full resume reconstruction. Since elspeth-5129406607 the tripwire
        fires INSIDE the entry guard itself (the checkpoint-currency check
        needs the manager, before any mutation), so ``rebase_sequence`` is
        never reached — admission is proven by the exception TYPE: a refused
        run raises ``NonResumableRunError`` instead.
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

        # Slice-4 flip: RUNNING + expired seat is NOW RESUMABLE.  The
        # tripwire raises OrchestrationInvariantError to prove the status
        # guard admitted the run (not NonResumableRunError any more).
        with pytest.raises(OrchestrationInvariantError, match="CheckpointManager is required"):
            coordinator.resume(
                _resume_point_for("run-dead-leader"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        # Admission confirmed by exception TYPE (refusals raise
        # NonResumableRunError). No mutation occurs: the tripwire fires in
        # the checkpoint-currency entry guard, before rebase_sequence
        # (elspeth-5129406607).
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

        # Admission proven by exception TYPE (refusals raise
        # NonResumableRunError); the None-manager tripwire fires inside the
        # checkpoint-currency entry guard before any mutation
        # (elspeth-5129406607), so rebase_sequence is never reached.
        assert checkpoints.rebase_calls == []


class TestResumeCheckpointCurrencyGuard:
    """resume() re-verifies checkpoint currency + topology at entry (elspeth-5129406607).

    ``RecoveryManager.get_resume_point()`` is ADVISORY like ``can_resume()``
    — a caller may hand-build a ``ResumePoint`` — so ``resume()`` re-checks
    at the enforcing boundary that (a) the supplied checkpoint is the run's
    LATEST resume baseline and (b) its recorded topology matches the graph
    the run is about to resume under. Both are READ-ONLY refusals that fire
    BEFORE the first mutation (``rebase_sequence`` / the seat CAS), mirroring
    the elspeth-2f23292372 status guard above.
    """

    def test_missing_latest_checkpoint_refused_before_any_mutation(self, db: LandscapeDB) -> None:
        """A run with no checkpoint rows cannot validate a supplied resume point."""
        _insert_run(db, "run-no-cp", status=RunStatus.FAILED)
        coordinator, checkpoints = _currency_coordinator(db, latest=None)

        with pytest.raises(NonResumableRunError, match=r"no checkpoint") as exc_info:
            coordinator.resume(
                _full_resume_point("run-no-cp"),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == "run-no-cp"
        assert checkpoints.rebase_calls == [], "refusal must fire BEFORE rebase_sequence (the first mutation)"

    def test_stale_checkpoint_id_refused_before_any_mutation(self, db: LandscapeDB) -> None:
        """A resume point naming a superseded checkpoint is refused."""
        _insert_run(db, "run-stale-id", status=RunStatus.FAILED)
        coordinator, checkpoints = _currency_coordinator(
            db,
            latest=_latest_checkpoint("run-stale-id", checkpoint_id="cp-latest", sequence_number=9),
        )

        with pytest.raises(NonResumableRunError, match=r"not the run's latest resume point") as exc_info:
            coordinator.resume(
                _full_resume_point("run-stale-id", checkpoint_id="cp-old", sequence_number=9),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert "cp-old" in exc_info.value.reason
        assert "cp-latest" in exc_info.value.reason
        assert checkpoints.rebase_calls == []

    def test_stale_sequence_number_refused_before_any_mutation(self, db: LandscapeDB) -> None:
        """A resume point whose sequence lags the latest checkpoint is refused.

        ``resume_point.sequence_number`` is the field ``rebase_sequence``
        consumes, so a desynced hand-built point must be caught even when the
        checkpoint_id happens to match.
        """
        _insert_run(db, "run-stale-seq", status=RunStatus.FAILED)
        coordinator, checkpoints = _currency_coordinator(
            db,
            latest=_latest_checkpoint("run-stale-seq", checkpoint_id="cp-1", sequence_number=9),
        )

        with pytest.raises(NonResumableRunError, match=r"not the run's latest resume point"):
            coordinator.resume(
                _full_resume_point("run-stale-seq", checkpoint_id="cp-1", sequence_number=7),
                cast(Any, None),
                cast(Any, None),
                payload_store=MockPayloadStore(),
            )

        assert checkpoints.rebase_calls == []

    def test_divergent_topology_refused_before_any_mutation(self, db: LandscapeDB) -> None:
        """A checkpoint recorded under a different pipeline topology is refused."""
        from elspeth.engine.orchestrator import PipelineConfig
        from tests.fixtures.base_classes import as_sink, as_source
        from tests.fixtures.pipeline import build_production_graph
        from tests.fixtures.plugins import CollectSink, ListSource

        config = PipelineConfig(
            sources={"primary": as_source(ListSource([{"value": 1}]))},
            transforms=[],
            sinks={"default": as_sink(CollectSink("default"))},
        )
        graph = build_production_graph(config)

        _insert_run(db, "run-topo-drift", status=RunStatus.FAILED)
        coordinator, checkpoints = _currency_coordinator(
            db,
            latest=_latest_checkpoint("run-topo-drift"),
        )

        with pytest.raises(NonResumableRunError, match=r"configuration changed") as exc_info:
            coordinator.resume(
                _full_resume_point("run-topo-drift", topology_hash="stale-topology-hash"),
                cast(Any, config),
                cast(Any, graph),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == "run-topo-drift"
        assert checkpoints.rebase_calls == []

    def test_stored_latest_topology_refused_even_when_hand_built_resume_point_claims_current_hash(self, db: LandscapeDB) -> None:
        """The enforcing guard validates the stored latest checkpoint, not caller-supplied checkpoint fields."""
        from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
        from elspeth.engine.orchestrator import PipelineConfig
        from tests.fixtures.base_classes import as_sink, as_source
        from tests.fixtures.pipeline import build_production_graph
        from tests.fixtures.plugins import CollectSink, ListSource

        config = PipelineConfig(
            sources={"primary": as_source(ListSource([{"value": 1}]))},
            transforms=[],
            sinks={"default": as_sink(CollectSink("default"))},
        )
        graph = build_production_graph(config)
        current_hash = CheckpointCompatibilityValidator().compute_full_topology_hash(graph)

        _insert_run(db, "run-stored-topo-drift", status=RunStatus.FAILED)
        coordinator, checkpoints = _currency_coordinator(
            db,
            latest=_latest_checkpoint("run-stored-topo-drift", topology_hash="stale-stored-topology-hash"),
        )

        with pytest.raises(NonResumableRunError, match=r"configuration changed") as exc_info:
            coordinator.resume(
                _full_resume_point("run-stored-topo-drift", topology_hash=current_hash),
                cast(Any, config),
                cast(Any, graph),
                payload_store=MockPayloadStore(),
            )

        assert exc_info.value.run_id == "run-stored-topo-drift"
        assert checkpoints.rebase_calls == []

    def test_current_checkpoint_and_matching_topology_admitted(self, db: LandscapeDB) -> None:
        """Positive control: a current, topology-matching point passes the guard.

        Admission is proven by ``rebase_sequence`` being reached (the first
        mutation after the guard); the resume then fails downstream inside
        ``reconstruct_resume_state`` on this fixture's bare runs row, which is
        expected — the guard must refuse stale points without over-blocking
        current ones.
        """
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
        from elspeth.engine.orchestrator import PipelineConfig
        from tests.fixtures.base_classes import as_sink, as_source
        from tests.fixtures.pipeline import build_production_graph
        from tests.fixtures.plugins import CollectSink, ListSource

        config = PipelineConfig(
            sources={"primary": as_source(ListSource([{"value": 1}]))},
            transforms=[],
            sinks={"default": as_sink(CollectSink("default"))},
        )
        graph = build_production_graph(config)
        current_hash = CheckpointCompatibilityValidator().compute_full_topology_hash(graph)

        _insert_run(db, "run-current-cp", status=RunStatus.FAILED)
        coordinator, checkpoints = _currency_coordinator(
            db,
            latest=_latest_checkpoint("run-current-cp", topology_hash=current_hash),
        )

        with pytest.raises(AuditIntegrityError, match=r"no runtime VAL manifest stored"):
            coordinator.resume(
                _full_resume_point("run-current-cp", topology_hash=current_hash),
                cast(Any, config),
                cast(Any, graph),
                payload_store=MockPayloadStore(),
            )

        assert checkpoints.rebase_calls == [7], "a current resume point must be admitted past the guard"
