"""Deterministic stale-token fence suite (ADR-030 §H, slice-2 step 4).

For every leader-fenced verb: bump ``leader_epoch`` directly (simulating a
takeover that deposed this worker), then call the verb with the now-stale
:class:`CoordinationToken` and assert the designed refusal contract —

1. :class:`RunLeadershipLostError` raised;
2. exactly one ``fence_refusal`` event recorded (fresh connection, naming the
   verb in ``context_json``);
3. ZERO payload mutation (the verify-and-extend fence is the FIRST statement
   of the verb's IMMEDIATE transaction, so rowcount-0 unwinds everything).

Fenced verbs covered: ``complete_run``, ``update_run_status``,
``create_checkpoint``, ``delete_checkpoints``, ``complete_barrier`` (both the
strict F1 arm and the legacy partial-release wrapper arm),
``ingest_row_with_initial_claim`` (woken-mid-ingest: atomic rollback, no
orphan ``rows`` row), the fenced ``create_row_with_token`` arm,
``recover_expired_leases``,
``terminalize_pending_sinks_with_terminal_outcomes``, and the §C.4 row-7
per-terminalization-batch fences on ``mark_pending_sink_terminal``/``_many``.

Plus the owner-strictness refusals of the strict pending-sink
terminalization (``mark_pending_sink_terminal``/``_many``): the required
keyword, owner mismatch, and the removed NULL-owner acceptance all refuse
with zero mutation — the owner CAS protects even token-less callers; the
epoch fence stacks on top when a token is threaded.

The §C.4 "fence doubles as the seat heartbeat" property and the §D
quiescence predicate are pinned by the valid-token tests at the bottom.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, select, update

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import (
    AuditIntegrityError,
    OrchestrationInvariantError,
    RunLeadershipLostError,
)
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    checkpoints_table,
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    runs_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.factories import make_graph_linear
from tests.fixtures.landscape import make_landscape_db

RUN_ID = "run-fence-1"
WORKER = f"worker:{RUN_ID}:deadbeef"
NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)
NODE_ID = "transform-1"
SOURCE_NODE_ID = "source-1"


def _payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


@pytest.fixture
def token(db: LandscapeDB) -> CoordinationToken:
    """Seed a RUNNING run + nodes and mint the epoch-1 leader seat."""
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="cfg",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.RUNNING.value,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type in ((SOURCE_NODE_ID, NodeType.SOURCE), (NODE_ID, NodeType.TRANSFORM)):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name="test",
                    node_type=node_type.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="cfg",
                    config_json="{}",
                    registered_at=NOW,
                )
            )
    return RunCoordinationRepository(db.engine).register_run_leader(run_id=RUN_ID, worker_id=WORKER, now=NOW, window_seconds=80.0)


def _bump_epoch(db: LandscapeDB) -> None:
    """Depose the leader: a takeover bumped the seat epoch out from under it."""
    with db.engine.begin() as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == RUN_ID)
            .values(leader_epoch=run_coordination_table.c.leader_epoch + 1)
        )


def _fence_refusals(db: LandscapeDB, verb: str) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        rows = (
            conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == RUN_ID)
                .where(run_coordination_events_table.c.event_type == "fence_refusal")
                .order_by(run_coordination_events_table.c.seq)
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows if json.loads(str(row["context_json"])).get("verb") == verb]


def _seed_row_and_token(db: LandscapeDB, *, sequence: int) -> tuple[str, str]:
    token_id = f"token-{sequence}"
    row_id = f"row-{sequence}"
    with db.engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id=SOURCE_NODE_ID,
                row_index=sequence,
                source_row_index=sequence,
                ingest_sequence=sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=RUN_ID, created_at=NOW))
    return token_id, row_id


def _work_item_row(db: LandscapeDB, token_id: str) -> dict[str, object]:
    with db.engine.connect() as conn:
        row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
    return dict(row)


def _enqueue_and_claim(db: LandscapeDB, repo: TokenSchedulerRepository, *, sequence: int, owner: str) -> tuple[str, str, str]:
    """READY → LEASED row for ``owner``; returns (token_id, row_id, work_item_id)."""
    token_id, row_id = _seed_row_and_token(db, sequence=sequence)
    repo.enqueue_ready(
        run_id=RUN_ID,
        token_id=token_id,
        row_id=row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=sequence,
        row_payload_json=_payload_json(),
        available_at=NOW,
    )
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner=owner, lease_seconds=60, now=NOW)
    assert claimed is not None and claimed.token_id == token_id
    return token_id, row_id, claimed.work_item_id


class TestStaleTokenFenceRefusals:
    """Every fenced verb refuses a stale epoch: raise + event + zero mutation."""

    def test_complete_run_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        factory = RecorderFactory(db)
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            factory.run_lifecycle.complete_run(RUN_ID, RunStatus.COMPLETED, token=token)
        with db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == RUN_ID)).scalar_one()
        assert status == RunStatus.RUNNING.value, "a deposed leader must not stamp a terminal status"
        assert len(_fence_refusals(db, "complete_run")) == 1

    def test_update_run_status_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        factory = RecorderFactory(db)
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            factory.run_lifecycle.update_run_status(RUN_ID, RunStatus.FAILED, token=token)
        with db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == RUN_ID)).scalar_one()
        assert status == RunStatus.RUNNING.value
        assert len(_fence_refusals(db, "update_run_status")) == 1

    def test_create_checkpoint_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        manager = CheckpointManager(db)
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            manager.create_checkpoint(
                run_id=RUN_ID,
                sequence_number=1,
                barrier_scalars=None,
                graph=make_graph_linear(),
                coordination_token=token,
            )
        with db.engine.connect() as conn:
            count = len(conn.execute(select(checkpoints_table.c.checkpoint_id)).all())
        assert count == 0, "the refused checkpoint INSERT must roll back with the fence"
        assert len(_fence_refusals(db, "create_checkpoint")) == 1

    def test_delete_checkpoints_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        manager = CheckpointManager(db)
        manager.create_checkpoint(
            run_id=RUN_ID,
            sequence_number=0,
            barrier_scalars=None,
            graph=make_graph_linear(),
            coordination_token=token,
        )
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            manager.delete_checkpoints(RUN_ID, coordination_token=token)
        with db.engine.connect() as conn:
            count = len(conn.execute(select(checkpoints_table.c.checkpoint_id)).all())
        assert count == 1, "a deposed leader must not destroy the new leader's resume anchors"
        assert len(_fence_refusals(db, "delete_checkpoints")) == 1

    def test_complete_barrier_strict_arm_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner=WORKER)
        repo.mark_blocked(work_item_id=work_item_id, queue_key=None, barrier_key="b1", now=NOW, expected_lease_owner=WORKER)
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            repo.complete_barrier(
                run_id=RUN_ID,
                barrier_key="b1",
                consumed_token_ids=(token_id,),
                emitted_pending_sink=(),
                emitted_ready=(),
                now=NOW,
                coordination_token=token,
            )
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.BLOCKED.value, "refusal before any journal mutation"
        assert len(_fence_refusals(db, "complete_barrier")) == 1

    def test_complete_barrier_legacy_wrapper_arm_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner=WORKER)
        repo.mark_blocked(work_item_id=work_item_id, queue_key=None, barrier_key="b1", now=NOW, expected_lease_owner=WORKER)
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            repo.mark_blocked_barrier_terminal(
                run_id=RUN_ID,
                barrier_key="b1",
                token_ids=(token_id,),
                now=NOW,
                coordination_token=token,
            )
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.BLOCKED.value
        assert len(_fence_refusals(db, "complete_barrier")) == 1

    def test_recover_expired_leases_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, _work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner="peer-worker")
        _bump_epoch(db)
        sweep_time = NOW + timedelta(seconds=120)  # the peer lease (60 s) is expired
        with pytest.raises(RunLeadershipLostError):
            repo.recover_expired_leases(
                run_id=RUN_ID,
                now=sweep_time,
                caller_owner=WORKER,
                coordination_token=token,
            )
        row = _work_item_row(db, token_id)
        assert row["status"] == TokenWorkStatus.LEASED.value, "a deposed leader cannot rotate attempts under the new one"
        assert row["lease_owner"] == "peer-worker"
        assert len(_fence_refusals(db, "recover_expired_leases")) == 1

    def test_terminalize_pending_sinks_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner=WORKER)
        repo.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=_payload_json(),
            sink_name="sink-a",
            outcome="success",
            path="completed",
            error_hash=None,
            error_message=None,
            now=NOW,
            expected_lease_owner=WORKER,
        )
        # Durable terminal outcome witness: without the fence this row WOULD
        # be repaired — proving the refusal is the fence, not a missing match.
        with db.engine.begin() as conn:
            conn.execute(
                insert(token_outcomes_table).values(
                    outcome_id="outcome-1",
                    run_id=RUN_ID,
                    token_id=token_id,
                    outcome="success",
                    path="completed",
                    completed=1,
                    recorded_at=NOW,
                    sink_name="sink-a",
                )
            )
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            repo.terminalize_pending_sinks_with_terminal_outcomes(
                run_id=RUN_ID,
                now=NOW,
                caller_owner=WORKER,
                coordination_token=token,
            )
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value
        assert len(_fence_refusals(db, "terminalize_pending_sinks_with_terminal_outcomes")) == 1

    def test_mark_pending_sink_terminal_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """§C.4 row 7: the epoch fence sits ON TOP of the strict owner CAS —
        a matching owner with a stale epoch is still refused."""
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner=WORKER)
        repo.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=_payload_json(),
            sink_name="sink-a",
            outcome="success",
            path="completed",
            error_hash=None,
            error_message=None,
            now=NOW,
            expected_lease_owner=WORKER,
        )
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            repo.mark_pending_sink_terminal(
                run_id=RUN_ID,
                token_id=token_id,
                now=NOW,
                expected_lease_owner=WORKER,
                coordination_token=token,
            )
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value
        assert len(_fence_refusals(db, "mark_pending_sink_terminal")) == 1

    def test_mark_pending_sink_terminal_many_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner=WORKER)
        repo.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=_payload_json(),
            sink_name="sink-a",
            outcome="success",
            path="completed",
            error_hash=None,
            error_message=None,
            now=NOW,
            expected_lease_owner=WORKER,
        )
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            repo.mark_pending_sink_terminal_many(
                run_id=RUN_ID,
                token_ids=(token_id,),
                now=NOW,
                expected_lease_owner=WORKER,
                coordination_token=token,
            )
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value
        assert len(_fence_refusals(db, "mark_pending_sink_terminal_many")) == 1

    def test_ingest_woken_mid_ingest_atomic_rollback(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """§C.4 row 9: a deposed leader woken mid-ingest leaves NO orphan rows row."""
        repo = TokenSchedulerRepository(db.engine)
        data_flow = RecorderFactory(db).data_flow
        _bump_epoch(db)

        def insert_row_and_token(conn):  # type: ignore[no-untyped-def]
            return data_flow.insert_row_with_token_on(
                conn,
                run_id=RUN_ID,
                source_node_id=SOURCE_NODE_ID,
                row_index=0,
                data={"id": 1},
                source_row_index=0,
                ingest_sequence=0,
                row_id="row-ingest",
                token_id="token-ingest",
            )

        with pytest.raises(RunLeadershipLostError):
            repo.ingest_row_with_initial_claim(
                coordination_token=token,
                now=NOW,
                insert_row_and_token=insert_row_and_token,
                token_id="token-ingest",
                row_id="row-ingest",
                node_id=NODE_ID,
                step_index=1,
                ingest_sequence=0,
                row_payload_json=_payload_json(),
                lease_owner=WORKER,
                lease_seconds=60,
            )
        with db.engine.connect() as conn:
            rows = conn.execute(select(rows_table.c.row_id)).scalars().all()
            items = conn.execute(select(token_work_items_table.c.work_item_id)).scalars().all()
        assert rows == [], "the rows insert rolls back with everything else — no orphan rows row exists"
        assert items == []
        assert len(_fence_refusals(db, "ingest_row_with_initial_claim")) == 1

    def test_fenced_create_row_with_token_refused(self, db: LandscapeDB, token: CoordinationToken) -> None:
        data_flow = RecorderFactory(db).data_flow
        _bump_epoch(db)
        with pytest.raises(RunLeadershipLostError):
            data_flow.create_row_with_token(
                RUN_ID,
                SOURCE_NODE_ID,
                0,
                {"id": 1},
                source_row_index=0,
                ingest_sequence=0,
                coordination_token=token,
            )
        with db.engine.connect() as conn:
            rows = conn.execute(select(rows_table.c.row_id)).scalars().all()
        assert rows == []
        assert len(_fence_refusals(db, "create_row_with_token")) == 1


class TestStrictPendingSinkOwnerCAS:
    """Strict owner CAS on pending-sink terminalization.

    These arms exercise the OWNER CAS in isolation (no ``coordination_token``
    → the unfenced legacy arm). The verbs ALSO carry the §C.4 row-7
    per-terminalization-batch epoch fence when a token is threaded — the
    fenced refusals are pinned in :class:`TestStaleTokenFenceRefusals` below
    and e2e in tests/e2e/recovery/test_suspended_winner_fences.py.
    """

    def test_expected_lease_owner_is_a_required_keyword(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """The strictness signature pin (design :449): ``expected_lease_owner``
        is keyword-only and REQUIRED on both verbs — omission is a TypeError,
        not a silent owner-blind terminalization."""
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        with pytest.raises(TypeError, match="expected_lease_owner"):
            repo.mark_pending_sink_terminal(run_id=RUN_ID, token_id=token_id, now=NOW)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="expected_lease_owner"):
            repo.mark_pending_sink_terminal_many(run_id=RUN_ID, token_ids=(token_id,), now=NOW)  # type: ignore[call-arg]
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value

    def _parked_handoff(self, db: LandscapeDB, *, owner: str) -> tuple[TokenSchedulerRepository, str]:
        repo = TokenSchedulerRepository(db.engine)
        token_id, _row_id, work_item_id = _enqueue_and_claim(db, repo, sequence=0, owner=owner)
        repo.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=_payload_json(),
            sink_name="sink-a",
            outcome="success",
            path="completed",
            error_hash=None,
            error_message=None,
            now=NOW,
            expected_lease_owner=owner,
        )
        return repo, token_id

    def test_attributed_park_keeps_owner_without_lease(self, db: LandscapeDB, token: CoordinationToken) -> None:
        _repo, token_id = self._parked_handoff(db, owner=WORKER)
        row = _work_item_row(db, token_id)
        assert row["status"] == TokenWorkStatus.PENDING_SINK.value
        assert row["lease_owner"] == WORKER, "parked, owner-attributed"
        assert row["lease_expires_at"] is None, "…but not leased"

    def test_owner_mismatch_refuses_with_zero_mutation(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        # Epoch fence passes (valid token), owner CAS refuses (wrong owner) → 0.
        terminalized = repo.mark_pending_sink_terminal(
            run_id=RUN_ID, token_id=token_id, now=NOW, expected_lease_owner="some-other-worker", coordination_token=token
        )
        assert terminalized == 0
        row = _work_item_row(db, token_id)
        assert row["status"] == TokenWorkStatus.PENDING_SINK.value
        assert row["lease_owner"] == WORKER

    def test_null_owner_acceptance_is_removed(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """A NULL-parked handoff (the reap arm's park) no longer terminalizes."""
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        with db.engine.begin() as conn:
            conn.execute(update(token_work_items_table).where(token_work_items_table.c.token_id == token_id).values(lease_owner=None))
        terminalized = repo.mark_pending_sink_terminal(
            run_id=RUN_ID, token_id=token_id, now=NOW, expected_lease_owner=WORKER, coordination_token=token
        )
        assert terminalized == 0, "the historical NULL-owner acceptance arm is deleted"
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value

    def test_matching_owner_terminalizes(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        terminalized = repo.mark_pending_sink_terminal(
            run_id=RUN_ID, token_id=token_id, now=NOW, expected_lease_owner=WORKER, coordination_token=token
        )
        assert terminalized == 1
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.TERMINAL.value

    def test_many_owner_mismatch_refuses_batch(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        with pytest.raises(AuditIntegrityError, match="strict owner CAS"):
            repo.mark_pending_sink_terminal_many(
                run_id=RUN_ID, token_ids=(token_id,), now=NOW, expected_lease_owner="some-other-worker", coordination_token=token
            )
        assert _work_item_row(db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value

    def test_many_null_park_refuses_batch(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        with db.engine.begin() as conn:
            conn.execute(update(token_work_items_table).where(token_work_items_table.c.token_id == token_id).values(lease_owner=None))
        with pytest.raises(AuditIntegrityError, match="strict owner CAS"):
            repo.mark_pending_sink_terminal_many(
                run_id=RUN_ID, token_ids=(token_id,), now=NOW, expected_lease_owner=WORKER, coordination_token=token
            )

    def test_reclaim_restores_attribution_for_reaped_handoff(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """The reap arm parks NULL; claim_pending_sink restores attribution."""
        repo, token_id = self._parked_handoff(db, owner=WORKER)
        with db.engine.begin() as conn:
            conn.execute(update(token_work_items_table).where(token_work_items_table.c.token_id == token_id).values(lease_owner=None))
        reclaimed = repo.claim_pending_sink(run_id=RUN_ID, lease_owner="resume-worker", lease_seconds=60, now=NOW)
        assert reclaimed is not None and reclaimed.token_id == token_id
        terminalized = repo.mark_pending_sink_terminal(
            run_id=RUN_ID, token_id=token_id, now=NOW, expected_lease_owner="resume-worker", coordination_token=token
        )
        assert terminalized == 1


class TestValidTokenFenceSemantics:
    """The fence's positive contract: extend-on-verify and §D quiescence."""

    def test_fenced_verb_extends_the_seat(self, db: LandscapeDB, token: CoordinationToken) -> None:
        manager = CheckpointManager(db)
        # Pin the seat expiry to the distant past so the extension is
        # observable regardless of the wall clock (the manager fences with
        # datetime.now(UTC), not the fixture clock).
        before = datetime(2020, 1, 1, tzinfo=UTC)
        with db.engine.begin() as conn:
            conn.execute(
                update(run_coordination_table).where(run_coordination_table.c.run_id == RUN_ID).values(leader_heartbeat_expires_at=before)
            )
        manager.create_checkpoint(
            run_id=RUN_ID,
            sequence_number=1,
            barrier_scalars=None,
            graph=make_graph_linear(),
            coordination_token=token,
        )
        with db.engine.connect() as conn:
            after = conn.execute(
                select(run_coordination_table.c.leader_heartbeat_expires_at).where(run_coordination_table.c.run_id == RUN_ID)
            ).scalar_one()
        assert after.replace(tzinfo=UTC) > before, "every fenced verb doubles as the seat heartbeat (verify-AND-EXTEND)"
        assert _fence_refusals(db, "create_checkpoint") == []

    def test_complete_run_quiescence_predicate_refuses_residual_work(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """§D: a SUCCESS finalize over residual scheduler work is refused in-statement."""
        repo = TokenSchedulerRepository(db.engine)
        token_id, row_id = _seed_row_and_token(db, sequence=0)
        repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id=NODE_ID,
            step_index=1,
            ingest_sequence=0,
            row_payload_json=_payload_json(),
            available_at=NOW,
        )
        factory = RecorderFactory(db)
        with pytest.raises(OrchestrationInvariantError, match="residual scheduler work"):
            factory.run_lifecycle.complete_run(RUN_ID, RunStatus.COMPLETED, token=token)
        with db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == RUN_ID)).scalar_one()
        assert status == RunStatus.RUNNING.value
        # FAILED is exempt from quiescence: the journal stays intact for resume.
        run = factory.run_lifecycle.complete_run(RUN_ID, RunStatus.FAILED, token=token)
        assert run.status == RunStatus.FAILED

    def test_complete_run_writes_finalize_event(self, db: LandscapeDB, token: CoordinationToken) -> None:
        factory = RecorderFactory(db)
        factory.run_lifecycle.complete_run(RUN_ID, RunStatus.COMPLETED, token=token)
        with db.engine.connect() as conn:
            events = (
                conn.execute(
                    select(run_coordination_events_table)
                    .where(run_coordination_events_table.c.run_id == RUN_ID)
                    .where(run_coordination_events_table.c.event_type == "finalize")
                )
                .mappings()
                .all()
            )
        assert len(events) == 1
        assert events[0]["worker_id"] == WORKER
        assert events[0]["leader_epoch"] == token.leader_epoch
        assert json.loads(str(events[0]["context_json"]))["status"] == RunStatus.COMPLETED.value
