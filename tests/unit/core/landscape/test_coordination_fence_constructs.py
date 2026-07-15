"""Dedicated unit tests for the two SHARED fence constructs (ADR-030 §C.4 / §G).

Design :411: each construct has ONE definition and ONE dedicated unit test —
the ``blocked_barrier_hold_clause`` hygiene pattern — because the predicates
appear in many verbs and per-verb hand-rolled copies would drift.

1. ``active_worker_fence_clause`` (schema module, sibling of
   ``blocked_barrier_hold_clause``): the membership EXISTS predicate that
   slice 4 compiles into ``claim_ready`` / ``claim_pending_sink`` CAS UPDATEs
   and ``enqueue_ready``'s INSERT…SELECT. Slice 2 lands the CONSTRUCT and its
   test ONLY — the negative pin at the bottom proves the claim verbs are NOT
   yet membership-fenced (no claim-refusal behavior is asserted here; that is
   slice 4, design :491).

2. ``verify_and_extend_leader_fence`` (coordination repository): the leader
   epoch verify-and-extend UPDATE CAS, emitted as the FIRST statement of
   every leader-fenced transaction (design :244-255). It executes on the
   CALLER's connection inside ``begin_write`` — no autonomous commit — so a
   later rollback unwinds the extension with the payload.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import Parameter, signature

import pytest
from sqlalchemy import CheckConstraint, delete, insert, select, update

from elspeth.contracts import NodeType, PipelineRow, RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import RunLeadershipLostError, RunWorkerEvictedError
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.database import LandscapeDB, begin_write
from elspeth.core.landscape.run_coordination_repository import (
    RunCoordinationRepository,
    verify_and_extend_leader_fence,
)
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    active_worker_fence_clause,
    claim_verb_fence_clause,
    metadata,
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)
RUN_1 = "run-fence-construct-1"
RUN_2 = "run-fence-construct-2"
NODE_ID = "transform-1"
SOURCE_NODE_ID = "source-1"


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


def _insert_run(db: LandscapeDB, run_id: str) -> None:
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
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
                    run_id=run_id,
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


def _insert_worker(db: LandscapeDB, *, worker_id: str, run_id: str, status: str) -> None:
    with db.engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=run_id,
                role="follower",
                status=status,
                registered_at=NOW,
                heartbeat_expires_at=NOW + timedelta(hours=1),
                evicted_at=NOW if status == "evicted" else None,
            )
        )


def _seed_ready_item(db: LandscapeDB, run_id: str, *, sequence: int = 0) -> str:
    """One READY token_work_items row via the production enqueue. Returns work_item_id."""
    token_id = f"token-{run_id}-{sequence}"
    row_id = f"row-{run_id}-{sequence}"
    with db.engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=run_id,
                source_node_id=SOURCE_NODE_ID,
                row_index=sequence,
                source_row_index=sequence,
                ingest_sequence=sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=run_id, created_at=NOW))
    repo = TokenSchedulerRepository(db.engine)
    repo.enqueue_ready(
        run_id=run_id,
        token_id=token_id,
        row_id=row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=sequence,
        row_payload_json=TokenSchedulerRepository.serialize_row_payload(
            PipelineRow({"id": sequence}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
        ),
        available_at=NOW,
    )
    with db.engine.connect() as conn:
        return str(
            conn.execute(select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.token_id == token_id)).scalar_one()
        )


def _seed_pending_sink_item(db: LandscapeDB, run_id: str, *, sequence: int = 0) -> str:
    from elspeth.contracts.scheduler import TokenWorkStatus

    work_item_id = _seed_ready_item(db, run_id, sequence=sequence)
    with db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == work_item_id)
            .values(
                status=TokenWorkStatus.PENDING_SINK.value,
                pending_sink_name="sink-a",
                pending_outcome=TerminalOutcome.SUCCESS.value,
                pending_path=TerminalPath.DEFAULT_FLOW.value,
                pending_error_hash=None,
                pending_error_message=None,
                updated_at=NOW + timedelta(seconds=1),
            )
        )
    return work_item_id


def _seed_unscheduled_item(db: LandscapeDB, run_id: str, *, sequence: int) -> dict[str, object]:
    """Insert the durable row/token prerequisites for one not-yet-queued item."""
    token_id = f"token-{run_id}-{sequence}"
    row_id = f"row-{run_id}-{sequence}"
    with db.engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=run_id,
                source_node_id=SOURCE_NODE_ID,
                row_index=sequence,
                source_row_index=sequence,
                ingest_sequence=sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=run_id, created_at=NOW))
    return {
        "run_id": run_id,
        "token_id": token_id,
        "row_id": row_id,
        "node_id": NODE_ID,
        "step_index": 1,
        "ingest_sequence": sequence,
        "row_payload_json": TokenSchedulerRepository.serialize_row_payload(
            PipelineRow({"id": sequence}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
        ),
        "available_at": NOW,
        "lease_seconds": 60,
        "now": NOW,
    }


def _full_durable_snapshot(db: LandscapeDB) -> dict[str, tuple[tuple[object, ...], ...]]:
    """Backend-portable value snapshot of every durable Landscape table."""
    with db.engine.connect() as conn:
        return {
            table.name: tuple(sorted((tuple(row) for row in conn.execute(select(table)).all()), key=repr))
            for table in metadata.sorted_tables
        }


class TestActiveWorkerFenceClause:
    """Design :257-265, :411 — the CONSTRUCT only (compilation is slice 4)."""

    def test_compiled_shape_is_correlated_exists_over_run_workers(self) -> None:
        clause = active_worker_fence_clause(worker_id="worker-x", run_id="run-x")
        sql = str(clause.compile(compile_kwargs={"literal_binds": True}))
        assert "EXISTS" in sql.upper()
        assert "run_workers" in sql
        assert "worker_id" in sql
        assert "run_id" in sql
        assert "status = 'active'" in sql

    def test_active_literal_matches_the_run_workers_status_check(self) -> None:
        """The ``blocked_barrier_hold_clause`` literal-parity discipline: the
        'active' literal the clause compiles MUST be a member of the
        ``ck_run_workers_status`` CHECK's value set — drift between the two
        would make the fence silently always-False."""
        status_checks = [
            constraint
            for constraint in run_workers_table.constraints
            if isinstance(constraint, CheckConstraint) and constraint.name == "ck_run_workers_status"
        ]
        assert len(status_checks) == 1
        check_sql = str(status_checks[0].sqltext)
        for literal in ("'active'", "'departed'", "'evicted'"):
            assert literal in check_sql
        clause_sql = str(active_worker_fence_clause(worker_id="w", run_id="r").compile(compile_kwargs={"literal_binds": True}))
        assert "'active'" in clause_sql


class TestClaimVerbFenceClause:
    """Claim verbs are lenient only for true N=0 registry compatibility."""

    @pytest.mark.parametrize(
        ("registered_worker", "caller", "expected_rowcount"),
        [
            (None, "worker-absent", 1),  # N=0 unit-test mode remains allowed
            ("worker-active", "worker-active", 1),  # active registered caller
            ("worker-active", "worker-absent", 0),  # absent caller cannot bypass active registry
            ("worker-evicted", "worker-absent", 0),  # absent caller cannot bypass any non-empty registry
            ("worker-evicted", "worker-evicted", 0),  # non-active caller remains fenced
        ],
    )
    def test_claim_verb_allows_absent_worker_only_when_run_has_no_workers(
        self,
        db: LandscapeDB,
        registered_worker: str | None,
        caller: str,
        expected_rowcount: int,
    ) -> None:
        _insert_run(db, RUN_1)
        if registered_worker is not None:
            status = "evicted" if registered_worker == "worker-evicted" else "active"
            _insert_worker(db, worker_id=registered_worker, run_id=RUN_1, status=status)
        work_item_id = _seed_ready_item(db, RUN_1)

        with begin_write(db.engine) as conn:
            result = conn.execute(
                update(token_work_items_table)
                .where(
                    token_work_items_table.c.work_item_id == work_item_id,
                    token_work_items_table.c.status == "ready",
                    claim_verb_fence_clause(worker_id=caller, run_id=RUN_1),
                )
                .values(updated_at=NOW + timedelta(seconds=1))
            )

        assert result.rowcount == expected_rowcount

    def test_claim_ready_absent_worker_does_not_claim_when_run_has_active_workers(self, db: LandscapeDB) -> None:
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        work_item_id = _seed_ready_item(db, RUN_1)
        repo = TokenSchedulerRepository(db.engine)

        claimed = repo.claim_ready(run_id=RUN_1, lease_owner="worker-absent", lease_seconds=60, now=NOW)

        assert claimed is None
        with db.engine.connect() as conn:
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        assert row["status"] == TokenWorkStatus.READY.value
        assert row["lease_owner"] is None

    def test_claim_pending_sink_absent_worker_does_not_claim_when_run_has_active_workers(self, db: LandscapeDB) -> None:
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        work_item_id = _seed_pending_sink_item(db, RUN_1)
        repo = TokenSchedulerRepository(db.engine)

        claimed = repo.claim_pending_sink(run_id=RUN_1, lease_owner="worker-absent", lease_seconds=60, now=NOW)

        assert claimed is None
        with db.engine.connect() as conn:
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        assert row["status"] == TokenWorkStatus.PENDING_SINK.value
        assert row["lease_owner"] is None

    @pytest.mark.parametrize(
        ("caller", "caller_run", "expected_rowcount"),
        [
            ("worker-active", RUN_1, 1),  # A: active, this run
            ("worker-evicted", RUN_1, 0),  # B: evicted, this run
            ("worker-departed", RUN_1, 0),  # C: departed, this run
            ("worker-absent", RUN_1, 0),  # no registry row at all
            ("worker-other-run", RUN_1, 0),  # D: active, but in the OTHER run
        ],
    )
    def test_claim_shaped_update_against_real_epoch_21_db(
        self, db: LandscapeDB, caller: str, caller_run: str, expected_rowcount: int
    ) -> None:
        """Behavioral matrix on a real epoch-21 SQLite DB: the clause embedded
        in a representative claim-shaped UPDATE over one seeded READY row."""
        _insert_run(db, RUN_1)
        _insert_run(db, RUN_2)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        _insert_worker(db, worker_id="worker-evicted", run_id=RUN_1, status="evicted")
        _insert_worker(db, worker_id="worker-departed", run_id=RUN_1, status="departed")
        _insert_worker(db, worker_id="worker-other-run", run_id=RUN_2, status="active")
        work_item_id = _seed_ready_item(db, RUN_1)

        with begin_write(db.engine) as conn:
            result = conn.execute(
                update(token_work_items_table)
                .where(
                    token_work_items_table.c.work_item_id == work_item_id,
                    token_work_items_table.c.status == "ready",
                    active_worker_fence_clause(worker_id=caller, run_id=caller_run),
                )
                .values(updated_at=NOW + timedelta(seconds=1))
            )
            assert result.rowcount == expected_rowcount

    def test_membership_fence_compiled_into_claim_verbs_in_slice_4(self, db: LandscapeDB) -> None:
        """Slice-4 flip (design :491): the clause IS compiled into
        ``claim_ready``/``claim_pending_sink``/``enqueue_ready`` — an EVICTED
        worker is refused with ``RunWorkerEvictedError`` for all three verbs.

        Replaces the slice-2 negative pin (EVICTED worker could claim).
        """
        from elspeth.contracts.errors import RunWorkerEvictedError
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-evicted", run_id=RUN_1, status="evicted")
        _seed_ready_item(db, RUN_1)
        repo = TokenSchedulerRepository(db.engine)

        # claim_ready: evicted worker is refused
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            repo.claim_ready(run_id=RUN_1, lease_owner="worker-evicted", lease_seconds=60, now=NOW)
        assert exc_info.value.worker_id == "worker-evicted"
        assert exc_info.value.run_id == RUN_1

        # The READY row is untouched (zero mutation on fence failure).
        with db.engine.connect() as conn:
            status = conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == RUN_1)).scalar_one()
        assert status == TokenWorkStatus.READY.value

        # enqueue_ready: evicted worker is refused (worker_id kwarg carries the fence).
        # Re-use the same schema seed (different sequence to avoid work_item_id collision).
        _seed_ready_item(db, RUN_1, sequence=1)
        with pytest.raises(RunWorkerEvictedError):
            repo.enqueue_ready(
                run_id=RUN_1,
                token_id="token-new",
                row_id="row-new",
                node_id=NODE_ID,
                step_index=1,
                ingest_sequence=99,
                row_payload_json=TokenSchedulerRepository.serialize_row_payload(
                    PipelineRow({"id": 99}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
                ),
                available_at=NOW,
                worker_id="worker-evicted",
            )

        # An ABSENT worker (no registry row at all) calling enqueue_ready with
        # worker_id is also refused (absent → fence fails).
        with pytest.raises(RunWorkerEvictedError) as exc_info2:
            repo.enqueue_ready(
                run_id=RUN_1,
                token_id="token-absent",
                row_id="row-absent",
                node_id=NODE_ID,
                step_index=1,
                ingest_sequence=100,
                row_payload_json=TokenSchedulerRepository.serialize_row_payload(
                    PipelineRow({"id": 100}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
                ),
                available_at=NOW,
                worker_id="worker-absent",
            )
        assert exc_info2.value.worker_id == "worker-absent"

        # An ACTIVE worker can still claim (positive control).
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        claimed = repo.claim_ready(run_id=RUN_1, lease_owner="worker-active", lease_seconds=60, now=NOW)
        assert claimed is not None, "active worker must succeed"


class TestEnqueueReadyClaimedMembershipFence:
    """The standalone enqueue-and-claim path is strict; legacy is explicit."""

    @pytest.mark.parametrize("caller_status", [None, "evicted"], ids=["absent", "evicted"])
    def test_absent_or_evicted_identity_is_refused_with_full_zero_mutation(
        self,
        db: LandscapeDB,
        caller_status: str | None,
    ) -> None:
        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        caller = "worker-absent" if caller_status is None else "worker-evicted"
        if caller_status is not None:
            _insert_worker(db, worker_id=caller, run_id=RUN_1, status=caller_status)
        enqueue = _seed_unscheduled_item(db, RUN_1, sequence=10)
        before = _full_durable_snapshot(db)

        with pytest.raises(RunWorkerEvictedError) as exc_info:
            TokenSchedulerRepository(db.engine).enqueue_ready_claimed(**enqueue, lease_owner=caller)

        assert exc_info.value.worker_id == caller
        assert exc_info.value.run_id == RUN_1
        assert _full_durable_snapshot(db) == before

    def test_active_member_enqueues_claims_and_records_both_events(self, db: LandscapeDB) -> None:
        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        enqueue = _seed_unscheduled_item(db, RUN_1, sequence=11)

        claimed = TokenSchedulerRepository(db.engine).enqueue_ready_claimed(**enqueue, lease_owner="worker-active")

        assert claimed.status is TokenWorkStatus.LEASED
        assert claimed.lease_owner == "worker-active"
        with db.engine.connect() as conn:
            event_types = set(
                conn.execute(select(scheduler_events_table.c.event_type).where(scheduler_events_table.c.run_id == RUN_1)).scalars()
            )
        assert event_types == {SchedulerEventType.ENQUEUE.value, SchedulerEventType.CLAIM_READY.value}

    def test_active_member_future_item_stays_ready_without_misreported_eviction(self, db: LandscapeDB) -> None:
        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        enqueue = _seed_unscheduled_item(db, RUN_1, sequence=15)
        future = NOW + timedelta(seconds=60)
        enqueue["available_at"] = future

        scheduled = TokenSchedulerRepository(db.engine).enqueue_ready_claimed(**enqueue, lease_owner="worker-active")

        assert scheduled.status is TokenWorkStatus.READY
        assert scheduled.lease_owner is None
        assert scheduled.available_at == future
        with db.engine.connect() as conn:
            event_types = set(
                conn.execute(select(scheduler_events_table.c.event_type).where(scheduler_events_table.c.run_id == RUN_1)).scalars()
            )
        assert event_types == {SchedulerEventType.ENQUEUE.value}

    def test_explicit_legacy_unfenced_helper_preserves_n0_test_mode(self, db: LandscapeDB) -> None:
        _insert_run(db, RUN_1)
        enqueue = _seed_unscheduled_item(db, RUN_1, sequence=12)

        claimed = TokenSchedulerRepository(db.engine).enqueue_ready_claimed_legacy_unfenced(
            **enqueue,
            lease_owner="legacy-test-owner",
        )

        assert claimed.status is TokenWorkStatus.LEASED
        assert claimed.lease_owner == "legacy-test-owner"

    def test_membership_cas_rolls_back_when_member_is_evicted_after_entry_guard(
        self,
        db: LandscapeDB,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The claim UPDATE rechecks membership after the initial strict guard."""
        from elspeth.core.landscape.scheduler import queue as queue_module

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        enqueue = _seed_unscheduled_item(db, RUN_1, sequence=13)
        before = _full_durable_snapshot(db)
        real_insert = queue_module.insert_work_item_idempotent

        def evict_then_insert(*args: object, **kwargs: object) -> bool:
            conn = args[0]
            conn.execute(
                update(run_workers_table).where(run_workers_table.c.worker_id == "worker-active").values(status="evicted", evicted_at=NOW)
            )
            return real_insert(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(queue_module, "insert_work_item_idempotent", evict_then_insert)

        with pytest.raises(RunWorkerEvictedError):
            TokenSchedulerRepository(db.engine).enqueue_ready_claimed(**enqueue, lease_owner="worker-active")

        assert _full_durable_snapshot(db) == before

    def test_strict_membership_cas_rolls_back_when_sole_member_is_deleted_after_entry_guard(
        self,
        db: LandscapeDB,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Production enqueue must not fall through to the lenient N=0 claim arm."""
        from elspeth.core.landscape.scheduler import queue as queue_module

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-active", run_id=RUN_1, status="active")
        enqueue = _seed_unscheduled_item(db, RUN_1, sequence=14)
        before = _full_durable_snapshot(db)
        real_insert = queue_module.insert_work_item_idempotent

        def delete_member_then_insert(*args: object, **kwargs: object) -> bool:
            conn = args[0]
            conn.execute(delete(run_workers_table).where(run_workers_table.c.worker_id == "worker-active"))
            return real_insert(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(queue_module, "insert_work_item_idempotent", delete_member_then_insert)

        with pytest.raises(RunWorkerEvictedError) as exc_info:
            TokenSchedulerRepository(db.engine).enqueue_ready_claimed(**enqueue, lease_owner="worker-active")

        assert exc_info.value.worker_id == "worker-active"
        assert exc_info.value.run_id == RUN_1
        assert _full_durable_snapshot(db) == before


class TestHeartbeatLeaseMembershipFence:
    """Heartbeat callers choose strict membership or explicit legacy N=0."""

    def test_public_repository_layers_require_explicit_fence_intent(self) -> None:
        from elspeth.core.landscape.scheduler.leases import SchedulerLeaseRepository

        for heartbeat in (SchedulerLeaseRepository.heartbeat_lease, TokenSchedulerRepository.heartbeat_lease):
            parameter = signature(heartbeat).parameters["membership_fenced"]
            assert parameter.default is Parameter.empty

    @staticmethod
    def _leased_item(db: LandscapeDB, *, worker_id: str) -> tuple[TokenSchedulerRepository, str]:
        work_item_id = _seed_ready_item(db, RUN_1)
        repo = TokenSchedulerRepository(db.engine)
        claimed = repo.claim_ready(run_id=RUN_1, lease_owner=worker_id, lease_seconds=60, now=NOW)
        assert claimed is not None and claimed.work_item_id == work_item_id
        return repo, work_item_id

    @staticmethod
    def _durable_state(db: LandscapeDB, *, work_item_id: str, worker_id: str) -> tuple[object, ...]:
        """Snapshot every surface a refused heartbeat must leave unchanged."""
        with db.engine.connect() as conn:
            item = (
                conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
            )
            worker = conn.execute(select(run_workers_table).where(run_workers_table.c.worker_id == worker_id)).mappings().one_or_none()
            scheduler_events = tuple(
                conn.execute(
                    select(scheduler_events_table.c.event_id)
                    .where(scheduler_events_table.c.run_id == RUN_1)
                    .order_by(scheduler_events_table.c.event_id)
                ).scalars()
            )
            coordination_events = tuple(
                conn.execute(
                    select(run_coordination_events_table.c.event_id)
                    .where(run_coordination_events_table.c.run_id == RUN_1)
                    .order_by(run_coordination_events_table.c.event_id)
                ).scalars()
            )
        return dict(item), None if worker is None else dict(worker), scheduler_events, coordination_events

    @pytest.mark.parametrize("status", ["evicted", "departed"])
    def test_non_active_owner_is_refused_with_zero_durable_mutation(self, db: LandscapeDB, status: str) -> None:
        from elspeth.contracts.errors import RunWorkerEvictedError

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        with db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == "worker-a")
                .values(
                    status=status,
                    departed_at=NOW + timedelta(seconds=1) if status == "departed" else None,
                    evicted_at=NOW + timedelta(seconds=1) if status == "evicted" else None,
                    evicted_by_worker_id="worker-leader" if status == "evicted" else None,
                )
            )
        before = self._durable_state(db, work_item_id=work_item_id, worker_id="worker-a")

        with pytest.raises(RunWorkerEvictedError) as exc_info:
            repo.heartbeat_lease(
                run_id=RUN_1,
                work_item_id=work_item_id,
                lease_owner="worker-a",
                lease_seconds=60,
                now=NOW + timedelta(seconds=30),
                membership_fenced=True,
            )

        assert exc_info.value.worker_id == "worker-a"
        assert exc_info.value.run_id == RUN_1
        assert self._durable_state(db, work_item_id=work_item_id, worker_id="worker-a") == before

    def test_deleted_sole_membership_row_is_refused_with_zero_durable_mutation(self, db: LandscapeDB) -> None:
        from elspeth.contracts.errors import RunWorkerEvictedError

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        with db.engine.begin() as conn:
            conn.execute(delete(run_workers_table).where(run_workers_table.c.worker_id == "worker-a"))
        before = self._durable_state(db, work_item_id=work_item_id, worker_id="worker-a")

        with pytest.raises(RunWorkerEvictedError):
            repo.heartbeat_lease(
                run_id=RUN_1,
                work_item_id=work_item_id,
                lease_owner="worker-a",
                lease_seconds=60,
                now=NOW + timedelta(seconds=30),
                membership_fenced=True,
            )

        assert self._durable_state(db, work_item_id=work_item_id, worker_id="worker-a") == before

    def test_active_current_owner_can_extend_lease(self, db: LandscapeDB) -> None:
        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")

        expires_at = repo.heartbeat_lease(
            run_id=RUN_1,
            work_item_id=work_item_id,
            lease_owner="worker-a",
            lease_seconds=60,
            now=NOW + timedelta(seconds=30),
            membership_fenced=True,
        )

        assert expires_at == NOW + timedelta(seconds=90)

    def test_active_owner_can_revive_expired_lease_before_recovery(self, db: LandscapeDB) -> None:
        """Expiry alone is not ownership loss; recovery is the competing CAS."""
        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")

        expires_at = repo.heartbeat_lease(
            run_id=RUN_1,
            work_item_id=work_item_id,
            lease_owner="worker-a",
            lease_seconds=60,
            now=NOW + timedelta(seconds=61),
            membership_fenced=True,
        )

        assert expires_at == NOW + timedelta(seconds=121)

    def test_wrong_active_owner_uses_existing_lease_lost_path(self, db: LandscapeDB) -> None:
        from elspeth.contracts.errors import SchedulerLeaseLostError

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        _insert_worker(db, worker_id="worker-b", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")

        with pytest.raises(SchedulerLeaseLostError):
            repo.heartbeat_lease(
                run_id=RUN_1,
                work_item_id=work_item_id,
                lease_owner="worker-b",
                lease_seconds=60,
                now=NOW + timedelta(seconds=30),
                membership_fenced=True,
            )

    def test_recovered_lease_uses_existing_lease_lost_path_when_membership_active(self, db: LandscapeDB) -> None:
        from elspeth.contracts.errors import SchedulerLeaseLostError

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        recovered = repo.recover_expired_leases(
            run_id=RUN_1,
            now=NOW + timedelta(seconds=61),
            caller_owner="worker-reaper",
        )
        assert recovered == 1

        with pytest.raises(SchedulerLeaseLostError):
            repo.heartbeat_lease(
                run_id=RUN_1,
                work_item_id=work_item_id,
                lease_owner="worker-a",
                lease_seconds=60,
                now=NOW + timedelta(seconds=62),
                membership_fenced=True,
            )

    def test_explicit_unfenced_intent_preserves_direct_harness_n0(self, db: LandscapeDB) -> None:
        """The required False flag, not registry emptiness, selects legacy mode."""
        _insert_run(db, RUN_1)
        repo, work_item_id = self._leased_item(db, worker_id="direct-harness")

        expires_at = repo.heartbeat_lease(
            run_id=RUN_1,
            work_item_id=work_item_id,
            lease_owner="direct-harness",
            lease_seconds=60,
            now=NOW + timedelta(seconds=30),
            membership_fenced=False,
        )

        assert expires_at == NOW + timedelta(seconds=90)


class TestVerifyAndExtendLeaderFence:
    """Design :244-255 — the leader epoch verify-and-extend UPDATE CAS."""

    def _seat(self, db: LandscapeDB) -> CoordinationToken:
        _insert_run(db, RUN_1)
        return RunCoordinationRepository(db.engine).register_run_leader(
            run_id=RUN_1,
            worker_id="worker-leader",
            now=NOW,
            window_seconds=80.0,
        )

    def _expiry(self, db: LandscapeDB) -> datetime:
        with db.engine.connect() as conn:
            value: datetime = conn.execute(
                select(run_coordination_table.c.leader_heartbeat_expires_at).where(run_coordination_table.c.run_id == RUN_1)
            ).scalar_one()
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def test_match_extends_expiry_and_stamps_updated_at(self, db: LandscapeDB) -> None:
        token = self._seat(db)
        later = NOW + timedelta(seconds=30)
        with begin_write(db.engine) as conn:
            verify_and_extend_leader_fence(conn, token=token, now=later, window_seconds=120.0, verb="unit-test")
        assert self._expiry(db) == later + timedelta(seconds=120)
        with db.engine.connect() as conn:
            updated_at = conn.execute(
                select(run_coordination_table.c.updated_at).where(run_coordination_table.c.run_id == RUN_1)
            ).scalar_one()
        assert (updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=UTC)) == later

    def test_stale_epoch_raises_and_does_not_move_expiry(self, db: LandscapeDB) -> None:
        token = self._seat(db)
        before = self._expiry(db)
        stale = CoordinationToken(run_id=RUN_1, worker_id=token.worker_id, leader_epoch=token.leader_epoch + 1)
        with pytest.raises(RunLeadershipLostError) as exc_info, begin_write(db.engine) as conn:
            verify_and_extend_leader_fence(conn, token=stale, now=NOW + timedelta(seconds=30), window_seconds=120.0, verb="unit-test")
        assert exc_info.value.verb == "unit-test"
        assert self._expiry(db) == before

    def test_wrong_worker_with_correct_epoch_raises(self, db: LandscapeDB) -> None:
        token = self._seat(db)
        before = self._expiry(db)
        foreign = CoordinationToken(run_id=RUN_1, worker_id="worker-imposter", leader_epoch=token.leader_epoch)
        with pytest.raises(RunLeadershipLostError), begin_write(db.engine) as conn:
            verify_and_extend_leader_fence(conn, token=foreign, now=NOW + timedelta(seconds=30), window_seconds=120.0, verb="unit-test")
        assert self._expiry(db) == before

    def test_verify_rides_the_payload_transaction_no_autonomous_commit(self, db: LandscapeDB) -> None:
        """Slice-1 discipline: the verify executes on the CALLER's connection
        inside ``begin_write``. A rollback AFTER a successful verify leaves
        the expiry unmoved — the extension is part of the payload
        transaction, never an autonomous commit."""
        token = self._seat(db)
        before = self._expiry(db)

        class _Boom(Exception):
            pass

        with pytest.raises(_Boom), begin_write(db.engine) as conn:
            verify_and_extend_leader_fence(conn, token=token, now=NOW + timedelta(seconds=30), window_seconds=120.0, verb="unit-test")
            raise _Boom

        assert self._expiry(db) == before, "the successful verify rolled back with the payload"


class TestDispositionMembershipFence:
    """ADR-030 §G parity for DISPOSITION verbs (filigree elspeth-ba7b2cc25d).

    ``claim_ready`` / ``claim_pending_sink`` / ``enqueue_ready`` gained the
    membership fence in slices 4-5; the disposition verbs (``mark_blocked`` /
    ``mark_terminal`` / ``mark_failed`` / ``mark_pending_sink``) were the lone
    unfenced exception. When the caller threads ``worker_id``, the LENIENT
    ``claim_verb_fence_clause`` rides the same UPDATE WHERE: an evicted or
    departed worker is refused with ``RunWorkerEvictedError`` and ZERO
    mutation, while the N=0 OR-branch (no registered workers at all) keeps
    unregistered/unit-test dispositions working.
    """

    def _leased_item(self, db: LandscapeDB, *, worker_id: str, sequence: int = 0) -> tuple[TokenSchedulerRepository, str]:
        work_item_id = _seed_ready_item(db, RUN_1, sequence=sequence)
        repo = TokenSchedulerRepository(db.engine)
        claimed = repo.claim_ready(run_id=RUN_1, lease_owner=worker_id, lease_seconds=60, now=NOW)
        assert claimed is not None and claimed.work_item_id == work_item_id
        return repo, work_item_id

    @staticmethod
    def _set_worker_status(db: LandscapeDB, worker_id: str, status: str) -> None:
        with db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == worker_id)
                .values(status=status, evicted_at=NOW if status == "evicted" else None)
            )

    @staticmethod
    def _item_state(db: LandscapeDB, work_item_id: str) -> tuple[str, str | None]:
        with db.engine.connect() as conn:
            row = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                    token_work_items_table.c.work_item_id == work_item_id
                )
            ).one()
        return str(row.status), row.lease_owner

    def test_evicted_worker_is_refused_on_every_disposition_verb_with_zero_mutation(self, db: LandscapeDB) -> None:
        from elspeth.contracts.errors import RunWorkerEvictedError
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        self._set_worker_status(db, "worker-a", "evicted")

        dispositions = {
            "mark_terminal": lambda: repo.mark_terminal(
                work_item_id=work_item_id, now=NOW, expected_lease_owner="worker-a", worker_id="worker-a"
            ),
            "mark_failed": lambda: repo.mark_failed(
                work_item_id=work_item_id, now=NOW, expected_lease_owner="worker-a", worker_id="worker-a"
            ),
            "mark_blocked": lambda: repo.mark_blocked(
                work_item_id=work_item_id,
                queue_key=None,
                barrier_key="barrier-1",
                now=NOW,
                expected_lease_owner="worker-a",
                worker_id="worker-a",
            ),
            "mark_pending_sink": lambda: repo.mark_pending_sink(
                work_item_id=work_item_id,
                row_payload_json="{}",
                sink_name="sink-a",
                outcome="success",
                path=TerminalPath.DEFAULT_FLOW.value,
                error_hash=None,
                error_message=None,
                now=NOW,
                expected_lease_owner="worker-a",
                worker_id="worker-a",
            ),
        }
        for verb, disposition in dispositions.items():
            with pytest.raises(RunWorkerEvictedError) as exc_info:
                disposition()
            assert exc_info.value.worker_id == "worker-a", verb
            assert exc_info.value.run_id == RUN_1, verb
            status, lease_owner = self._item_state(db, work_item_id)
            assert status == TokenWorkStatus.LEASED.value, f"{verb} mutated status after eviction"
            assert lease_owner == "worker-a", f"{verb} mutated lease_owner after eviction"

    def test_departed_worker_is_refused_too(self, db: LandscapeDB) -> None:
        from elspeth.contracts.errors import RunWorkerEvictedError

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        self._set_worker_status(db, "worker-a", "departed")
        with pytest.raises(RunWorkerEvictedError):
            repo.mark_terminal(work_item_id=work_item_id, now=NOW, expected_lease_owner="worker-a", worker_id="worker-a")

    def test_fenced_disposition_succeeds_when_run_has_no_workers(self, db: LandscapeDB) -> None:
        """The LENIENT clause's N=0 OR-branch: a disposition WITH worker_id
        supplied still succeeds when the run has no registered workers at all
        (the reason this fence uses claim_verb_fence_clause, NOT the strict
        active_worker_fence_clause)."""
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        repo, work_item_id = self._leased_item(db, worker_id="worker-unregistered")
        item = repo.mark_terminal(
            work_item_id=work_item_id, now=NOW, expected_lease_owner="worker-unregistered", worker_id="worker-unregistered"
        )
        assert item.status is TokenWorkStatus.TERMINAL

    def test_active_worker_disposition_succeeds_with_fence(self, db: LandscapeDB) -> None:
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        item = repo.mark_terminal(work_item_id=work_item_id, now=NOW, expected_lease_owner="worker-a", worker_id="worker-a")
        assert item.status is TokenWorkStatus.TERMINAL

    def test_unfenced_disposition_keeps_legacy_behavior_when_worker_id_omitted(self, db: LandscapeDB) -> None:
        """Default ``worker_id=None`` keeps the fence OFF — the processor gates
        the fence on ``_scheduler_lease_owner_registered``, so legacy/N=0
        callers that never thread an identity are unaffected."""
        from elspeth.contracts.scheduler import TokenWorkStatus

        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-a", run_id=RUN_1, status="active")
        repo, work_item_id = self._leased_item(db, worker_id="worker-a")
        self._set_worker_status(db, "worker-a", "evicted")
        item = repo.mark_terminal(work_item_id=work_item_id, now=NOW, expected_lease_owner="worker-a")
        assert item.status is TokenWorkStatus.TERMINAL
