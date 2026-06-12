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

import pytest
from sqlalchemy import CheckConstraint, insert, select, update

from elspeth.contracts import NodeType, PipelineRow, RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import RunLeadershipLostError
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.database import LandscapeDB, begin_write
from elspeth.core.landscape.run_coordination_repository import (
    RunCoordinationRepository,
    verify_and_extend_leader_fence,
)
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    active_worker_fence_clause,
    nodes_table,
    rows_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
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

    def test_negative_pin_claim_verbs_are_not_membership_fenced_in_slice_2(self, db: LandscapeDB) -> None:
        """Slice-scope pin (design :491): the clause is NOT yet compiled into
        ``claim_ready``/``claim_pending_sink``/``enqueue_ready`` — an EVICTED
        worker can still claim in slice 2. When slice 4 lands the membership
        fences, THIS test flips to the ``RunWorkerEvictedError`` contract."""
        _insert_run(db, RUN_1)
        _insert_worker(db, worker_id="worker-evicted", run_id=RUN_1, status="evicted")
        _seed_ready_item(db, RUN_1)
        claimed = TokenSchedulerRepository(db.engine).claim_ready(
            run_id=RUN_1,
            lease_owner="worker-evicted",
            lease_seconds=60,
            now=NOW,
        )
        assert claimed is not None, "slice 2 deliberately leaves fresh claims membership-unfenced"


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
