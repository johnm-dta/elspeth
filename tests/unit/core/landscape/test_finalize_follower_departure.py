"""Tests for ADR-030 §D finalize follower-departure flip and
§G enqueue_ready membership fence (slice 5, task f + task e).

§D: complete_run() flips remaining active non-leader run_workers rows to
'departed' inside the finalize IMMEDIATE transaction and emits a
'worker_depart' event with context reason='run_finalized' for each.

§G: enqueue_ready membership fence refuses an evicted OR departed worker
(any non-active row, or absent row) and admits an active one, for both
leader and follower roles (the fence is role-agnostic).

All tests use a real in-memory SQLite Tier-1 engine via make_landscape_db().
Clocks are pinned via the NOW constant.  run_workers rows are seeded via
raw INSERT so liveness TTLs are deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, select, update

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.coordination import mint_worker_id
from elspeth.contracts.errors import RunWorkerEvictedError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.model_loaders import RunLoader
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "run-finalize-follower-test-1"
NOW = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
WINDOW = 80.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_running_run(db: LandscapeDB, *, run_id: str = RUN_ID) -> str:
    """Seed a RUNNING run with a live leader seat. Returns the leader worker_id."""
    leader_id = mint_worker_id(run_id)
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                started_at=NOW,
                config_hash="cfg-hash",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.RUNNING.value,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        conn.execute(
            insert(run_coordination_table).values(
                run_id=run_id,
                leader_worker_id=leader_id,
                leader_epoch=1,
                leader_heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
                updated_at=NOW,
            )
        )
        conn.execute(
            insert(run_workers_table).values(
                worker_id=leader_id,
                run_id=run_id,
                role="leader",
                status="active",
                registered_at=NOW,
                heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
            )
        )
    return leader_id


def _seed_follower(db: LandscapeDB, *, run_id: str = RUN_ID) -> str:
    """Seed an active follower run_workers row. Returns the follower worker_id."""
    follower_id = mint_worker_id(run_id)
    with db.engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=follower_id,
                run_id=run_id,
                role="follower",
                status="active",
                registered_at=NOW,
                heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
            )
        )
    return follower_id


def _get_workers(db: LandscapeDB, run_id: str = RUN_ID) -> list[dict]:
    with db.engine.connect() as conn:
        rows = conn.execute(select(run_workers_table).where(run_workers_table.c.run_id == run_id)).mappings().all()
    return [dict(r) for r in rows]


def _get_coord_events(db: LandscapeDB, run_id: str = RUN_ID) -> list[dict]:
    with db.engine.connect() as conn:
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


def _make_lifecycle(db: LandscapeDB) -> RunLifecycleRepository:
    return RunLifecycleRepository(db, DatabaseOps(db), RunLoader())


def _seed_transform_node(db: LandscapeDB, *, run_id: str = RUN_ID) -> None:
    """Seed a transform node so the scheduler repo can validate references."""
    with db.engine.begin() as conn:
        conn.execute(
            insert(nodes_table).values(
                run_id=run_id,
                node_id="transform-1",
                plugin_name="identity",
                node_type=NodeType.TRANSFORM.value,
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="cfg-hash",
                config_json="{}",
                registered_at=NOW,
            )
        )


def _seed_row_and_token(
    db: LandscapeDB,
    *,
    run_id: str = RUN_ID,
    row_id: str = "row-1",
    token_id: str = "token-1",
    ingest_sequence: int = 0,
) -> None:
    """Seed a rows + tokens row for enqueue_ready reference validation."""
    with db.engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=run_id,
                source_node_id="source-1",
                row_index=ingest_sequence,
                source_row_index=ingest_sequence,
                ingest_sequence=ingest_sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=NOW,
            )
        )
        conn.execute(
            insert(tokens_table).values(
                token_id=token_id,
                row_id=row_id,
                run_id=run_id,
                created_at=NOW,
            )
        )


def _seed_source_node(db: LandscapeDB, *, run_id: str = RUN_ID) -> None:
    with db.engine.begin() as conn:
        conn.execute(
            insert(nodes_table).values(
                run_id=run_id,
                node_id="source-1",
                plugin_name="csv",
                node_type=NodeType.SOURCE.value,
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="cfg-hash",
                config_json="{}",
                registered_at=NOW,
            )
        )


def _row_payload() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


# ---------------------------------------------------------------------------
# §D finalize follower-departure flip
# ---------------------------------------------------------------------------


class TestFinalizeFollowerDeparture:
    """complete_run() atomically departs active follower rows (§D flip)."""

    def test_active_follower_flipped_to_departed_on_complete_run(self) -> None:
        """One active follower: status flipped to 'departed', leader unchanged."""
        db = make_landscape_db()
        _seed_running_run(db)
        follower_id = _seed_follower(db)

        lifecycle = _make_lifecycle(db)
        lifecycle.complete_run(RUN_ID, RunStatus.FAILED)  # FAILED: no quiescence check

        workers = _get_workers(db)
        by_id = {w["worker_id"]: w for w in workers}

        follower_row = by_id[follower_id]
        assert follower_row["status"] == "departed", "follower must be 'departed' after finalize"
        assert follower_row["departed_at"] is not None

    def test_leader_row_not_touched_by_follower_flip(self) -> None:
        """The leader run_workers row is NOT flipped to departed by §D (role filter)."""
        db = make_landscape_db()
        leader_id = _seed_running_run(db)
        _seed_follower(db)

        lifecycle = _make_lifecycle(db)
        lifecycle.complete_run(RUN_ID, RunStatus.FAILED)

        workers = _get_workers(db)
        by_id = {w["worker_id"]: w for w in workers}
        leader_row = by_id[leader_id]
        # Leader row is still 'active' after complete_run — it gets departed by
        # the normal shutdown ceremony (release_seat + depart_worker), not §D.
        assert leader_row["status"] == "active", "§D flip must not touch the leader row"

    def test_worker_depart_event_emitted_per_follower(self) -> None:
        """A 'worker_depart' event with reason='run_finalized' is written per follower."""
        db = make_landscape_db()
        _seed_running_run(db)
        follower_id = _seed_follower(db)

        lifecycle = _make_lifecycle(db)
        lifecycle.complete_run(RUN_ID, RunStatus.FAILED)

        events = _get_coord_events(db)
        depart_events = [e for e in events if e["event_type"] == "worker_depart"]
        assert len(depart_events) == 1, "exactly one worker_depart event per follower"
        evt = depart_events[0]
        assert evt["worker_id"] == follower_id
        assert evt["context_json"] is not None
        import json

        ctx = json.loads(evt["context_json"])
        assert ctx.get("reason") == "run_finalized"

    def test_two_followers_both_departed_and_evented(self) -> None:
        """Multiple active followers: each gets a departed row and event."""
        db = make_landscape_db()
        _seed_running_run(db)
        fid1 = _seed_follower(db)
        fid2 = _seed_follower(db)

        lifecycle = _make_lifecycle(db)
        lifecycle.complete_run(RUN_ID, RunStatus.FAILED)

        workers = _get_workers(db)
        by_id = {w["worker_id"]: w for w in workers}
        assert by_id[fid1]["status"] == "departed"
        assert by_id[fid2]["status"] == "departed"

        events = _get_coord_events(db)
        depart_events = [e for e in events if e["event_type"] == "worker_depart"]
        departed_ids = {e["worker_id"] for e in depart_events}
        assert fid1 in departed_ids and fid2 in departed_ids

    def test_no_op_at_n1_no_followers(self) -> None:
        """N=1 run with no followers: §D flip is a no-op (zero depart events)."""
        db = make_landscape_db()
        _seed_running_run(db)

        lifecycle = _make_lifecycle(db)
        lifecycle.complete_run(RUN_ID, RunStatus.FAILED)

        events = _get_coord_events(db)
        depart_events = [e for e in events if e["event_type"] == "worker_depart"]
        assert len(depart_events) == 0, "no follower depart events for N=1 run"

    def test_already_departed_follower_not_double_evented(self) -> None:
        """A follower that already departed (e.g., clean exit) is not re-departed."""
        db = make_landscape_db()
        _seed_running_run(db)
        follower_id = _seed_follower(db)

        # Pre-depart the follower (simulates a clean follower exit before finalize).
        with db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table).where(run_workers_table.c.worker_id == follower_id).values(status="departed", departed_at=NOW)
            )

        lifecycle = _make_lifecycle(db)
        lifecycle.complete_run(RUN_ID, RunStatus.FAILED)

        # The §D WHERE status='active' filter excludes already-departed rows.
        events = _get_coord_events(db)
        depart_events = [e for e in events if e["event_type"] == "worker_depart"]
        assert len(depart_events) == 0, "already-departed follower must not generate a second event"


# ---------------------------------------------------------------------------
# §G enqueue_ready membership fence — leader and follower roles
# ---------------------------------------------------------------------------


class TestEnqueueReadyMembershipFenceRoles:
    """enqueue_ready membership fence is role-agnostic (§G verb table).

    The fence checks run_workers.status == 'active' — it does not filter by
    role. Tests explicitly cover both leader and follower role seeds to pin
    that the fence holds for each.
    """

    def _make_scheduler(self, db: LandscapeDB) -> TokenSchedulerRepository:
        return TokenSchedulerRepository(db.engine)

    def _enqueue(
        self,
        repo: TokenSchedulerRepository,
        *,
        run_id: str = RUN_ID,
        token_id: str = "token-1",
        row_id: str = "row-1",
        worker_id: str | None = None,
        ingest_sequence: int = 0,
    ) -> None:
        repo.enqueue_ready(
            run_id=run_id,
            token_id=token_id,
            row_id=row_id,
            node_id="transform-1",
            step_index=1,
            ingest_sequence=ingest_sequence,
            row_payload_json=_row_payload(),
            available_at=NOW,
            worker_id=worker_id,
        )

    def _setup_db(self) -> LandscapeDB:
        db = make_landscape_db()
        _seed_running_run(db)
        _seed_source_node(db)
        _seed_transform_node(db)
        _seed_row_and_token(db)
        return db

    def _seed_worker(
        self,
        db: LandscapeDB,
        *,
        worker_id: str,
        role: str,
        status: str,
        run_id: str = RUN_ID,
    ) -> None:
        with db.engine.begin() as conn:
            conn.execute(
                insert(run_workers_table).values(
                    worker_id=worker_id,
                    run_id=run_id,
                    role=role,
                    status=status,
                    registered_at=NOW,
                    heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
                    evicted_at=NOW if status == "evicted" else None,
                    evicted_by_worker_id="evictor" if status == "evicted" else None,
                )
            )

    def test_active_leader_can_enqueue(self) -> None:
        """An active LEADER row passes the membership fence."""
        db = self._setup_db()
        leader_id = f"worker:{RUN_ID}:leader-test"
        self._seed_worker(db, worker_id=leader_id, role="leader", status="active")

        repo = self._make_scheduler(db)
        # Must not raise.
        self._enqueue(repo, worker_id=leader_id)

    def test_active_follower_can_enqueue(self) -> None:
        """An active FOLLOWER row passes the membership fence."""
        db = self._setup_db()
        # Seed a second row+token for the follower to enqueue (avoids
        # work_item_id collision with the active-leader test).
        _seed_row_and_token(db, row_id="row-f", token_id="token-f", ingest_sequence=1)
        follower_id = f"worker:{RUN_ID}:follower-test"
        self._seed_worker(db, worker_id=follower_id, role="follower", status="active")

        repo = self._make_scheduler(db)
        # Must not raise.
        self._enqueue(repo, worker_id=follower_id, token_id="token-f", row_id="row-f", ingest_sequence=1)

    def test_evicted_leader_raises(self) -> None:
        """An EVICTED leader row is refused (RunWorkerEvictedError)."""
        db = self._setup_db()
        leader_id = f"worker:{RUN_ID}:leader-evicted"
        self._seed_worker(db, worker_id=leader_id, role="leader", status="evicted")

        repo = self._make_scheduler(db)
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            self._enqueue(repo, worker_id=leader_id)

        assert exc_info.value.worker_id == leader_id
        assert exc_info.value.run_id == RUN_ID

    def test_evicted_follower_raises(self) -> None:
        """An EVICTED follower row is refused (RunWorkerEvictedError)."""
        db = self._setup_db()
        follower_id = f"worker:{RUN_ID}:follower-evicted"
        self._seed_worker(db, worker_id=follower_id, role="follower", status="evicted")

        repo = self._make_scheduler(db)
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            self._enqueue(repo, worker_id=follower_id)

        assert exc_info.value.worker_id == follower_id

    def test_departed_follower_raises(self) -> None:
        """A DEPARTED follower row (finalize-flipped) is refused after departure."""
        db = self._setup_db()
        follower_id = f"worker:{RUN_ID}:follower-departed"
        self._seed_worker(db, worker_id=follower_id, role="follower", status="departed")

        repo = self._make_scheduler(db)
        with pytest.raises(RunWorkerEvictedError):
            self._enqueue(repo, worker_id=follower_id)

    def test_absent_worker_raises(self) -> None:
        """A worker_id with NO run_workers row is refused (absent = fence fails)."""
        db = self._setup_db()
        absent_id = f"worker:{RUN_ID}:absent-worker"

        repo = self._make_scheduler(db)
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            self._enqueue(repo, worker_id=absent_id)

        assert exc_info.value.worker_id == absent_id

    def test_no_worker_id_bypasses_fence(self) -> None:
        """worker_id=None preserves the legacy unfenced path (N=1 / tests)."""
        db = self._setup_db()

        repo = self._make_scheduler(db)
        # Must not raise even though there are no run_workers rows.
        self._enqueue(repo, worker_id=None)

    def test_eviction_leaves_zero_ready_rows(self) -> None:
        """An evicted worker's enqueue attempt leaves NO READY row in the DB."""
        from elspeth.core.landscape.schema import token_work_items_table

        db = self._setup_db()
        follower_id = f"worker:{RUN_ID}:follower-evicted-2"
        self._seed_worker(db, worker_id=follower_id, role="follower", status="evicted")

        repo = self._make_scheduler(db)
        with pytest.raises(RunWorkerEvictedError):
            self._enqueue(repo, worker_id=follower_id)

        # The fence fires BEFORE the INSERT — zero mutation.
        with db.engine.connect() as conn:
            count = conn.execute(select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.run_id == RUN_ID)).fetchall()
        assert len(count) == 0, "evicted enqueue must leave zero READY rows (fence before INSERT)"
