"""Statement-order regression for membership-fence / eviction serialization.

Ticket elspeth-6903f82511: the membership fence rode the claim/heartbeat CAS
UPDATE as an unlocked EXISTS predicate, and ``evict_worker`` read its
no-unexpired-leases precondition before touching the target's ``run_workers``
row.  On PostgreSQL (MVCC, no lock from predicate reads) that let a fenced
claim or heartbeat renewal commit around an in-flight eviction — leaving an
evicted worker holding a live lease.

The serialization contract is an acquisition ORDER, observable as statement
order on any dialect:

* fenced claim/heartbeat verbs issue a ``run_workers`` membership row lock
  (``SELECT … FOR SHARE`` on PostgreSQL; the SELECT compiles lock-free on
  SQLite where ``BEGIN IMMEDIATE`` already serializes writers) BEFORE their
  ``token_work_items`` CAS UPDATE;
* ``evict_worker`` locks the target's ``run_workers`` row BEFORE its
  ``token_work_items`` live-lease precondition read.

The real PostgreSQL interleaving proof lives in
``tests/testcontainer/core/test_scheduler_lease_eviction_postgres.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, event, insert
from sqlalchemy.engine import Engine

from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-fence-order"
NOW = datetime(2026, 7, 16, 10, 0, 0, tzinfo=UTC)
WINDOW = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
GRACE = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
LEADER_ID = "leader-w"
WORKER_ID = "worker-w"


def _make_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _seed(engine: Tier1Engine, *, worker_heartbeat_expires_at: datetime) -> CoordinationToken:
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type, plugin in (("source-a", "source", "csv"), ("transform-1", "transform", "identity")):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name=plugin,
                    node_type=node_type,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=NOW,
                )
            )
        conn.execute(
            insert(run_coordination_table).values(
                run_id=RUN_ID,
                leader_worker_id=LEADER_ID,
                leader_epoch=1,
                leader_heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
                updated_at=NOW,
            )
        )
        conn.execute(
            insert(run_workers_table).values(
                worker_id=LEADER_ID,
                run_id=RUN_ID,
                role="leader",
                status="active",
                registered_at=NOW,
                heartbeat_expires_at=NOW + timedelta(seconds=WINDOW),
            )
        )
        conn.execute(
            insert(run_workers_table).values(
                worker_id=WORKER_ID,
                run_id=RUN_ID,
                role="follower",
                status="active",
                registered_at=NOW,
                heartbeat_expires_at=worker_heartbeat_expires_at,
            )
        )
    return CoordinationToken(run_id=RUN_ID, worker_id=LEADER_ID, leader_epoch=1)


def _enqueue_ready_item(engine: Tier1Engine, *, token_id: str = "tok-1") -> str:
    row_id = f"row-{token_id}"
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id="source-a",
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash=f"hash-{token_id}",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=RUN_ID, created_at=NOW))
    repo = TokenSchedulerRepository(engine)
    payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
    repo.enqueue_ready(
        run_id=RUN_ID,
        token_id=token_id,
        row_id=row_id,
        node_id="transform-1",
        step_index=1,
        ingest_sequence=0,
        row_payload_json=payload,
        available_at=NOW,
    )
    return row_id


@contextmanager
def _recorded_statements(engine: Engine) -> Iterator[list[str]]:
    statements: list[str] = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        statements.append(" ".join(statement.upper().split()))

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def _first_index(statements: list[str], *, startswith: str, contains: str) -> int:
    for index, statement in enumerate(statements):
        if statement.startswith(startswith) and contains in statement:
            return index
    raise AssertionError(f"no statement starting with {startswith!r} containing {contains!r} in: {statements}")


def _assert_membership_lock_before_cas(statements: list[str], *, verb: str) -> None:
    membership_read = _first_index(statements, startswith="SELECT", contains="FROM RUN_WORKERS")
    cas_update = _first_index(statements, startswith="UPDATE TOKEN_WORK_ITEMS", contains="TOKEN_WORK_ITEMS")
    assert membership_read < cas_update, (
        f"{verb} must lock the caller's run_workers membership row BEFORE its "
        "token_work_items CAS UPDATE so the fence serializes with worker "
        f"eviction (elspeth-6903f82511); observed statements: {statements}"
    )


class TestMembershipFenceLockOrder:
    """Fenced lease verbs lock the membership row before their CAS UPDATE."""

    def test_claim_ready_locks_membership_row_before_cas(self) -> None:
        engine = _make_engine()
        _seed(engine, worker_heartbeat_expires_at=NOW + timedelta(seconds=WINDOW))
        _enqueue_ready_item(engine)
        repo = TokenSchedulerRepository(engine)

        with _recorded_statements(engine) as statements:
            item = repo.claim_ready(run_id=RUN_ID, lease_owner=WORKER_ID, lease_seconds=300, now=NOW)
        assert item is not None
        _assert_membership_lock_before_cas(statements, verb="claim_ready")

    def test_heartbeat_lease_locks_membership_row_before_cas(self) -> None:
        engine = _make_engine()
        _seed(engine, worker_heartbeat_expires_at=NOW + timedelta(seconds=WINDOW))
        _enqueue_ready_item(engine)
        repo = TokenSchedulerRepository(engine)
        item = repo.claim_ready(run_id=RUN_ID, lease_owner=WORKER_ID, lease_seconds=300, now=NOW)
        assert item is not None

        with _recorded_statements(engine) as statements:
            renewed = repo.heartbeat_lease(
                run_id=RUN_ID,
                work_item_id=item.work_item_id,
                lease_owner=WORKER_ID,
                lease_seconds=300,
                now=NOW + timedelta(seconds=5),
                membership_fenced=True,
            )
        assert renewed == NOW + timedelta(seconds=305)
        _assert_membership_lock_before_cas(statements, verb="heartbeat_lease")

    def test_claim_pending_sink_locks_membership_row_before_cas(self) -> None:
        engine = _make_engine()
        _seed(engine, worker_heartbeat_expires_at=NOW + timedelta(seconds=WINDOW))
        _enqueue_ready_item(engine)
        repo = TokenSchedulerRepository(engine)
        item = repo.claim_ready(run_id=RUN_ID, lease_owner=WORKER_ID, lease_seconds=300, now=NOW)
        assert item is not None
        # Promote the leased row to a complete durable PENDING_SINK bundle
        # (same promotion shape as the elspeth-28aaa36a62 ABA regression).
        with engine.begin() as conn:
            conn.execute(
                token_work_items_table.update()
                .where(token_work_items_table.c.work_item_id == item.work_item_id)
                .values(
                    status="pending_sink",
                    pending_sink_name="sink-a",
                    pending_outcome="success",
                    pending_path="default_flow",
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )

        with _recorded_statements(engine) as statements:
            claimed = repo.claim_pending_sink(run_id=RUN_ID, lease_owner=WORKER_ID, lease_seconds=300, now=NOW)
        assert claimed is not None
        _assert_membership_lock_before_cas(statements, verb="claim_pending_sink")


class TestEvictWorkerLockOrder:
    """evict_worker locks the target row before its live-lease precondition."""

    def test_evict_worker_locks_target_row_before_live_lease_read(self) -> None:
        engine = _make_engine()
        token = _seed(engine, worker_heartbeat_expires_at=NOW - timedelta(seconds=GRACE + 10))
        coord = RunCoordinationRepository(engine)

        with _recorded_statements(engine) as statements:
            evicted = coord.evict_worker(
                token=token,
                target_worker_id=WORKER_ID,
                now=NOW,
                grace_seconds=GRACE,
                window_seconds=WINDOW,
            )
        assert evicted is True

        membership_read = _first_index(statements, startswith="SELECT", contains="FROM RUN_WORKERS")
        live_lease_read = _first_index(statements, startswith="SELECT", contains="FROM TOKEN_WORK_ITEMS")
        assert membership_read < live_lease_read, (
            "evict_worker must lock the target's run_workers row BEFORE its "
            "no-unexpired-leases precondition read so in-flight fenced lease "
            "writes are committed-and-visible or blocked "
            f"(elspeth-6903f82511); observed statements: {statements}"
        )

    def test_evict_worker_absent_target_is_benign_skip(self) -> None:
        engine = _make_engine()
        token = _seed(engine, worker_heartbeat_expires_at=NOW - timedelta(seconds=GRACE + 10))
        coord = RunCoordinationRepository(engine)

        evicted = coord.evict_worker(
            token=token,
            target_worker_id="never-registered",
            now=NOW,
            grace_seconds=GRACE,
            window_seconds=WINDOW,
        )
        assert evicted is False
