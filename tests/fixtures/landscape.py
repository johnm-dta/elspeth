# tests/fixtures/landscape.py
"""Landscape database and RecorderFactory fixtures.

All fixtures are function-scoped for full test isolation.
No module-scoped databases — every test gets a fresh database.

Factory hierarchy:
    make_landscape_db()          → bare LandscapeDB
    make_factory()               → LandscapeDB + RecorderFactory
    make_recorder_with_run()     → LandscapeDB + RecorderFactory + run + source node
    register_test_node()         → add additional nodes to an existing run
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.payload_store import PayloadStore
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from tests.fixtures.stores import MockPayloadStore

# Shared default for schema_config across all factory-created nodes
_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def make_landscape_db() -> LandscapeDB:
    """Factory for in-memory LandscapeDB."""
    return LandscapeDB.in_memory()


def expire_leader_seat(db: LandscapeDB, run_id: str) -> None:
    """Lapse the epoch-21 leader seat ``begin_run`` minted for ``run_id``.

    ADR-030 §B.4: a hard-killed leader never releases its ``run_coordination``
    seat — it stays HELD until the liveness window (80 s) lapses, and resume's
    takeover CAS requires vacant-or-expired. Fixtures that craft a crashed run
    via ``begin_run(...)`` + direct status writes (instead of running the real
    engine, whose ceremony arms release the seat) call this to produce the
    post-window image deterministically rather than sleeping out the window.
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update

    from elspeth.core.landscape.schema import run_coordination_table

    with db.engine.begin() as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == run_id)
            .values(leader_heartbeat_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )


def insert_crashed_leader_seat(conn: Any, *, run_id: str) -> None:
    """Insert the expired ``run_coordination`` seat row a crashed leader leaves.

    For fixtures that craft the ``runs`` row via raw SQL (bypassing
    ``begin_run``, which at epoch 21 mints the seat atomically with the run):
    without a seat row, resume's takeover CAS refuses with
    ``AuditIntegrityError`` ("no run_coordination seat row"). Call on the same
    connection/transaction that inserted the ``runs`` row (FK).
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import insert as sa_insert

    from elspeth.core.landscape.schema import run_coordination_table

    lapsed = datetime.now(UTC) - timedelta(seconds=1)
    conn.execute(
        sa_insert(run_coordination_table).values(
            run_id=run_id,
            leader_worker_id=f"worker:{run_id}:crashed-leader",
            leader_epoch=1,
            leader_heartbeat_expires_at=lapsed,
            updated_at=lapsed,
        )
    )


def leader_coordination_token(factory: RecorderFactory, run_id: str) -> CoordinationToken:
    """The run's OWN leader token (the epoch-1 seat ``begin_run`` minted).

    ADR-030 slice 3: the journal-first barrier intake's adoption verbs are
    leader-fenced with NO unfenced arm, so any test that drives barrier work
    through a directly-constructed ``RowProcessor`` must bind the coordination
    token. ``begin_run`` always mints the seat (self-minted worker identity at
    epoch 1); this helper reads it back. The fence predicate is identity+epoch
    only — an expired seat still passes its own leader's fence.
    """
    from datetime import UTC, datetime

    leader = factory.run_coordination.live_leader(run_id=run_id, now=datetime.now(UTC))
    if leader is None:
        raise AssertionError(f"run {run_id!r} has no run_coordination seat; begin_run mints one — was the run created via raw SQL?")
    return CoordinationToken(run_id=run_id, worker_id=leader.leader_worker_id, leader_epoch=leader.leader_epoch)


def make_factory(db: LandscapeDB | None = None, *, payload_store: PayloadStore | None = None) -> RecorderFactory:
    """Factory for RecorderFactory.

    Always wires a payload store so expand_token / coalesce_tokens can
    persist per-token payloads (required since epoch 11). Tests that don't
    care about the stored bytes get a fresh MockPayloadStore automatically.
    Pass an explicit payload_store to inspect stored payloads in assertions.
    """
    if db is None:
        db = make_landscape_db()
    if payload_store is None:
        payload_store = MockPayloadStore()
    return RecorderFactory(db, payload_store=payload_store)


# =============================================================================
# RecorderSetup — The 80% setup pattern as a single factory call
# =============================================================================


@dataclass
class RecorderSetup:
    """Result from make_recorder_with_run().

    Plain @dataclass — test scaffolding, not audit records.
    Note: db and factory are mutable objects; frozen=True would only prevent
    reference reassignment without providing an immutability guarantee.
    """

    db: LandscapeDB
    factory: RecorderFactory
    run_id: str
    source_node_id: str

    @property
    def run_lifecycle(self) -> RunLifecycleRepository:
        return self.factory.run_lifecycle

    @property
    def execution(self) -> ExecutionRepository:
        return self.factory.execution

    @property
    def data_flow(self) -> DataFlowRepository:
        return self.factory.data_flow

    @property
    def query(self) -> QueryRepository:
        return self.factory.query


def make_recorder_with_run(
    *,
    run_id: str | None = None,
    source_node_id: str | None = None,
    source_plugin_name: str = "source",
    canonical_version: str = "v1",
    payload_store: PayloadStore | None = None,
) -> RecorderSetup:
    """Create LandscapeDB + RecorderFactory + run + source node in one call.

    Covers the 80% setup pattern: db → factory → begin_run → register_node(SOURCE).
    Tests needing additional nodes (transforms, sinks, aggregations) can call
    factory.data_flow.register_node() on the returned factory, or use register_test_node().

    Always call this inside individual test methods or setup_method(), never
    setup_class(). It creates a fresh in-memory DB per call for test isolation.

    Args:
        run_id: Explicit run ID for deterministic tests. Auto-generated if None.
        source_node_id: Explicit source node ID. Auto-generated if None.
        source_plugin_name: Plugin name for the source node (default "source").
        canonical_version: Version string for begin_run (default "v1").
            Some tests (e.g., test_processor.py) use "sha256-rfc8785-v1".
        payload_store: Payload store to inject (defaults to MockPayloadStore).
            Pass a specific MockPayloadStore instance to inspect stored payloads.
            Tests that explicitly test "no payload store" behavior must pass
            RecorderFactory(db) directly rather than using this helper.
    """
    db = make_landscape_db()
    factory = make_factory(db, payload_store=payload_store)

    # Build kwargs, only passing explicit IDs if provided
    begin_kwargs: dict[str, Any] = {
        "config": {},
        "canonical_version": canonical_version,
    }
    if run_id is not None:
        begin_kwargs["run_id"] = run_id

    run = factory.run_lifecycle.begin_run(**begin_kwargs)

    register_kwargs: dict[str, Any] = {
        "run_id": run.run_id,
        "plugin_name": source_plugin_name,
        "node_type": NodeType.SOURCE,
        "plugin_version": "1.0",
        "config": {},
        "schema_config": _OBSERVED_SCHEMA,
    }
    if source_node_id is not None:
        register_kwargs["node_id"] = source_node_id

    node = factory.data_flow.register_node(**register_kwargs)

    setup = RecorderSetup(
        db=db,
        factory=factory,
        run_id=run.run_id,
        source_node_id=node.node_id,
    )

    # Offensive programming: verify round-trip invariant.
    # If this assertion fails, the factory itself is broken.
    assert setup.run_id == run.run_id, f"Factory bug: returned run_id {setup.run_id!r} != begin_run result {run.run_id!r}"
    assert setup.source_node_id == node.node_id, (
        f"Factory bug: returned source_node_id {setup.source_node_id!r} != register_node result {node.node_id!r}"
    )

    return setup


def register_test_node(
    data_flow: DataFlowRepository,
    run_id: str,
    node_id: str,
    *,
    node_type: NodeType = NodeType.TRANSFORM,
    plugin_name: str = "transform",
) -> str:
    """Register an additional test node with sensible defaults.

    For the 20% variant pattern where tests need 2-5 additional nodes
    after make_recorder_with_run() creates the source.

    Defaults plugin_version="1.0", config={}, schema_config=observed.
    Returns the node_id for convenience.
    """
    node = data_flow.register_node(
        run_id=run_id,
        plugin_name=plugin_name,
        node_type=node_type,
        plugin_version="1.0",
        config={},
        node_id=node_id,
        schema_config=_OBSERVED_SCHEMA,
    )
    return node.node_id


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture
def landscape_db() -> LandscapeDB:
    """Function-scoped in-memory LandscapeDB — fresh per test."""
    return make_landscape_db()


@pytest.fixture
def landscape_factory(landscape_db: LandscapeDB) -> RecorderFactory:
    """Function-scoped RecorderFactory."""
    return RecorderFactory(landscape_db)


@pytest.fixture
def landscape_factory_with_payload_store(landscape_db: LandscapeDB, tmp_path: Any) -> RecorderFactory:
    """RecorderFactory with real filesystem payload store."""
    from elspeth.core.payload_store import FilesystemPayloadStore

    payload_dir = tmp_path / "payloads"
    payload_store = FilesystemPayloadStore(payload_dir)
    return RecorderFactory(landscape_db, payload_store=payload_store)
