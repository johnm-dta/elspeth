"""Unit tests for CheckpointManager unhappy paths and ordering behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import Connection, select
from sqlalchemy.engine import Row

from elspeth.contracts import Checkpoint, Determinism, NodeType, RunStatus
from elspeth.contracts.barrier_scalars import (
    AggregationNodeScalars,
    BarrierScalars,
    CoalescePendingScalars,
)
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.checkpoint.manager import (
    CheckpointManager,
    IncompatibleCheckpointError,
    _validate_barrier_scalars_json_size,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import checkpoints_table, nodes_table, rows_table, runs_table, tokens_table
from tests.fixtures.factories import make_graph_linear
from tests.fixtures.landscape import make_landscape_db


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


@pytest.fixture
def checkpoint_manager(db: LandscapeDB) -> CheckpointManager:
    return CheckpointManager(db)


def _insert_checkpoint_prereqs(
    conn: Connection,
    *,
    run_id: str = "run-001",
    node_id: str = "node-001",
    row_id: str = "row-001",
    token_id: str = "tok-001",
) -> None:
    now = datetime.now(UTC)
    conn.execute(
        runs_table.insert().values(
            run_id=run_id,
            started_at=now,
            config_hash="cfg",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.RUNNING,
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
    )
    conn.execute(
        nodes_table.insert().values(
            node_id=node_id,
            run_id=run_id,
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="node_cfg",
            config_json="{}",
            registered_at=now,
        )
    )
    conn.execute(
        rows_table.insert().values(
            row_id=row_id,
            run_id=run_id,
            source_node_id=node_id,
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_data_hash="hash",
            created_at=now,
        )
    )
    conn.execute(
        tokens_table.insert().values(
            token_id=token_id,
            row_id=row_id,
            run_id=run_id,
            created_at=now,
        )
    )


def _select_checkpoint(db: LandscapeDB, checkpoint_id: str) -> Row[Any]:
    with db.engine.connect() as conn:
        row = conn.execute(select(checkpoints_table).where(checkpoints_table.c.checkpoint_id == checkpoint_id)).fetchone()
    assert row is not None
    return row


def test_create_checkpoint_requires_graph(checkpoint_manager: CheckpointManager) -> None:
    with pytest.raises(ValueError, match="graph parameter is required"):
        checkpoint_manager.create_checkpoint(
            run_id="run-001",
            sequence_number=1,
            barrier_scalars=None,
            graph=None,  # type: ignore[arg-type]  # testing None rejection
        )


def test_validate_barrier_scalars_json_size_rejects_large_payload() -> None:
    """Serialized barrier-scalars size guard lives at the manager boundary (hard-fail)."""
    with pytest.raises(OrchestrationInvariantError, match="exceeds 10MB limit"):
        _validate_barrier_scalars_json_size("x" * 10_000_001)


def test_create_checkpoint_persists_barrier_scalars(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    """F1 Task 1.2: the checkpoint row carries BarrierScalars only (format_version 5)."""
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    scalars = BarrierScalars(
        aggregation={"agg-1": AggregationNodeScalars(count_fire_offset=2.25, condition_fire_offset=None)},
        coalesce={},
    )
    cp = checkpoint_manager.create_checkpoint(
        run_id="run-001",
        sequence_number=1,
        barrier_scalars=scalars,
        graph=graph,
    )

    row = _select_checkpoint(db, cp.checkpoint_id)
    assert row.barrier_scalars_json is not None
    persisted = json.loads(row.barrier_scalars_json)
    # Wire shape: per-node envelope carries _version alongside the offsets.
    assert persisted["aggregation"]["agg-1"]["count_fire_offset"] == 2.25
    assert persisted["aggregation"]["agg-1"]["condition_fire_offset"] is None
    assert cp.format_version == 5
    assert cp.barrier_scalars_json == row.barrier_scalars_json


def test_create_checkpoint_empty_scalars_persist_null(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    """Empty BarrierScalars (has_state=False) persists NULL, same as None."""
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    cp_none = checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=1, barrier_scalars=None, graph=graph)
    cp_empty = checkpoint_manager.create_checkpoint(
        run_id="run-001",
        sequence_number=2,
        barrier_scalars=BarrierScalars(aggregation={}, coalesce={}),
        graph=graph,
    )

    assert cp_none.barrier_scalars_json is None
    assert cp_empty.barrier_scalars_json is None
    assert _select_checkpoint(db, cp_none.checkpoint_id).barrier_scalars_json is None
    assert _select_checkpoint(db, cp_empty.checkpoint_id).barrier_scalars_json is None


def test_get_checkpoints_returns_ascending_sequence_order(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=5, barrier_scalars=None, graph=graph)
    checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=1, barrier_scalars=None, graph=graph)
    checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=3, barrier_scalars=None, graph=graph)

    checkpoints = checkpoint_manager.get_checkpoints("run-001")
    assert [cp.sequence_number for cp in checkpoints] == [1, 3, 5]


def test_create_checkpoint_rejects_duplicate_sequence_for_run(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    """Duplicate per-run checkpoint sequence numbers would make resume order ambiguous."""
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=1, barrier_scalars=None, graph=graph)

    with pytest.raises(OrchestrationInvariantError, match="Duplicate checkpoint sequence_number"):
        checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=1, barrier_scalars=None, graph=graph)


def test_create_checkpoint_round_trips_coalesce_scalars(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    """Coalesce lost-branch scalars round-trip through persistence unchanged."""
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint = checkpoint_manager.create_checkpoint(
        run_id="run-001",
        sequence_number=1,
        barrier_scalars=BarrierScalars(
            aggregation={},
            coalesce={("merge_paths", "row-001"): CoalescePendingScalars(lost_branches={"branch_b": "timeout"})},
        ),
        graph=graph,
    )

    assert checkpoint.barrier_scalars_json is not None
    loaded = checkpoint_manager.get_latest_checkpoint("run-001")
    assert loaded is not None
    assert loaded.barrier_scalars_json == checkpoint.barrier_scalars_json
    restored = BarrierScalars.from_dict(json.loads(loaded.barrier_scalars_json))
    assert restored.coalesce[("merge_paths", "row-001")].lost_branches == {"branch_b": "timeout"}


def test_validate_checkpoint_compatibility_rejects_missing_format_version(checkpoint_manager: CheckpointManager) -> None:
    checkpoint = Checkpoint(
        checkpoint_id="cp-test",
        run_id="run-001",
        sequence_number=1,
        created_at=datetime.now(UTC),
        upstream_topology_hash="a" * 64,
        format_version=None,
    )

    with pytest.raises(IncompatibleCheckpointError, match="missing format_version"):
        checkpoint_manager._validate_checkpoint_compatibility(checkpoint)


@pytest.mark.parametrize(
    "version",
    [Checkpoint.CURRENT_FORMAT_VERSION - 1, Checkpoint.CURRENT_FORMAT_VERSION + 1],
)
def test_validate_checkpoint_compatibility_rejects_mismatched_version(checkpoint_manager: CheckpointManager, version: int) -> None:
    checkpoint = Checkpoint(
        checkpoint_id="cp-test",
        run_id="run-001",
        sequence_number=1,
        created_at=datetime.now(UTC),
        upstream_topology_hash="a" * 64,
        format_version=version,
    )

    with pytest.raises(IncompatibleCheckpointError, match="incompatible format version"):
        checkpoint_manager._validate_checkpoint_compatibility(checkpoint)


def test_delete_checkpoints_removes_all_for_run(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=1, barrier_scalars=None, graph=graph)
    checkpoint_manager.create_checkpoint(run_id="run-001", sequence_number=2, barrier_scalars=None, graph=graph)

    deleted = checkpoint_manager.delete_checkpoints("run-001")
    assert deleted == 2
    assert checkpoint_manager.get_checkpoints("run-001") == []
