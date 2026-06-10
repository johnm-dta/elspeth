"""Unit tests for CheckpointManager unhappy paths and ordering behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection

from elspeth.contracts import Checkpoint, Determinism, NodeType, RunStatus
from elspeth.contracts.coalesce_checkpoint import (
    CoalesceCheckpointState,
    CoalescePendingCheckpoint,
    CoalesceTokenCheckpoint,
)
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.checkpoint.manager import (
    CheckpointManager,
    IncompatibleCheckpointError,
    _validate_checkpoint_state_json_size,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table, tokens_table
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


def test_create_checkpoint_requires_graph(checkpoint_manager: CheckpointManager) -> None:
    with pytest.raises(ValueError, match="graph parameter is required"):
        checkpoint_manager.create_checkpoint(
            run_id="run-001",
            sequence_number=1,
            graph=None,  # type: ignore[arg-type]  # testing None rejection
        )


def test_validate_checkpoint_state_json_size_rejects_large_aggregation_state() -> None:
    """Serialized aggregation-size guard lives at the manager boundary."""
    with pytest.raises(OrchestrationInvariantError, match="exceeds 10MB limit"):
        _validate_checkpoint_state_json_size(
            state_name="aggregation",
            serialized="x" * 10_000_001,
            total_rows=7,
            node_count=2,
        )


def test_validate_checkpoint_state_json_size_rejects_large_coalesce_state() -> None:
    """Serialized coalesce-size guard lives at the manager boundary."""
    with pytest.raises(RuntimeError, match=r"Coalesce checkpoint size .* exceeds 10MB limit"):
        _validate_checkpoint_state_json_size(
            state_name="coalesce",
            serialized="x" * 10_000_001,
            pending_joins=3,
        )


def test_get_checkpoints_returns_ascending_sequence_order(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint_manager.create_checkpoint("run-001", 5, graph)
    checkpoint_manager.create_checkpoint("run-001", 1, graph)
    checkpoint_manager.create_checkpoint("run-001", 3, graph)

    checkpoints = checkpoint_manager.get_checkpoints("run-001")
    assert [cp.sequence_number for cp in checkpoints] == [1, 3, 5]


def test_create_checkpoint_rejects_duplicate_sequence_for_run(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    """Duplicate per-run checkpoint sequence numbers would make resume order ambiguous."""
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint_manager.create_checkpoint("run-001", 1, graph)

    with pytest.raises(OrchestrationInvariantError, match="Duplicate checkpoint sequence_number"):
        checkpoint_manager.create_checkpoint("run-001", 1, graph)


def test_create_checkpoint_round_trips_coalesce_state(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
    with db.connection() as conn:
        _insert_checkpoint_prereqs(conn)

    graph = make_graph_linear("node-001")
    checkpoint = checkpoint_manager.create_checkpoint(
        "run-001",
        1,
        graph,
        coalesce_state=CoalesceCheckpointState(
            version="1.0",
            pending=(
                CoalescePendingCheckpoint(
                    coalesce_name="merge_paths",
                    row_id="row-001",
                    elapsed_age_seconds=2.0,
                    branches={
                        "branch_a": CoalesceTokenCheckpoint(
                            token_id="tok-001",
                            row_id="row-001",
                            branch_name="branch_a",
                            fork_group_id="fork-001",
                            join_group_id=None,
                            expand_group_id=None,
                            row_data={"value": 1},
                            contract={"mode": "OBSERVED", "locked": True, "fields": [], "version_hash": "vh-1"},
                            state_id="state-001",
                            arrival_offset_seconds=0.0,
                        )
                    },
                    lost_branches={},
                ),
            ),
            completed_keys=(),
        ),
    )

    assert checkpoint.coalesce_state_json is not None
    loaded = checkpoint_manager.get_latest_checkpoint("run-001")
    assert loaded is not None
    assert loaded.coalesce_state_json == checkpoint.coalesce_state_json


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
    checkpoint_manager.create_checkpoint("run-001", 1, graph)
    checkpoint_manager.create_checkpoint("run-001", 2, graph)

    deleted = checkpoint_manager.delete_checkpoints("run-001")
    assert deleted == 2
    assert checkpoint_manager.get_checkpoints("run-001") == []
