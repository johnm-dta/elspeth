"""Shared fixtures for checkpoint tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.contracts import Determinism, NodeType, RowOutcome, RunStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore


def _create_test_graph(checkpoint_node: str = "sink-node") -> ExecutionGraph:
    """Create a minimal test graph for checkpoint tests.

    Args:
        checkpoint_node: The node ID where checkpoint will be created.
                        If not in default nodes, it will be added as a sink.
    """
    graph = ExecutionGraph()
    graph.add_node("source-node", node_type=NodeType.SOURCE, plugin_name="test-source", config={})
    graph.add_node("transform-node", node_type=NodeType.TRANSFORM, plugin_name="test-transform", config={})

    # Add the checkpoint node if it's custom
    if checkpoint_node not in ["source-node", "transform-node", "sink-node"]:
        graph.add_node(checkpoint_node, node_type=NodeType.SINK, plugin_name="test-sink", config={})
        graph.add_edge("source-node", "transform-node", label="continue")
        graph.add_edge("transform-node", checkpoint_node, label="continue")
    else:
        graph.add_node("sink-node", node_type=NodeType.SINK, plugin_name="test-sink", config={})
        graph.add_edge("source-node", "transform-node", label="continue")
        graph.add_edge("transform-node", "sink-node", label="continue")

    return graph


@pytest.fixture
def db(tmp_path: Path) -> LandscapeDB:
    """Create test database."""
    return LandscapeDB(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture
def payload_store(tmp_path: Path) -> FilesystemPayloadStore:
    """Create test payload store."""
    return FilesystemPayloadStore(tmp_path / "payloads")


@pytest.fixture
def checkpoint_manager(db: LandscapeDB) -> CheckpointManager:
    """Create checkpoint manager."""
    return CheckpointManager(db)


@pytest.fixture
def recovery_manager(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
    """Create recovery manager."""
    return RecoveryManager(db, checkpoint_manager)


@pytest.fixture
def run_with_checkpoint_and_payloads(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    payload_store: FilesystemPayloadStore,
) -> str:
    """Create a failed run with checkpoint and payload data.

    Creates 5 rows (0-4), checkpoint at row 2, so rows 3-4 are unprocessed.
    All rows have payload data stored.
    """
    run_id = "test-run-resume"
    now = datetime.now(UTC)

    with db.engine.connect() as conn:
        # Create run (failed status)
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=RunStatus.FAILED,
            )
        )

        # Create source node
        conn.execute(
            nodes_table.insert().values(
                node_id="source-node",
                run_id=run_id,
                plugin_name="csv",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.IO_READ,
                config_hash="x",
                config_json="{}",
                registered_at=now,
            )
        )

        # Create sink node (needed for token outcomes)
        conn.execute(
            nodes_table.insert().values(
                node_id="sink-node",
                run_id=run_id,
                plugin_name="csv_sink",
                node_type=NodeType.SINK,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="x",
                config_json="{}",
                registered_at=now,
            )
        )

        # Create rows with payload data
        for i in range(5):
            row_id = f"row-{i:03d}"
            row_data = {"id": i, "value": f"data-{i}"}

            # Store payload - CORRECT SIGNATURE: store(content) returns hash
            payload_bytes = json.dumps(row_data).encode("utf-8")
            payload_ref = payload_store.store(payload_bytes)

            conn.execute(
                rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="source-node",
                    row_index=i,
                    source_data_hash=f"hash{i}",
                    source_data_ref=payload_ref,  # Reference to stored payload
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id=f"tok-{i:03d}",
                    row_id=row_id,
                    created_at=now,
                )
            )

        # Record terminal outcomes for rows 0, 1, 2 (completed before checkpoint)
        # Rows 3-4 have no outcomes (unprocessed)
        for i in range(3):
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id=f"outcome-{i:03d}",
                    run_id=run_id,
                    token_id=f"tok-{i:03d}",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink-node",
                )
            )

        conn.commit()

    # Create checkpoint at row 2 (rows 3-4 are unprocessed)
    graph = _create_test_graph()
    checkpoint_manager.create_checkpoint(
        run_id=run_id,
        token_id="tok-002",
        node_id="source-node",
        sequence_number=2,
        graph=graph,
    )

    return run_id
