"""Tests for CheckpointManager."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.checkpoint.manager import CheckpointManager, IncompatibleCheckpointError

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph


class TestCheckpointManager:
    """Tests for checkpoint creation and loading."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        """Create CheckpointManager with test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")  # Tables auto-created
        return CheckpointManager(db)

    @pytest.fixture
    def mock_graph(self) -> "ExecutionGraph":
        """Create a simple mock graph for testing."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("node-001", node_type=NodeType.TRANSFORM, plugin_name="test", config={})
        return graph

    @pytest.fixture
    def setup_run(self, manager: CheckpointManager) -> str:
        """Create a run with some tokens for checkpoint tests."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        # Insert test run, node, row, token via schema tables
        db = manager._db
        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id="run-001",
                    started_at=datetime.now(UTC),
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.RUNNING,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-001",
                    run_id="run-001",
                    plugin_name="test_transform",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=datetime.now(UTC),
                )
            )
            conn.execute(
                rows_table.insert().values(
                    row_id="row-001",
                    run_id="run-001",
                    source_node_id="node-001",
                    row_index=0,
                    source_data_hash="hash1",
                    created_at=datetime.now(UTC),
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-001",
                    row_id="row-001",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()
        return "run-001"

    def test_create_checkpoint(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Can create a checkpoint with all fields correctly populated.

        Verifies all checkpoint fields, not just sequence number, to ensure
        resume compatibility and audit lineage integrity.
        """
        from elspeth.core.canonical import compute_full_topology_hash, stable_hash

        checkpoint = manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            graph=mock_graph,
        )

        # Basic identity fields
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.run_id == "run-001"
        assert checkpoint.sequence_number == 1

        # Token and node identity (critical for resume correctness)
        assert checkpoint.token_id == "tok-001"
        assert checkpoint.node_id == "node-001"

        # Timestamp must be timezone-aware (audit requirement)
        assert checkpoint.created_at is not None
        assert checkpoint.created_at.tzinfo is not None

        # Topology hashes must match expected values (resume compatibility)
        # Note: CheckpointManager uses compute_full_topology_hash for ALL branches
        expected_upstream_hash = compute_full_topology_hash(mock_graph)
        node_info = mock_graph.get_node_info("node-001")
        expected_config_hash = stable_hash(node_info.config)

        assert checkpoint.upstream_topology_hash == expected_upstream_hash
        assert checkpoint.checkpoint_node_config_hash == expected_config_hash

    def test_get_latest_checkpoint(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Can retrieve the latest checkpoint for a run."""
        # Create multiple checkpoints
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 3, mock_graph)

        latest = manager.get_latest_checkpoint("run-001")

        assert latest is not None
        assert latest.sequence_number == 3

    def test_get_latest_checkpoint_no_checkpoints(self, manager: CheckpointManager) -> None:
        """Returns None when no checkpoints exist."""
        latest = manager.get_latest_checkpoint("nonexistent-run")
        assert latest is None

    def test_checkpoint_with_aggregation_state(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Can store aggregation state in checkpoint."""
        agg_state = {"buffer": [1, 2, 3], "count": 3}

        manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            graph=mock_graph,
            aggregation_state=agg_state,
        )

        loaded = manager.get_latest_checkpoint("run-001")
        assert loaded is not None
        assert loaded.aggregation_state_json is not None
        assert json.loads(loaded.aggregation_state_json) == agg_state

    def test_checkpoint_with_empty_aggregation_state_preserved(
        self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph"
    ) -> None:
        """Empty aggregation state {} is distinct from None (no state).

        Regression test for truthiness bug: empty dict {} should serialize to "{}"
        not None. This ensures resume can distinguish "state is empty" from
        "no aggregation state was provided".

        See: docs/bugs/closed/P2-2026-01-19-checkpoint-empty-aggregation-state-dropped.md
        """
        # Create checkpoint with empty dict (NOT None)
        manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            graph=mock_graph,
            aggregation_state={},  # Empty dict, not None
        )

        loaded = manager.get_latest_checkpoint("run-001")
        assert loaded is not None
        # Critical: empty dict should serialize to "{}", NOT None
        assert loaded.aggregation_state_json == "{}", (
            f"Empty aggregation state should serialize to '{{}}', got {loaded.aggregation_state_json!r}. "
            "This is likely a truthiness bug where `if aggregation_state:` was used "
            "instead of `if aggregation_state is not None:`"
        )

    def test_checkpoint_with_none_aggregation_state_is_null(
        self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph"
    ) -> None:
        """None aggregation state stays as NULL/None in the database.

        Complements test_checkpoint_with_empty_aggregation_state_preserved to
        verify the full contract: {} → "{}" and None → NULL.
        """
        # Create checkpoint with explicit None
        manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            graph=mock_graph,
            aggregation_state=None,
        )

        loaded = manager.get_latest_checkpoint("run-001")
        assert loaded is not None
        # None should stay as None (NULL in DB)
        assert loaded.aggregation_state_json is None, f"None aggregation state should stay as None, got {loaded.aggregation_state_json!r}"

    def test_get_checkpoints_ordered(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Get all checkpoints ordered by sequence number."""
        manager.create_checkpoint("run-001", "tok-001", "node-001", 3, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2, mock_graph)

        checkpoints = manager.get_checkpoints("run-001")

        assert len(checkpoints) == 3
        assert [c.sequence_number for c in checkpoints] == [1, 2, 3]

    def test_delete_checkpoints(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Delete all checkpoints for a run."""
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2, mock_graph)

        deleted = manager.delete_checkpoints("run-001")

        assert deleted == 2
        assert manager.get_latest_checkpoint("run-001") is None

    def test_delete_checkpoints_no_checkpoints(self, manager: CheckpointManager) -> None:
        """Delete returns 0 when no checkpoints exist."""
        deleted = manager.delete_checkpoints("nonexistent-run")
        assert deleted == 0

    def test_old_checkpoint_rejected(self, manager: CheckpointManager) -> None:
        """Checkpoints without format_version should be rejected.

        Unversioned checkpoints predate deterministic node IDs and cannot be
        safely resumed.
        """
        from elspeth.core.landscape.schema import (
            checkpoints_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        # Set up required foreign key references
        run_id = "run-old"
        created_at = datetime(2026, 1, 23, 23, 59, 59, tzinfo=UTC)

        with manager._db.engine.connect() as conn:
            # Create run
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=created_at,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )
            # Create node
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-old",
                    run_id=run_id,
                    plugin_name="test_transform",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=created_at,
                )
            )
            # Create row
            conn.execute(
                rows_table.insert().values(
                    row_id="row-old",
                    run_id=run_id,
                    source_node_id="node-old",
                    row_index=0,
                    source_data_hash="hash1",
                    created_at=created_at,
                )
            )
            # Create token
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-old",
                    row_id="row-old",
                    created_at=created_at,
                )
            )
            # Create checkpoint with old date
            # Include topology hashes (added in Bug #12) to satisfy NOT NULL constraints
            checkpoint_id = "cp-old-checkpoint"
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    token_id="tok-old",
                    node_id="node-old",
                    sequence_number=1,
                    aggregation_state_json=None,
                    upstream_topology_hash="old-upstream-hash",  # Bug #12: required field
                    checkpoint_node_config_hash="old-node-config-hash",  # Bug #12: required field
                    created_at=created_at,
                )
            )
            conn.commit()

        # Attempting to load should raise IncompatibleCheckpointError
        with pytest.raises(IncompatibleCheckpointError) as exc_info:
            manager.get_latest_checkpoint(run_id)

        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert checkpoint_id in error_msg
        assert "format_version" in error_msg
        assert "Resume not supported" in error_msg

    def test_new_checkpoint_accepted(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Checkpoints created on or after 2026-01-24 should be accepted.

        New checkpoints use deterministic hash-based node IDs which are
        compatible with the current system.
        """
        # Create a checkpoint with new created_at date (on cutoff)
        new_date = datetime(2026, 1, 24, 0, 0, 0, tzinfo=UTC)  # Exactly at cutoff

        from elspeth.contracts import Checkpoint
        from elspeth.core.landscape.schema import checkpoints_table

        checkpoint_id = "cp-new-checkpoint"

        with manager._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id="run-001",
                    token_id="tok-001",
                    node_id="node-001",
                    sequence_number=1,
                    aggregation_state_json=None,
                    upstream_topology_hash="new-upstream-hash",  # Bug #12: required field
                    checkpoint_node_config_hash="new-node-config-hash",  # Bug #12: required field
                    created_at=new_date,
                    format_version=Checkpoint.CURRENT_FORMAT_VERSION,
                )
            )
            conn.commit()

        # Should load successfully without exception
        checkpoint = manager.get_latest_checkpoint("run-001")
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == checkpoint_id
        # SQLite may return timezone-naive datetime, so compare the datetime values
        assert checkpoint.created_at.replace(tzinfo=None) == new_date.replace(tzinfo=None)

    def test_checkpoint_after_cutoff_accepted(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Checkpoints created after 2026-01-24 should be accepted."""
        # Create a checkpoint well after cutoff
        future_date = datetime(2026, 2, 1, 12, 30, 0, tzinfo=UTC)

        from elspeth.contracts import Checkpoint
        from elspeth.core.landscape.schema import checkpoints_table

        checkpoint_id = "cp-future-checkpoint"

        with manager._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id="run-001",
                    token_id="tok-001",
                    node_id="node-001",
                    sequence_number=1,
                    aggregation_state_json=None,
                    upstream_topology_hash="future-upstream-hash",  # Bug #12: required field
                    checkpoint_node_config_hash="future-node-config-hash",  # Bug #12: required field
                    created_at=future_date,
                    format_version=Checkpoint.CURRENT_FORMAT_VERSION,
                )
            )
            conn.commit()

        # Should load successfully
        checkpoint = manager.get_latest_checkpoint("run-001")
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == checkpoint_id

    def test_versioned_checkpoint_accepted(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Checkpoints with current format_version should be accepted.

        New checkpoints include format_version=2 which explicitly indicates
        compatibility with deterministic node IDs.
        """
        from elspeth.contracts import Checkpoint
        from elspeth.core.landscape.schema import checkpoints_table

        checkpoint_id = "cp-versioned"
        now = datetime.now(UTC)

        with manager._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id="run-001",
                    token_id="tok-001",
                    node_id="node-001",
                    sequence_number=1,
                    aggregation_state_json=None,
                    upstream_topology_hash="versioned-upstream-hash",
                    checkpoint_node_config_hash="versioned-node-config-hash",
                    created_at=now,
                    format_version=Checkpoint.CURRENT_FORMAT_VERSION,  # Explicitly versioned
                )
            )
            conn.commit()

        # Should load successfully
        checkpoint = manager.get_latest_checkpoint("run-001")
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == checkpoint_id
        assert checkpoint.format_version == Checkpoint.CURRENT_FORMAT_VERSION

    def test_old_format_version_rejected(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Checkpoints with old format_version should be rejected.

        format_version=1 indicates pre-deterministic node IDs, which are
        incompatible with the current system.
        """
        from elspeth.contracts import Checkpoint
        from elspeth.core.landscape.schema import checkpoints_table

        # Use a recent date to ensure the rejection is based on version, not date
        recent_date = datetime.now(UTC)
        checkpoint_id = "cp-old-version"

        with manager._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id="run-001",
                    token_id="tok-001",
                    node_id="node-001",
                    sequence_number=1,
                    aggregation_state_json=None,
                    upstream_topology_hash="old-version-upstream-hash",
                    checkpoint_node_config_hash="old-version-node-config-hash",
                    created_at=recent_date,  # Recent date - rejection should be version-based
                    format_version=1,  # Old format version
                )
            )
            conn.commit()

        # Attempting to load should raise IncompatibleCheckpointError
        with pytest.raises(IncompatibleCheckpointError) as exc_info:
            manager.get_latest_checkpoint("run-001")

        # Verify error message mentions version incompatibility
        error_msg = str(exc_info.value)
        assert checkpoint_id in error_msg
        assert "format version" in error_msg.lower()
        assert "v1" in error_msg  # Old version
        assert f"v{Checkpoint.CURRENT_FORMAT_VERSION}" in error_msg  # Required version
        # P2b fix: Now rejects both older AND newer versions (exact match required)
        assert "exact format version match" in error_msg

    def test_newer_format_version_rejected(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """P2b fix: Checkpoints with NEWER format_version should also be rejected.

        Cross-version resume is not supported in either direction. If a checkpoint
        was created by a future version of the software, we cannot safely resume
        because the newer version may have changes we don't understand.

        This test verifies the P2b fix: version comparison changed from `<` to `!=`.
        """
        from elspeth.contracts import Checkpoint
        from elspeth.core.landscape.schema import checkpoints_table

        recent_date = datetime.now(UTC)
        checkpoint_id = "cp-newer-version"
        future_version = Checkpoint.CURRENT_FORMAT_VERSION + 1  # Newer than current

        with manager._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id="run-001",
                    token_id="tok-001",
                    node_id="node-001",
                    sequence_number=1,
                    aggregation_state_json=None,
                    upstream_topology_hash="newer-version-upstream-hash",
                    checkpoint_node_config_hash="newer-version-node-config-hash",
                    created_at=recent_date,
                    format_version=future_version,  # NEWER format version
                )
            )
            conn.commit()

        # P2b fix: Attempting to load should raise IncompatibleCheckpointError
        # Previous behavior: Would have accepted (only rejected older versions)
        with pytest.raises(IncompatibleCheckpointError) as exc_info:
            manager.get_latest_checkpoint("run-001")

        error_msg = str(exc_info.value)
        assert checkpoint_id in error_msg
        assert "format version" in error_msg.lower()
        assert f"v{future_version}" in error_msg  # Newer version in checkpoint
        assert f"v{Checkpoint.CURRENT_FORMAT_VERSION}" in error_msg  # Current version
        assert "exact format version match" in error_msg

    def test_create_checkpoint_requires_graph(self, manager: CheckpointManager, setup_run: str) -> None:
        """Bug #9: Verify checkpoint creation fails if graph parameter is None.

        Parameter validation must catch missing graph at function entry,
        not later during hash computation.
        """
        with pytest.raises(ValueError, match="graph parameter is required"):
            manager.create_checkpoint(
                run_id=setup_run,
                token_id="t1",
                node_id="node1",
                sequence_number=0,
                graph=None,  # type: ignore  # Intentionally passing None to test validation
            )

    def test_create_checkpoint_validates_node_exists_in_graph(
        self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph"
    ) -> None:
        """Bug #9: Verify checkpoint creation fails if node_id not in graph.

        Must validate that node_id exists in graph before attempting
        to compute topology hashes.
        """
        # mock_graph has nodes: "source", "transform", "sink"
        with pytest.raises(ValueError, match="does not exist in graph"):
            manager.create_checkpoint(
                run_id=setup_run,
                token_id="t1",
                node_id="nonexistent_node",  # Not in graph
                sequence_number=0,
                graph=mock_graph,
            )

    def test_create_checkpoint_with_empty_graph_fails(self, manager: CheckpointManager, setup_run: str) -> None:
        """Bug #9: Verify checkpoint creation fails with empty graph.

        Even if graph exists, if it has no nodes, node validation should fail.
        """
        from elspeth.core.dag import ExecutionGraph

        empty_graph = ExecutionGraph()

        with pytest.raises(ValueError, match="does not exist in graph"):
            manager.create_checkpoint(
                run_id=setup_run,
                token_id="t1",
                node_id="any_node",  # No nodes exist in empty graph
                sequence_number=0,
                graph=empty_graph,
            )
