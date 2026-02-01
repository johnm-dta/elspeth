"""Mutation gap tests for core/checkpoint/manager.py.

Tests targeting specific mutation survivors:
- Line 53-54: checkpoint_id prefix format (cp- prefix)
- Line 96: get_latest_checkpoint returns DESC order (highest sequence, not lowest)
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.checkpoint.manager import CheckpointManager

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph


class TestCheckpointIdFormat:
    """Tests for checkpoint ID generation format.

    Targets line 54: checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"
    Mutant might change the prefix or format.
    """

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        """Create CheckpointManager with test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
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
        """Create a run with tokens for checkpoint tests."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

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

    def test_checkpoint_id_starts_with_cp_prefix(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Line 54: checkpoint_id must start with 'cp-' prefix.

        Ensures mutants that change the prefix (e.g., 'cp-' to 'XX-') are killed.
        """
        checkpoint = manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            graph=mock_graph,
            sequence_number=1,
        )

        assert checkpoint.checkpoint_id.startswith("cp-"), f"checkpoint_id must start with 'cp-' prefix, got: {checkpoint.checkpoint_id}"

    def test_checkpoint_id_has_correct_format(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Line 78: checkpoint_id format is 'cp-' + full 32 hex chars (UUID).

        Format: cp-{uuid.uuid4().hex} = "cp-" + exactly 32 hex characters.
        """
        checkpoint = manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            graph=mock_graph,
            sequence_number=1,
        )

        # Should be cp- + 32 hex chars = 35 chars total
        assert len(checkpoint.checkpoint_id) == 35, f"checkpoint_id should be 35 chars (cp- + 32 hex), got {len(checkpoint.checkpoint_id)}"

        # Extract hex portion (after "cp-")
        hex_portion = checkpoint.checkpoint_id[3:]
        assert len(hex_portion) == 32, f"hex portion should be 32 chars, got {len(hex_portion)}"

        # Verify it's valid hex
        try:
            int(hex_portion, 16)
        except ValueError:
            pytest.fail(f"checkpoint_id hex portion is not valid hex: {hex_portion}")


class TestGetLatestCheckpointOrdering:
    """Tests for get_latest_checkpoint DESC ordering.

    Targets line 96: .order_by(desc(checkpoints_table.c.sequence_number))
    Mutant might change desc() to asc(), returning oldest instead of latest.
    """

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        """Create CheckpointManager with test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
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
        """Create a run with tokens for checkpoint tests."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

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

    def test_get_latest_returns_highest_sequence_not_lowest(
        self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph"
    ) -> None:
        """Line 96: get_latest_checkpoint must use DESC order.

        Creates checkpoints with sequence numbers [1, 5, 3] and verifies
        the method returns sequence 5 (highest), not 1 (lowest/first inserted).

        This kills mutants that change desc() to asc().
        """
        # Create checkpoints out of order to test sorting
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 5, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 3, mock_graph)

        latest = manager.get_latest_checkpoint("run-001")

        assert latest is not None
        # Critical: must be 5 (highest), not 1 (first inserted) or 3 (last inserted)
        assert latest.sequence_number == 5, (
            f"get_latest_checkpoint should return highest sequence (5), got {latest.sequence_number}. "
            "This suggests DESC ordering is broken."
        )

    def test_get_latest_with_only_one_checkpoint(self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph") -> None:
        """Line 96: With single checkpoint, should return that checkpoint.

        Edge case: ordering shouldn't matter with only one record.
        """
        manager.create_checkpoint("run-001", "tok-001", "node-001", 42, mock_graph)

        latest = manager.get_latest_checkpoint("run-001")

        assert latest is not None
        assert latest.sequence_number == 42

    def test_get_latest_returns_most_recent_not_first_created(
        self, manager: CheckpointManager, setup_run: str, mock_graph: "ExecutionGraph"
    ) -> None:
        """Line 96: Must return highest sequence_number, not first inserted.

        Additional test: insert in ascending order to verify it's not just
        returning the first row.
        """
        # Create in ascending order
        manager.create_checkpoint("run-001", "tok-001", "node-001", 10, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 20, mock_graph)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 30, mock_graph)

        latest = manager.get_latest_checkpoint("run-001")

        assert latest is not None
        assert latest.sequence_number == 30, f"Expected sequence 30 (highest), got {latest.sequence_number}"

    def test_get_latest_filters_by_run_id(self, manager: CheckpointManager, mock_graph: "ExecutionGraph") -> None:
        """get_latest_checkpoint must filter by run_id, not return from other runs.

        A regression that drops the run_id filter would return checkpoints from
        the wrong run, causing incorrect resume behavior. This test creates
        checkpoints for two different runs and verifies correct filtering.

        Identified by quality audit as P1 missing edge case.
        """
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        db = manager._db

        # Set up run-001 with sequence 10
        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id="run-001",
                    started_at=datetime.now(UTC),
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.RUNNING,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-001",
                    run_id="run-001",
                    plugin_name="test",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
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
            conn.execute(tokens_table.insert().values(token_id="tok-001", row_id="row-001", created_at=datetime.now(UTC)))
            conn.commit()

        # Set up run-002 with sequence 99 (higher than run-001)
        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id="run-002",
                    started_at=datetime.now(UTC),
                    config_hash="def",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.RUNNING,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-002",
                    run_id="run-002",
                    plugin_name="test",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=datetime.now(UTC),
                )
            )
            conn.execute(
                rows_table.insert().values(
                    row_id="row-002",
                    run_id="run-002",
                    source_node_id="node-002",
                    row_index=0,
                    source_data_hash="hash2",
                    created_at=datetime.now(UTC),
                )
            )
            conn.execute(tokens_table.insert().values(token_id="tok-002", row_id="row-002", created_at=datetime.now(UTC)))
            conn.commit()

        # Create graph with node-002 for run-002
        graph_002 = mock_graph.__class__()
        graph_002.add_node("node-002", node_type=NodeType.TRANSFORM, plugin_name="test", config={})

        # Create checkpoints
        manager.create_checkpoint("run-001", "tok-001", "node-001", 10, mock_graph)
        manager.create_checkpoint("run-002", "tok-002", "node-002", 99, graph_002)

        # Verify filtering - run-001 should return its checkpoint, not run-002's
        latest_run1 = manager.get_latest_checkpoint("run-001")
        assert latest_run1 is not None
        assert latest_run1.run_id == "run-001", f"Expected run-001, got {latest_run1.run_id}"
        assert latest_run1.sequence_number == 10

        # Verify run-002 returns its own checkpoint
        latest_run2 = manager.get_latest_checkpoint("run-002")
        assert latest_run2 is not None
        assert latest_run2.run_id == "run-002", f"Expected run-002, got {latest_run2.run_id}"
        assert latest_run2.sequence_number == 99
