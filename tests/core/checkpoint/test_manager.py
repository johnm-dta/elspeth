"""Tests for CheckpointManager."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.core.checkpoint.manager import CheckpointManager


class TestCheckpointManager:
    """Tests for checkpoint creation and loading."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        """Create CheckpointManager with test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")  # Tables auto-created
        return CheckpointManager(db)

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
                    status="running",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-001",
                    run_id="run-001",
                    plugin_name="test_transform",
                    node_type="transform",
                    plugin_version="1.0.0",
                    determinism="deterministic",
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

    def test_create_checkpoint(
        self, manager: CheckpointManager, setup_run: str
    ) -> None:
        """Can create a checkpoint."""
        checkpoint = manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
        )

        assert checkpoint.checkpoint_id is not None
        assert checkpoint.run_id == "run-001"
        assert checkpoint.sequence_number == 1

    def test_get_latest_checkpoint(
        self, manager: CheckpointManager, setup_run: str
    ) -> None:
        """Can retrieve the latest checkpoint for a run."""
        # Create multiple checkpoints
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 3)

        latest = manager.get_latest_checkpoint("run-001")

        assert latest is not None
        assert latest.sequence_number == 3

    def test_get_latest_checkpoint_no_checkpoints(
        self, manager: CheckpointManager
    ) -> None:
        """Returns None when no checkpoints exist."""
        latest = manager.get_latest_checkpoint("nonexistent-run")
        assert latest is None

    def test_checkpoint_with_aggregation_state(
        self, manager: CheckpointManager, setup_run: str
    ) -> None:
        """Can store aggregation state in checkpoint."""
        agg_state = {"buffer": [1, 2, 3], "count": 3}

        manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            aggregation_state=agg_state,
        )

        loaded = manager.get_latest_checkpoint("run-001")
        assert loaded is not None
        assert loaded.aggregation_state_json is not None
        assert json.loads(loaded.aggregation_state_json) == agg_state

    def test_get_checkpoints_ordered(
        self, manager: CheckpointManager, setup_run: str
    ) -> None:
        """Get all checkpoints ordered by sequence number."""
        manager.create_checkpoint("run-001", "tok-001", "node-001", 3)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2)

        checkpoints = manager.get_checkpoints("run-001")

        assert len(checkpoints) == 3
        assert [c.sequence_number for c in checkpoints] == [1, 2, 3]

    def test_delete_checkpoints(
        self, manager: CheckpointManager, setup_run: str
    ) -> None:
        """Delete all checkpoints for a run."""
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2)

        deleted = manager.delete_checkpoints("run-001")

        assert deleted == 2
        assert manager.get_latest_checkpoint("run-001") is None

    def test_delete_checkpoints_no_checkpoints(
        self, manager: CheckpointManager
    ) -> None:
        """Delete returns 0 when no checkpoints exist."""
        deleted = manager.delete_checkpoints("nonexistent-run")
        assert deleted == 0
