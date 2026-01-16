# tests/integration/test_checkpoint_recovery.py
"""Integration tests for checkpoint and recovery.

End-to-end test that verifies checkpoint creation, simulated crash,
and recovery work together correctly.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.landscape.database import LandscapeDB


class TestCheckpointRecoveryIntegration:
    """End-to-end checkpoint/recovery tests."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and payload store."""
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        checkpoint_settings = CheckpointSettings(frequency="every_row")

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "checkpoint_settings": checkpoint_settings,
        }

    def test_full_checkpoint_recovery_cycle(self, test_env: dict[str, Any]) -> None:
        """Complete cycle: run -> checkpoint -> crash -> recover -> complete."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        # 1. Set up a run with rows and checkpoints
        run_id = self._setup_partial_run(db, checkpoint_mgr)

        # 2. Verify checkpoint exists
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        assert checkpoint.sequence_number > 0

        # 3. Verify can resume
        check = recovery_mgr.can_resume(run_id)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # 4. Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id)
        assert resume_point is not None

        # 5. Get unprocessed rows (setup creates 5 rows 0-4, checkpoint at sequence 2)
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert len(unprocessed) == 2  # rows 3 and 4

    def test_checkpoint_sequence_ordering(self, test_env: dict[str, Any]) -> None:
        """Verify checkpoints are ordered by sequence number."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr)

        # Create additional checkpoints at different sequence numbers
        # Use token IDs that were created by _setup_partial_run (tok-001-003, tok-001-004)
        # and use matching node_id from _setup_partial_run (default run_suffix="001")
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-001-003",
            node_id="node-001",
            sequence_number=3,
        )
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-001-004",
            node_id="node-001",
            sequence_number=4,
        )

        # Get all checkpoints
        checkpoints = checkpoint_mgr.get_checkpoints(run_id)
        assert len(checkpoints) >= 3  # At least 3 checkpoints (2, 3, 4)

        # Verify ordering
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].sequence_number < checkpoints[i + 1].sequence_number

        # Latest checkpoint should be the highest sequence
        latest = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert latest is not None
        assert latest.sequence_number == max(c.sequence_number for c in checkpoints)

    def test_recovery_with_aggregation_state(self, test_env: dict[str, Any]) -> None:
        """Verify aggregation state is preserved across recovery."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr)

        # Create checkpoint with aggregation state
        # Use token ID that was created by _setup_partial_run (tok-001-003)
        agg_state = {
            "buffer": [{"id": 1, "value": 100}, {"id": 2, "value": 200}],
            "count": 2,
            "sum": 300,
        }
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-001-003",
            node_id="node-001",
            sequence_number=3,
            aggregation_state=agg_state,
        )

        # Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id)
        assert resume_point is not None
        assert resume_point.aggregation_state is not None
        assert resume_point.aggregation_state == agg_state
        assert resume_point.aggregation_state["count"] == 2
        assert resume_point.aggregation_state["sum"] == 300

    def test_checkpoint_cleanup_after_completion(
        self, test_env: dict[str, Any]
    ) -> None:
        """Verify checkpoints are cleaned up after successful run."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr)

        # Verify checkpoints exist
        checkpoints = checkpoint_mgr.get_checkpoints(run_id)
        assert len(checkpoints) > 0

        # Delete checkpoints (simulating run completion)
        deleted_count = checkpoint_mgr.delete_checkpoints(run_id)
        assert deleted_count > 0

        # Verify no checkpoints remain
        remaining = checkpoint_mgr.get_checkpoints(run_id)
        assert len(remaining) == 0

    def test_recovery_respects_checkpoint_boundary(
        self, test_env: dict[str, Any]
    ) -> None:
        """Verify recovery resumes from correct checkpoint position."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr)

        # Resume point should match the checkpoint
        resume_point = recovery_mgr.get_resume_point(run_id)
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)

        assert resume_point is not None
        assert checkpoint is not None
        assert resume_point.token_id == checkpoint.token_id
        assert resume_point.node_id == checkpoint.node_id
        assert resume_point.sequence_number == checkpoint.sequence_number

    def test_multiple_runs_independent_checkpoints(
        self, test_env: dict[str, Any]
    ) -> None:
        """Verify checkpoints are isolated per run."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # Create two runs with different checkpoint states
        run_id_1 = self._setup_partial_run(db, checkpoint_mgr, run_suffix="001")
        run_id_2 = self._setup_partial_run(db, checkpoint_mgr, run_suffix="002")

        # Each run should have its own checkpoints
        cp_1 = checkpoint_mgr.get_checkpoints(run_id_1)
        cp_2 = checkpoint_mgr.get_checkpoints(run_id_2)

        assert len(cp_1) > 0
        assert len(cp_2) > 0

        # Deleting one run's checkpoints doesn't affect the other
        checkpoint_mgr.delete_checkpoints(run_id_1)
        assert len(checkpoint_mgr.get_checkpoints(run_id_1)) == 0
        assert len(checkpoint_mgr.get_checkpoints(run_id_2)) > 0

    def _setup_partial_run(
        self,
        db: LandscapeDB,
        checkpoint_mgr: CheckpointManager,
        *,
        run_suffix: str = "001",
    ) -> str:
        """Helper to create a partially-completed run with checkpoints."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        run_id = f"test-run-{run_suffix}"
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
                    status="failed",
                )
            )

            # Create node
            conn.execute(
                nodes_table.insert().values(
                    node_id=f"node-{run_suffix}",
                    run_id=run_id,
                    plugin_name="test",
                    node_type="transform",
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create multiple rows
            for i in range(5):
                row_id = f"row-{run_suffix}-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id=f"node-{run_suffix}",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-{run_suffix}-{i:03d}",
                        row_id=row_id,
                        created_at=now,
                    )
                )

            conn.commit()

        # Create checkpoint at row 2 (simulating partial progress)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=f"tok-{run_suffix}-002",
            node_id=f"node-{run_suffix}",
            sequence_number=2,
        )

        return run_id
