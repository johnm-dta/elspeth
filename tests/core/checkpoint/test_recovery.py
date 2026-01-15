"""Tests for checkpoint recovery protocol."""

from datetime import UTC, datetime

import pytest


class TestRecoveryProtocol:
    """Tests for resuming runs from checkpoints."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db):
        from elspeth.core.checkpoint import CheckpointManager

        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db, checkpoint_manager):
        from elspeth.core.checkpoint import RecoveryManager

        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def failed_run_with_checkpoint(self, landscape_db, checkpoint_manager):
        """Create a failed run that has checkpoints."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        run_id = "failed-run-001"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status="failed",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-001",
                    run_id=run_id,
                    plugin_name="test",
                    node_type="transform",
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                rows_table.insert().values(
                    row_id="row-001",
                    run_id=run_id,
                    source_node_id="node-001",
                    row_index=0,
                    source_data_hash="hash1",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-001", row_id="row-001", created_at=now
                )
            )
            conn.commit()

        checkpoint_manager.create_checkpoint(run_id, "tok-001", "node-001", 1)
        return run_id

    @pytest.fixture
    def completed_run(self, landscape_db):
        """Create a completed run (cannot be resumed)."""
        from elspeth.core.landscape.schema import runs_table

        run_id = "completed-run-001"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    completed_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status="completed",
                )
            )
            conn.commit()
        return run_id

    @pytest.fixture
    def failed_run_no_checkpoint(self, landscape_db):
        """Create a failed run without checkpoints."""
        from elspeth.core.landscape.schema import runs_table

        run_id = "failed-no-cp-001"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status="failed",
                )
            )
            conn.commit()
        return run_id

    @pytest.fixture
    def running_run(self, landscape_db):
        """Create a running run (cannot be resumed - still in progress)."""
        from elspeth.core.landscape.schema import runs_table

        run_id = "running-run-001"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status="running",
                )
            )
            conn.commit()
        return run_id

    def test_can_resume_returns_true_for_failed_run_with_checkpoint(
        self, recovery_manager, failed_run_with_checkpoint
    ) -> None:
        can_resume, reason = recovery_manager.can_resume(failed_run_with_checkpoint)
        assert can_resume is True
        assert reason is None

    def test_can_resume_returns_false_for_completed_run(
        self, recovery_manager, completed_run
    ) -> None:
        can_resume, reason = recovery_manager.can_resume(completed_run)
        assert can_resume is False
        assert "completed" in reason.lower()

    def test_can_resume_returns_false_without_checkpoint(
        self, recovery_manager, failed_run_no_checkpoint
    ) -> None:
        can_resume, reason = recovery_manager.can_resume(failed_run_no_checkpoint)
        assert can_resume is False
        assert "no checkpoint" in reason.lower()

    def test_can_resume_returns_false_for_running_run(
        self, recovery_manager, running_run
    ) -> None:
        can_resume, reason = recovery_manager.can_resume(running_run)
        assert can_resume is False
        assert "in progress" in reason.lower()

    def test_can_resume_returns_false_for_nonexistent_run(
        self, recovery_manager
    ) -> None:
        can_resume, reason = recovery_manager.can_resume("nonexistent-run")
        assert can_resume is False
        assert "not found" in reason.lower()

    def test_get_resume_point(
        self, recovery_manager, failed_run_with_checkpoint
    ) -> None:
        resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint)
        assert resume_point is not None
        assert resume_point.token_id is not None
        assert resume_point.node_id is not None
        assert resume_point.sequence_number > 0

    def test_get_resume_point_returns_none_for_unresumable_run(
        self, recovery_manager, completed_run
    ) -> None:
        resume_point = recovery_manager.get_resume_point(completed_run)
        assert resume_point is None

    def test_get_resume_point_includes_checkpoint(
        self, recovery_manager, failed_run_with_checkpoint
    ) -> None:
        resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint)
        assert resume_point is not None
        assert resume_point.checkpoint is not None
        assert resume_point.checkpoint.run_id == failed_run_with_checkpoint

    def test_get_resume_point_with_aggregation_state(
        self, landscape_db, checkpoint_manager
    ) -> None:
        """Resume point includes deserialized aggregation state."""
        from elspeth.core.checkpoint import RecoveryManager
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        run_id = "agg-state-run"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status="failed",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-agg",
                    run_id=run_id,
                    plugin_name="test",
                    node_type="aggregation",
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                rows_table.insert().values(
                    row_id="row-agg",
                    run_id=run_id,
                    source_node_id="node-agg",
                    row_index=0,
                    source_data_hash="hash1",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-agg", row_id="row-agg", created_at=now
                )
            )
            conn.commit()

        # Create checkpoint with aggregation state
        agg_state = {"buffer": [1, 2, 3], "count": 3}
        checkpoint_manager.create_checkpoint(
            run_id, "tok-agg", "node-agg", 5, aggregation_state=agg_state
        )

        recovery_manager = RecoveryManager(landscape_db, checkpoint_manager)
        resume_point = recovery_manager.get_resume_point(run_id)

        assert resume_point is not None
        assert resume_point.aggregation_state == agg_state
        assert resume_point.sequence_number == 5


class TestGetUnprocessedRows:
    """Unit tests for RecoveryManager.get_unprocessed_rows()."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db):
        from elspeth.core.checkpoint import CheckpointManager

        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db, checkpoint_manager):
        from elspeth.core.checkpoint import RecoveryManager

        return RecoveryManager(landscape_db, checkpoint_manager)

    def _setup_run_with_rows(
        self, landscape_db, checkpoint_manager, *, create_checkpoint: bool = True
    ) -> str:
        """Helper to create a run with multiple rows and optionally a checkpoint."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        run_id = "unprocessed-test-run"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status="failed",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-unproc",
                    run_id=run_id,
                    plugin_name="test",
                    node_type="transform",
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=now,
                )
            )
            # Create 5 rows (indices 0-4)
            for i in range(5):
                row_id = f"row-unproc-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id="node-unproc",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-unproc-{i:03d}",
                        row_id=row_id,
                        created_at=now,
                    )
                )
            conn.commit()

        if create_checkpoint:
            # Checkpoint at row index 2 (rows 0, 1, 2 are processed)
            checkpoint_manager.create_checkpoint(
                run_id, "tok-unproc-002", "node-unproc", 2
            )

        return run_id

    def test_returns_correct_rows_when_checkpoint_exists(
        self, landscape_db, checkpoint_manager, recovery_manager
    ) -> None:
        """Returns rows with index > checkpoint.sequence_number."""
        run_id = self._setup_run_with_rows(
            landscape_db, checkpoint_manager, create_checkpoint=True
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Checkpoint at sequence 2, so rows 3 and 4 are unprocessed
        assert len(unprocessed) == 2
        assert "row-unproc-003" in unprocessed
        assert "row-unproc-004" in unprocessed

    def test_returns_empty_list_when_no_checkpoint_exists(
        self, landscape_db, checkpoint_manager, recovery_manager
    ) -> None:
        """Returns empty list when there is no checkpoint for the run."""
        run_id = self._setup_run_with_rows(
            landscape_db, checkpoint_manager, create_checkpoint=False
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert unprocessed == []

    def test_returns_empty_list_when_all_rows_processed(
        self, landscape_db, checkpoint_manager, recovery_manager
    ) -> None:
        """Returns empty list when checkpoint is at or beyond all rows."""
        run_id = self._setup_run_with_rows(
            landscape_db, checkpoint_manager, create_checkpoint=False
        )

        # Create checkpoint at sequence 4 (the last row)
        checkpoint_manager.create_checkpoint(run_id, "tok-unproc-004", "node-unproc", 4)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert unprocessed == []

    def test_handles_nonexistent_run_id_gracefully(self, recovery_manager) -> None:
        """Returns empty list for a run_id that does not exist."""
        unprocessed = recovery_manager.get_unprocessed_rows("nonexistent-run-id")

        assert unprocessed == []
