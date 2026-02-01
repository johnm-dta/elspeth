"""Tests for checkpoint recovery protocol."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB


class TestRecoveryProtocol:
    """Tests for resuming runs from checkpoints."""

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a simple mock graph for recovery tests."""
        graph = ExecutionGraph()
        graph.add_node("node-001", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node(
            "node-agg",
            node_type=NodeType.AGGREGATION,
            plugin_name="test",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "options": {"schema": {"fields": "dynamic"}},
                "schema": {"fields": "dynamic"},
            },
        )
        return graph

    @pytest.fixture
    def failed_run_with_checkpoint(
        self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager, mock_graph: ExecutionGraph
    ) -> str:
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
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-001",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
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
            conn.execute(tokens_table.insert().values(token_id="tok-001", row_id="row-001", created_at=now))
            conn.commit()

        # Create checkpoint with graph to include topology data
        checkpoint_manager.create_checkpoint(run_id, "tok-001", "node-001", 1, graph=mock_graph)
        return run_id

    @pytest.fixture
    def completed_run(self, landscape_db: LandscapeDB) -> str:
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
                    status=RunStatus.COMPLETED,
                )
            )
            conn.commit()
        return run_id

    @pytest.fixture
    def failed_run_no_checkpoint(self, landscape_db: LandscapeDB) -> str:
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
                    status=RunStatus.FAILED,
                )
            )
            conn.commit()
        return run_id

    @pytest.fixture
    def running_run(self, landscape_db: LandscapeDB) -> str:
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
                    status=RunStatus.RUNNING,
                )
            )
            conn.commit()
        return run_id

    def test_can_resume_returns_true_for_failed_run_with_checkpoint(
        self, recovery_manager: RecoveryManager, failed_run_with_checkpoint: str, mock_graph: ExecutionGraph
    ) -> None:
        check = recovery_manager.can_resume(failed_run_with_checkpoint, mock_graph)
        assert check.can_resume is True
        assert check.reason is None

    def test_can_resume_returns_false_for_completed_run(
        self, recovery_manager: RecoveryManager, completed_run: str, mock_graph: ExecutionGraph
    ) -> None:
        check = recovery_manager.can_resume(completed_run, mock_graph)
        assert check.can_resume is False
        assert check.reason is not None
        assert "completed" in check.reason.lower()

    def test_can_resume_returns_false_without_checkpoint(
        self, recovery_manager: RecoveryManager, failed_run_no_checkpoint: str, mock_graph: ExecutionGraph
    ) -> None:
        check = recovery_manager.can_resume(failed_run_no_checkpoint, mock_graph)
        assert check.can_resume is False
        assert check.reason is not None
        assert "no checkpoint" in check.reason.lower()

    def test_can_resume_returns_false_for_running_run(
        self, recovery_manager: RecoveryManager, running_run: str, mock_graph: ExecutionGraph
    ) -> None:
        check = recovery_manager.can_resume(running_run, mock_graph)
        assert check.can_resume is False
        assert check.reason is not None
        assert "in progress" in check.reason.lower()

    def test_can_resume_returns_false_for_nonexistent_run(self, recovery_manager: RecoveryManager, mock_graph: ExecutionGraph) -> None:
        check = recovery_manager.can_resume("nonexistent-run", mock_graph)
        assert check.can_resume is False
        assert check.reason is not None
        assert "not found" in check.reason.lower()

    def test_get_resume_point(self, recovery_manager: RecoveryManager, failed_run_with_checkpoint: str, mock_graph: ExecutionGraph) -> None:
        """Resume point returns exact checkpoint values, not just non-null.

        The fixture creates checkpoint with tok-001, node-001, sequence 1.
        A regression returning wrong values (but still non-null) would break resume.
        """
        resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint, mock_graph)

        assert resume_point is not None

        # Assert exact values from fixture, not just non-null/positive
        assert resume_point.token_id == "tok-001"
        assert resume_point.node_id == "node-001"
        assert resume_point.sequence_number == 1

    def test_get_resume_point_returns_none_for_unresumable_run(
        self, recovery_manager: RecoveryManager, completed_run: str, mock_graph: ExecutionGraph
    ) -> None:
        resume_point = recovery_manager.get_resume_point(completed_run, mock_graph)
        assert resume_point is None

    def test_get_resume_point_includes_checkpoint(
        self, recovery_manager: RecoveryManager, failed_run_with_checkpoint: str, mock_graph: ExecutionGraph
    ) -> None:
        resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint, mock_graph)
        assert resume_point is not None
        assert resume_point.checkpoint is not None
        assert resume_point.checkpoint.run_id == failed_run_with_checkpoint

    def test_get_resume_point_with_aggregation_state(
        self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager, mock_graph: ExecutionGraph
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
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-agg",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.AGGREGATION,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
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
            conn.execute(tokens_table.insert().values(token_id="tok-agg", row_id="row-agg", created_at=now))
            conn.commit()

        # Create checkpoint with aggregation state and graph for topology validation
        agg_state = {"buffer": [1, 2, 3], "count": 3}
        checkpoint_manager.create_checkpoint(run_id, "tok-agg", "node-agg", 5, aggregation_state=agg_state, graph=mock_graph)

        recovery_manager = RecoveryManager(landscape_db, checkpoint_manager)
        resume_point = recovery_manager.get_resume_point(run_id, mock_graph)

        assert resume_point is not None
        assert resume_point.aggregation_state == agg_state
        assert resume_point.sequence_number == 5


class TestGetUnprocessedRows:
    """Unit tests for RecoveryManager.get_unprocessed_rows()."""

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    def _setup_run_with_rows(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        *,
        create_checkpoint: bool = True,
    ) -> str:
        """Helper to create a run with multiple rows and optionally a checkpoint.

        When create_checkpoint=True:
        - Creates 5 rows (indices 0-4)
        - Records terminal outcomes for rows 0, 1, 2 (completed)
        - Checkpoint at row 2
        - Expected unprocessed: rows 3, 4
        """
        from elspeth.contracts import RowOutcome
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            token_outcomes_table,
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
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-unproc",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
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
            # Record terminal outcomes for rows 0, 1, 2 (completed before checkpoint)
            # This matches production behavior where outcomes are recorded on sink write
            with landscape_db.engine.connect() as conn:
                for i in range(3):
                    conn.execute(
                        token_outcomes_table.insert().values(
                            outcome_id=f"outcome-unproc-{i:03d}",
                            run_id=run_id,
                            token_id=f"tok-unproc-{i:03d}",
                            outcome=RowOutcome.COMPLETED.value,
                            is_terminal=1,
                            recorded_at=now,
                            sink_name="output",
                        )
                    )
                conn.commit()

            # Checkpoint at row index 2 (rows 0, 1, 2 are processed)
            from tests.core.checkpoint.conftest import _create_test_graph

            graph = _create_test_graph(checkpoint_node="node-unproc")
            checkpoint_manager.create_checkpoint(run_id, "tok-unproc-002", "node-unproc", 2, graph=graph)

        return run_id

    def test_returns_correct_rows_when_checkpoint_exists(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Returns rows with index > checkpoint.sequence_number."""
        run_id = self._setup_run_with_rows(landscape_db, checkpoint_manager, create_checkpoint=True)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Checkpoint at sequence 2, so rows 3 and 4 are unprocessed
        assert len(unprocessed) == 2
        assert "row-unproc-003" in unprocessed
        assert "row-unproc-004" in unprocessed

    def test_returns_empty_list_when_no_checkpoint_exists(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Returns empty list when there is no checkpoint for the run."""
        run_id = self._setup_run_with_rows(landscape_db, checkpoint_manager, create_checkpoint=False)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert unprocessed == []

    def test_returns_empty_list_when_all_rows_processed(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Returns empty list when all rows have terminal outcomes."""
        from elspeth.contracts import RowOutcome
        from elspeth.core.landscape.schema import token_outcomes_table

        run_id = self._setup_run_with_rows(landscape_db, checkpoint_manager, create_checkpoint=False)
        now = datetime.now(UTC)

        # Record terminal outcomes for ALL 5 rows
        with landscape_db.engine.connect() as conn:
            for i in range(5):
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"outcome-all-{i:03d}",
                        run_id=run_id,
                        token_id=f"tok-unproc-{i:03d}",
                        outcome=RowOutcome.COMPLETED.value,
                        is_terminal=1,
                        recorded_at=now,
                        sink_name="output",
                    )
                )
            conn.commit()

        # Create checkpoint at sequence 4 (the last row)
        from tests.core.checkpoint.conftest import _create_test_graph

        graph = _create_test_graph(checkpoint_node="node-unproc")
        checkpoint_manager.create_checkpoint(run_id, "tok-unproc-004", "node-unproc", 4, graph=graph)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert unprocessed == []

    def test_handles_nonexistent_run_id_gracefully(self, recovery_manager: RecoveryManager) -> None:
        """Returns empty list for a run_id that does not exist."""
        unprocessed = recovery_manager.get_unprocessed_rows("nonexistent-run-id")

        assert unprocessed == []


class TestResumeCheck:
    """Tests for ResumeCheck dataclass."""

    def test_can_resume_true(self) -> None:
        """can_resume=True should have no reason."""
        from elspeth.core.checkpoint.recovery import ResumeCheck

        check = ResumeCheck(can_resume=True)
        assert check.can_resume is True
        assert check.reason is None

    def test_can_resume_false_with_reason(self) -> None:
        """can_resume=False must have a reason."""
        from elspeth.core.checkpoint.recovery import ResumeCheck

        check = ResumeCheck(can_resume=False, reason="Run completed")
        assert check.can_resume is False
        assert check.reason == "Run completed"

    def test_can_resume_true_with_reason_raises(self) -> None:
        """can_resume=True with reason should raise."""
        from elspeth.core.checkpoint.recovery import ResumeCheck

        with pytest.raises(ValueError, match="should not have a reason"):
            ResumeCheck(can_resume=True, reason="unexpected")

    def test_can_resume_false_without_reason_raises(self) -> None:
        """can_resume=False without reason should raise."""
        from elspeth.core.checkpoint.recovery import ResumeCheck

        with pytest.raises(ValueError, match="must have a reason"):
            ResumeCheck(can_resume=False)

    def test_frozen(self) -> None:
        """ResumeCheck should be immutable."""
        from elspeth.core.checkpoint.recovery import ResumeCheck

        check = ResumeCheck(can_resume=True)
        with pytest.raises(AttributeError):
            check.can_resume = False  # type: ignore[misc]


class TestGetUnprocessedRowsForkScenarios:
    """Tests that verify correct row boundary in fork scenarios.

    These tests expose the bug where sequence_number != row_index.
    """

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    def _setup_fork_scenario(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> str:
        """Create scenario where row 0 forks to 3 tokens, all complete.

        Row 0: forks to 3 tokens (tok-0-a, tok-0-b, tok-0-c) - all have terminal outcomes
        Rows 1-4: No tokens created yet (not started)

        Expected unprocessed: rows 1, 2, 3, 4
        """
        from elspeth.contracts import RowOutcome
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            token_outcomes_table,
            tokens_table,
        )

        run_id = "fork-test-run"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            # Create run
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                )
            )
            # Create node
            conn.execute(
                nodes_table.insert().values(
                    node_id="gate-fork",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.GATE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=now,
                )
            )
            # Create 5 source rows (indices 0-4)
            for i in range(5):
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"row-{i:03d}",
                        run_id=run_id,
                        source_node_id="gate-fork",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
            # Row 0 forks to 3 tokens (simulating fork gate)
            conn.execute(tokens_table.insert().values(token_id="tok-0-a", row_id="row-000", created_at=now))
            conn.execute(tokens_table.insert().values(token_id="tok-0-b", row_id="row-000", created_at=now))
            conn.execute(tokens_table.insert().values(token_id="tok-0-c", row_id="row-000", created_at=now))

            # Record terminal outcomes for all 3 forked tokens
            for suffix in ["a", "b", "c"]:
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"outcome-0-{suffix}",
                        run_id=run_id,
                        token_id=f"tok-0-{suffix}",
                        outcome=RowOutcome.COMPLETED.value,
                        is_terminal=1,
                        recorded_at=now,
                        sink_name=f"sink_{suffix}",
                    )
                )

            conn.commit()

        # Checkpoint at token tok-0-c with sequence_number=3
        # (simulating 3 terminal token events from one source row)
        from tests.core.checkpoint.conftest import _create_test_graph

        graph = _create_test_graph(checkpoint_node="gate-fork")
        checkpoint_manager.create_checkpoint(run_id, "tok-0-c", "gate-fork", sequence_number=3, graph=graph)

        return run_id

    def test_fork_scenario_does_not_skip_unprocessed_rows(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Fork: Row 0 -> 3 tokens. Resume must process rows 1-4, not skip them."""
        run_id = self._setup_fork_scenario(landscape_db, checkpoint_manager)

        # BUG: Old code returns [] because row_index(1,2,3,4) > sequence_number(3) is only true for row 4
        # FIX: Should return rows 1,2,3,4 because row_index > 0 (the checkpointed row's index)
        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # All rows after row 0 should be unprocessed
        assert len(unprocessed) == 4, f"Expected 4 unprocessed rows, got {len(unprocessed)}: {unprocessed}"
        assert "row-001" in unprocessed
        assert "row-002" in unprocessed
        assert "row-003" in unprocessed
        assert "row-004" in unprocessed


class TestGetUnprocessedRowsFailureScenarios:
    """Tests for rows that failed/quarantined without checkpointing."""

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    def _setup_failure_scenario(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> str:
        """Create scenario: rows 0,1 processed, row 2 failed, rows 3,4 pending.

        Row 0: completed (terminal outcome)
        Row 1: completed (terminal outcome)
        Row 2: failed - token exists but NO outcome (crashed before sink)
        Rows 3, 4: not started (no tokens)

        Expected unprocessed: rows 2, 3, 4
        """
        from elspeth.contracts import RowOutcome
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            token_outcomes_table,
            tokens_table,
        )

        run_id = "failure-test-run"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="abc",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="transform-1",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=now,
                )
            )
            # Create 5 source rows
            for i in range(5):
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"row-{i:03d}",
                        run_id=run_id,
                        source_node_id="transform-1",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
            # Tokens for rows 0, 1, 2 (row 2 failed before checkpoint)
            for i in range(3):
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-{i:03d}",
                        row_id=f"row-{i:03d}",
                        created_at=now,
                    )
                )

            # Record terminal outcomes for rows 0 and 1 ONLY
            # Row 2 failed before reaching sink - no outcome recorded
            for i in range(2):
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"outcome-fail-{i:03d}",
                        run_id=run_id,
                        token_id=f"tok-{i:03d}",
                        outcome=RowOutcome.COMPLETED.value,
                        is_terminal=1,
                        recorded_at=now,
                        sink_name="output",
                    )
                )

            conn.commit()

        # Checkpoint at row 1 (row 2 failed before it could checkpoint)
        # sequence_number=2 but we're at row_index=1
        from tests.core.checkpoint.conftest import _create_test_graph

        graph = _create_test_graph(checkpoint_node="transform-1")
        checkpoint_manager.create_checkpoint(run_id, "tok-001", "transform-1", sequence_number=2, graph=graph)

        return run_id

    def test_failure_scenario_includes_failed_row_in_resume(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Failure: Row 2 failed after row 1 checkpointed. Resume must include rows 2,3,4."""
        run_id = self._setup_failure_scenario(landscape_db, checkpoint_manager)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Rows after row 1 (the checkpointed row) should be unprocessed
        assert len(unprocessed) == 3, f"Expected 3 unprocessed rows, got {len(unprocessed)}: {unprocessed}"
        assert "row-002" in unprocessed  # The failed row - must be retried
        assert "row-003" in unprocessed
        assert "row-004" in unprocessed
