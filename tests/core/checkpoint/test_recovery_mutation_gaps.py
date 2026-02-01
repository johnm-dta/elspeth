# tests/core/checkpoint/test_recovery_mutation_gaps.py
"""Tests specifically targeting mutation testing gaps in recovery.py.

These tests were written to kill surviving mutants found during mutation testing.
Each test targets specific lines where mutations survived, indicating weak coverage.

Mutation testing run: 2026-01-23 (partial, 59% complete)
Survivors in recovery.py: 13 unique lines
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.checkpoint.recovery import RecoveryManager, ResumeCheck, ResumePoint
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.landscape import LandscapeDB


def _create_run_with_checkpoint_prerequisites(
    db: LandscapeDB,
    run_id: str = "test-run",
    token_id: str = "test-token",
    row_id: str = "test-row",
    node_id: str = "test-node",
    status: RunStatus = RunStatus.FAILED,
) -> None:
    """Helper to create run, node, row, and token needed for checkpoint FK constraints.

    This creates the minimum required data for a checkpoint to be valid:
    - Run entry
    - Node entry (source)
    - Row entry (points to source node)
    - Token entry (points to row)
    """
    now = datetime.now(UTC)
    with db.engine.connect() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test-hash",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=status,
            )
        )
        conn.execute(
            nodes_table.insert().values(
                node_id=node_id,
                run_id=run_id,
                plugin_name="test",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="x",
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
                source_data_hash="hash",
                created_at=now,
            )
        )
        conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id=row_id,
                created_at=now,
            )
        )
        conn.commit()


# =============================================================================
# Tests for ResumeCheck dataclass validation (lines 31, 35, 37)
# Mutations: changing field types, boolean logic in __post_init__
# =============================================================================


class TestResumeCheckDataclass:
    """Verify ResumeCheck dataclass invariants are enforced.

    The dataclass has constraints:
    - can_resume=True MUST have reason=None
    - can_resume=False MUST have reason set
    """

    def test_can_resume_true_with_no_reason_is_valid(self) -> None:
        """Line 31: can_resume=True with reason=None should work."""
        check = ResumeCheck(can_resume=True)
        assert check.can_resume is True
        assert check.reason is None

    def test_can_resume_true_with_reason_raises(self) -> None:
        """Line 35: can_resume=True with reason set should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ResumeCheck(can_resume=True, reason="Should not have a reason")

        assert "can_resume=True should not have a reason" in str(exc_info.value)

    def test_can_resume_false_with_reason_is_valid(self) -> None:
        """Line 37: can_resume=False with reason set should work."""
        check = ResumeCheck(can_resume=False, reason="Run not found")
        assert check.can_resume is False
        assert check.reason == "Run not found"

    def test_can_resume_false_without_reason_raises(self) -> None:
        """Line 37: can_resume=False without reason should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ResumeCheck(can_resume=False)  # Missing reason

        assert "can_resume=False must have a reason" in str(exc_info.value)

    def test_can_resume_field_is_bool(self) -> None:
        """Line 31: Ensure can_resume is actually a boolean field."""
        check = ResumeCheck(can_resume=True)
        assert isinstance(check.can_resume, bool)

        check_false = ResumeCheck(can_resume=False, reason="test")
        assert isinstance(check_false.can_resume, bool)


# =============================================================================
# Tests for RecoveryManager.can_resume() branch coverage (lines 100, 103, 106, 110)
# Mutations: changing None checks, status comparisons
# =============================================================================


class TestCanResumeBranches:
    """Test all branches in RecoveryManager.can_resume()."""

    @pytest.fixture
    def in_memory_db(self) -> LandscapeDB:
        """Create in-memory database."""
        from elspeth.core.landscape import LandscapeDB

        return LandscapeDB.in_memory()

    @pytest.fixture
    def checkpoint_manager(self, in_memory_db: LandscapeDB) -> CheckpointManager:
        """Create checkpoint manager."""
        from elspeth.core.checkpoint import CheckpointManager

        return CheckpointManager(in_memory_db)

    @pytest.fixture
    def recovery_manager(self, in_memory_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        """Create recovery manager."""
        return RecoveryManager(in_memory_db, checkpoint_manager)

    @pytest.fixture
    def recorder(self, in_memory_db: LandscapeDB) -> Any:
        """Create landscape recorder."""
        from elspeth.core.landscape import LandscapeRecorder

        return LandscapeRecorder(in_memory_db)

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a simple mock graph for recovery tests."""
        graph = ExecutionGraph()
        graph.add_node("test-node", node_type=NodeType.TRANSFORM, plugin_name="test")
        return graph

    def test_nonexistent_run_returns_cannot_resume(self, recovery_manager: RecoveryManager, mock_graph: ExecutionGraph) -> None:
        """Line 100: Non-existent run returns can_resume=False."""
        result = recovery_manager.can_resume("nonexistent-run-id", mock_graph)

        assert result.can_resume is False
        assert "not found" in result.reason.lower()  # type: ignore[union-attr]

    def test_completed_run_returns_cannot_resume(
        self, recovery_manager: RecoveryManager, recorder: Any, mock_graph: ExecutionGraph
    ) -> None:
        """Line 103: Completed run returns can_resume=False."""
        # Create a completed run
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
            status=RunStatus.COMPLETED,
        )

        result = recovery_manager.can_resume(run.run_id, mock_graph)

        assert result.can_resume is False
        assert "completed" in result.reason.lower()  # type: ignore[union-attr]

    def test_running_run_returns_cannot_resume(self, recovery_manager: RecoveryManager, recorder: Any, mock_graph: ExecutionGraph) -> None:
        """Line 106: Running run returns can_resume=False."""
        # Create a running run
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
            status=RunStatus.RUNNING,
        )

        result = recovery_manager.can_resume(run.run_id, mock_graph)

        assert result.can_resume is False
        assert "in progress" in result.reason.lower()  # type: ignore[union-attr]

    def test_failed_run_without_checkpoint_returns_cannot_resume(
        self, recovery_manager: RecoveryManager, recorder: Any, mock_graph: ExecutionGraph
    ) -> None:
        """Line 110: Failed run without checkpoint returns can_resume=False."""
        # Create a failed run (no checkpoint)
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
            status=RunStatus.FAILED,
        )

        result = recovery_manager.can_resume(run.run_id, mock_graph)

        assert result.can_resume is False
        assert "checkpoint" in result.reason.lower()  # type: ignore[union-attr]

    def test_failed_run_with_checkpoint_returns_can_resume(
        self,
        recovery_manager: RecoveryManager,
        in_memory_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        mock_graph: ExecutionGraph,
    ) -> None:
        """Full success path: Failed run with checkpoint returns can_resume=True."""
        run_id = "test-failed-run"
        token_id = "test-token"

        # Create all prerequisite records (run, node, row, token)
        _create_run_with_checkpoint_prerequisites(
            in_memory_db,
            run_id=run_id,
            token_id=token_id,
            status=RunStatus.FAILED,
        )

        # Create a checkpoint for it
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=token_id,
            node_id="test-node",
            sequence_number=1,
            graph=mock_graph,
        )

        result = recovery_manager.can_resume(run_id, mock_graph)

        assert result.can_resume is True
        assert result.reason is None


# =============================================================================
# Tests for get_resume_point() (line 138)
# =============================================================================


class TestGetResumePoint:
    """Test get_resume_point returns correct ResumePoint or None."""

    @pytest.fixture
    def in_memory_db(self) -> LandscapeDB:
        """Create in-memory database."""
        from elspeth.core.landscape import LandscapeDB

        return LandscapeDB.in_memory()

    @pytest.fixture
    def checkpoint_manager(self, in_memory_db: LandscapeDB) -> CheckpointManager:
        """Create checkpoint manager."""
        from elspeth.core.checkpoint import CheckpointManager

        return CheckpointManager(in_memory_db)

    @pytest.fixture
    def recovery_manager(self, in_memory_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        """Create recovery manager."""
        return RecoveryManager(in_memory_db, checkpoint_manager)

    @pytest.fixture
    def recorder(self, in_memory_db: LandscapeDB) -> Any:
        """Create landscape recorder."""
        from elspeth.core.landscape import LandscapeRecorder

        return LandscapeRecorder(in_memory_db)

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a simple mock graph for recovery tests."""
        graph = ExecutionGraph()
        graph.add_node("test-node", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("test-node-456", node_type=NodeType.TRANSFORM, plugin_name="test")
        return graph

    def test_get_resume_point_returns_none_for_nonresumable(self, recovery_manager: RecoveryManager, mock_graph: ExecutionGraph) -> None:
        """Line 138 area: Non-resumable run returns None."""
        result = recovery_manager.get_resume_point("nonexistent-run", mock_graph)

        assert result is None

    def test_get_resume_point_returns_resume_point_for_resumable(
        self,
        recovery_manager: RecoveryManager,
        in_memory_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        mock_graph: ExecutionGraph,
    ) -> None:
        """Line 138: Resumable run returns ResumePoint with all fields."""
        run_id = "test-resume-run"
        token_id = "test-token-123"

        # Create all prerequisite records
        _create_run_with_checkpoint_prerequisites(
            in_memory_db,
            run_id=run_id,
            token_id=token_id,
            node_id="test-node-456",
            status=RunStatus.FAILED,
        )

        # Create a checkpoint with aggregation state
        agg_state = {"count": 42, "sum": 100.0}
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=token_id,
            node_id="test-node-456",
            sequence_number=5,
            aggregation_state=agg_state,  # Accepts dict, serialized internally
            graph=mock_graph,
        )

        result = recovery_manager.get_resume_point(run_id, mock_graph)

        assert result is not None
        assert isinstance(result, ResumePoint)
        assert result.token_id == token_id
        assert result.node_id == "test-node-456"
        assert result.sequence_number == 5
        assert result.aggregation_state == agg_state

    def test_get_resume_point_with_no_aggregation_state(
        self,
        recovery_manager: RecoveryManager,
        in_memory_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        mock_graph: ExecutionGraph,
    ) -> None:
        """ResumePoint with no aggregation state has aggregation_state=None."""
        run_id = "test-no-agg-run"
        token_id = "test-token"

        # Create all prerequisite records
        _create_run_with_checkpoint_prerequisites(
            in_memory_db,
            run_id=run_id,
            token_id=token_id,
            status=RunStatus.FAILED,
        )

        # Checkpoint without aggregation_state_json
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=token_id,
            node_id="test-node",
            sequence_number=1,
            graph=mock_graph,
        )

        result = recovery_manager.get_resume_point(run_id, mock_graph)

        assert result is not None
        assert result.aggregation_state is None


# =============================================================================
# Tests for get_unprocessed_row_data() error paths (lines 186, 192, 199)
# Mutations: changing None checks, exception type
# =============================================================================


class TestGetUnprocessedRowDataErrors:
    """Test error handling in get_unprocessed_row_data()."""

    @pytest.fixture
    def in_memory_db(self) -> LandscapeDB:
        """Create in-memory database."""
        from elspeth.core.landscape import LandscapeDB

        return LandscapeDB.in_memory()

    @pytest.fixture
    def checkpoint_manager(self, in_memory_db: LandscapeDB) -> CheckpointManager:
        """Create checkpoint manager."""
        from elspeth.core.checkpoint import CheckpointManager

        return CheckpointManager(in_memory_db)

    @pytest.fixture
    def recovery_manager(self, in_memory_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        """Create recovery manager."""
        return RecoveryManager(in_memory_db, checkpoint_manager)

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a simple mock graph for recovery tests."""
        graph = ExecutionGraph()
        graph.add_node("test-node", node_type=NodeType.TRANSFORM, plugin_name="test")
        return graph

    def test_empty_unprocessed_rows_returns_empty_list(self, recovery_manager: RecoveryManager, mock_graph: ExecutionGraph) -> None:
        """Line 174-175: When no unprocessed rows, return empty list."""
        # Mock get_unprocessed_rows to return empty
        recovery_manager.get_unprocessed_rows = MagicMock(return_value=[])  # type: ignore[method-assign]

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        mock_payload_store = MagicMock()
        result = recovery_manager.get_unprocessed_row_data("run-id", mock_payload_store, source_schema_class=mock_schema)

        assert result == []

    def test_row_not_found_raises_value_error(
        self, recovery_manager: RecoveryManager, in_memory_db: LandscapeDB, mock_graph: ExecutionGraph
    ) -> None:
        """Line 186: Row not in database raises ValueError."""
        # Mock get_unprocessed_rows to return a row_id that doesn't exist
        recovery_manager.get_unprocessed_rows = MagicMock(return_value=["nonexistent-row"])  # type: ignore[method-assign]

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        mock_payload_store = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            recovery_manager.get_unprocessed_row_data("run-id", mock_payload_store, source_schema_class=mock_schema)

        assert "not found in database" in str(exc_info.value)
        assert "nonexistent-row" in str(exc_info.value)

    def test_missing_source_data_ref_raises_value_error(
        self, recovery_manager: RecoveryManager, in_memory_db: LandscapeDB, mock_graph: ExecutionGraph
    ) -> None:
        """Line 192: Row with no source_data_ref raises ValueError."""
        run_id = "test-no-ref-run"
        row_id = "test-row-no-ref"
        now = datetime.now(UTC)

        # Create run, node, and row WITHOUT source_data_ref
        with in_memory_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test-hash",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )
            # Row with source_data_ref=None (simulating legacy or bug)
            conn.execute(
                rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="source-node",
                    row_index=0,
                    source_data_hash="hash123",
                    source_data_ref=None,  # No ref!
                    created_at=now,
                )
            )
            conn.commit()

        # Mock get_unprocessed_rows to return this row
        recovery_manager.get_unprocessed_rows = MagicMock(return_value=[row_id])  # type: ignore[method-assign]

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        mock_payload_store = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            recovery_manager.get_unprocessed_row_data(run_id, mock_payload_store, source_schema_class=mock_schema)

        assert "no source_data_ref" in str(exc_info.value)
        assert "cannot resume without payload" in str(exc_info.value)

    def test_purged_payload_raises_value_error(
        self, recovery_manager: RecoveryManager, in_memory_db: LandscapeDB, mock_graph: ExecutionGraph
    ) -> None:
        """Line 199: Purged payload (KeyError) raises ValueError."""
        run_id = "test-purged-run"
        row_id = "test-row-purged"
        now = datetime.now(UTC)

        # Create run, node, and row WITH source_data_ref
        with in_memory_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test-hash",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="source-node",
                    row_index=0,
                    source_data_hash="hash123",
                    source_data_ref="ref-that-was-purged",
                    created_at=now,
                )
            )
            conn.commit()

        # Mock get_unprocessed_rows to return this row
        recovery_manager.get_unprocessed_rows = MagicMock(return_value=[row_id])  # type: ignore[method-assign]

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        # Mock payload store to raise KeyError (simulating purged data)
        mock_payload_store = MagicMock()
        mock_payload_store.retrieve.side_effect = KeyError("ref-that-was-purged")

        with pytest.raises(ValueError) as exc_info:
            recovery_manager.get_unprocessed_row_data(run_id, mock_payload_store, source_schema_class=mock_schema)

        assert "purged" in str(exc_info.value).lower()
        assert "cannot resume" in str(exc_info.value)


# =============================================================================
# Tests for get_unprocessed_rows() error path (lines 243, 244)
# Mutations: changing RuntimeError, error message
# =============================================================================


class TestGetUnprocessedRowsErrors:
    """Test error handling in get_unprocessed_rows()."""

    @pytest.fixture
    def in_memory_db(self) -> LandscapeDB:
        """Create in-memory database."""
        from elspeth.core.landscape import LandscapeDB

        return LandscapeDB.in_memory()

    @pytest.fixture
    def checkpoint_manager(self, in_memory_db: LandscapeDB) -> CheckpointManager:
        """Create checkpoint manager."""
        from elspeth.core.checkpoint import CheckpointManager

        return CheckpointManager(in_memory_db)

    @pytest.fixture
    def recovery_manager(self, in_memory_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        """Create recovery manager."""
        return RecoveryManager(in_memory_db, checkpoint_manager)

    @pytest.fixture
    def recorder(self, in_memory_db: LandscapeDB) -> Any:
        """Create landscape recorder."""
        from elspeth.core.landscape import LandscapeRecorder

        return LandscapeRecorder(in_memory_db)

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a simple mock graph for recovery tests."""
        graph = ExecutionGraph()
        graph.add_node("test-node", node_type=NodeType.TRANSFORM, plugin_name="test")
        return graph

    def test_no_checkpoint_returns_empty_list(self, recovery_manager: RecoveryManager, mock_graph: ExecutionGraph) -> None:
        """Lines 223-225: No checkpoint returns empty list."""
        result = recovery_manager.get_unprocessed_rows("nonexistent-run")

        assert result == []

    def test_checkpoint_with_deleted_token_returns_row_as_unprocessed(
        self,
        recovery_manager: RecoveryManager,
        in_memory_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        mock_graph: ExecutionGraph,
    ) -> None:
        """Deleted token results in row being returned as unprocessed.

        After P1-2026-01-22-recovery-skips-rows-multi-sink fix, the implementation
        uses token_outcomes to determine which rows are processed, not token lookups.
        If a token is deleted (DB corruption), the row won't have terminal outcomes,
        so it will correctly be returned as unprocessed for reprocessing.

        This is the correct behavior: corrupted data should be recoverable, not crash.
        """
        from sqlalchemy import text

        run_id = "test-corrupt-run"
        token_id = "token-to-delete"
        row_id = "test-row"

        # Create all prerequisite records
        _create_run_with_checkpoint_prerequisites(
            in_memory_db,
            run_id=run_id,
            token_id=token_id,
            row_id=row_id,
            status=RunStatus.FAILED,
        )

        # Create a checkpoint pointing to the token
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=token_id,
            node_id="test-node",
            sequence_number=1,
            graph=mock_graph,
        )

        # Simulate database corruption by deleting the token
        # Must disable FK constraints to do this in SQLite
        with in_memory_db.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = OFF"))
            conn.execute(text(f"DELETE FROM tokens WHERE token_id = '{token_id}'"))
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.commit()

        # With outcome-based detection, the row has no terminal outcome,
        # so it should be returned as unprocessed (correct recovery behavior)
        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert len(unprocessed) == 1
        assert row_id in unprocessed
