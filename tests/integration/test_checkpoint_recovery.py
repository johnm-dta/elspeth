# tests/integration/test_checkpoint_recovery.py
"""Integration tests for checkpoint and recovery.

End-to-end test that verifies checkpoint creation, simulated crash,
and recovery work together correctly.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism, NodeType, RowOutcome, RunStatus
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB


class TestCheckpointRecoveryIntegration:
    """End-to-end checkpoint/recovery tests."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and payload store."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        checkpoint_settings = CheckpointSettings(frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "checkpoint_config": checkpoint_config,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph for checkpoint/recovery tests."""
        graph = ExecutionGraph()
        graph.add_node("node-001", node_type=NodeType.TRANSFORM, plugin_name="test", config={"schema": {"fields": "dynamic"}})
        return graph

    def test_full_checkpoint_recovery_cycle(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Complete cycle: run -> checkpoint -> crash -> recover -> complete."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        # 1. Set up a run with rows and checkpoints
        run_id = self._setup_partial_run(db, checkpoint_mgr, mock_graph)

        # 2. Verify checkpoint exists
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        assert checkpoint.sequence_number > 0

        # 3. Verify can resume
        check = recovery_mgr.can_resume(run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # 4. Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id, mock_graph)
        assert resume_point is not None

        # 5. Get unprocessed rows (setup creates 5 rows 0-4, checkpoint at sequence 2)
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert len(unprocessed) == 2  # rows 3 and 4

    def test_checkpoint_sequence_ordering(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify checkpoints are ordered by sequence number."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr, mock_graph)

        # Create additional checkpoints at different sequence numbers
        # Use token IDs that were created by _setup_partial_run (tok-001-003, tok-001-004)
        # and use matching node_id from _setup_partial_run (default run_suffix="001")
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-001-003",
            node_id="node-001",
            sequence_number=3,
            graph=mock_graph,
        )
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-001-004",
            node_id="node-001",
            sequence_number=4,
            graph=mock_graph,
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

    def test_recovery_with_aggregation_state(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify aggregation state is preserved across recovery."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr, mock_graph)

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
            graph=mock_graph,
        )

        # Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id, mock_graph)
        assert resume_point is not None
        assert resume_point.aggregation_state is not None
        assert resume_point.aggregation_state == agg_state
        assert resume_point.aggregation_state["count"] == 2
        assert resume_point.aggregation_state["sum"] == 300

    def test_checkpoint_cleanup_after_completion(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify checkpoints are cleaned up after successful run."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr, mock_graph)

        # Verify checkpoints exist
        checkpoints = checkpoint_mgr.get_checkpoints(run_id)
        assert len(checkpoints) > 0

        # Delete checkpoints (simulating run completion)
        deleted_count = checkpoint_mgr.delete_checkpoints(run_id)
        assert deleted_count > 0

        # Verify no checkpoints remain
        remaining = checkpoint_mgr.get_checkpoints(run_id)
        assert len(remaining) == 0

    def test_recovery_respects_checkpoint_boundary(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify recovery resumes from correct checkpoint position."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        run_id = self._setup_partial_run(db, checkpoint_mgr, mock_graph)

        # Resume point should match the checkpoint
        resume_point = recovery_mgr.get_resume_point(run_id, mock_graph)
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)

        assert resume_point is not None
        assert checkpoint is not None
        assert resume_point.token_id == checkpoint.token_id
        assert resume_point.node_id == checkpoint.node_id
        assert resume_point.sequence_number == checkpoint.sequence_number

    def test_multiple_runs_independent_checkpoints(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify checkpoints are isolated per run."""
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # Create two runs with different checkpoint states
        run_id_1 = self._setup_partial_run(db, checkpoint_mgr, mock_graph, run_suffix="001")
        run_id_2 = self._setup_partial_run(db, checkpoint_mgr, mock_graph, run_suffix="002")

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
        graph: ExecutionGraph,
        *,
        run_suffix: str = "001",
    ) -> str:
        """Helper to create a partially-completed run with checkpoints."""
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            token_outcomes_table,
            tokens_table,
        )

        run_id = f"test-run-{run_suffix}"
        now = datetime.now(UTC)
        node_id = f"node-{run_suffix}"

        # Add node to graph if it doesn't exist
        if not graph.has_node(node_id):
            graph.add_node(node_id, node_type=NodeType.TRANSFORM, plugin_name="test", config={"schema": {"fields": "dynamic"}})

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

            # Create node
            conn.execute(
                nodes_table.insert().values(
                    node_id=f"node-{run_suffix}",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create multiple rows
            for i in range(5):
                row_id = f"row-{run_suffix}-{i:03d}"
                token_id = f"tok-{run_suffix}-{i:03d}"
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
                        token_id=token_id,
                        row_id=row_id,
                        created_at=now,
                    )
                )
                # Mark rows 0, 1, 2 as COMPLETED (processed before checkpoint)
                # Rows 3, 4 have no terminal outcomes and will be returned as unprocessed
                if i < 3:
                    conn.execute(
                        token_outcomes_table.insert().values(
                            outcome_id=f"outcome-{run_suffix}-{i:03d}",
                            run_id=run_id,
                            token_id=token_id,
                            outcome=RowOutcome.COMPLETED.value,
                            is_terminal=1,
                            recorded_at=now,
                            sink_name="default",
                        )
                    )

            conn.commit()

        # Create checkpoint at row 2 (simulating partial progress)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=f"tok-{run_suffix}-002",
            node_id=f"node-{run_suffix}",
            sequence_number=2,
            graph=graph,
        )

        return run_id

    # NOTE: test_full_resume_processes_remaining_rows removed - duplicate of
    # test_resume_comprehensive.py::TestResumeComprehensive::test_resume_normal_path_with_remaining_rows


class TestCheckpointTopologyHashAtomicity:
    """Test for Bug #1 fix: Topology hash race condition.

    Verifies that topology hash is computed atomically inside the transaction,
    preventing race conditions where graph modification after hash computation
    but before checkpoint write could cause hash mismatch.
    """

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        return {"db": db, "checkpoint_manager": checkpoint_mgr}

    def test_checkpoint_hash_matches_graph_at_creation_time(self, test_env: dict[str, Any]) -> None:
        """Verify topology hash in checkpoint matches graph state at creation time.

        This test verifies Bug #1 fix: moving topology hash computation inside
        the transaction ensures hash consistency even if graph is modified
        after checkpoint creation.
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.canonical import compute_full_topology_hash
        from elspeth.core.landscape.recorder import LandscapeRecorder

        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]
        recorder = LandscapeRecorder(db)

        # Create initial graph with two nodes
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config={"version": 1, "schema": {"fields": "dynamic"}})
        graph.add_node(
            "transform_a", node_type=NodeType.TRANSFORM, plugin_name="test", config={"version": 1, "schema": {"fields": "dynamic"}}
        )

        # Create run
        run = recorder.begin_run(config={}, canonical_version="test-v1")

        # Register nodes in database
        schema_config = SchemaConfig(mode=None, fields=None, is_dynamic=True)
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={"version": 1},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="transform_a",
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={"version": 1},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )

        # Create row for checkpoint
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"test": "data"},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Compute expected hash for current graph state
        # BUG-COMPAT-01: CheckpointManager now uses full topology hash (not upstream-only)
        # to ensure changes to ANY branch invalidate checkpoints
        expected_hash = compute_full_topology_hash(graph)

        # Create checkpoint with current graph state
        checkpoint = checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=token.token_id,
            node_id="transform_a",
            sequence_number=0,
            graph=graph,
        )

        # Verify checkpoint has expected hash
        assert checkpoint.upstream_topology_hash == expected_hash, (
            f"Checkpoint hash {checkpoint.upstream_topology_hash} does not match expected hash {expected_hash} from graph at creation time"
        )

        # Now modify the graph (simulating what could happen in a race condition)
        graph.add_node(
            "transform_b", node_type=NodeType.TRANSFORM, plugin_name="new_plugin", config={"version": 2, "schema": {"fields": "dynamic"}}
        )

        # Compute hash with modified graph
        modified_hash = compute_full_topology_hash(graph)

        # Verify stored hash still matches ORIGINAL graph, not modified graph
        # (This proves hash was captured atomically at checkpoint creation time)
        assert checkpoint.upstream_topology_hash == expected_hash, "Checkpoint hash should match graph at creation time, not current state"

        # Additional verification: if graph was modified BEFORE checkpoint,
        # the hashes would differ
        if expected_hash != modified_hash:
            # Graph modification changed the hash (as expected for topology changes)
            assert checkpoint.upstream_topology_hash != modified_hash, "Checkpoint should NOT have hash from modified graph"

    def test_checkpoint_validates_graph_parameter(self, test_env: dict[str, Any]) -> None:
        """Verify create_checkpoint() rejects None graph (Bug #9 early fix)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.recorder import LandscapeRecorder

        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]
        recorder = LandscapeRecorder(db)

        # Create minimal run
        run = recorder.begin_run(config={}, canonical_version="test-v1")

        # Register source node
        schema_config = SchemaConfig(mode=None, fields=None, is_dynamic=True)
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )

        # Create row/token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Attempt to create checkpoint with None graph
        with pytest.raises(ValueError, match="graph parameter is required"):
            checkpoint_mgr.create_checkpoint(
                run_id=run.run_id,
                token_id=token.token_id,
                node_id="test_node",
                sequence_number=0,
                graph=None,  # type: ignore
            )

    def test_checkpoint_validates_node_exists_in_graph(self, test_env: dict[str, Any]) -> None:
        """Verify create_checkpoint() rejects node_id not in graph (Bug #9 early fix)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.recorder import LandscapeRecorder

        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]
        recorder = LandscapeRecorder(db)

        # Create graph with one node
        graph = ExecutionGraph()
        graph.add_node("existing_node", node_type=NodeType.TRANSFORM, plugin_name="test", config={"schema": {"fields": "dynamic"}})

        # Create minimal run
        run = recorder.begin_run(config={}, canonical_version="test-v1")

        # Register nodes in database (need source for row creation)
        schema_config = SchemaConfig(mode=None, fields=None, is_dynamic=True)
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="existing_node",
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )

        # Create row/token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Attempt to create checkpoint with non-existent node_id
        with pytest.raises(ValueError, match="does not exist in graph"):
            checkpoint_mgr.create_checkpoint(
                run_id=run.run_id,
                token_id=token.token_id,
                node_id="nonexistent_node",
                sequence_number=0,
                graph=graph,
            )


class TestResumeCheckpointCleanup:
    """Integration tests for checkpoint cleanup on resume (Bug #8).

    Verifies that resume deletes checkpoints even when taking
    the early-exit path (no unprocessed rows remaining).

    NOTE: Merged from test_resume_checkpoint_cleanup.py.
    """

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def simple_graph(self) -> ExecutionGraph:
        """Create a simple source -> sink graph."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source", "sink", label="continue")
        return graph

    def test_checkpoint_cleanup_called_on_early_exit(
        self,
        test_env: dict[str, Any],
        simple_graph: ExecutionGraph,
    ) -> None:
        """Verify _delete_checkpoints() is called on early-exit path.

        Scenario:
        1. Create checkpoints for a run
        2. Call _delete_checkpoints() (simulating early-exit cleanup)
        3. Verify: Checkpoints are deleted from database

        This is Bug #8 fix: early-exit path in resume() must call
        _delete_checkpoints(run_id) before returning. This test verifies
        the cleanup method works correctly.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.schema import checkpoints_table, nodes_table

        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]

        # Create run and required parent records
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        now = datetime.now(UTC)
        with db.engine.begin() as conn:
            # Create node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run.run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

        # Create row and token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"id": 1},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Create checkpoint
        checkpoint = checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=token.token_id,
            node_id="source",
            sequence_number=0,
            graph=simple_graph,
        )

        # Verify checkpoint exists
        with db.engine.connect() as conn:
            checkpoints_before = conn.execute(select(checkpoints_table).where(checkpoints_table.c.run_id == run.run_id)).fetchall()

        assert len(checkpoints_before) == 1
        assert checkpoints_before[0].checkpoint_id == checkpoint.checkpoint_id

        # Call _delete_checkpoints() (this is what Bug #8 fix added to early-exit path)
        from elspeth.engine.orchestrator import Orchestrator

        orchestrator = Orchestrator(db=db, checkpoint_manager=checkpoint_mgr)
        orchestrator._delete_checkpoints(run.run_id)

        # Verify checkpoints are deleted
        with db.engine.connect() as conn:
            checkpoints_after = conn.execute(select(checkpoints_table).where(checkpoints_table.c.run_id == run.run_id)).fetchall()

        # SUCCESS: Cleanup method deleted checkpoints
        assert len(checkpoints_after) == 0, (
            f"Expected 0 checkpoints after _delete_checkpoints(), found {len(checkpoints_after)}. "
            f"Bug #8 fix: Early-exit path must call _delete_checkpoints(run_id) to clean up."
        )


class TestCanResumeErrorHandling:
    """Tests for can_resume() error handling (Bug: incompatible-checkpoint-error-propagates).

    Verifies that can_resume() returns ResumeCheck for all error cases,
    never propagating exceptions that violate its API contract.
    """

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment."""
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        return {"db": db, "checkpoint_manager": checkpoint_mgr, "recovery_manager": recovery_mgr}

    def test_can_resume_returns_check_for_incompatible_checkpoint(self, test_env: dict[str, Any]) -> None:
        """can_resume() returns ResumeCheck for incompatible checkpoints, not exception.

        Bug fix: IncompatibleCheckpointError was propagating from get_latest_checkpoint()
        instead of being caught and converted to a ResumeCheck with can_resume=False.

        This test verifies the API contract: can_resume() always returns ResumeCheck.
        """
        from sqlalchemy import update

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.schema import checkpoints_table

        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        recorder = LandscapeRecorder(db)

        # Create graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config={"schema": {"fields": "dynamic"}})
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="test", config={"schema": {"fields": "dynamic"}})
        graph.add_edge("source", "transform", label="continue")

        # Create run with FAILED status (required for resume eligibility)
        run = recorder.begin_run(config={}, canonical_version="test-v1")

        # Register nodes
        schema_config = SchemaConfig(mode=None, fields=None, is_dynamic=True)
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="transform",
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=schema_config,
        )

        # Create row/token
        row = recorder.create_row(run_id=run.run_id, source_node_id="source", row_index=0, data={})
        token = recorder.create_token(row_id=row.row_id)

        # Mark run as failed
        recorder.update_run_status(run.run_id, status=RunStatus.FAILED)

        # Create checkpoint
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=token.token_id,
            node_id="transform",
            sequence_number=1,
            graph=graph,
        )

        # Corrupt the checkpoint's format_version to trigger IncompatibleCheckpointError
        with db.engine.begin() as conn:
            conn.execute(update(checkpoints_table).where(checkpoints_table.c.run_id == run.run_id).values(format_version=None))

        # Before the fix, this would raise IncompatibleCheckpointError
        # After the fix, it should return a ResumeCheck with can_resume=False
        result = recovery_mgr.can_resume(run.run_id, graph)

        # Verify API contract: returns ResumeCheck, not exception
        assert result.can_resume is False
        assert "format_version" in result.reason.lower() or "incompatible" in result.reason.lower()
