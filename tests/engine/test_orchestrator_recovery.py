"""Tests for orchestrator crash recovery."""

import json
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism, NodeType, PluginSchema, RoutingMode, RunStatus
from elspeth.contracts.enums import BatchStatus
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig


class RowSchema(PluginSchema):
    """Schema for row data used in recovery tests."""

    id: int
    value: int


class TestOrchestratorResume:
    """Tests for Orchestrator.resume() crash recovery."""

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def orchestrator(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> Orchestrator:
        return Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_manager,
        )

    @pytest.fixture
    def payload_store(self, tmp_path: Path) -> FilesystemPayloadStore:
        """Create test payload store."""
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph for recovery tests."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        agg_config = {
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {"schema": {"fields": "dynamic"}},
            "schema": {"fields": "dynamic"},
        }
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        graph.add_node("agg_node", node_type=NodeType.AGGREGATION, plugin_name="test_agg", config=agg_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json", config=schema_config)
        graph.add_edge("source", "agg_node", label="continue")
        graph.add_edge("agg_node", "sink", label="continue")
        return graph

    @pytest.fixture
    def failed_run_with_batch(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        mock_graph: ExecutionGraph,
        payload_store: FilesystemPayloadStore,
    ) -> dict[str, Any]:
        """Create a failed run with an incomplete batch."""
        # Pass payload_store so rows get source_data_ref recorded for resume
        recorder = LandscapeRecorder(landscape_db, payload_store=payload_store)

        # Create run with source schema for resume type restoration
        source_schema_json = json.dumps(RowSchema.model_json_schema())
        run = recorder.begin_run(config={}, canonical_version="v1", source_schema_json=source_schema_json)

        # Register nodes
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="null",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_make_dynamic_schema(),
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_make_dynamic_schema(),
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink",
            plugin_name="csv",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_make_dynamic_schema(),
        )

        # Register edges (required for resume edge_map)
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="source",
            to_node_id="agg_node",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="agg_node",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Create rows and tokens
        rows = []
        tokens = []
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 100},
            )
            rows.append(row)
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch with members
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        original_members = []
        for i, token in enumerate(tokens):
            recorder.add_batch_member(batch.batch_id, token.token_id, ordinal=i)
            original_members.append(token.token_id)

        # Checkpoint with aggregation state
        agg_state = {"buffer": [0, 100, 200], "sum": 300, "count": 3}
        checkpoint_manager.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="agg_node",
            sequence_number=2,
            aggregation_state=agg_state,
            graph=mock_graph,
        )

        # Simulate crash mid-flush
        recorder.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)
        recorder.complete_run(run.run_id, status=RunStatus.FAILED)

        return {
            "run_id": run.run_id,
            "batch_id": batch.batch_id,
            "batch_attempt": batch.attempt,
            "original_members": original_members,
            "agg_state": agg_state,
        }

    def test_resume_retries_failed_batches(
        self,
        orchestrator: Orchestrator,
        landscape_db: LandscapeDB,
        failed_run_with_batch: dict[str, Any],
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        plugin_manager,
        mock_graph: ExecutionGraph,
    ) -> None:
        """resume() retries batches that were executing when crash occurred.

        P1 Fix: Stronger assertions on:
        - Attempt number incremented
        - Batch members preserved
        - Run completion status
        """
        run_id = failed_run_with_batch["run_id"]
        original_batch_id = failed_run_with_batch["batch_id"]
        original_attempt = failed_run_with_batch["batch_attempt"]
        original_members = failed_run_with_batch["original_members"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, mock_graph)
        assert resume_point is not None

        # Create minimal config for resume
        config = self._create_minimal_config()
        graph = self._create_minimal_graph(plugin_manager)

        # Act
        result = orchestrator.resume(resume_point, config, graph, payload_store=payload_store)

        # Assert: Original batch marked failed
        recorder = LandscapeRecorder(landscape_db)

        original_batch = recorder.get_batch(original_batch_id)
        assert original_batch is not None
        assert original_batch.status == BatchStatus.FAILED, f"Original batch should be FAILED, got {original_batch.status}"

        # P1 Fix: Find retry batch and verify attempt increment
        all_batches = recorder.get_batches(run_id, node_id="agg_node")
        retry_batches = [b for b in all_batches if b.attempt > original_attempt]
        assert len(retry_batches) >= 1, "Should have at least one retry batch"

        retry_batch = retry_batches[0]
        assert retry_batch.attempt == original_attempt + 1, (
            f"Retry batch attempt should be {original_attempt + 1}, got {retry_batch.attempt}"
        )

        # P1 Fix: Verify batch members are preserved in retry
        retry_members = recorder.get_batch_members(retry_batch.batch_id)
        retry_member_token_ids = {m.token_id for m in retry_members}
        original_member_set = set(original_members)
        assert retry_member_token_ids == original_member_set, (
            f"Retry batch should have same members. Expected {original_member_set}, got {retry_member_token_ids}"
        )

        # P1 Fix: Verify run completed successfully (or appropriate status)
        run = recorder.get_run(run_id)
        assert run is not None
        # After resume, run should be completed (or still failed if retry failed)
        assert run.status in (RunStatus.COMPLETED, RunStatus.FAILED), f"Run should be COMPLETED or FAILED after resume, got {run.status}"

        # Verify result reflects the resumed run
        assert result.run_id == run_id

    def _create_minimal_config(self) -> PipelineConfig:
        """Create minimal config for resume testing.

        Uses real plugin types for proper protocol compliance.
        """
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({"schema": {"fields": "dynamic"}})
        sink = JSONSink({"path": "/tmp/test_recovery.json", "schema": {"fields": "dynamic"}, "mode": "write"})

        return PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
        )

    def _create_minimal_graph(self, plugin_manager) -> ExecutionGraph:
        """Create minimal execution graph for resume testing.

        Uses manual node IDs matching the failed_run_with_batch fixture
        to ensure checkpoint/resume compatibility.
        """
        from elspeth.contracts.types import SinkName

        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        agg_config = {
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {"schema": {"fields": "dynamic"}},
            "schema": {"fields": "dynamic"},
        }

        # Must match the node IDs registered in failed_run_with_batch
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        graph.add_node("agg_node", node_type=NodeType.AGGREGATION, plugin_name="test_agg", config=agg_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json", config=schema_config)
        graph.add_edge("source", "agg_node", label="continue")
        graph.add_edge("agg_node", "sink", label="continue")

        # Manually populate ID maps that from_plugin_instances() normally creates
        # Maps sink_name (from PipelineConfig) -> node_id (in graph)
        graph._sink_id_map = {SinkName("default"): "sink"}
        graph._default_sink = "default"

        return graph


def _make_dynamic_schema() -> SchemaConfig:
    """Create a dynamic schema config for test nodes."""
    return SchemaConfig(mode=None, fields=None, is_dynamic=True)
