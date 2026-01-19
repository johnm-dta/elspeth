"""Tests for orchestrator crash recovery."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism, NodeType
from elspeth.contracts.enums import BatchStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig


class TestOrchestratorResume:
    """Tests for Orchestrator.resume() crash recovery."""

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(
        self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager
    ) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def orchestrator(
        self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager
    ) -> Orchestrator:
        return Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_manager,
        )

    @pytest.fixture
    def failed_run_with_batch(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> dict[str, Any]:
        """Create a failed run with an incomplete batch."""
        recorder = LandscapeRecorder(landscape_db)

        # Create run
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="test_source",
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
        for i, token in enumerate(tokens):
            recorder.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Checkpoint with aggregation state
        agg_state = {"buffer": [0, 100, 200], "sum": 300, "count": 3}
        checkpoint_manager.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="agg_node",
            sequence_number=2,
            aggregation_state=agg_state,
        )

        # Simulate crash mid-flush
        recorder.update_batch_status(batch.batch_id, "executing")
        recorder.complete_run(run.run_id, status="failed")

        return {
            "run_id": run.run_id,
            "batch_id": batch.batch_id,
            "agg_state": agg_state,
        }

    def test_resume_method_exists(
        self,
        orchestrator: Orchestrator,
    ) -> None:
        """Orchestrator has resume() method."""
        assert hasattr(orchestrator, "resume")
        assert callable(orchestrator.resume)

    def test_resume_retries_failed_batches(
        self,
        orchestrator: Orchestrator,
        landscape_db: LandscapeDB,
        failed_run_with_batch: dict[str, Any],
        recovery_manager: RecoveryManager,
    ) -> None:
        """resume() retries batches that were executing when crash occurred."""
        run_id = failed_run_with_batch["run_id"]
        original_batch_id = failed_run_with_batch["batch_id"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id)
        assert resume_point is not None

        # Create minimal config for resume
        config = self._create_minimal_config()
        graph = self._create_minimal_graph()

        # Act
        orchestrator.resume(resume_point, config, graph)

        # Assert: Original batch marked failed, retry batch created
        recorder = LandscapeRecorder(landscape_db)

        original_batch = recorder.get_batch(original_batch_id)
        assert original_batch.status == BatchStatus.FAILED

        # Find retry batch
        all_batches = recorder.get_batches(run_id, node_id="agg_node")
        retry_batches = [b for b in all_batches if b.attempt > 0]
        assert len(retry_batches) >= 1

    def _create_minimal_config(self) -> PipelineConfig:
        """Create minimal config for resume testing."""
        # Mock source and sink
        source = Mock()
        source.name = "test_source"
        source.plugin_version = "1.0"
        source.determinism = Determinism.DETERMINISTIC
        source.output_schema = {"fields": "dynamic"}
        source.load = Mock(return_value=[])

        sink = Mock()
        sink.name = "default"
        sink.plugin_version = "1.0"
        sink.determinism = Determinism.DETERMINISTIC
        sink.input_schema = {"fields": "dynamic"}
        sink.write_batch = Mock()

        return PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
        )

    def _create_minimal_graph(self) -> ExecutionGraph:
        """Create minimal execution graph for resume testing."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="test_source"),
            sinks={"default": SinkSettings(plugin="test_sink")},
            output_sink="default",
        )
        return ExecutionGraph.from_config(settings)


def _make_dynamic_schema():
    """Create a dynamic schema config for test nodes."""
    from elspeth.contracts.schema import SchemaConfig

    return SchemaConfig(mode=None, fields=None, is_dynamic=True)
