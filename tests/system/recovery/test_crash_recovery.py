"""System tests for crash recovery and resume scenarios.

These tests verify that ELSPETH can recover from crashes and resume
processing, producing the same results as uninterrupted runs.

Per the test regime plan: "Recovery idempotence - Resume produces same
result as uninterrupted run."
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from elspeth.contracts import Determinism, NodeType, PluginSchema, RoutingMode, RowOutcome, RunStatus, SourceRow
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult


class _InputSchema(PluginSchema):
    """Input schema for test transforms."""

    id: str
    value: int


class _FailOnceTransform(BaseTransform):
    """Transform that fails on first attempt for specific rows, succeeds on retry.

    Used to test retry and recovery behavior.
    """

    name = "fail_once"
    determinism = Determinism.DETERMINISTIC
    input_schema = _InputSchema
    output_schema = _InputSchema

    _attempt_count: ClassVar[dict[str, int]] = {}
    _fail_row_ids: ClassVar[set[str]] = set()

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    @classmethod
    def configure(cls, fail_row_ids: set[str]) -> None:
        """Configure which row IDs should fail on first attempt."""
        cls._fail_row_ids = fail_row_ids
        cls._attempt_count.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset state for test isolation."""
        cls._attempt_count.clear()
        cls._fail_row_ids.clear()

    def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        row_id = row.get("id", "unknown")
        self._attempt_count[row_id] = self._attempt_count.get(row_id, 0) + 1

        if row_id in self._fail_row_ids and self._attempt_count[row_id] == 1:
            return TransformResult.error({"reason": "simulated_failure"})

        return TransformResult.success({**row, "attempts": self._attempt_count[row_id]}, success_reason={"action": "test"})


def _build_linear_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for testing."""
    graph = ExecutionGraph()

    schema_config = {"schema": {"fields": "dynamic"}}
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name=config.source.name, config=schema_config)

    transform_ids: dict[int, NodeID] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = NodeID(f"transform_{i}")
        transform_ids[i] = node_id
        graph.add_node(node_id, node_type=NodeType.TRANSFORM, plugin_name=t.name, config=schema_config)
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in config.sinks.items():
        node_id = NodeID(f"sink_{sink_name}")
        sink_ids[SinkName(sink_name)] = node_id
        graph.add_node(node_id, node_type=NodeType.SINK, plugin_name=sink.name, config=schema_config)

    if SinkName("default") in sink_ids:
        graph.add_edge(prev, sink_ids[SinkName("default")], label="continue", mode=RoutingMode.MOVE)

    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._default_sink = SinkName("default") if SinkName("default") in sink_ids else next(iter(sink_ids))
    graph._route_resolution_map = {}

    return graph


class TestResumeIdempotence:
    """Tests for resume idempotence - same results whether interrupted or not."""

    def test_resume_produces_same_result(self, tmp_path: Path, payload_store) -> None:
        """Resume after interruption produces same final output.

        This test verifies the recovery idempotence property:
        - Run pipeline completely (baseline)
        - Run pipeline, interrupt at row 2, resume
        - Both should produce identical output

        Steps:
        1. Run pipeline A: source -> transform -> sink (collects 5 rows)
        2. Run pipeline B: source -> transform -> sink, checkpoint at row 2
        3. Mark run B as failed (simulating crash)
        4. Resume run B from checkpoint
        5. Verify: pipeline A output == (pipeline B pre-crash + pipeline B resume)
        """
        import json
        from collections.abc import Iterator

        from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        # ===== Setup: Create schema and plugins =====
        class RowSchema(PluginSchema):
            id: int
            value: int

        class ListSource(_TestSourceBase):
            """Source that yields rows from a list."""

            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class DoublerTransform(BaseTransform):
            """Transform that doubles the value field."""

            name = "doubler"
            input_schema = RowSchema
            output_schema = RowSchema
            determinism = Determinism.DETERMINISTIC

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success({**row, "value": row["value"] * 2}, success_reason={"action": "doubler"})

        class CollectSink(_TestSinkBase):
            """Sink that collects rows."""

            name = "collect_sink"
            results: ClassVar[list[dict[str, Any]]] = []

            def __init__(self) -> None:
                CollectSink.results = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                CollectSink.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc")

            def close(self) -> None:
                pass

        # Test data: 5 rows
        source_data = [{"id": i, "value": (i + 1) * 10} for i in range(5)]
        # Expected after doubler: values = [20, 40, 60, 80, 100]

        # ===== Pipeline A: Run completely (baseline) =====
        db_a = LandscapeDB(f"sqlite:///{tmp_path}/baseline.db")
        source_a = ListSource(source_data)
        transform_a = DoublerTransform()
        sink_a = CollectSink()

        config_a = PipelineConfig(
            source=as_source(source_a),
            transforms=[transform_a],  # type: ignore[list-item]
            sinks={"default": as_sink(sink_a)},
        )

        orchestrator_a = Orchestrator(db_a)
        result_a = orchestrator_a.run(
            config_a,
            graph=_build_linear_graph(config_a),
            payload_store=payload_store,
        )

        assert result_a.status == "completed"
        assert result_a.rows_processed == 5
        baseline_output = list(CollectSink.results)
        assert len(baseline_output) == 5

        db_a.close()

        # ===== Pipeline B: Run with checkpoint at row 2, simulate crash, resume =====
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig

        db_b = LandscapeDB(f"sqlite:///{tmp_path}/resume_test.db")
        checkpoint_mgr = CheckpointManager(db_b)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db_b, payload_store=payload_store)

        # Phase 1: Create a "crashed" run with first 3 rows processed (checkpointed at row 2)
        run = recorder.begin_run(
            config={"test": "resume"},
            canonical_version="sha256-rfc8785-v1",
        )
        run_id = run.run_id

        # Store source schema for resume
        with db_b.engine.connect() as conn:
            from elspeth.core.landscape.schema import runs_table

            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(
                    source_schema_json=json.dumps(
                        {
                            "properties": {"id": {"type": "integer"}, "value": {"type": "integer"}},
                            "required": ["id", "value"],
                        }
                    )
                )
            )
            conn.commit()

        # Register nodes
        recorder.register_node(
            run_id=run_id,
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source",
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode=None, fields=None, is_dynamic=True),
        )
        recorder.register_node(
            run_id=run_id,
            plugin_name="doubler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform_0",
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode=None, fields=None, is_dynamic=True),
        )
        recorder.register_node(
            run_id=run_id,
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink_default",
            determinism=Determinism.IO_WRITE,
            schema_config=SchemaConfig(mode=None, fields=None, is_dynamic=True),
        )
        recorder.register_edge(
            run_id=run_id,
            from_node_id="source",
            to_node_id="transform_0",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        recorder.register_edge(
            run_id=run_id,
            from_node_id="transform_0",
            to_node_id="sink_default",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Create all 5 rows with payloads
        row_ids = []
        token_ids = []
        for i, row_data in enumerate(source_data):
            payload_ref = payload_store.store(json.dumps(row_data).encode("utf-8"))
            row = recorder.create_row(
                run_id=run_id,
                source_node_id="source",
                row_index=i,
                data=row_data,
                payload_ref=payload_ref,
            )
            row_ids.append(row.row_id)
            token = recorder.create_token(row_id=row.row_id)
            token_ids.append(token.token_id)

        # Build graph for checkpoint
        graph_b = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph_b.add_node("source", node_type=NodeType.SOURCE, plugin_name="list_source", config=schema_config)
        graph_b.add_node("transform_0", node_type=NodeType.TRANSFORM, plugin_name="doubler", config=schema_config)
        graph_b.add_node("sink_default", node_type=NodeType.SINK, plugin_name="collect_sink", config=schema_config)
        graph_b.add_edge("source", "transform_0", label="continue", mode=RoutingMode.MOVE)
        graph_b.add_edge("transform_0", "sink_default", label="continue", mode=RoutingMode.MOVE)
        graph_b._sink_id_map = {SinkName("default"): NodeID("sink_default")}
        graph_b._transform_id_map = {0: NodeID("transform_0")}
        graph_b._default_sink = SinkName("default")
        graph_b._route_resolution_map = {}
        graph_b._config_gate_id_map = {}

        # "Simulate" that first 3 rows were processed and checkpointed at row 2
        pre_crash_output = [{"id": i, "value": (i + 1) * 10 * 2} for i in range(3)]

        # Record terminal outcomes for first 3 rows (recovery uses these)
        for i in range(3):
            recorder.record_token_outcome(
                token_id=token_ids[i],
                run_id=run_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="default",
            )

        # Create checkpoint at row 2 (0-indexed, so rows 0, 1, 2 are processed)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=token_ids[2],  # Checkpoint at row 2
            node_id="sink_default",  # Checkpoint at sink (durable)
            sequence_number=3,  # 3 rows processed
            graph=graph_b,
        )

        # Mark run as failed (simulating crash)
        recorder.complete_run(run_id, status=RunStatus.FAILED)

        # Phase 2: Resume and process remaining rows
        recovery_mgr = RecoveryManager(db_b, checkpoint_mgr)

        # Verify can resume
        check = recovery_mgr.can_resume(run_id, graph_b)
        assert check.can_resume, f"Cannot resume: {check.reason}"

        # Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id, graph_b)
        assert resume_point is not None

        # Create fresh plugins for resume
        CollectSink.results = []  # Clear for resume
        source_b = ListSource(source_data)  # Same data
        transform_b = DoublerTransform()
        sink_b = CollectSink()

        config_b = PipelineConfig(
            source=as_source(source_b),
            transforms=[transform_b],  # type: ignore[list-item]
            sinks={"default": as_sink(sink_b)},
        )

        # Build graph for resume
        resume_graph = _build_linear_graph(config_b)

        orchestrator_b = Orchestrator(
            db_b,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        # Resume the run
        result_b = orchestrator_b.resume(
            resume_point,
            config_b,
            resume_graph,
            payload_store=payload_store,
        )

        assert result_b.status == "completed"
        # Only 2 rows should be reprocessed (rows 3 and 4)
        assert result_b.rows_processed == 2

        resumed_output = list(CollectSink.results)
        assert len(resumed_output) == 2

        # ===== Verification: Combined output matches baseline =====
        # Pre-crash output (rows 0, 1, 2) + resumed output (rows 3, 4) = baseline
        combined_output = pre_crash_output + resumed_output
        assert len(combined_output) == 5
        assert combined_output == baseline_output, (
            f"Resume did not produce same result as uninterrupted run.\nExpected: {baseline_output}\nGot: {combined_output}"
        )

        db_b.close()


class TestRetryBehavior:
    """Tests for retry behavior during processing."""

    def test_pipeline_with_failed_transform_records_failure(self, tmp_path: Path, payload_store) -> None:
        """A pipeline that has a failing transform records the failure in the audit trail.

        When a transform returns TransformResult.error():
        1. The error is recorded in the transform_errors table
        2. The row is routed to the on_error sink (or discarded)
        3. The error_details from the TransformResult are preserved

        This test verifies that transform errors are properly captured for audit.
        """
        from collections.abc import Iterator

        from sqlalchemy import select

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.landscape.schema import transform_errors_table
        from elspeth.plugins.results import TransformResult

        # Create a transform that fails for a specific row with on_error configured
        class ErroringTransform(BaseTransform):
            """Transform that returns error for specific row IDs."""

            name = "erroring_transform"
            input_schema = _InputSchema
            output_schema = _InputSchema
            determinism = Determinism.DETERMINISTIC
            _on_error = "discard"  # Route errors to discard (required for error results)

            def __init__(self, fail_ids: set[str]) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self._fail_ids = fail_ids

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                if row["id"] in self._fail_ids:
                    return TransformResult.error(
                        {
                            "reason": "validation_failed",
                            "error": f"Row {row['id']} failed validation",
                        }
                    )
                return TransformResult.success(row, success_reason={"action": "test"})

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        class TestSource(_TestSourceBase):
            name = "test_source"
            output_schema = _InputSchema

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": "row_1", "value": 100})
                yield SourceRow.valid({"id": "row_2", "value": 200})  # Will fail
                yield SourceRow.valid({"id": "row_3", "value": 300})

            def close(self) -> None:
                pass

        source = TestSource()

        class TestSink(_TestSinkBase):
            name = "collect_sink"
            results: ClassVar[list[dict[str, Any]]] = []

            def __init__(self) -> None:
                TestSink.results = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                TestSink.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # Configure transform to fail on "row_2"
        transform = ErroringTransform(fail_ids={"row_2"})

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],  # type: ignore[list-item]
            sinks={"default": as_sink(TestSink())},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_linear_graph(config), payload_store=payload_store)

        # Pipeline completes (errors are handled via routing, not as failures)
        assert result.status == "completed"
        assert result.rows_processed == 3

        # Only 2 rows make it to the sink (row_2 was discarded due to error)
        assert len(TestSink.results) == 2
        sink_ids = {r["id"] for r in TestSink.results}
        assert sink_ids == {"row_1", "row_3"}

        # Verify: The error was recorded in the transform_errors table
        with db.engine.connect() as conn:
            errors = conn.execute(select(transform_errors_table).where(transform_errors_table.c.run_id == result.run_id)).fetchall()

        # Should have exactly 1 error recorded
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"

        # Verify error details
        error = errors[0]
        assert error.destination == "discard"  # Error was discarded

        # Error details should contain our custom error info
        import json

        error_details = json.loads(error.error_details_json)
        assert error_details["reason"] == "validation_failed"
        assert error_details["error"] == "Row row_2 failed validation"

        db.close()


class TestCheckpointRecovery:
    """Tests for checkpoint-based recovery."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and checkpoint manager."""
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph for checkpoint recovery tests."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config=schema_config)
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="test", config=schema_config)
        return graph

    def test_checkpoint_preserves_partial_progress(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Checkpoint saves progress so resume doesn't re-process rows.

        This verifies:
        1. Create a run with 5 rows
        2. Checkpoint at row 2
        3. Simulate crash (run failed)
        4. get_unprocessed_rows() returns only rows 3-4 (not 0-2)
        """
        from datetime import UTC, datetime

        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            token_outcomes_table,
            tokens_table,
        )

        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]

        run_id = "checkpoint-partial-progress-test"
        now = datetime.now(UTC)

        # Create the run and rows directly in the database
        with db.engine.connect() as conn:
            # Create run
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

            # Create source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create transform node
            conn.execute(
                nodes_table.insert().values(
                    node_id="transform",
                    run_id=run_id,
                    plugin_name="test_transform",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create 5 rows with tokens
            for i in range(5):
                row_id = f"row-{i:03d}"
                token_id = f"tok-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id="source",
                        row_index=i,
                        source_data_hash=f"hash-{i}",
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
                if i < 3:
                    conn.execute(
                        token_outcomes_table.insert().values(
                            outcome_id=f"outcome-{i:03d}",
                            run_id=run_id,
                            token_id=token_id,
                            outcome=RowOutcome.COMPLETED.value,
                            is_terminal=1,
                            recorded_at=now,
                            sink_name="default",
                        )
                    )
            conn.commit()

        # Checkpoint at row 2 (token tok-002)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="transform",
            sequence_number=2,
            graph=mock_graph,
        )

        # Verify checkpoint exists
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        assert checkpoint.sequence_number == 2

        # Verify can_resume returns True
        check = recovery_mgr.can_resume(run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Verify unprocessed rows are only 3 and 4 (not 0, 1, 2)
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert len(unprocessed) == 2
        assert unprocessed == ["row-003", "row-004"]

    def test_checkpoint_across_process_restart(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Checkpoint survives process restart (file-based).

        This verifies:
        1. Create checkpoint with a persistent database
        2. Close the database
        3. Reopen the database (simulating process restart)
        4. Checkpoint data is still available and valid
        """
        from datetime import UTC, datetime

        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        tmp_path = test_env["tmp_path"]
        db_path = tmp_path / "restart_test.db"

        # PHASE 1: Create database, run, and checkpoint
        db1 = LandscapeDB(f"sqlite:///{db_path}")
        checkpoint_mgr1 = CheckpointManager(db1)

        run_id = "checkpoint-restart-test"
        now = datetime.now(UTC)

        with db1.engine.connect() as conn:
            # Create run
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

            # Create source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create transform node
            conn.execute(
                nodes_table.insert().values(
                    node_id="transform",
                    run_id=run_id,
                    plugin_name="test_transform",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create a row and token
            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="source",
                    row_index=0,
                    source_data_hash="hash-0",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    created_at=now,
                )
            )
            conn.commit()

        # Create checkpoint
        original_checkpoint = checkpoint_mgr1.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="transform",
            sequence_number=0,
            graph=mock_graph,
            aggregation_state={"buffer": [1, 2, 3], "sum": 6},
        )

        # Close database (simulate process exit)
        db1.close()

        # PHASE 2: Reopen database (simulate process restart)
        db2 = LandscapeDB(f"sqlite:///{db_path}")
        checkpoint_mgr2 = CheckpointManager(db2)
        recovery_mgr2 = RecoveryManager(db2, checkpoint_mgr2)

        # Verify checkpoint is still there
        restored_checkpoint = checkpoint_mgr2.get_latest_checkpoint(run_id)
        assert restored_checkpoint is not None

        # Verify all checkpoint fields match
        assert restored_checkpoint.checkpoint_id == original_checkpoint.checkpoint_id
        assert restored_checkpoint.run_id == original_checkpoint.run_id
        assert restored_checkpoint.token_id == original_checkpoint.token_id
        assert restored_checkpoint.node_id == original_checkpoint.node_id
        assert restored_checkpoint.sequence_number == original_checkpoint.sequence_number
        assert restored_checkpoint.upstream_topology_hash == original_checkpoint.upstream_topology_hash
        assert restored_checkpoint.checkpoint_node_config_hash == original_checkpoint.checkpoint_node_config_hash

        # Verify can_resume works with restored checkpoint
        check = recovery_mgr2.can_resume(run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Verify resume point with aggregation state
        resume_point = recovery_mgr2.get_resume_point(run_id, mock_graph)
        assert resume_point is not None
        assert resume_point.aggregation_state is not None
        assert resume_point.aggregation_state["buffer"] == [1, 2, 3]
        assert resume_point.aggregation_state["sum"] == 6

        db2.close()


class TestAggregationRecovery:
    """Tests for recovery of aggregation-in-progress."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and checkpoint manager."""
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        recorder = LandscapeRecorder(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "recorder": recorder,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph for aggregation recovery tests."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        agg_config = {
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {"schema": {"fields": "dynamic"}},
            "schema": {"fields": "dynamic"},
        }
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config=schema_config)
        graph.add_node("aggregator", node_type=NodeType.AGGREGATION, plugin_name="sum_agg", config=agg_config)
        return graph

    def test_aggregation_state_recovers(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Aggregation state is recovered after crash.

        Aggregations hold state (collected rows). Recovery must restore
        this state to produce correct results.

        This verifies:
        1. Create run with partial aggregation (3 rows collected, trigger at 5)
        2. Checkpoint with aggregation state (buffer, count, sum)
        3. Simulate crash (run failed)
        4. Recovery restores aggregation state exactly
        """
        from datetime import UTC, datetime

        from elspeth.core.landscape.schema import nodes_table

        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        recorder = test_env["recorder"]

        # Create run
        run = recorder.begin_run(
            config={"aggregation": {"trigger": {"count": 5}}},
            canonical_version="sha256-rfc8785-v1",
        )

        # Register nodes using raw SQL
        now = datetime.now(UTC)
        with db.engine.connect() as conn:
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
            conn.execute(
                nodes_table.insert().values(
                    node_id="aggregator",
                    run_id=run.run_id,
                    plugin_name="sum_agg",
                    node_type=NodeType.AGGREGATION,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.commit()

        # Create 3 rows (partial aggregation - trigger is at 5)
        tokens = []
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": (i + 1) * 100},  # values: 100, 200, 300
            )
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create aggregation state (buffer of 3 rows, sum=600)
        agg_state = {
            "buffer": [
                {"id": 0, "value": 100},
                {"id": 1, "value": 200},
                {"id": 2, "value": 300},
            ],
            "count": 3,
            "sum": 600,
            "expected_trigger": 5,
        }

        # Create checkpoint with aggregation state
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="aggregator",
            sequence_number=2,
            aggregation_state=agg_state,
            graph=mock_graph,
        )

        # Simulate crash
        recorder.complete_run(run.run_id, status=RunStatus.FAILED)

        # Verify can resume
        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Get resume point with aggregation state
        resume_point = recovery_mgr.get_resume_point(run.run_id, mock_graph)
        assert resume_point is not None

        # Verify aggregation state is restored exactly
        assert resume_point.aggregation_state is not None
        restored_state = resume_point.aggregation_state

        assert restored_state["count"] == 3
        assert restored_state["sum"] == 600
        assert restored_state["expected_trigger"] == 5
        assert len(restored_state["buffer"]) == 3

        # Verify buffer contents
        assert restored_state["buffer"][0] == {"id": 0, "value": 100}
        assert restored_state["buffer"][1] == {"id": 1, "value": 200}
        assert restored_state["buffer"][2] == {"id": 2, "value": 300}
