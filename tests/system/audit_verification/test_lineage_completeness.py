"""System tests for audit lineage completeness.

These tests verify that every row processed through a full pipeline
has complete lineage available via explain queries.

Per ELSPETH's guiding principle: "I don't know what happened" is never
an acceptable answer for any output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from elspeth.contracts import Determinism, NodeType, PluginSchema, RoutingMode, SourceRow
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


class _OutputSchema(PluginSchema):
    """Output schema for test transforms."""

    id: str
    value: int
    enriched: bool | None = None
    processed_by: str | None = None


class _PassthroughTransform(BaseTransform):
    """Transform that passes data through unchanged."""

    name = "passthrough"
    determinism = Determinism.DETERMINISTIC
    input_schema = _InputSchema
    output_schema = _InputSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        return TransformResult.success(row, success_reason={"action": "passthrough"})


class _EnrichingTransform(BaseTransform):
    """Transform that adds a field to the data."""

    name = "enricher"
    determinism = Determinism.DETERMINISTIC
    input_schema = _InputSchema
    output_schema = _OutputSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        enriched = {**row, "enriched": True, "processed_by": self.name}
        return TransformResult.success(enriched, success_reason={"action": "enrich"})


def _build_linear_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for testing.

    Creates: source -> transforms... -> sinks
    """
    graph = ExecutionGraph()
    schema_config = {"schema": {"fields": "dynamic"}}

    # Add source
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name=config.source.name, config=schema_config)

    # Add transforms
    transform_ids: dict[int, NodeID] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = NodeID(f"transform_{i}")
        transform_ids[i] = node_id
        graph.add_node(node_id, node_type=NodeType.TRANSFORM, plugin_name=t.name, config=schema_config)
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks
    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in config.sinks.items():
        node_id = NodeID(f"sink_{sink_name}")
        sink_ids[SinkName(sink_name)] = node_id
        graph.add_node(node_id, node_type=NodeType.SINK, plugin_name=sink.name, config=schema_config)

    # Connect last transform to default sink
    if SinkName("default") in sink_ids:
        graph.add_edge(prev, sink_ids[SinkName("default")], label="continue", mode=RoutingMode.MOVE)

    # Populate internal maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._default_sink = SinkName("default") if SinkName("default") in sink_ids else next(iter(sink_ids))
    graph._route_resolution_map = {}

    return graph


class TestLineageCompleteness:
    """Tests for verifying lineage is complete for all processed rows."""

    def test_simple_pipeline_runs(self, tmp_path: Path, payload_store) -> None:
        """Simple pipeline: source -> transform -> sink runs successfully."""
        from elspeth.contracts import ArtifactDescriptor

        # Setup database (use SQLAlchemy connection string)
        db = LandscapeDB.in_memory()

        # Build source
        class TestSource(_TestSourceBase):
            name = "test_source"
            output_schema = _InputSchema

            def load(self, ctx: Any) -> Any:
                yield SourceRow.valid({"id": "row_1", "value": 100})
                yield SourceRow.valid({"id": "row_2", "value": 200})
                yield SourceRow.valid({"id": "row_3", "value": 300})

            def close(self) -> None:
                pass

        source = TestSource()

        # Build sink that collects output
        class TestSink(_TestSinkBase):
            name = "collect_sink"
            results: ClassVar[list[dict[str, Any]]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                TestSink.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        TestSink.results.clear()

        # Pipeline config
        config = PipelineConfig(
            source=as_source(source),
            transforms=[_PassthroughTransform()],  # type: ignore[list-item]
            sinks={"default": as_sink(TestSink())},
        )

        # Run pipeline
        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_linear_graph(config), payload_store=payload_store)

        # Verify all rows completed
        assert result.status == "completed"
        assert result.rows_processed == 3
        assert len(TestSink.results) == 3

        db.close()

    def test_multi_transform_pipeline_runs(self, tmp_path: Path, payload_store) -> None:
        """Multi-transform pipeline runs successfully."""
        from elspeth.contracts import ArtifactDescriptor

        # Setup database
        db = LandscapeDB.in_memory()

        # Build source
        class TestSource(_TestSourceBase):
            name = "test_source"
            output_schema = _InputSchema

            def load(self, ctx: Any) -> Any:
                yield SourceRow.valid({"id": "doc_1", "value": 10})
                yield SourceRow.valid({"id": "doc_2", "value": 20})

            def close(self) -> None:
                pass

        source = TestSource()

        # Build sink
        class TestSink(_TestSinkBase):
            name = "collect_sink"
            results: ClassVar[list[dict[str, Any]]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                TestSink.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        TestSink.results.clear()

        # Pipeline with multiple transforms
        config = PipelineConfig(
            source=as_source(source),
            transforms=[
                _PassthroughTransform(),  # type: ignore[list-item]
                _EnrichingTransform(),  # type: ignore[list-item]
            ],
            sinks={"default": as_sink(TestSink())},
        )

        # Run pipeline
        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_linear_graph(config), payload_store=payload_store)

        # Verify all rows completed
        assert result.status == "completed"
        assert result.rows_processed == 2

        # Verify output contains enriched data
        assert len(TestSink.results) == 2
        for row in TestSink.results:
            assert row.get("enriched") is True
            assert row.get("processed_by") == "enricher"

        db.close()


class TestLineageAfterRetention:
    """Tests for lineage availability after payload retention."""

    def test_lineage_available_after_payload_purge(self, tmp_path: Path, payload_store) -> None:
        """Lineage hashes remain after payload data is purged.

        ELSPETH design: Hashes survive payload deletion - integrity is
        always verifiable even after payloads are purged for storage reasons.
        """
        from datetime import UTC, datetime, timedelta

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.row_data import RowDataState
        from elspeth.core.payload_store import FilesystemPayloadStore
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB(f"sqlite:///{tmp_path}/lineage.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        class TestSource(_TestSourceBase):
            name = "test_source"
            output_schema = _InputSchema

            def load(self, ctx: Any) -> Any:
                yield SourceRow.valid({"id": "row_1", "value": 100})

            def close(self) -> None:
                pass

        class TestSink(_TestSinkBase):
            name = "collect_sink"

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = TestSource()
        sink = TestSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[_PassthroughTransform()],  # type: ignore[list-item]
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_linear_graph(config), payload_store=payload_store)

        recorder = LandscapeRecorder(db, payload_store=payload_store)
        rows = recorder.get_rows(result.run_id)
        assert rows, "Expected at least one row recorded"
        row = rows[0]
        assert row.source_data_ref is not None
        assert row.source_data_hash is not None

        # Ensure payload exists before purge
        assert payload_store.exists(row.source_data_ref)
        before = recorder.get_row_data(row.row_id)
        assert before.state == RowDataState.AVAILABLE

        # Purge payloads (treat run as expired)
        purge_manager = PurgeManager(db, payload_store)
        as_of = datetime.now(UTC) + timedelta(minutes=1)
        refs = purge_manager.find_expired_payload_refs(retention_days=0, as_of=as_of)
        assert row.source_data_ref in refs
        purge_manager.purge_payloads(refs)

        # Payload should now be purged, but hash remains
        after = recorder.get_row_data(row.row_id)
        assert after.state == RowDataState.PURGED
        lineage = recorder.explain_row(result.run_id, row.row_id)
        assert lineage is not None
        assert lineage.source_data_hash == row.source_data_hash
        assert lineage.payload_available is False
        assert lineage.source_data is None

        db.close()


class TestExplainQueryFunctionality:
    """Tests for explain query functionality."""

    def test_explain_returns_source_data(self, tmp_path: Path, payload_store) -> None:
        """Explain query returns original source data for a row.

        This verifies the fundamental audit requirement: for any processed row,
        we can trace back to the exact source data that was ingested.

        Uses PayloadStore to persist source data, as required by CLAUDE.md:
        "Source entry - Raw data stored before any processing" (non-negotiable)
        """
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.landscape.lineage import explain
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Setup database and payload store
        db = LandscapeDB.in_memory()
        payload_path = tmp_path / "payloads"
        payload_path.mkdir()
        payload_store = FilesystemPayloadStore(payload_path)

        # Source data we want to trace
        source_data = {"id": "trace_me", "value": 42}

        # Build source
        class TestSource(_TestSourceBase):
            name = "test_source"
            output_schema = _InputSchema

            def load(self, ctx: Any) -> Any:
                yield SourceRow.valid(source_data)

            def close(self) -> None:
                pass

        source = TestSource()

        # Build sink
        class TestSink(_TestSinkBase):
            name = "collect_sink"
            results: ClassVar[list[dict[str, Any]]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                TestSink.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        TestSink.results.clear()

        # Pipeline config with a transform
        config = PipelineConfig(
            source=as_source(source),
            transforms=[_PassthroughTransform()],  # type: ignore[list-item]
            sinks={"default": as_sink(TestSink())},
        )

        # Run pipeline with payload_store to persist source data
        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config,
            graph=_build_linear_graph(config),
            payload_store=payload_store,
        )

        assert result.status == "completed"
        assert result.rows_processed == 1

        # Query lineage via explain (with payload_store for data retrieval)
        recorder = LandscapeRecorder(db, payload_store=payload_store)
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 1

        row = rows[0]
        lineage = explain(recorder, run_id=result.run_id, row_id=row.row_id)

        # Verify explain returns lineage with source data
        assert lineage is not None, "Explain must return lineage for processed row"
        assert lineage.source_row is not None, "Lineage must include source_row"
        assert lineage.source_row.payload_available is True, "Payload must be available"
        assert lineage.source_row.source_data == source_data, (
            f"Source data mismatch: expected {source_data}, got {lineage.source_row.source_data}"
        )

        db.close()

    def test_explain_returns_transform_history(self, tmp_path: Path, payload_store) -> None:
        """Explain query returns transformation history for a row.

        This verifies that explain() returns node_states showing each transform
        the row passed through, enabling full audit trail reconstruction.
        """
        from elspeth.contracts import ArtifactDescriptor, NodeStateStatus
        from elspeth.core.landscape.lineage import explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        # Setup database
        db = LandscapeDB.in_memory()

        # Build source
        class TestSource(_TestSourceBase):
            name = "test_source"
            output_schema = _InputSchema

            def load(self, ctx: Any) -> Any:
                yield SourceRow.valid({"id": "history_row", "value": 100})

            def close(self) -> None:
                pass

        source = TestSource()

        # Build sink
        class TestSink(_TestSinkBase):
            name = "collect_sink"
            results: ClassVar[list[dict[str, Any]]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                TestSink.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        TestSink.results.clear()

        # Pipeline with multiple transforms to verify history ordering
        config = PipelineConfig(
            source=as_source(source),
            transforms=[
                _PassthroughTransform(),  # type: ignore[list-item]
                _EnrichingTransform(),  # type: ignore[list-item]
            ],
            sinks={"default": as_sink(TestSink())},
        )

        # Run pipeline
        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_linear_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 1

        # Verify enriched output (proves transforms ran)
        assert len(TestSink.results) == 1
        assert TestSink.results[0]["enriched"] is True

        # Query lineage via explain
        recorder = LandscapeRecorder(db)
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 1

        row = rows[0]
        lineage = explain(recorder, run_id=result.run_id, row_id=row.row_id)

        # Verify explain returns transform history
        assert lineage is not None, "Explain must return lineage for processed row"
        assert len(lineage.node_states) >= 2, f"Expected at least 2 node_states (for 2 transforms), got {len(lineage.node_states)}"

        # Verify node_states are ordered by step_index
        step_indices = [state.step_index for state in lineage.node_states]
        assert step_indices == sorted(step_indices), f"Node states should be ordered by step_index: {step_indices}"

        # Verify all states completed successfully
        for state in lineage.node_states:
            assert state.status == NodeStateStatus.COMPLETED, (
                f"Node state at step {state.step_index} has status {state.status}, expected COMPLETED"
            )

        db.close()
