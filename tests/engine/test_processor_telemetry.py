# tests/engine/test_processor_telemetry.py
"""Tests for telemetry event emission in the RowProcessor.

Tests verify:
1. TransformCompleted is emitted after transform execution
2. GateEvaluated is emitted after gate evaluation
3. TokenCompleted is emitted when tokens reach terminal state
4. Events are emitted ONLY AFTER Landscape recording succeeds
5. TelemetryManager is optional (no telemetry when not provided)
6. Aggregation flushes emit TransformCompleted telemetry (P3-2026-01-31)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
from unittest.mock import MagicMock

from pydantic import ConfigDict

from elspeth.contracts import (
    ArtifactDescriptor,
    Determinism,
    NodeID,
    NodeType,
    PluginSchema,
    RoutingMode,
    RowOutcome,
    SinkName,
    SourceRow,
)
from elspeth.contracts.enums import NodeStateStatus, RunStatus, TelemetryGranularity
from elspeth.contracts.events import (
    GateEvaluated,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import GateResult, RoutingAction, TransformResult
from elspeth.telemetry import TelemetryManager
from tests.conftest import as_gate

# =============================================================================
# Test Fixtures
# =============================================================================


class DynamicSchema(PluginSchema):
    """Simple schema for testing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class PassthroughTransform(BaseTransform):
    """Transform that passes through rows unchanged."""

    name = "passthrough"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "passthrough"})


class FailingTransform(BaseTransform):
    """Transform that returns an error (with on_error configured)."""

    name = "failing"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    plugin_version = "1.0.0"
    _on_error = "discard"  # Route errors to discard (quarantine)

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.error({"reason": "intentional_failure"})


class SimpleGate:
    """Gate that routes all rows to the default path (mock implementation)."""

    name = "simple_gate"
    node_id: str | None = None
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
    routes: ClassVar[dict[str, str]] = {}  # No routes - always continues
    fork_to: list[str] | None = None  # No forking

    def evaluate(self, row: Any, ctx: Any) -> GateResult:
        return GateResult(row=row, action=RoutingAction.continue_())

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook called at start of run."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook called at end of run."""
        pass

    def close(self) -> None:
        """Cleanup method required by orchestrator."""
        pass


@dataclass
class MockTelemetryConfig:
    """Mock RuntimeTelemetryProtocol implementation for testing."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False

    @property
    def backpressure_mode(self) -> Any:
        return None

    @property
    def exporter_configs(self) -> tuple:
        return ()


class RecordingExporter:
    """Exporter that records all events for test verification."""

    def __init__(self, name: str = "recording"):
        self._name = name
        self.events: list[TelemetryEvent] = []

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def create_minimal_graph() -> ExecutionGraph:
    """Create a minimal valid execution graph."""
    graph = ExecutionGraph()
    schema_config = {"schema": {"fields": "dynamic"}}
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
    graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink", config=schema_config)
    graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
    graph._transform_id_map = {0: NodeID("transform")}
    graph._sink_id_map = {SinkName("output"): NodeID("sink")}
    graph._default_sink = "output"
    return graph


def create_graph_with_gate() -> ExecutionGraph:
    """Create a graph with a gate transform."""
    graph = ExecutionGraph()
    schema_config = {"schema": {"fields": "dynamic"}}
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
    graph.add_node("gate", node_type=NodeType.GATE, plugin_name="simple_gate", config=schema_config)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink", config=schema_config)
    graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("gate", "sink", label="continue", mode=RoutingMode.MOVE)
    graph._transform_id_map = {0: NodeID("gate")}
    graph._sink_id_map = {SinkName("output"): NodeID("sink")}
    graph._default_sink = "output"
    return graph


def create_graph_with_failing_transform() -> ExecutionGraph:
    """Create a graph with a failing transform."""
    graph = ExecutionGraph()
    schema_config = {"schema": {"fields": "dynamic"}}
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
    graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="failing", config=schema_config)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink", config=schema_config)
    graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
    graph._transform_id_map = {0: NodeID("transform")}
    graph._sink_id_map = {SinkName("output"): NodeID("sink")}
    graph._default_sink = "output"
    return graph


def create_mock_source(rows: list[dict[str, Any]]) -> MagicMock:
    """Create a mock source that yields specified rows."""
    mock_source = MagicMock()
    mock_source.name = "test_source"
    mock_source._on_validation_failure = "discard"
    mock_source.determinism = Determinism.IO_READ
    mock_source.plugin_version = "1.0.0"

    schema_mock = MagicMock()
    schema_mock.model_json_schema.return_value = {"type": "object"}
    mock_source.output_schema = schema_mock

    mock_source.load.return_value = iter([SourceRow.valid(row) for row in rows])
    mock_source.get_field_resolution.return_value = (None, None)

    return mock_source


def create_mock_sink() -> MagicMock:
    """Create a mock sink."""
    mock_sink = MagicMock()
    mock_sink.name = "test_sink"
    mock_sink.determinism = Determinism.IO_WRITE
    mock_sink.plugin_version = "1.0.0"

    schema_mock = MagicMock()
    schema_mock.model_json_schema.return_value = {"type": "object"}
    mock_sink.input_schema = schema_mock
    mock_sink.write.return_value = ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc123")

    return mock_sink


# =============================================================================
# TransformCompleted Event Tests
# =============================================================================


class TestTransformCompletedTelemetry:
    """Tests for TransformCompleted telemetry event emission."""

    def test_transform_completed_emitted_for_successful_transform(self, landscape_db: LandscapeDB, payload_store) -> None:
        """TransformCompleted is emitted after successful transform execution."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[PassthroughTransform()],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Verify TransformCompleted was emitted
        transform_events = [e for e in exporter.events if isinstance(e, TransformCompleted)]
        assert len(transform_events) == 1

        event = transform_events[0]
        assert event.plugin_name == "passthrough"
        assert event.status == NodeStateStatus.COMPLETED
        assert event.duration_ms >= 0
        assert event.input_hash is not None
        assert event.output_hash is not None

    def test_transform_completed_emitted_for_failed_transform(self, landscape_db: LandscapeDB, payload_store) -> None:
        """TransformCompleted is emitted with FAILED status for transform errors."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[FailingTransform()],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_graph_with_failing_transform(), payload_store=payload_store)

        # Verify TransformCompleted was emitted with FAILED status
        transform_events = [e for e in exporter.events if isinstance(e, TransformCompleted)]
        assert len(transform_events) == 1

        event = transform_events[0]
        assert event.plugin_name == "failing"
        assert event.status == NodeStateStatus.FAILED

    def test_multiple_transforms_emit_multiple_events(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Each transform execution emits its own TransformCompleted event."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        # Create graph with two transforms
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
        graph.add_node("transform1", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("transform2", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink", config=schema_config)
        graph.add_edge("source", "transform1", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform1", "transform2", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform2", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {0: NodeID("transform1"), 1: NodeID("transform2")}
        graph._sink_id_map = {SinkName("output"): NodeID("sink")}
        graph._default_sink = "output"

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[PassthroughTransform(), PassthroughTransform()],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Verify two TransformCompleted events were emitted
        transform_events = [e for e in exporter.events if isinstance(e, TransformCompleted)]
        assert len(transform_events) == 2


# =============================================================================
# GateEvaluated Event Tests
# =============================================================================


class TestGateEvaluatedTelemetry:
    """Tests for GateEvaluated telemetry event emission."""

    def test_gate_evaluated_emitted_for_gate(self, landscape_db: LandscapeDB, payload_store) -> None:
        """GateEvaluated is emitted after gate evaluation."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[as_gate(SimpleGate())],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_graph_with_gate(), payload_store=payload_store)

        # Verify GateEvaluated was emitted
        gate_events = [e for e in exporter.events if isinstance(e, GateEvaluated)]
        assert len(gate_events) == 1

        event = gate_events[0]
        assert event.plugin_name == "simple_gate"
        assert event.destinations == ("continue",)

    def test_gate_evaluated_contains_routing_mode(self, landscape_db: LandscapeDB, payload_store) -> None:
        """GateEvaluated event includes the routing mode."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[as_gate(SimpleGate())],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_graph_with_gate(), payload_store=payload_store)

        gate_events = [e for e in exporter.events if isinstance(e, GateEvaluated)]
        assert len(gate_events) == 1

        event = gate_events[0]
        assert event.routing_mode is not None


# =============================================================================
# TokenCompleted Event Tests
# =============================================================================


class TestTokenCompletedTelemetry:
    """Tests for TokenCompleted telemetry event emission."""

    def test_token_completed_emitted_for_quarantined_token(self, landscape_db: LandscapeDB, payload_store) -> None:
        """TokenCompleted is emitted when a token is quarantined."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[FailingTransform()],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_graph_with_failing_transform(), payload_store=payload_store)

        # Verify TokenCompleted was emitted with QUARANTINED outcome
        token_events = [e for e in exporter.events if isinstance(e, TokenCompleted)]
        quarantined_events = [e for e in token_events if e.outcome == RowOutcome.QUARANTINED]
        assert len(quarantined_events) == 1

    def test_token_completed_contains_outcome(self, landscape_db: LandscapeDB, payload_store) -> None:
        """TokenCompleted event contains the correct outcome."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[FailingTransform()],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_graph_with_failing_transform(), payload_store=payload_store)

        # Find the TokenCompleted event
        token_events = [e for e in exporter.events if isinstance(e, TokenCompleted)]
        assert len(token_events) >= 1

        # Check it has valid outcome
        event = token_events[0]
        assert event.outcome is not None
        assert event.row_id is not None
        assert event.token_id is not None


# =============================================================================
# No Telemetry When Manager Not Provided Tests
# =============================================================================


class TestNoTelemetryWithoutManager:
    """Tests verifying no telemetry is emitted without a TelemetryManager."""

    def test_no_events_without_telemetry_manager(self, landscape_db: LandscapeDB, payload_store) -> None:
        """No telemetry events emitted when TelemetryManager is not provided."""
        # No telemetry_manager provided
        orchestrator = Orchestrator(landscape_db)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[PassthroughTransform()],
            sinks={"output": create_mock_sink()},
        )

        # Should complete without error
        result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)
        assert result.status == RunStatus.COMPLETED


# =============================================================================
# Event Ordering Tests
# =============================================================================


class TestTelemetryEventOrdering:
    """Tests verifying telemetry events are emitted in correct order."""

    def test_transform_completed_before_token_completed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """TransformCompleted should be emitted before TokenCompleted for same row."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[FailingTransform()],  # Will fail and quarantine
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_graph_with_failing_transform(), payload_store=payload_store)

        # Get events in order
        transform_events = [i for i, e in enumerate(exporter.events) if isinstance(e, TransformCompleted)]
        token_events = [i for i, e in enumerate(exporter.events) if isinstance(e, TokenCompleted)]

        # TransformCompleted should come before TokenCompleted
        if transform_events and token_events:
            assert transform_events[0] < token_events[0]

    def test_all_row_events_share_same_run_id(self, landscape_db: LandscapeDB, payload_store) -> None:
        """All row-level telemetry events share the same run_id."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        config = PipelineConfig(
            source=create_mock_source([{"id": 1}]),
            transforms=[PassthroughTransform()],
            sinks={"output": create_mock_sink()},
        )

        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # All events should have the same run_id
        run_ids = {e.run_id for e in exporter.events}
        assert len(run_ids) == 1


# =============================================================================
# Aggregation Flush Telemetry Tests (P3-2026-01-31)
# =============================================================================


class BatchAwareTransformForTelemetry(BaseTransform):
    """Batch-aware transform for telemetry testing."""

    name = "batch_telemetry"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    is_batch_aware = True
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        if isinstance(row, list):
            # Batch mode - aggregate
            total = sum(r.get("value", 0) for r in row)
            return TransformResult.success(
                {"batch_total": total, "count": len(row)},
                success_reason={"action": "batch_aggregate"},
            )
        else:
            # Single row mode
            return TransformResult.success(row, success_reason={"action": "passthrough"})


class FailingBatchAwareTransform(BaseTransform):
    """Batch-aware transform that fails during batch processing.

    Used to test telemetry ordering in failed flush paths.
    """

    name = "failing_batch"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    is_batch_aware = True
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        if isinstance(row, list):
            # Batch mode - return error to simulate flush failure
            return TransformResult.error({"reason": "intentional_batch_failure"})
        else:
            # Single row mode - succeed (shouldn't be called in batch mode)
            return TransformResult.success(row)


class PassthroughBatchAwareTransform(BaseTransform):
    """Batch-aware transform that enriches rows for passthrough mode testing.

    Unlike BatchAwareTransformForTelemetry which aggregates N rows → 1 row,
    this transform returns N enriched rows (1:1 mapping) for passthrough mode.
    """

    name = "batch_passthrough"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    is_batch_aware = True
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        if isinstance(row, list):
            # Batch mode - enrich each row with batch metadata
            enriched_rows = [{**r, "batch_processed": True, "batch_size": len(row)} for r in row]
            return TransformResult.success_multi(
                enriched_rows,
                success_reason={"action": "batch_passthrough"},
            )
        else:
            # Single row mode
            return TransformResult.success(
                {**row, "batch_processed": False},
                success_reason={"action": "passthrough"},
            )


class TelemetryTestSource:
    """Simple test source for telemetry tests."""

    name = "telemetry_test_source"
    output_schema = DynamicSchema
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    node_id: str | None = None
    determinism = Determinism.IO_READ
    plugin_version = "1.0.0"
    _on_validation_failure = "discard"

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.config = {"schema": {"fields": "dynamic"}}

    def load(self, ctx: Any) -> Any:
        for row in self._rows:
            yield SourceRow.valid(row)

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def get_field_resolution(self) -> tuple[dict[str, str], str | None] | None:
        return None


class TelemetryTestSink:
    """Simple test sink for telemetry tests."""

    name = "telemetry_test_sink"
    input_schema = DynamicSchema
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    node_id: str | None = None
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    idempotent = True

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.config = {"schema": {"fields": "dynamic"}}

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        for row in rows:
            self.rows.append(row)
        return ArtifactDescriptor.for_file(
            path="memory://test",
            size_bytes=0,
            content_hash="test123",
        )

    def flush(self) -> None:
        """Flush pending writes (no-op for memory sink)."""
        pass

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass


class TestAggregationFlushTelemetry:
    """Tests for TransformCompleted telemetry on aggregation flushes.

    Bug: P3-2026-01-31-aggregation-flush-missing-telemetry
    - Regular transforms emit TransformCompleted telemetry
    - Aggregation flushes did NOT emit TransformCompleted
    - Fix: Emit TransformCompleted for each buffered token after successful flush
    """

    def test_aggregation_count_flush_emits_transform_completed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Aggregation count trigger flush should emit TransformCompleted for each buffered token."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        # Create a batch-aware transform
        transform = BatchAwareTransformForTelemetry()

        # Create source with 3 rows (will trigger count=3 flush)
        source = TelemetryTestSource(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
            ]
        )

        sink = TelemetryTestSink()

        # Build graph
        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        # Get transform node_id
        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # Configure aggregation with count=3 trigger
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_telemetry",
            trigger=TriggerConfig(count=3),  # Will trigger after 3 rows
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Verify TransformCompleted events were emitted for buffered tokens
        transform_events = [e for e in exporter.events if isinstance(e, TransformCompleted)]

        # We should have 3 TransformCompleted events - one for each buffered token
        assert len(transform_events) == 3, (
            f"Expected 3 TransformCompleted events (one per buffered token), "
            f"got {len(transform_events)}. "
            f"Bug P3-2026-01-31: Aggregation flush should emit TransformCompleted."
        )

        # All should reference the batch_telemetry transform
        for event in transform_events:
            assert event.plugin_name == "batch_telemetry"
            assert event.status == NodeStateStatus.COMPLETED

    def test_aggregation_end_of_source_flush_emits_transform_completed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """End-of-source flush should emit TransformCompleted for buffered tokens."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        transform = BatchAwareTransformForTelemetry()

        # Create source with 2 rows (won't hit count=10, will flush at end-of-source)
        source = TelemetryTestSource(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
            ]
        )

        sink = TelemetryTestSink()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # High count threshold - won't trigger during processing, only at end-of-source
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_telemetry",
            trigger=TriggerConfig(count=10),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Verify TransformCompleted events for the 2 buffered tokens
        transform_events = [e for e in exporter.events if isinstance(e, TransformCompleted)]

        assert len(transform_events) == 2, (
            f"Expected 2 TransformCompleted events for end-of-source flush, "
            f"got {len(transform_events)}. "
            f"Bug P3-2026-01-31: End-of-source flush should emit TransformCompleted."
        )

    def test_transform_mode_aggregation_ordering_bug(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Bug P2-2026-02-01: TransformCompleted must precede TokenCompleted for aggregation.

        In transform-mode aggregation, tokens become terminal (CONSUMED_IN_BATCH) when buffered,
        but TransformCompleted only fires at flush time. This causes TransformCompleted to
        arrive AFTER TokenCompleted for buffered (non-triggering) tokens.
        """
        from elspeth.core.config import AggregationSettings, TriggerConfig

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        transform = BatchAwareTransformForTelemetry()

        # Create source with 3 rows - tokens 1 and 2 will buffer, token 3 triggers flush
        source = TelemetryTestSource(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
            ]
        )

        sink = TelemetryTestSink()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # Configure aggregation with count=3 trigger in TRANSFORM mode
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_telemetry",
            trigger=TriggerConfig(count=3),
            output_mode="transform",  # Crucial: transform mode makes tokens terminal on buffer
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Get events and check ordering per token
        events = exporter.events

        # Find all TransformCompleted and TokenCompleted events
        transform_completed_events = [(i, e) for i, e in enumerate(events) if isinstance(e, TransformCompleted)]
        token_completed_events = [(i, e) for i, e in enumerate(events) if isinstance(e, TokenCompleted)]

        # For EACH token, TransformCompleted MUST come before TokenCompleted
        for tc_idx, tc_event in token_completed_events:
            token_id = tc_event.token_id

            # Find matching TransformCompleted for this token
            matching_transform = [(idx, e) for idx, e in transform_completed_events if e.token_id == token_id]

            if matching_transform:
                tf_idx, _tf_event = matching_transform[0]
                assert tf_idx < tc_idx, (
                    f"Bug P2-2026-02-01: TransformCompleted (index {tf_idx}) arrived AFTER "
                    f"TokenCompleted (index {tc_idx}) for token {token_id}. "
                    f"Transform-mode aggregation emits TokenCompleted at buffer time, "
                    f"but TransformCompleted at flush time - ordering is reversed."
                )

    def test_transform_mode_aggregation_batch_size_one_ordering(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Bug P2-2026-02-01: Batch size=1 triggers flush immediately - verify ordering.

        Edge case: when count=1, every token triggers flush immediately.
        There's no "buffering" period, so ordering should still be correct.
        """
        from elspeth.core.config import AggregationSettings, TriggerConfig

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        transform = BatchAwareTransformForTelemetry()

        # 3 rows, each triggers flush immediately (count=1)
        source = TelemetryTestSource(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
            ]
        )

        sink = TelemetryTestSink()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # Batch size = 1: every token triggers flush immediately
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_telemetry",
            trigger=TriggerConfig(count=1),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Verify ordering for each token
        events = exporter.events
        transform_completed_events = [(i, e) for i, e in enumerate(events) if isinstance(e, TransformCompleted)]
        token_completed_events = [(i, e) for i, e in enumerate(events) if isinstance(e, TokenCompleted)]

        # Should have 3 TransformCompleted (one per token, since batch_size=1)
        assert len(transform_completed_events) == 3

        # For each token, TransformCompleted must precede TokenCompleted
        for tc_idx, tc_event in token_completed_events:
            token_id = tc_event.token_id
            matching_transform = [(idx, e) for idx, e in transform_completed_events if e.token_id == token_id]
            if matching_transform:
                tf_idx, _ = matching_transform[0]
                assert tf_idx < tc_idx, (
                    f"Batch size=1 edge case: TransformCompleted ({tf_idx}) must precede TokenCompleted ({tc_idx}) for token {token_id}"
                )

    def test_passthrough_mode_no_ordering_issue(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Negative test: Passthrough mode should NOT have the P2-2026-02-01 ordering bug.

        In passthrough mode, tokens get BUFFERED (non-terminal) when buffered, then
        COMPLETED when flush succeeds. TokenCompleted only fires at flush time, so
        ordering is naturally correct (TransformCompleted, then TokenCompleted).
        """
        from elspeth.core.config import AggregationSettings, TriggerConfig

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        # Use passthrough-compatible transform (returns N rows for N inputs)
        transform = PassthroughBatchAwareTransform()

        source = TelemetryTestSource(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
            ]
        )

        sink = TelemetryTestSink()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # PASSTHROUGH mode - tokens continue after flush, not consumed
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_passthrough",
            trigger=TriggerConfig(count=3),
            output_mode="passthrough",  # Key difference from transform mode
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        events = exporter.events
        transform_completed_events = [(i, e) for i, e in enumerate(events) if isinstance(e, TransformCompleted)]
        token_completed_events = [(i, e) for i, e in enumerate(events) if isinstance(e, TokenCompleted)]

        # Passthrough mode: tokens get COMPLETED at flush, so TokenCompleted fires then
        # Verify ordering is correct (TransformCompleted before TokenCompleted)
        for tc_idx, tc_event in token_completed_events:
            token_id = tc_event.token_id
            matching_transform = [(idx, e) for idx, e in transform_completed_events if e.token_id == token_id]
            if matching_transform:
                tf_idx, _ = matching_transform[0]
                assert tf_idx < tc_idx, (
                    f"Passthrough mode should NOT have ordering bug: TransformCompleted ({tf_idx}) "
                    f"must precede TokenCompleted ({tc_idx}) for token {token_id}"
                )

    def test_transform_mode_failed_flush_emits_token_completed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Bug P2-2026-02-01: Failed flush must still emit TokenCompleted for buffered tokens.

        On error path in transform mode:
        - TransformCompleted is NOT emitted (transform didn't succeed)
        - TokenCompleted MUST be emitted (tokens are terminal with CONSUMED_IN_BATCH)
        """
        from elspeth.core.config import AggregationSettings, TriggerConfig

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        # Use failing batch-aware transform
        transform = FailingBatchAwareTransform()

        source = TelemetryTestSource(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
            ]
        )

        sink = TelemetryTestSink()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # Transform mode with count=3 trigger
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="failing_batch",
            trigger=TriggerConfig(count=3),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Run completes (with failures recorded)
        assert result.status == RunStatus.COMPLETED

        events = exporter.events
        token_completed_events = [e for e in events if isinstance(e, TokenCompleted)]
        transform_completed_events = [e for e in events if isinstance(e, TransformCompleted)]

        # On failed flush, TransformCompleted is NOT emitted (transform didn't succeed)
        assert len(transform_completed_events) == 0, "TransformCompleted should NOT be emitted on failed flush"

        # But TokenCompleted MUST be emitted for all 3 buffered tokens
        # (they have CONSUMED_IN_BATCH outcome from buffer time)
        assert len(token_completed_events) == 3, (
            f"Expected 3 TokenCompleted events for buffered tokens, got {len(token_completed_events)}. "
            f"Bug P2-2026-02-01: TokenCompleted must be emitted even on failed flush."
        )

        # All should have CONSUMED_IN_BATCH outcome (not FAILED - that would violate
        # unique terminal outcome constraint)
        for event in token_completed_events:
            assert event.outcome == RowOutcome.CONSUMED_IN_BATCH, (
                f"Expected CONSUMED_IN_BATCH outcome for token {event.token_id}, got {event.outcome}"
            )
