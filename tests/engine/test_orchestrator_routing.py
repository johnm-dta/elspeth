# tests/engine/test_orchestrator_routing.py
"""Tests for Orchestrator routing behavior.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import Determinism, SourceRow
from elspeth.core.landscape import LandscapeDB
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="module")
def routing_db() -> LandscapeDB:
    """Module-scoped in-memory database for routing tests.

    Tests must use unique run_ids to avoid conflicts.
    """
    return LandscapeDB.in_memory()


class TestOrchestratorInvalidRouting:
    """Test that invalid routing fails explicitly instead of silently."""

    def test_gate_routing_to_unknown_sink_raises_error(self, routing_db: LandscapeDB, payload_store) -> None:
        """Gate routing to non-existent sink must fail loudly, not silently."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.dag import GraphValidationError
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
        )

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # Config-driven gate that always routes to a non-existent sink
        misrouting_gate = GateSettings(
            name="misrouting_gate",
            condition="True",  # Always routes
            routes={
                "true": "nonexistent_sink",
                "false": "continue",
            },  # Invalid sink for error test
        )

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},  # Note: "nonexistent_sink" is NOT here
            gates=[misrouting_gate],
        )

        orchestrator = Orchestrator(routing_db)

        # This MUST fail loudly - silent counting was the bug
        # Config-driven gates are validated at pipeline init via GraphValidationError,
        # catching the misconfiguration before any rows are processed
        with pytest.raises(GraphValidationError, match="nonexistent_sink"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)


class TestOrchestratorOutputSinkRouting:
    """Verify completed rows go to the configured output_sink, not hardcoded 'default'."""

    def test_completed_rows_go_to_output_sink(self, plugin_manager: Any, routing_db: LandscapeDB, payload_store) -> None:
        """Rows that complete the pipeline go to the output_sink from config."""
        from unittest.mock import MagicMock

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        # Config with default_sink="results" (NOT "default")
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"fields": "dynamic"}}),
                "errors": SinkSettings(plugin="json", options={"path": "errors.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="results",
        )
        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Mock source that yields one row
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source._on_validation_failure = "discard"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        schema_mock = MagicMock()

        schema_mock.model_json_schema.return_value = {"type": "object"}

        mock_source.output_schema = schema_mock
        mock_source.load.return_value = iter([SourceRow.valid({"id": 1, "value": "test"})])
        mock_source.get_field_resolution.return_value = None

        # Mock sinks - track what gets written
        mock_results_sink = MagicMock()
        mock_results_sink.name = "csv"
        mock_results_sink.determinism = Determinism.IO_WRITE
        mock_results_sink.plugin_version = "1.0.0"
        mock_results_sink.write = MagicMock(return_value=ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc123"))

        mock_errors_sink = MagicMock()
        mock_errors_sink.name = "csv"
        mock_errors_sink.determinism = Determinism.IO_WRITE
        mock_errors_sink.plugin_version = "1.0.0"
        mock_errors_sink.write = MagicMock(return_value=ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc123"))

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={
                "results": mock_results_sink,
                "errors": mock_errors_sink,
            },
        )

        orchestrator = Orchestrator(routing_db)
        result = orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Row should go to "results" sink, not "default"
        assert result.rows_processed == 1
        assert result.rows_succeeded == 1
        assert mock_results_sink.write.called, "results sink should receive completed rows"
        assert not mock_errors_sink.write.called, "errors sink should not receive completed rows"


class TestOrchestratorGateRouting:
    """Test that gate routing works with route labels."""

    def test_gate_routes_to_named_sink(self, routing_db: LandscapeDB, payload_store) -> None:
        """Gate can route rows to a named sink using route labels."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class RowSchema(PluginSchema):
            id: int
            score: float

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []
                self.write_called = False

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.write_called = True
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc123")

            def close(self) -> None:
                pass

        # Config-driven gate: always routes to "flagged" sink
        routing_gate = GateSettings(
            name="test_gate",
            condition="True",  # Always routes
            routes={"true": "flagged", "false": "continue"},
        )

        source = ListSource([{"id": 1, "score": 0.2}])
        results_sink = CollectSink()
        flagged_sink = CollectSink()

        pipeline_config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"results": as_sink(results_sink), "flagged": as_sink(flagged_sink)},
            gates=[routing_gate],
        )

        orchestrator = Orchestrator(routing_db)
        result = orchestrator.run(pipeline_config, graph=build_production_graph(pipeline_config), payload_store=payload_store)

        # Row should be routed, not completed
        assert result.rows_processed == 1
        assert result.rows_routed == 1
        assert flagged_sink.write_called, "flagged sink should receive routed row"
        assert not results_sink.write_called, "results sink should not receive routed row"


class TestRouteValidation:
    """Test that route destinations are validated at initialization.

    MED-003: Route validation should happen BEFORE any rows are processed,
    not during row processing. This prevents partial runs where config errors
    are discovered after processing some rows.
    """

    def test_valid_routes_pass_validation(self, routing_db: LandscapeDB, payload_store) -> None:
        """Valid route configurations should pass validation without error."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Config-driven gate: routes values > 50 to "quarantine" sink, else continues
        routing_gate = GateSettings(
            name="routing_gate",
            condition="row['value'] > 50",
            routes={"true": "quarantine", "false": "continue"},
        )

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 100}])
        default_sink = CollectSink()
        quarantine_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(quarantine_sink),
            },
            gates=[routing_gate],
        )

        orchestrator = Orchestrator(routing_db)
        # Should not raise - routes are valid
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert len(default_sink.results) == 1  # value=10 continues
        assert len(quarantine_sink.results) == 1  # value=100 routed

    def test_invalid_route_destination_fails_at_init(self, routing_db: LandscapeDB, payload_store) -> None:
        """Route to non-existent sink should fail before processing any rows."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.dag import GraphValidationError
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
        )

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data
                self.load_called = False

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                self.load_called = True
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Config-driven gate: routes values > 50 to "quarantine" (which doesn't exist)
        safety_gate = GateSettings(
            name="safety_gate",
            condition="row['value'] > 50",
            routes={"true": "quarantine", "false": "continue"},
        )

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 100}])
        default_sink = CollectSink()
        # Note: NO quarantine sink provided!

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink)},  # Only default, no quarantine
            gates=[safety_gate],
        )

        orchestrator = Orchestrator(routing_db)

        # Should fail at initialization with clear error message
        with pytest.raises(GraphValidationError) as exc_info:
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify error message contains helpful information
        error_msg = str(exc_info.value)
        assert "safety_gate" in error_msg  # Gate name
        assert "quarantine" in error_msg  # Invalid destination

        # Verify no rows were processed
        assert not source.load_called, "Source should not be loaded on validation failure"
        assert len(default_sink.results) == 0, "No rows should be written on failure"

    def test_error_message_includes_route_label(self, routing_db: LandscapeDB, payload_store) -> None:
        """Error message should include the route label for debugging."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.dag import GraphValidationError
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
        )

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Config-driven gate: always routes to "high_scores" (which doesn't exist)
        threshold_gate = GateSettings(
            name="threshold_gate",
            condition="True",  # Always routes
            routes={"true": "high_scores", "false": "continue"},  # Non-existent sink
        )

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}])
        default_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink), "errors": as_sink(CollectSink())},
            gates=[threshold_gate],
        )

        orchestrator = Orchestrator(routing_db)

        with pytest.raises(GraphValidationError) as exc_info:
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        error_msg = str(exc_info.value)
        # Should include destination (route target)
        assert "high_scores" in error_msg
        # Should include gate name
        assert "threshold_gate" in error_msg

    def test_continue_routes_are_not_validated_as_sinks(self, routing_db: LandscapeDB, payload_store) -> None:
        """Routes that resolve to 'continue' should not be validated as sinks."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Config-driven gate: always continues (no routing to sink)
        filter_gate = GateSettings(
            name="filter_gate",
            condition="True",  # Always evaluates to true
            routes={
                "true": "continue",
                "false": "continue",
            },  # "continue" is not a sink
        )

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}])
        default_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink)},
            gates=[filter_gate],
        )

        orchestrator = Orchestrator(routing_db)
        # Should not raise - "continue" is a valid routing target
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 1
