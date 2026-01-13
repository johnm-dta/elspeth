# tests/engine/test_orchestrator.py
"""Tests for Orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from elspeth.engine.orchestrator import PipelineConfig


def _build_test_graph(config: PipelineConfig) -> "ExecutionGraph":
    """Build a simple graph for testing (temporary until from_config is wired).

    Creates a linear graph matching the PipelineConfig structure:
    source -> transforms... -> sinks

    For gates, creates additional edges to all sinks (gates can route anywhere).
    """
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()

    # Add source
    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    # Add transforms and populate transform_id_map
    transform_ids: dict[int, str] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        transform_ids[i] = node_id
        is_gate = hasattr(t, "evaluate")
        graph.add_node(
            node_id,
            node_type="gate" if is_gate else "transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode="move")
        prev = node_id

    # Add sinks and populate sink_id_map
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)
        graph.add_edge(prev, node_id, label=sink_name, mode="move")

        # Gates can route to any sink, so add edges from all gates
        for i, t in enumerate(config.transforms):
            if hasattr(t, "evaluate"):
                gate_id = f"transform_{i}"
                if gate_id != prev:  # Don't duplicate edge
                    graph.add_edge(gate_id, node_id, label=sink_name, mode="move")

    # Populate internal ID maps so get_sink_id_map() and get_transform_id_map() work
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids

    # Set output_sink - use "default" if present, otherwise first sink
    if "default" in sink_ids:
        graph._output_sink = "default"
    elif sink_ids:
        graph._output_sink = next(iter(sink_ids))

    return graph


class TestOrchestrator:
    """Full run orchestration."""

    def test_run_simple_pipeline(self) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class ListSource:
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row, ctx):
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []  # Instance attribute, not class attribute

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = DoubleTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        assert run_result.status == "completed"
        assert run_result.rows_processed == 3
        assert len(sink.results) == 3
        assert sink.results[0] == {"value": 1, "doubled": 2}

    def test_run_with_gate_routing(self) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row, ctx):
                if row["value"] > 50:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        gate = ThresholdGate()
        default_sink = CollectSink()
        high_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": default_sink, "high": high_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        assert run_result.status == "completed"
        # value=10 and value=30 go to default, value=100 goes to high
        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1


class TestOrchestratorAuditTrail:
    """Verify audit trail is recorded correctly."""

    def test_run_records_landscape_entries(self) -> None:
        """Verify that run creates proper audit trail."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "test_sink"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 42}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        # Query Landscape to verify audit trail
        recorder = LandscapeRecorder(db)
        run = recorder.get_run(run_result.run_id)

        assert run is not None
        assert run.status == "completed"

        # Verify nodes were registered
        nodes = recorder.get_nodes(run_result.run_id)
        assert len(nodes) == 3  # source, transform, sink

        node_names = [n.plugin_name for n in nodes]
        assert "test_source" in node_names
        assert "identity" in node_names
        assert "test_sink" in node_names


class TestOrchestratorErrorHandling:
    """Test error handling in orchestration."""

    def test_run_marks_failed_on_transform_exception(self) -> None:
        """If a transform raises, run status should be failed in Landscape."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class ExplodingTransform:
            name = "exploding"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def process(self, row, ctx):
                raise RuntimeError("Transform exploded!")

        class CollectSink:
            name = "test_sink"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 42}])
        transform = ExplodingTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError, match="Transform exploded!"):
            orchestrator.run(config, graph=_build_test_graph(config))

        # Verify run was marked as failed in Landscape audit trail
        # Query for all runs and find the one that was created
        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1, "Expected exactly one run in Landscape"

        failed_run = runs[0]
        assert failed_run.status == "failed", (
            f"Landscape audit trail must record status='failed', "
            f"got status='{failed_run.status}'"
        )


class TestOrchestratorMultipleTransforms:
    """Test pipelines with multiple transforms."""

    def test_run_multiple_transforms_in_sequence(self) -> None:
        """Test that multiple transforms execute in order."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class NumberSchema(PluginSchema):
            value: int

        class ListSource:
            name = "numbers"
            output_schema = NumberSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class AddOneTransform:
            name = "add_one"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] + 1})

        class MultiplyTwoTransform:
            name = "multiply_two"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] * 2})

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 5}])
        transform1 = AddOneTransform()
        transform2 = MultiplyTwoTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform1, transform2],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        assert run_result.status == "completed"
        assert len(sink.results) == 1
        # (5 + 1) * 2 = 12
        assert sink.results[0]["value"] == 12


class TestOrchestratorEmptyPipeline:
    """Test edge cases."""

    def test_run_no_transforms(self) -> None:
        """Test pipeline with source directly to sink."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "direct"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 99}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        assert run_result.status == "completed"
        assert run_result.rows_processed == 1
        assert len(sink.results) == 1
        assert sink.results[0] == {"value": 99}

    def test_run_empty_source(self) -> None:
        """Test pipeline with no rows from source."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class EmptySource:
            name = "empty"
            output_schema = ValueSchema

            def load(self, ctx):
                return iter([])

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = EmptySource()
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        assert run_result.status == "completed"
        assert run_result.rows_processed == 0
        assert len(sink.results) == 0


class TestOrchestratorInvalidRouting:
    """Test that invalid routing fails explicitly instead of silently."""

    def test_gate_routing_to_unknown_sink_raises_error(self) -> None:
        """Gate routing to non-existent sink must fail loudly, not silently."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.executors import MissingEdgeError
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class MisroutingGate:
            """Gate that routes to a sink that doesn't exist."""

            name = "misrouting_gate"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row, ctx):
                # Route to a sink that wasn't configured
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink("nonexistent_sink"),
                )

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 42}])
        gate = MisroutingGate()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": sink},  # Note: "nonexistent_sink" is NOT here
        )

        orchestrator = Orchestrator(db)

        # This MUST fail loudly - silent counting was the bug
        # The GateExecutor catches this first via MissingEdgeError,
        # which is even better since it happens at the routing level
        with pytest.raises(MissingEdgeError, match="nonexistent_sink"):
            orchestrator.run(config, graph=_build_test_graph(config))


class TestOrchestratorAcceptsGraph:
    """Orchestrator accepts ExecutionGraph parameter."""

    def test_orchestrator_uses_graph_node_ids(self) -> None:
        """Orchestrator uses node IDs from graph, not generated IDs."""
        from unittest.mock import MagicMock

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Build config and graph from settings
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )
        graph = ExecutionGraph.from_config(settings)

        # Create mock source and sink
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])  # Empty source

        mock_sink = MagicMock()
        mock_sink.name = "csv"

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
        )

        orchestrator = Orchestrator(db)
        orchestrator.run(pipeline_config, graph=graph)

        # Source should have node_id set from graph
        assert hasattr(mock_source, "node_id")
        assert mock_source.node_id == graph.get_source()

        # Sink should have node_id set from graph's sink_id_map
        sink_id_map = graph.get_sink_id_map()
        assert hasattr(mock_sink, "node_id")
        assert mock_sink.node_id == sink_id_map["output"]

    def test_orchestrator_run_accepts_graph(self) -> None:
        """Orchestrator.run() accepts graph parameter."""
        import inspect

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator

        db = LandscapeDB.in_memory()

        # Build a simple graph
        graph = ExecutionGraph()
        graph.add_node("source_1", node_type="source", plugin_name="csv")
        graph.add_node("sink_1", node_type="sink", plugin_name="csv")
        graph.add_edge("source_1", "sink_1", label="continue", mode="move")

        orchestrator = Orchestrator(db)

        # Should accept graph parameter (signature check)
        sig = inspect.signature(orchestrator.run)
        assert "graph" in sig.parameters

    def test_orchestrator_run_requires_graph(self) -> None:
        """Orchestrator.run() raises ValueError if graph is None."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class DummySource:
            name = "dummy"
            output_schema = ValueSchema

            def load(self, ctx):
                yield from []

            def close(self):
                pass

        class DummySink:
            name = "dummy"

            def __init__(self):
                self.results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        config = PipelineConfig(
            source=DummySource(),
            transforms=[],
            sinks={"default": DummySink()},
        )

        orchestrator = Orchestrator(db)

        # graph=None should raise ValueError
        with pytest.raises(ValueError, match="ExecutionGraph is required"):
            orchestrator.run(config, graph=None)


class TestOrchestratorOutputSinkRouting:
    """Verify completed rows go to the configured output_sink, not hardcoded 'default'."""

    def test_completed_rows_go_to_output_sink(self) -> None:
        """Rows that complete the pipeline go to the output_sink from config."""
        from unittest.mock import MagicMock

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Config with output_sink="results" (NOT "default")
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "errors": SinkSettings(plugin="csv"),
            },
            output_sink="results",
        )
        graph = ExecutionGraph.from_config(settings)

        # Mock source that yields one row
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1, "value": "test"}])

        # Mock sinks - track what gets written
        mock_results_sink = MagicMock()
        mock_results_sink.name = "csv"
        mock_results_sink.write = MagicMock(
            return_value={"path": "memory", "size_bytes": 0, "content_hash": "abc123"}
        )

        mock_errors_sink = MagicMock()
        mock_errors_sink.name = "csv"
        mock_errors_sink.write = MagicMock(
            return_value={"path": "memory", "size_bytes": 0, "content_hash": "abc123"}
        )

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={
                "results": mock_results_sink,
                "errors": mock_errors_sink,
            },
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline_config, graph=graph)

        # Row should go to "results" sink, not "default"
        assert result.rows_processed == 1
        assert result.rows_succeeded == 1
        assert mock_results_sink.write.called, "results sink should receive completed rows"
        assert not mock_errors_sink.write.called, "errors sink should not receive completed rows"


class TestOrchestratorGateRouting:
    """Test that gate routing works with route labels."""

    def test_gate_routes_to_named_sink(self) -> None:
        """Gate can route rows to a named sink using route labels."""
        from unittest.mock import MagicMock

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="test_gate",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )
        graph = ExecutionGraph.from_config(settings)

        # Mock source
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1, "score": 0.2}])

        # Mock gate that routes to "flagged"
        mock_gate = MagicMock()
        mock_gate.name = "test_gate"
        mock_gate.evaluate.return_value = GateResult(
            row={"id": 1, "score": 0.2},
            action=RoutingAction.route_to_sink("flagged", reason={"score": "low"}),
        )

        # Mock sinks - must return proper artifact info from write()
        mock_results = MagicMock()
        mock_results.name = "csv"
        mock_results.write.return_value = {
            "path": "memory",
            "size_bytes": 0,
            "content_hash": "abc123",
        }

        mock_flagged = MagicMock()
        mock_flagged.name = "csv"
        mock_flagged.write.return_value = {
            "path": "memory",
            "size_bytes": 0,
            "content_hash": "abc123",
        }

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[mock_gate],
            sinks={"results": mock_results, "flagged": mock_flagged},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline_config, graph=graph)

        # Row should be routed, not completed
        assert result.rows_processed == 1
        assert result.rows_routed == 1
        assert mock_flagged.write.called, "flagged sink should receive routed row"
        assert not mock_results.write.called, "results sink should not receive routed row"
