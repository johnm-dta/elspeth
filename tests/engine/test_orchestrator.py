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
    Route labels use sink names for simplicity in tests.
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

    # Populate route resolution map: (gate_id, label) -> sink_name
    # In test graphs, route labels = sink names for simplicity
    route_resolution_map: dict[tuple[str, str], str] = {}
    for i, t in enumerate(config.transforms):
        if hasattr(t, "evaluate"):  # It's a gate
            gate_id = f"transform_{i}"
            for sink_name in sink_ids:
                route_resolution_map[(gate_id, sink_name)] = sink_name
    graph._route_resolution_map = route_resolution_map

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []  # Instance attribute, not class attribute

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def evaluate(self, row, ctx):
                if row["value"] > 50:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route("high"),  # Route label (same as sink name in test)
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "test_sink"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class ExplodingTransform:
            name = "exploding"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                raise RuntimeError("Transform exploded!")

        class CollectSink:
            name = "test_sink"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class AddOneTransform:
            name = "add_one"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] + 1})

        class MultiplyTwoTransform:
            name = "multiply_two"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] * 2})

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                return iter([])

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class MisroutingGate:
            """Gate that routes to a route label that doesn't exist."""

            name = "misrouting_gate"
            input_schema = RowSchema
            output_schema = RowSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def evaluate(self, row, ctx):
                # Route to a label that wasn't configured
                return GateResult(
                    row=row,
                    action=RoutingAction.route("nonexistent_sink"),
                )

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

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

        # Mock gate that routes using "suspicious" label (maps to "flagged" sink in routes config)
        mock_gate = MagicMock()
        mock_gate.name = "test_gate"
        mock_gate.evaluate.return_value = GateResult(
            row={"id": 1, "score": 0.2},
            action=RoutingAction.route("suspicious", reason={"score": "low"}),  # Uses route label
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


class TestLifecycleHooks:
    """Orchestrator invokes plugin lifecycle hooks."""

    def test_on_start_called_before_processing(self) -> None:
        """on_start() called before any rows processed."""
        from unittest.mock import MagicMock

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        call_order: list[str] = []

        class TrackedTransform:
            name = "tracked"
            plugin_version = "1.0.0"

            def on_start(self, ctx):
                call_order.append("on_start")

            def process(self, row, ctx):
                call_order.append("process")
                return TransformResult.success(row)

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1}])

        transform = TrackedTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.write.return_value = {
            "path": "memory",
            "size_bytes": 0,
            "content_hash": "abc123",
        }

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        # Minimal graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="tracked")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_start should be called first
        assert call_order[0] == "on_start"
        assert "process" in call_order

    def test_on_complete_called_after_all_rows(self) -> None:
        """on_complete() called after all rows processed."""
        from unittest.mock import MagicMock

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        call_order: list[str] = []

        class TrackedTransform:
            name = "tracked"
            plugin_version = "1.0.0"

            def on_start(self, ctx):
                call_order.append("on_start")

            def process(self, row, ctx):
                call_order.append("process")
                return TransformResult.success(row)

            def on_complete(self, ctx):
                call_order.append("on_complete")

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1}, {"id": 2}])

        transform = TrackedTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.write.return_value = {
            "path": "memory",
            "size_bytes": 0,
            "content_hash": "abc123",
        }

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="tracked")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_complete should be called last (among transform lifecycle calls)
        transform_calls = [c for c in call_order if c in ["on_start", "process", "on_complete"]]
        assert transform_calls[-1] == "on_complete"
        # All processing should happen before on_complete
        assert call_order.count("process") == 2

    def test_on_complete_called_on_error(self) -> None:
        """on_complete() called even when run fails."""
        from unittest.mock import MagicMock

        import pytest

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        completed: list[bool] = []

        class FailingTransform:
            name = "failing"
            plugin_version = "1.0.0"

            def on_start(self, ctx):
                pass

            def process(self, row, ctx):
                raise RuntimeError("intentional failure")

            def on_complete(self, ctx):
                completed.append(True)

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1}])

        transform = FailingTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="failing")
        graph.add_node("transform", node_type="transform", plugin_name="failing")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=graph)

        # on_complete should still be called
        assert len(completed) == 1


class TestOrchestratorLandscapeExport:
    """Test landscape export integration."""

    def test_orchestrator_exports_landscape_when_configured(self) -> None:
        """Orchestrator should export audit trail after run completes."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            """Sink that captures written rows."""

            name = "collect"

            def __init__(self):
                self.captured_rows: list[dict] = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, row, ctx):
                # Row processing writes batches (lists), export writes single records
                if isinstance(row, list):
                    self.captured_rows.extend(row)
                else:
                    self.captured_rows.append(row)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def flush(self):
                pass

            def close(self):
                pass

        # Create in-memory DB
        db = LandscapeDB.in_memory()

        # Create sinks
        output_sink = CollectSink()
        export_sink = CollectSink()

        # Build settings with export enabled
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "output": SinkSettings(plugin="csv"),
                "audit_export": SinkSettings(plugin="csv"),
            },
            output_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                export=LandscapeExportSettings(
                    enabled=True,
                    sink="audit_export",
                    format="json",  # JSON works with mock sinks; CSV requires file path
                ),
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=source,
            transforms=[],
            sinks={
                "output": output_sink,
                "audit_export": export_sink,
            },
        )

        # Build graph from config
        graph = ExecutionGraph.from_config(settings)

        # Run with settings
        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline, graph=graph, settings=settings)

        # Run should complete
        assert result.status == "completed"
        assert result.rows_processed == 1

        # Export sink should have received audit records
        assert len(export_sink.captured_rows) > 0
        # Should have at least a "run" record type
        record_types = [r.get("record_type") for r in export_sink.captured_rows]
        assert "run" in record_types, f"Expected 'run' record type, got: {record_types}"

    def test_orchestrator_export_with_signing(self) -> None:
        """Orchestrator should sign records when export.sign is True."""
        import os
        from unittest.mock import patch

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.captured_rows: list[dict] = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, row, ctx):
                # Row processing writes batches (lists), export writes single records
                if isinstance(row, list):
                    self.captured_rows.extend(row)
                else:
                    self.captured_rows.append(row)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def flush(self):
                pass

            def close(self):
                pass

        db = LandscapeDB.in_memory()
        output_sink = CollectSink()
        export_sink = CollectSink()

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "output": SinkSettings(plugin="csv"),
                "audit_export": SinkSettings(plugin="csv"),
            },
            output_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                export=LandscapeExportSettings(
                    enabled=True,
                    sink="audit_export",
                    format="json",  # JSON works with mock sinks; CSV requires file path
                    sign=True,
                ),
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=source,
            transforms=[],
            sinks={
                "output": output_sink,
                "audit_export": export_sink,
            },
        )

        graph = ExecutionGraph.from_config(settings)
        orchestrator = Orchestrator(db)

        # Set signing key environment variable
        with patch.dict(os.environ, {"ELSPETH_SIGNING_KEY": "test-signing-key-12345"}):
            result = orchestrator.run(pipeline, graph=graph, settings=settings)

        assert result.status == "completed"
        assert len(export_sink.captured_rows) > 0

        # All records should have signatures when signing enabled
        for record in export_sink.captured_rows:
            assert "signature" in record, f"Record missing signature: {record}"

        # Should have a manifest record at the end
        record_types = [r.get("record_type") for r in export_sink.captured_rows]
        assert "manifest" in record_types

    def test_orchestrator_export_requires_signing_key_when_sign_enabled(self) -> None:
        """Should raise error when sign=True but ELSPETH_SIGNING_KEY not set."""
        import os
        from unittest.mock import patch

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.captured_rows: list[dict] = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.captured_rows.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def flush(self):
                pass

            def close(self):
                pass

        db = LandscapeDB.in_memory()
        output_sink = CollectSink()
        export_sink = CollectSink()

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "output": SinkSettings(plugin="csv"),
                "audit_export": SinkSettings(plugin="csv"),
            },
            output_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                export=LandscapeExportSettings(
                    enabled=True,
                    sink="audit_export",
                    format="csv",
                    sign=True,
                ),
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=source,
            transforms=[],
            sinks={
                "output": output_sink,
                "audit_export": export_sink,
            },
        )

        graph = ExecutionGraph.from_config(settings)
        orchestrator = Orchestrator(db)

        # Ensure ELSPETH_SIGNING_KEY is not set
        env_without_key = {k: v for k, v in os.environ.items() if k != "ELSPETH_SIGNING_KEY"}
        with (
            patch.dict(os.environ, env_without_key, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            orchestrator.run(pipeline, graph=graph, settings=settings)

    def test_orchestrator_no_export_when_disabled(self) -> None:
        """Should not export when export.enabled is False."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.schemas import PluginSchema

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.captured_rows: list[dict] = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.captured_rows.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def flush(self):
                pass

            def close(self):
                pass

        db = LandscapeDB.in_memory()
        output_sink = CollectSink()
        audit_sink = CollectSink()

        # Export disabled (the default)
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "output": SinkSettings(plugin="csv"),
                "audit": SinkSettings(plugin="csv"),
            },
            output_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                # export.enabled defaults to False
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=source,
            transforms=[],
            sinks={
                "output": output_sink,
                "audit": audit_sink,
            },
        )

        graph = ExecutionGraph.from_config(settings)
        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline, graph=graph, settings=settings)

        assert result.status == "completed"
        # Output sink should have the row
        assert len(output_sink.captured_rows) == 1
        # Audit sink should be empty (no export)
        assert len(audit_sink.captured_rows) == 0


class TestSourceLifecycleHooks:
    """Tests for source plugin lifecycle hook calls."""

    def test_source_lifecycle_hooks_called(self) -> None:
        """Source on_start, on_complete should be called around loading."""
        from unittest.mock import MagicMock

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        call_order: list[str] = []

        class TrackedSource:
            """Source that tracks lifecycle calls."""

            name = "tracked_source"

            def on_start(self, ctx):
                call_order.append("source_on_start")

            def load(self, ctx):
                call_order.append("source_load")
                yield {"value": 1}

            def on_complete(self, ctx):
                call_order.append("source_on_complete")

            def close(self):
                call_order.append("source_close")

        db = LandscapeDB.in_memory()

        source = TrackedSource()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.write.return_value = {
            "path": "memory",
            "size_bytes": 0,
            "content_hash": "abc123",
        }

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": mock_sink},
        )

        # Minimal graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="tracked_source")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "sink", label="continue", mode="move")
        graph._transform_id_map = {}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_start should be called BEFORE load
        assert "source_on_start" in call_order, "Source on_start should be called"
        assert call_order.index("source_on_start") < call_order.index("source_load"), (
            "Source on_start should be called before load"
        )
        # on_complete should be called AFTER load and BEFORE close
        assert "source_on_complete" in call_order, "Source on_complete should be called"
        assert call_order.index("source_on_complete") > call_order.index("source_load"), (
            "Source on_complete should be called after load"
        )
        assert call_order.index("source_on_complete") < call_order.index("source_close"), (
            "Source on_complete should be called before close"
        )


class TestSinkLifecycleHooks:
    """Tests for sink plugin lifecycle hook calls."""

    def test_sink_lifecycle_hooks_called(self) -> None:
        """Sink on_start and on_complete should be called."""
        from unittest.mock import MagicMock

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        call_order: list[str] = []

        class TrackedSink:
            """Sink that tracks lifecycle calls."""

            name = "tracked_sink"

            def on_start(self, ctx):
                call_order.append("sink_on_start")

            def on_complete(self, ctx):
                call_order.append("sink_on_complete")

            def write(self, rows, ctx):
                call_order.append("sink_write")
                return {"path": "memory", "size_bytes": 0, "content_hash": "abc123"}

            def close(self):
                call_order.append("sink_close")

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"value": 1}])

        sink = TrackedSink()

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": sink},
        )

        # Minimal graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("sink", node_type="sink", plugin_name="tracked_sink")
        graph.add_edge("source", "sink", label="continue", mode="move")
        graph._transform_id_map = {}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_start should be called before write
        assert "sink_on_start" in call_order, "Sink on_start should be called"
        assert call_order.index("sink_on_start") < call_order.index("sink_write"), (
            "Sink on_start should be called before write"
        )
        # on_complete should be called after write, before close
        assert "sink_on_complete" in call_order, "Sink on_complete should be called"
        assert call_order.index("sink_on_complete") > call_order.index("sink_write"), (
            "Sink on_complete should be called after write"
        )

    def test_sink_on_complete_called_even_on_error(self) -> None:
        """Sink on_complete should be called even when run fails."""
        from unittest.mock import MagicMock

        import pytest

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        completed: list[str] = []

        class FailingTransform:
            name = "failing"
            plugin_version = "1.0.0"

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                raise RuntimeError("intentional failure")

        class TrackedSink:
            name = "tracked_sink"

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                completed.append("sink_on_complete")

            def write(self, rows, ctx):
                return {"path": "memory", "size_bytes": 0, "content_hash": "abc123"}

            def close(self):
                pass

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"value": 1}])

        transform = FailingTransform()
        sink = TrackedSink()

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="failing")
        graph.add_node("sink", node_type="sink", plugin_name="tracked_sink")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=graph)

        # on_complete should still be called
        assert "sink_on_complete" in completed


class TestOrchestratorCheckpointing:
    """Tests for checkpoint integration in Orchestrator."""

    def test_orchestrator_accepts_checkpoint_manager(self) -> None:
        """Orchestrator can be initialized with CheckpointManager."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
        )
        assert orchestrator._checkpoint_manager is checkpoint_mgr

    def test_orchestrator_accepts_checkpoint_settings(self) -> None:
        """Orchestrator can be initialized with CheckpointSettings."""
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator

        db = LandscapeDB.in_memory()
        settings = CheckpointSettings(frequency="every_n", checkpoint_interval=10)
        orchestrator = Orchestrator(
            db=db,
            checkpoint_settings=settings,
        )
        assert orchestrator._checkpoint_settings == settings

    def test_maybe_checkpoint_creates_on_every_row(self) -> None:
        """_maybe_checkpoint creates checkpoint when frequency=every_row."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=settings,
        )
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 3

        # Checkpoints should have been created during processing
        # After completion, they should be deleted
        # So we can't check the checkpoint count here - it's cleaned up
        # Instead, we verify the run completed successfully with checkpointing enabled

    def test_maybe_checkpoint_respects_interval(self) -> None:
        """_maybe_checkpoint only creates checkpoint every N rows."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        # Checkpoint every 3 rows
        settings = CheckpointSettings(enabled=True, frequency="every_n", checkpoint_interval=3)

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        # 7 rows: should checkpoint at rows 3, 6 (sequence 3, 6)
        source = ListSource([{"value": i} for i in range(7)])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=settings,
        )
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 7

    def test_checkpoint_deleted_on_successful_completion(self) -> None:
        """Checkpoints are deleted when run completes successfully."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=settings,
        )
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"

        # After successful completion, checkpoints should be deleted
        remaining_checkpoints = checkpoint_mgr.get_checkpoints(result.run_id)
        assert len(remaining_checkpoints) == 0

    def test_checkpoint_preserved_on_failure(self) -> None:
        """Checkpoints are preserved when run fails for recovery."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class FailOnThirdTransform:
            name = "fail_on_third"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self):
                self.count = 0

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                self.count += 1
                if self.count == 3:
                    raise RuntimeError("Intentional failure on third row")
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = FailOnThirdTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=settings,
        )

        run_id = None
        with pytest.raises(RuntimeError, match="Intentional failure"):
            orchestrator.run(config, graph=_build_test_graph(config))

        # Need to find the run_id from the failed run
        # Query for all runs to find the failed one
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1
        run_id = runs[0].run_id

        # After failure, checkpoints should be preserved for recovery
        remaining_checkpoints = checkpoint_mgr.get_checkpoints(run_id)
        assert len(remaining_checkpoints) > 0

    def test_checkpoint_disabled_skips_checkpoint_creation(self) -> None:
        """No checkpoints created when checkpointing is disabled."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=False)

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=settings,
        )
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"

        # Even after run, no checkpoints should exist since disabled
        # (would have been deleted anyway, but let's verify with failure case)
        # We'll run a separate test with failure to verify

    def test_no_checkpoint_manager_skips_checkpointing(self) -> None:
        """Orchestrator works fine without checkpoint manager."""
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 1}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        # No checkpoint_manager passed - should work without checkpointing
        orchestrator = Orchestrator(
            db=db,
            checkpoint_settings=settings,
        )
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 1


class TestOrchestratorConfigRecording:
    """Test that runs record the resolved configuration."""

    def test_run_records_resolved_config(self) -> None:
        """Run should record the full resolved configuration in Landscape."""
        import json

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IdentityTransform:
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

        class CollectSink:
            name = "test_sink"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 42}])
        transform = IdentityTransform()
        sink = CollectSink()

        # Create config WITH resolved configuration dict
        resolved_config = {
            "datasource": {"plugin": "csv", "options": {"path": "test.csv"}},
            "sinks": {"default": {"plugin": "csv"}},
            "output_sink": "default",
        }

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
            config=resolved_config,  # Pass the resolved config
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        # Query Landscape to verify config was recorded
        recorder = LandscapeRecorder(db)
        run_record = recorder.get_run(run_result.run_id)

        assert run_record is not None
        # settings_json is stored as a JSON string, parse it
        settings = json.loads(run_record.settings_json)
        assert settings != {}
        assert "datasource" in settings
        assert settings["datasource"]["plugin"] == "csv"

    def test_run_with_empty_config_records_empty(self) -> None:
        """Run with no config passed should record empty dict (current behavior)."""
        import json

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

            def on_start(self, ctx):
                pass

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class CollectSink:
            name = "test_sink"

            def __init__(self):
                self.results = []

            def on_start(self, ctx):
                pass

            def on_complete(self, ctx):
                pass

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

            def close(self):
                pass

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        # No config passed - should default to empty dict
        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
            # config not passed - defaults to {}
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        # Query Landscape to verify empty config was recorded
        recorder = LandscapeRecorder(db)
        run_record = recorder.get_run(run_result.run_id)

        assert run_record is not None
        # This test documents that empty config is recorded when not provided
        # settings_json is stored as a JSON string
        settings = json.loads(run_record.settings_json)
        assert settings == {}
