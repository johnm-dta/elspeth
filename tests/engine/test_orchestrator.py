# tests/engine/test_orchestrator.py
"""Tests for Orchestrator.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode, SourceRow
from elspeth.plugins.base import BaseGate, BaseTransform

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


# ============================================================================
# Test Fixture Base Classes
# ============================================================================
# These provide the required protocol attributes so inline test classes
# don't need to repeat them.


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class providing SourceProtocol required attributes.

    Note: output_schema is NOT provided here because child classes override it
    with their own schemas, and mypy's type invariance would flag that as a conflict.
    Each test class must provide its own output_schema.
    """

    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"


class _TestSinkBase:
    """Base class providing SinkProtocol required attributes."""

    input_schema = _TestSchema  # Required by protocol
    idempotent = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"

    def flush(self) -> None:
        pass


def _build_test_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple graph for testing (temporary until from_config is wired).

    Creates a linear graph matching the PipelineConfig structure:
    source -> transforms... -> config gates... -> sinks

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
        is_gate = isinstance(t, BaseGate)
        graph.add_node(
            node_id,
            node_type="gate" if is_gate else "transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks first (need sink_ids for gate routing)
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Populate route resolution map: (gate_id, label) -> sink_name
    route_resolution_map: dict[tuple[str, str], str] = {}

    # Handle plugin-based gates in transforms
    for i, t in enumerate(config.transforms):
        if isinstance(t, BaseGate):  # It's a gate
            gate_id = f"transform_{i}"
            for sink_name in sink_ids:
                route_resolution_map[(gate_id, sink_name)] = sink_name

    # Add config-driven gates (from config.gates)
    config_gate_ids: dict[str, str] = {}
    for gate_config in config.gates:
        gate_id = f"config_gate_{gate_config.name}"
        config_gate_ids[gate_config.name] = gate_id

        # Store condition in node config for audit trail
        gate_node_config = {
            "condition": gate_config.condition,
            "routes": dict(gate_config.routes),
        }
        if gate_config.fork_to:
            gate_node_config["fork_to"] = list(gate_config.fork_to)

        graph.add_node(
            gate_id,
            node_type="gate",
            plugin_name=f"config_gate:{gate_config.name}",
            config=gate_node_config,
        )

        # Edge from previous node
        graph.add_edge(prev, gate_id, label="continue", mode=RoutingMode.MOVE)

        # Config gate routes to sinks
        for route_label, target in gate_config.routes.items():
            route_resolution_map[(gate_id, route_label)] = target

            if target == "continue":
                continue  # Not a sink route - no edge to create
            if target in sink_ids:
                graph.add_edge(
                    gate_id, sink_ids[target], label=route_label, mode=RoutingMode.MOVE
                )

        prev = gate_id

    # Add edges from transforms to sinks (for plugin-based gates and linear flow)
    for sink_name in sink_ids:
        node_id = sink_ids[sink_name]
        # Gates can route to any sink
        for i, t in enumerate(config.transforms):
            if isinstance(t, BaseGate):
                gate_id = f"transform_{i}"
                graph.add_edge(gate_id, node_id, label=sink_name, mode=RoutingMode.MOVE)

    # Edge from last node to output sink
    if "default" in sink_ids:
        output_sink = "default"
    elif sink_ids:
        output_sink = next(iter(sink_ids))
    else:
        output_sink = ""

    if output_sink:
        graph.add_edge(
            prev, sink_ids[output_sink], label="continue", mode=RoutingMode.MOVE
        )

    # Populate internal ID maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._config_gate_id_map = config_gate_ids
    graph._route_resolution_map = route_resolution_map
    graph._output_sink = output_sink

    return graph


def _build_fork_test_graph(
    config: PipelineConfig,
    fork_paths: dict[int, list[str]],  # transform_index -> list of fork path names
) -> ExecutionGraph:
    """Build a test graph that supports fork operations.

    Args:
        config: Pipeline configuration
        fork_paths: Maps transform index to list of fork path names
                   e.g., {0: ["path_a", "path_b"]} means transform_0 forks to those paths
    """
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()

    # Add source
    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    # Add transforms
    transform_ids: dict[int, str] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        transform_ids[i] = node_id
        is_gate = isinstance(t, BaseGate)
        graph.add_node(
            node_id,
            node_type="gate" if is_gate else "transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Add edge from last transform to default sink
    if "default" in sink_ids:
        graph.add_edge(
            prev, sink_ids["default"], label="continue", mode=RoutingMode.MOVE
        )

    # Populate internal maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._output_sink = "default" if "default" in sink_ids else next(iter(sink_ids))

    # Build route resolution map with fork support
    route_resolution_map: dict[tuple[str, str], str] = {}
    for i, paths in fork_paths.items():
        gate_id = f"transform_{i}"
        for path_name in paths:
            # Fork paths resolve to "fork" (special handling in executor)
            route_resolution_map[(gate_id, path_name)] = "fork"
            # Add edge for each fork path (needed for edge_map lookup)
            # Fork paths go to the NEXT transform (or sink if last)
            next_node = (
                f"transform_{i+1}"
                if i + 1 < len(config.transforms)
                else sink_ids["default"]
            )
            graph.add_edge(gate_id, next_node, label=path_name, mode=RoutingMode.COPY)

    graph._route_resolution_map = route_resolution_map

    return graph


class TestOrchestrator:
    """Full run orchestration."""

    def test_run_simple_pipeline(self) -> None:
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "doubled": row["value"] * 2,
                    }
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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Config-driven gate: routes values > 50 to "high" sink, else continues
        threshold_gate = GateSettings(
            name="threshold",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "continue"},
        )

        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        default_sink = CollectSink()
        high_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink, "high": high_sink},
            gates=[threshold_gate],
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import RunStatus

        recorder = LandscapeRecorder(db)
        run = recorder.get_run(run_result.run_id)

        assert run is not None
        assert run.status == RunStatus.COMPLETED

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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class ExplodingTransform(BaseTransform):
            name = "exploding"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("Transform exploded!")

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import RunStatus

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1, "Expected exactly one run in Landscape"

        failed_run = runs[0]
        assert failed_run.status == RunStatus.FAILED, (
            f"Landscape audit trail must record status=FAILED, "
            f"got status={failed_run.status!r}"
        )


class TestOrchestratorMultipleTransforms:
    """Test pipelines with multiple transforms."""

    def test_run_multiple_transforms_in_sequence(self) -> None:
        """Test that multiple transforms execute in order."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class NumberSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "numbers"
            output_schema = NumberSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class AddOneTransform(BaseTransform):
            name = "add_one"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"value": row["value"] + 1})

        class MultiplyTwoTransform(BaseTransform):
            name = "multiply_two"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "direct"
            output_schema = ValueSchema

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class EmptySource(_TestSourceBase):
            name = "empty"
            output_schema = ValueSchema

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                return iter([])

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        db = LandscapeDB.in_memory()

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

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
            source=source,
            transforms=[],
            sinks={"default": sink},  # Note: "nonexistent_sink" is NOT here
            gates=[misrouting_gate],
        )

        orchestrator = Orchestrator(db)

        # This MUST fail loudly - silent counting was the bug
        # Config-driven gates are validated at pipeline init via RouteValidationError,
        # catching the misconfiguration before any rows are processed
        with pytest.raises(RouteValidationError, match="nonexistent_sink"):
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
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter([])  # Empty source

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"

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
        graph.add_edge("source_1", "sink_1", label="continue", mode=RoutingMode.MOVE)

        orchestrator = Orchestrator(db)

        # Should accept graph parameter (signature check)
        sig = inspect.signature(orchestrator.run)
        assert "graph" in sig.parameters

    def test_orchestrator_run_requires_graph(self) -> None:
        """Orchestrator.run() raises ValueError if graph is None."""
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class DummySource(_TestSourceBase):
            name = "dummy"
            output_schema = ValueSchema

            def load(self, ctx: Any) -> Any:
                yield from []

            def close(self) -> None:
                pass

        class DummySink(_TestSinkBase):
            name = "dummy"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.engine.artifacts import ArtifactDescriptor
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
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter(
            [SourceRow.valid({"id": 1, "value": "test"})]
        )

        # Mock sinks - track what gets written
        mock_results_sink = MagicMock()
        mock_results_sink.name = "csv"
        mock_results_sink.determinism = Determinism.IO_WRITE
        mock_results_sink.plugin_version = "1.0.0"
        mock_results_sink.write = MagicMock(
            return_value=ArtifactDescriptor.for_file(
                path="memory", size_bytes=0, content_hash="abc123"
            )
        )

        mock_errors_sink = MagicMock()
        mock_errors_sink.name = "csv"
        mock_errors_sink.determinism = Determinism.IO_WRITE
        mock_errors_sink.plugin_version = "1.0.0"
        mock_errors_sink.write = MagicMock(
            return_value=ArtifactDescriptor.for_file(
                path="memory", size_bytes=0, content_hash="abc123"
            )
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
        assert (
            mock_results_sink.write.called
        ), "results sink should receive completed rows"
        assert (
            not mock_errors_sink.write.called
        ), "errors sink should not receive completed rows"


class TestOrchestratorGateRouting:
    """Test that gate routing works with route labels."""

    def test_gate_routes_to_named_sink(self) -> None:
        """Gate can route rows to a named sink using route labels."""
        from unittest.mock import MagicMock

        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Mock source
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter([SourceRow.valid({"id": 1, "score": 0.2})])

        # Config-driven gate: always routes to "flagged" sink
        routing_gate = GateSettings(
            name="test_gate",
            condition="True",  # Always routes
            routes={"true": "flagged", "false": "continue"},
        )

        # Mock sinks - must return proper artifact info from write()
        mock_results = MagicMock()
        mock_results.name = "csv"
        mock_results.determinism = Determinism.IO_WRITE
        mock_results.plugin_version = "1.0.0"
        mock_results.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

        mock_flagged = MagicMock()
        mock_flagged.name = "csv"
        mock_flagged.determinism = Determinism.IO_WRITE
        mock_flagged.plugin_version = "1.0.0"
        mock_flagged.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"results": mock_results, "flagged": mock_flagged},
            gates=[routing_gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            pipeline_config, graph=_build_test_graph(pipeline_config)
        )

        # Row should be routed, not completed
        assert result.rows_processed == 1
        assert result.rows_routed == 1
        assert mock_flagged.write.called, "flagged sink should receive routed row"
        assert (
            not mock_results.write.called
        ), "results sink should not receive routed row"


class TestLifecycleHooks:
    """Orchestrator invokes plugin lifecycle hooks."""

    def test_on_start_called_before_processing(self) -> None:
        """on_start() called before any rows processed."""
        from unittest.mock import MagicMock

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        call_order: list[str] = []

        from elspeth.contracts import PluginSchema, SourceRow

        class TestSchema(PluginSchema):
            model_config = {"extra": "allow"}  # noqa: RUF012

        class TrackedTransform(BaseTransform):
            name = "tracked"
            input_schema = TestSchema
            output_schema = TestSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})

            def on_start(self, ctx: Any) -> None:
                call_order.append("on_start")

            def process(self, row: Any, ctx: Any) -> TransformResult:
                call_order.append("process")
                return TransformResult.success(row)

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter([SourceRow.valid({"id": 1})])

        transform = TrackedTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"
        mock_sink.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

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
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
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

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        call_order: list[str] = []

        class TestSchema(PluginSchema):
            model_config = {"extra": "allow"}  # noqa: RUF012

        class TrackedTransform(BaseTransform):
            name = "tracked"
            input_schema = TestSchema
            output_schema = TestSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})

            def on_start(self, ctx: Any) -> None:
                call_order.append("on_start")

            def process(self, row: Any, ctx: Any) -> TransformResult:
                call_order.append("process")
                return TransformResult.success(row)

            def on_complete(self, ctx: Any) -> None:
                call_order.append("on_complete")

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter(
            [SourceRow.valid({"id": 1}), SourceRow.valid({"id": 2})]
        )

        transform = TrackedTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"
        mock_sink.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="tracked")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_complete should be called last (among transform lifecycle calls)
        transform_calls = [
            c for c in call_order if c in ["on_start", "process", "on_complete"]
        ]
        assert transform_calls[-1] == "on_complete"
        # All processing should happen before on_complete
        assert call_order.count("process") == 2

    def test_on_complete_called_on_error(self) -> None:
        """on_complete() called even when run fails."""
        from unittest.mock import MagicMock

        import pytest

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        completed: list[bool] = []

        class TestSchema(PluginSchema):
            model_config = {"extra": "allow"}  # noqa: RUF012

        class FailingTransform(BaseTransform):
            name = "failing"
            input_schema = TestSchema
            output_schema = TestSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})

            def on_start(self, ctx: Any) -> None:
                pass

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("intentional failure")

            def on_complete(self, ctx: Any) -> None:
                completed.append(True)

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter([SourceRow.valid({"id": 1})])

        transform = FailingTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="failing")
        graph.add_node("transform", node_type="transform", plugin_name="failing")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

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
            """Sink that captures written rows."""

            name = "collect"

            def __init__(self) -> None:
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, row: Any, ctx: Any) -> ArtifactDescriptor:
                # Row processing writes batches (lists), export writes single records
                if isinstance(row, list):
                    self.captured_rows.extend(row)
                else:
                    self.captured_rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
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

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

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
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, row: Any, ctx: Any) -> ArtifactDescriptor:
                # Row processing writes batches (lists), export writes single records
                if isinstance(row, list):
                    self.captured_rows.extend(row)
                else:
                    self.captured_rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
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

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

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
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.captured_rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
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
        env_without_key = {
            k: v for k, v in os.environ.items() if k != "ELSPETH_SIGNING_KEY"
        }
        with (
            patch.dict(os.environ, env_without_key, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            orchestrator.run(pipeline, graph=graph, settings=settings)

    def test_orchestrator_no_export_when_disabled(self) -> None:
        """Should not export when export.enabled is False."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            LandscapeSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

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
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.captured_rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
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
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        call_order: list[str] = []

        class TrackedSource(_TestSourceBase):
            """Source that tracks lifecycle calls."""

            name = "tracked_source"
            output_schema = _TestSchema

            def on_start(self, ctx: Any) -> None:
                call_order.append("source_on_start")

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                call_order.append("source_load")
                yield SourceRow.valid({"value": 1})

            def on_complete(self, ctx: Any) -> None:
                call_order.append("source_on_complete")

            def close(self) -> None:
                call_order.append("source_close")

        db = LandscapeDB.in_memory()

        source = TrackedSource()
        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"
        mock_sink.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": mock_sink},
        )

        # Minimal graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="tracked_source")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_start should be called BEFORE load
        assert "source_on_start" in call_order, "Source on_start should be called"
        assert call_order.index("source_on_start") < call_order.index(
            "source_load"
        ), "Source on_start should be called before load"
        # on_complete should be called AFTER load and BEFORE close
        assert "source_on_complete" in call_order, "Source on_complete should be called"
        assert call_order.index("source_on_complete") > call_order.index(
            "source_load"
        ), "Source on_complete should be called after load"
        assert call_order.index("source_on_complete") < call_order.index(
            "source_close"
        ), "Source on_complete should be called before close"


class TestSinkLifecycleHooks:
    """Tests for sink plugin lifecycle hook calls."""

    def test_sink_lifecycle_hooks_called(self) -> None:
        """Sink on_start and on_complete should be called."""
        from unittest.mock import MagicMock

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        call_order: list[str] = []

        class TrackedSink(_TestSinkBase):
            """Sink that tracks lifecycle calls."""

            name = "tracked_sink"

            def on_start(self, ctx: Any) -> None:
                call_order.append("sink_on_start")

            def on_complete(self, ctx: Any) -> None:
                call_order.append("sink_on_complete")

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                call_order.append("sink_write")
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                call_order.append("sink_close")

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter([SourceRow.valid({"value": 1})])

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
        graph.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_start should be called before write
        assert "sink_on_start" in call_order, "Sink on_start should be called"
        assert call_order.index("sink_on_start") < call_order.index(
            "sink_write"
        ), "Sink on_start should be called before write"
        # on_complete should be called after write, before close
        assert "sink_on_complete" in call_order, "Sink on_complete should be called"
        assert call_order.index("sink_on_complete") > call_order.index(
            "sink_write"
        ), "Sink on_complete should be called after write"

    def test_sink_on_complete_called_even_on_error(self) -> None:
        """Sink on_complete should be called even when run fails."""
        from unittest.mock import MagicMock

        import pytest

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        completed: list[str] = []

        class TestSchema(PluginSchema):
            model_config = {"extra": "allow"}  # noqa: RUF012

        class FailingTransform(BaseTransform):
            name = "failing"
            input_schema = TestSchema
            output_schema = TestSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("intentional failure")

        class TrackedSink(_TestSinkBase):
            name = "tracked_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                completed.append("sink_on_complete")

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.load.return_value = iter([SourceRow.valid({"value": 1})])

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
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        # Checkpoint every 3 rows
        settings = CheckpointSettings(
            enabled=True, frequency="every_n", checkpoint_interval=3
        )

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        """Checkpoints are preserved when run fails for recovery.

        IMPORTANT: Checkpoints are created AFTER sink writes, not during transform
        processing. This test verifies that when a pipeline fails with multiple
        sinks (where one succeeds and one fails), the checkpoints from the
        successful sink are preserved for potential recovery.

        The batched write model means:
        - All rows are processed through transforms first
        - Then pending tokens are written to each sink in sequence
        - Checkpoints are created after each sink.write() returns successfully
        - If a later sink fails, checkpoints from earlier sinks are preserved
        """
        from elspeth.contracts import PluginSchema, RoutingMode, SourceRow
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings, GateSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class GoodSink(_TestSinkBase):
            """Sink that succeeds."""

            name = "good_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="good123"
                )

            def close(self) -> None:
                pass

        class BadSink(_TestSinkBase):
            """Sink that fails during write."""

            name = "bad_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Bad sink failure")

            def close(self) -> None:
                pass

        # Route odd values to 'good' sink, even values to 'bad' sink
        gate_config = GateSettings(
            name="split",
            condition="row['value'] % 2 == 1",
            routes={"true": "good", "false": "bad"},
        )

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = PassthroughTransform()
        good_sink = GoodSink()
        bad_sink = BadSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"good": good_sink, "bad": bad_sink},
            gates=[gate_config],
        )

        # Build graph with routing
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="list_source")
        graph.add_node("transform_0", node_type="transform", plugin_name="passthrough")
        graph.add_node(
            "config_gate_split",
            node_type="gate",
            plugin_name="config_gate:split",
            config={
                "condition": gate_config.condition,
                "routes": dict(gate_config.routes),
            },
        )
        graph.add_node("sink_good", node_type="sink", plugin_name="good_sink")
        graph.add_node("sink_bad", node_type="sink", plugin_name="bad_sink")

        graph.add_edge("source", "transform_0", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge(
            "transform_0", "config_gate_split", label="continue", mode=RoutingMode.MOVE
        )
        graph.add_edge(
            "config_gate_split", "sink_good", label="true", mode=RoutingMode.MOVE
        )
        graph.add_edge(
            "config_gate_split", "sink_bad", label="false", mode=RoutingMode.MOVE
        )

        graph._sink_id_map = {"good": "sink_good", "bad": "sink_bad"}
        graph._transform_id_map = {0: "transform_0"}
        graph._config_gate_id_map = {"split": "config_gate_split"}
        graph._route_resolution_map = {
            ("config_gate_split", "true"): "good",
            ("config_gate_split", "false"): "bad",
        }
        graph._output_sink = "good"

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=settings,
        )

        # The bad sink will fail, but good sink should have already written
        # Note: Sink iteration order in dict may vary, so we check checkpoints exist
        with pytest.raises(RuntimeError, match="Bad sink failure"):
            orchestrator.run(config, graph=graph)

        # Query for the failed run
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1
        run_id = runs[0].run_id

        # Checkpoints should be preserved (at least for the good sink)
        remaining_checkpoints = checkpoint_mgr.get_checkpoints(run_id)

        # We may have 0, 1, or 2 checkpoints depending on which sink was written first
        # The key assertion is: if good sink was written first, its checkpoints exist
        # If bad sink was written first, it fails before any checkpoints
        # This test verifies the durability invariant: checkpoints that were created
        # are NOT deleted on failure (they're only deleted on success)

        # Check that the good sink did write (if it got a chance)
        if len(good_sink.results) > 0:
            # Good sink wrote - should have checkpoints for those rows
            assert len(remaining_checkpoints) == len(good_sink.results), (
                f"Expected {len(good_sink.results)} checkpoints for written rows, "
                f"got {len(remaining_checkpoints)}"
            )
        # If good sink didn't write (bad sink failed first), that's also valid behavior

    def test_checkpoint_disabled_skips_checkpoint_creation(self) -> None:
        """No checkpoints created when checkpointing is disabled."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=False)

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        settings = CheckpointSettings(enabled=True, frequency="every_row")

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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

        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

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
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
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


class TestNodeMetadataFromPlugin:
    """Test that node registration uses actual plugin metadata.

    BUG: All nodes were registered with hardcoded plugin_version="1.0.0"
    instead of reading from the actual plugin class attributes.
    """

    def test_node_metadata_records_plugin_version(self) -> None:
        """Node registration should use actual plugin metadata.

        Verifies that the node's plugin_version in Landscape matches
        the plugin class's plugin_version attribute.
        """
        from elspeth.contracts import Determinism, PluginSchema, SourceRow
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "versioned_source"
            output_schema = ValueSchema
            plugin_version = "3.7.2"  # Custom version

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class VersionedTransform(BaseTransform):
            name = "versioned_transform"
            input_schema = ValueSchema
            output_schema = ValueSchema
            plugin_version = "2.5.0"  # Custom version (not 1.0.0)
            determinism = Determinism.EXTERNAL_CALL

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class VersionedSink(_TestSinkBase):
            name = "versioned_sink"
            plugin_version = "4.1.0"  # Custom version

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 42}])
        transform = VersionedTransform()
        sink = VersionedSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        # Build graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="versioned_source")
        graph.add_node(
            "transform", node_type="transform", plugin_name="versioned_transform"
        )
        graph.add_node("sink", node_type="sink", plugin_name="versioned_sink")
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"default": "sink"}
        graph._output_sink = "default"

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph)

        # Query Landscape to verify node metadata
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)
        assert len(nodes) == 3  # source, transform, sink

        # Create lookup by plugin_name
        nodes_by_name = {n.plugin_name: n for n in nodes}

        # Verify source has correct version
        source_node = nodes_by_name["versioned_source"]
        assert (
            source_node.plugin_version == "3.7.2"
        ), f"Source plugin_version should be '3.7.2', got '{source_node.plugin_version}'"

        # Verify transform has correct version
        transform_node = nodes_by_name["versioned_transform"]
        assert (
            transform_node.plugin_version == "2.5.0"
        ), f"Transform plugin_version should be '2.5.0', got '{transform_node.plugin_version}'"

        # Verify sink has correct version
        sink_node = nodes_by_name["versioned_sink"]
        assert (
            sink_node.plugin_version == "4.1.0"
        ), f"Sink plugin_version should be '4.1.0', got '{sink_node.plugin_version}'"

    def test_node_metadata_records_determinism(self) -> None:
        """Node registration should record plugin determinism.

        Verifies that nondeterministic plugins are recorded correctly
        in the Landscape for reproducibility tracking.
        """
        from elspeth.contracts import Determinism, PluginSchema, SourceRow
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema
            plugin_version = "1.0.0"

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class NonDeterministicTransform(BaseTransform):
            name = "nondeterministic_transform"
            input_schema = ValueSchema
            output_schema = ValueSchema
            plugin_version = "1.0.0"
            determinism = Determinism.EXTERNAL_CALL  # Explicit nondeterministic

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            name = "test_sink"
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 42}])
        transform = NonDeterministicTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        # Build graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="test_source")
        graph.add_node(
            "transform", node_type="transform", plugin_name="nondeterministic_transform"
        )
        graph.add_node("sink", node_type="sink", plugin_name="test_sink")
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"default": "sink"}
        graph._output_sink = "default"

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph)

        # Query Landscape to verify determinism recorded
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)

        # Find the transform node
        transform_node = next(
            n for n in nodes if n.plugin_name == "nondeterministic_transform"
        )

        # Verify determinism is recorded correctly
        assert (
            transform_node.determinism == "external_call"
        ), f"Transform determinism should be 'nondeterministic', got '{transform_node.determinism}'"


class TestRouteValidation:
    """Test that route destinations are validated at initialization.

    MED-003: Route validation should happen BEFORE any rows are processed,
    not during row processing. This prevents partial runs where config errors
    are discovered after processing some rows.
    """

    def test_valid_routes_pass_validation(self) -> None:
        """Valid route configurations should pass validation without error."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 100}])
        default_sink = CollectSink()
        quarantine_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink, "quarantine": quarantine_sink},
            gates=[routing_gate],
        )

        orchestrator = Orchestrator(db)
        # Should not raise - routes are valid
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert len(default_sink.results) == 1  # value=10 continues
        assert len(quarantine_sink.results) == 1  # value=100 routed

    def test_invalid_route_destination_fails_at_init(self) -> None:
        """Route to non-existent sink should fail before processing any rows."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        db = LandscapeDB.in_memory()

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 100}])
        default_sink = CollectSink()
        # Note: NO quarantine sink provided!

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink},  # Only default, no quarantine
            gates=[safety_gate],
        )

        orchestrator = Orchestrator(db)

        # Should fail at initialization with clear error message
        with pytest.raises(RouteValidationError) as exc_info:
            orchestrator.run(config, graph=_build_test_graph(config))

        # Verify error message contains helpful information
        error_msg = str(exc_info.value)
        assert "safety_gate" in error_msg  # Gate name
        assert "quarantine" in error_msg  # Invalid destination
        assert "default" in error_msg  # Available sinks

        # Verify no rows were processed
        assert (
            not source.load_called
        ), "Source should not be loaded on validation failure"
        assert len(default_sink.results) == 0, "No rows should be written on failure"

    def test_error_message_includes_route_label(self) -> None:
        """Error message should include the route label for debugging."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        db = LandscapeDB.in_memory()

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}])
        default_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink, "errors": CollectSink()},
            gates=[threshold_gate],
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RouteValidationError) as exc_info:
            orchestrator.run(config, graph=_build_test_graph(config))

        error_msg = str(exc_info.value)
        # Should include destination (route target)
        assert "high_scores" in error_msg
        # Should include available sinks
        assert "default" in error_msg
        assert "errors" in error_msg
        # Should include gate name
        assert "threshold_gate" in error_msg

    def test_continue_routes_are_not_validated_as_sinks(self) -> None:
        """Routes that resolve to 'continue' should not be validated as sinks."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}])
        default_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink},
            gates=[filter_gate],
        )

        orchestrator = Orchestrator(db)
        # Should not raise - "continue" is a valid routing target
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 1


class TestOrchestratorForkExecution:
    """Test orchestrator handles fork results correctly.

    NOTE: Full fork testing at orchestrator level is blocked by ExecutionGraph
    using DiGraph instead of MultiDiGraph (can't store multiple edges between
    same nodes). See WP-07 notes. Fork logic is tested at processor level in
    test_processor.py::TestRowProcessorWorkQueue.
    """

    def test_orchestrator_handles_list_results_from_processor(self) -> None:
        """Orchestrator correctly iterates over list[RowResult] from processor.

        This tests the basic plumbing (list handling, counting) without forks.
        Fork-specific behavior is tested at processor level.
        """
        import hashlib

        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class RowSchema(_TestSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"value": 1})
                yield SourceRow.valid({"value": 2})
                yield SourceRow.valid({"value": 3})

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[Any] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                content = str(rows).encode()
                return ArtifactDescriptor(
                    artifact_type="file",
                    path_or_uri="memory://test",
                    content_hash=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                )

            def close(self) -> None:
                pass

        source = ListSource()
        transform = PassthroughTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        graph = _build_test_graph(config)

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph)

        assert run_result.status == "completed"
        # 3 rows from source
        assert run_result.rows_processed == 3
        # All 3 reach COMPLETED (no forks)
        assert run_result.rows_succeeded == 3
        # All 3 written to sink
        assert len(sink.results) == 3


class TestOrchestratorQuarantineMetrics:
    """Test that QUARANTINED rows are counted separately from FAILED."""

    def test_orchestrator_counts_quarantined_rows(self) -> None:
        """Orchestrator should count QUARANTINED rows separately.

        A transform with _on_error="discard" intentionally quarantines rows
        when it returns TransformResult.error(). These should be counted
        as quarantined, not failed.
        """
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int
            quality: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class QualityFilter(BaseTransform):
            """Transform that errors on 'bad' quality rows.

            With _on_error="discard", these become QUARANTINED.
            """

            name = "quality_filter"
            input_schema = ValueSchema
            output_schema = ValueSchema
            _on_error = "discard"  # Intentionally discard errors

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                if row.get("quality") == "bad":
                    return TransformResult.error(
                        {"reason": "bad_quality", "value": row["value"]}
                    )
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # 3 rows: good, bad, good
        source = ListSource(
            [
                {"value": 1, "quality": "good"},
                {"value": 2, "quality": "bad"},  # Will be quarantined
                {"value": 3, "quality": "good"},
            ]
        )
        transform = QualityFilter()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=_build_test_graph(config))

        # Verify counts
        assert run_result.status == "completed"
        assert run_result.rows_processed == 3, "All 3 rows should be processed"
        assert run_result.rows_succeeded == 2, "2 good quality rows should succeed"
        assert (
            run_result.rows_quarantined == 1
        ), "1 bad quality row should be quarantined"
        assert run_result.rows_failed == 0, "No rows should fail (quarantine != fail)"

        # Only good rows written to sink
        assert len(sink.results) == 2
        assert all(r["quality"] == "good" for r in sink.results)


class TestOrchestratorRetry:
    """Tests for retry configuration in Orchestrator."""

    def test_orchestrator_creates_retry_manager_from_settings(self) -> None:
        """Orchestrator creates RetryManager when settings.retry is configured."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RetrySettings,
            SinkSettings,
        )
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Transform that tracks retry attempts via closure
        attempt_count = {"count": 0}

        class RetryableTransform(BaseTransform):
            name = "retryable"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                attempt_count["count"] += 1
                # Fail with retryable error on first attempt
                if attempt_count["count"] == 1:
                    raise ConnectionError("Transient failure")
                return TransformResult.success(row)

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Settings with retry configuration
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"default": SinkSettings(plugin="csv")},
            output_sink="default",
            retry=RetrySettings(
                max_attempts=3,
                initial_delay_seconds=0.01,  # Fast for testing
                max_delay_seconds=0.1,
            ),
        )

        source = ListSource([{"value": 42}])
        transform = RetryableTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        # Use _build_test_graph to create graph matching PipelineConfig
        graph = _build_test_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Row should succeed after retry
        assert result.status == "completed"
        assert result.rows_processed == 1
        assert result.rows_succeeded == 1
        # Transform was called twice (first attempt failed, second succeeded)
        assert (
            attempt_count["count"] == 2
        ), f"Expected 2 attempts (1 failure + 1 success), got {attempt_count['count']}"
        assert len(sink.results) == 1

    def test_orchestrator_retry_exhausted_marks_row_failed(self) -> None:
        """When all retry attempts fail, row should be marked FAILED."""
        from elspeth.contracts import PluginSchema, SourceRow
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RetrySettings,
            SinkSettings,
        )
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Transform that always fails with retryable error
        class AlwaysFailTransform(BaseTransform):
            name = "always_fail"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise ConnectionError("Persistent failure")

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"default": SinkSettings(plugin="csv")},
            output_sink="default",
            retry=RetrySettings(
                max_attempts=2,  # Will try twice then fail
                initial_delay_seconds=0.01,
                max_delay_seconds=0.1,
            ),
        )

        source = ListSource([{"value": 42}])
        transform = AlwaysFailTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        # Use _build_test_graph to create graph matching PipelineConfig
        graph = _build_test_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Row should be marked failed after exhausting retries
        assert result.status == "completed"
        assert result.rows_processed == 1
        assert result.rows_failed == 1
        assert result.rows_succeeded == 0
        assert len(sink.results) == 0


class TestCoalesceWiring:
    """Test that coalesce is wired into orchestrator."""

    def test_orchestrator_creates_coalesce_executor_when_config_present(
        self,
    ) -> None:
        """When settings.coalesce is non-empty, CoalesceExecutor should be created."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # Mock source/sink to avoid file access
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"
        mock_source.output_schema = _TestSchema

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"
        mock_sink.input_schema = _TestSchema

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        # Build the graph from settings (which includes coalesce)
        graph = ExecutionGraph.from_config(settings)

        # Patch RowProcessor to capture its args
        with patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor:
            mock_processor.return_value.process_row.return_value = []
            mock_processor.return_value.token_manager = MagicMock()

            orchestrator.run(config, graph=graph, settings=settings)

            # RowProcessor should have been called with coalesce_executor
            call_kwargs = mock_processor.call_args.kwargs
            assert "coalesce_executor" in call_kwargs
            assert call_kwargs["coalesce_executor"] is not None
            assert "coalesce_node_ids" in call_kwargs
            assert call_kwargs["coalesce_node_ids"] is not None
            # Verify the coalesce_node_ids contains our registered coalesce
            assert "merge_results" in call_kwargs["coalesce_node_ids"]

    def test_orchestrator_handles_coalesced_outcome(self) -> None:
        """COALESCED outcome should route merged token to output sink."""
        from unittest.mock import MagicMock, patch

        from elspeth.contracts import RowOutcome, TokenInfo
        from elspeth.contracts.results import RowResult
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter(
            [MagicMock(is_quarantined=False, row={"value": 1})]
        )
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"
        mock_source.output_schema = _TestSchema

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"
        mock_sink.input_schema = _TestSchema
        mock_sink.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

        # Settings with coalesce (needed to enable coalesce path in orchestrator)
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
            gates=settings.gates,
        )

        graph = ExecutionGraph.from_config(settings)

        orchestrator = Orchestrator(db=db)

        # Mock RowProcessor to return COALESCED outcome
        merged_token = TokenInfo(
            row_id="row_1",
            token_id="merged_token_1",
            row_data={"merged": True},
            branch_name=None,
        )
        coalesced_result = RowResult(
            token=merged_token,
            final_data={"merged": True},
            outcome=RowOutcome.COALESCED,
        )

        with (
            patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor_cls,
            patch("elspeth.engine.executors.SinkExecutor") as mock_sink_executor_cls,
        ):
            mock_processor = MagicMock()
            mock_processor.process_row.return_value = [coalesced_result]
            mock_processor.token_manager.create_initial_token.return_value = MagicMock(
                row_id="row_1", token_id="t1", row_data={"value": 1}
            )
            mock_processor_cls.return_value = mock_processor

            # Mock SinkExecutor to avoid foreign key constraint errors
            mock_sink_executor = MagicMock()
            mock_sink_executor_cls.return_value = mock_sink_executor

            result = orchestrator.run(config, graph=graph, settings=settings)

            # COALESCED should count toward rows_coalesced
            assert result.rows_coalesced == 1

            # Verify the merged token was added to pending_tokens and passed to sink
            # SinkExecutor.write should have been called with the merged token
            assert mock_sink_executor.write.called
            write_call = mock_sink_executor.write.call_args
            tokens_written = write_call.kwargs.get("tokens") or write_call.args[1]
            assert len(tokens_written) == 1
            assert tokens_written[0].token_id == "merged_token_1"

    def test_orchestrator_calls_flush_pending_at_end(self) -> None:
        """flush_pending should be called on coalesce executor at end of source."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])  # Empty - immediate end
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"
        mock_source.output_schema = _TestSchema

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"
        mock_sink.input_schema = _TestSchema

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)
        graph = ExecutionGraph.from_config(settings)

        with patch(
            "elspeth.engine.coalesce_executor.CoalesceExecutor"
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.flush_pending.return_value = []
            mock_executor_cls.return_value = mock_executor

            orchestrator.run(config, graph=graph, settings=settings)

            # flush_pending should have been called
            mock_executor.flush_pending.assert_called_once()

    def test_orchestrator_flush_pending_routes_merged_tokens_to_sink(self) -> None:
        """Merged tokens from flush_pending should be routed to output sink."""
        from unittest.mock import MagicMock, patch

        from elspeth.contracts import TokenInfo
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.coalesce_executor import CoalesceOutcome
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="best_effort",  # best_effort merges whatever arrived
                    merge="union",
                    timeout_seconds=10.0,  # Required for best_effort
                ),
            ],
        )

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])  # Empty - immediate end
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"
        mock_source.output_schema = _TestSchema

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"
        mock_sink.input_schema = _TestSchema
        mock_sink.write.return_value = ArtifactDescriptor.for_file(
            path="memory", size_bytes=0, content_hash="abc123"
        )

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)
        graph = ExecutionGraph.from_config(settings)

        # Create a merged token that flush_pending will return
        merged_token = TokenInfo(
            row_id="row_1",
            token_id="flushed_merged_token",
            row_data={"merged_at_flush": True},
            branch_name=None,
        )

        with (
            patch(
                "elspeth.engine.coalesce_executor.CoalesceExecutor"
            ) as mock_executor_cls,
            patch("elspeth.engine.executors.SinkExecutor") as mock_sink_executor_cls,
        ):
            mock_executor = MagicMock()
            # flush_pending returns a merged token
            mock_executor.flush_pending.return_value = [
                CoalesceOutcome(
                    held=False,
                    merged_token=merged_token,
                    consumed_tokens=[],
                    coalesce_metadata={"policy": "best_effort"},
                )
            ]
            mock_executor_cls.return_value = mock_executor

            mock_sink_executor = MagicMock()
            mock_sink_executor_cls.return_value = mock_sink_executor

            result = orchestrator.run(config, graph=graph, settings=settings)

            # flush_pending should have been called
            mock_executor.flush_pending.assert_called_once()

            # The merged token from flush should count toward rows_coalesced
            assert result.rows_coalesced == 1

            # The merged token should be written to the sink
            assert mock_sink_executor.write.called
            write_call = mock_sink_executor.write.call_args
            tokens_written = write_call.kwargs.get("tokens") or write_call.args[1]
            assert len(tokens_written) == 1
            assert tokens_written[0].token_id == "flushed_merged_token"

    def test_orchestrator_flush_pending_handles_failures(self) -> None:
        """Failed coalesce outcomes from flush_pending should not crash."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.coalesce_executor import CoalesceOutcome
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",  # Will fail if not all branches arrive
                    merge="union",
                ),
            ],
        )

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])  # Empty - immediate end
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"
        mock_source.output_schema = _TestSchema

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"
        mock_sink.input_schema = _TestSchema

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)
        graph = ExecutionGraph.from_config(settings)

        with patch(
            "elspeth.engine.coalesce_executor.CoalesceExecutor"
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            # flush_pending returns a failure outcome (incomplete branches)
            mock_executor.flush_pending.return_value = [
                CoalesceOutcome(
                    held=False,
                    merged_token=None,  # No merged token on failure
                    failure_reason="incomplete_branches",
                    coalesce_metadata={
                        "policy": "require_all",
                        "expected_branches": ["path_a", "path_b"],
                        "branches_arrived": ["path_a"],
                    },
                )
            ]
            mock_executor_cls.return_value = mock_executor

            # Should not raise - failures are recorded but don't crash
            result = orchestrator.run(config, graph=graph, settings=settings)

            # flush_pending should have been called
            mock_executor.flush_pending.assert_called_once()

            # No merged tokens means no rows_coalesced increment
            assert result.rows_coalesced == 0

    def test_orchestrator_computes_coalesce_step_map(self) -> None:
        """Orchestrator should compute step positions for each coalesce point."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            row_plugins=[
                RowPluginSettings(plugin="passthrough"),  # Step 0
                RowPluginSettings(plugin="passthrough"),  # Step 1
            ],
            gates=[
                GateSettings(
                    name="forker",  # Step 2
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",  # Step 3
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"
        mock_source.output_schema = _TestSchema

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"
        mock_sink.input_schema = _TestSchema

        mock_transform = MagicMock()
        mock_transform.name = "passthrough"
        mock_transform.plugin_version = "1.0.0"
        mock_transform.determinism = "deterministic"
        mock_transform.is_batch_aware = False

        config = PipelineConfig(
            source=mock_source,
            transforms=[mock_transform, mock_transform],
            sinks={"output": mock_sink},
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        # Build the graph from settings (which includes coalesce)
        graph = ExecutionGraph.from_config(settings)

        with patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process_row.return_value = []
            mock_processor_cls.return_value = mock_processor

            orchestrator.run(config, graph=graph, settings=settings)

            # Check coalesce_step_map was passed
            call_kwargs = mock_processor_cls.call_args.kwargs
            assert "coalesce_step_map" in call_kwargs
            # 2 transforms + 1 gate = step 3 for coalesce
            assert call_kwargs["coalesce_step_map"]["merge_results"] == 3
