# tests/engine/test_orchestrator_cleanup.py
"""Tests for transform/gate cleanup in orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode, SourceRow
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import as_sink, as_source


def _build_test_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple graph for testing.

    Creates a linear graph matching the PipelineConfig structure:
    source -> transforms... -> sinks
    """
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
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks and populate sink_id_map
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)
        graph.add_edge(prev, node_id, label=sink_name, mode=RoutingMode.MOVE)

        # Gates can route to any sink, so add edges from all gates
        for i, t in enumerate(config.transforms):
            if hasattr(t, "evaluate"):
                gate_id = f"transform_{i}"
                if gate_id != prev:  # Don't duplicate edge
                    graph.add_edge(
                        gate_id, node_id, label=sink_name, mode=RoutingMode.MOVE
                    )

    # Populate internal ID maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids

    # Populate route resolution map: (gate_id, label) -> sink_name
    route_resolution_map: dict[tuple[str, str], str] = {}
    for i, t in enumerate(config.transforms):
        if hasattr(t, "evaluate"):  # It's a gate
            gate_id = f"transform_{i}"
            for sink_name in sink_ids:
                route_resolution_map[(gate_id, sink_name)] = sink_name
    graph._route_resolution_map = route_resolution_map

    # Set output_sink
    if "default" in sink_ids:
        graph._output_sink = "default"
    elif sink_ids:
        graph._output_sink = next(iter(sink_ids))

    return graph


class ValueSchema(PluginSchema):
    """Simple schema for test rows."""

    value: int


class ListSource:
    """Test source that yields from a list."""

    name = "list_source"
    output_schema = ValueSchema
    node_id: str | None = None  # Required by SourceProtocol
    determinism = Determinism.DETERMINISTIC  # Required by SourceProtocol
    plugin_version = "1.0.0"  # Required by SourceProtocol

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Any:
        for _row in self._data:
            yield SourceRow.valid(_row)

    def close(self) -> None:
        pass


class FailingSource(ListSource):
    """Test source that raises an exception during load."""

    name = "failing_source"

    def load(self, ctx: Any) -> Any:
        raise RuntimeError("Source failed intentionally")


class CollectSink:
    """Test sink that collects results in memory."""

    name = "collect"
    input_schema = ValueSchema  # Required by SinkProtocol
    idempotent = True  # Required by SinkProtocol
    node_id: str | None = None  # Required by SinkProtocol
    determinism = Determinism.DETERMINISTIC  # Required by SinkProtocol
    plugin_version = "1.0"  # Required by SinkProtocol

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TrackingTransform(BaseTransform):
    """Transform that tracks whether close() was called."""

    input_schema = ValueSchema
    output_schema = ValueSchema

    def __init__(self, name: str = "tracking") -> None:
        super().__init__({})
        self.name = name  # type: ignore[misc]
        self.close_called = False
        self.close_call_count = 0

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row)

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1


class FailingCloseTransform(TrackingTransform):
    """Transform whose close() raises an error."""

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1
        raise RuntimeError("Close failed!")


class TestOrchestratorCleanup:
    """Tests for Orchestrator calling close() on plugins."""

    def test_transforms_closed_on_success(self) -> None:
        """All transforms should have close() called after successful run."""
        db = LandscapeDB.in_memory()

        transform_1 = TrackingTransform("transform_1")
        transform_2 = TrackingTransform("transform_2")

        source = ListSource([{"value": 1}, {"value": 2}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform_1, transform_2],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=_build_test_graph(config))

        # Verify close() was called on all transforms
        assert transform_1.close_called, "transform_1.close() was not called"
        assert (
            transform_1.close_call_count == 1
        ), "transform_1.close() called multiple times"
        assert transform_2.close_called, "transform_2.close() was not called"
        assert (
            transform_2.close_call_count == 1
        ), "transform_2.close() called multiple times"

    def test_transforms_closed_on_failure(self) -> None:
        """All transforms should have close() called even if run fails."""
        db = LandscapeDB.in_memory()

        transform_1 = TrackingTransform("transform_1")
        transform_2 = TrackingTransform("transform_2")

        # Use failing source
        source = FailingSource([{"value": 1}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform_1, transform_2],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError, match="Source failed intentionally"):
            orchestrator.run(config, graph=_build_test_graph(config))

        # Verify close() was called on all transforms even though run failed
        assert (
            transform_1.close_called
        ), "transform_1.close() was not called after failure"
        assert (
            transform_2.close_called
        ), "transform_2.close() was not called after failure"

    def test_cleanup_handles_missing_close_method(self) -> None:
        """Cleanup should handle transforms that use default close() method.

        BaseTransform provides a default no-op close() method, so transforms
        that don't override it still satisfy the protocol. This test verifies
        the cleanup process works correctly with the default implementation.
        """
        db = LandscapeDB.in_memory()

        # Transform using BaseTransform's default close() implementation
        class MinimalTransform(BaseTransform):
            name = "minimal"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

            # Uses default close() from BaseTransform (no-op)

        source = ListSource([{"value": 1}])
        transform = MinimalTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        # Should not raise even though transform has no close()
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"

    def test_cleanup_continues_if_one_close_fails(self) -> None:
        """If one transform's close() fails, others should still be closed.

        Cleanup should be best-effort - one plugin failure shouldn't prevent
        cleanup of other plugins.
        """
        db = LandscapeDB.in_memory()

        # First transform: close() raises an error
        transform_1 = FailingCloseTransform("failing_close")

        # Second transform: close() works normally
        transform_2 = TrackingTransform("normal_close")

        source = ListSource([{"value": 1}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform_1, transform_2],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        # Should complete without raising, despite first transform's close() failing
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        # Both close() methods should have been called
        assert transform_1.close_called, "failing transform's close() was not called"
        assert (
            transform_2.close_called
        ), "second transform's close() was not called despite first failing"
