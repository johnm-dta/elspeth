# tests/engine/test_orchestrator_cleanup.py
"""Tests for transform/gate cleanup in orchestrator."""

from __future__ import annotations

import pytest

from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.results import GateResult, RoutingAction, TransformResult
from elspeth.plugins.schemas import PluginSchema


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

    def __init__(self, data: list[dict]) -> None:
        self._data = data

    def on_start(self, ctx) -> None:
        pass

    def on_complete(self, ctx) -> None:
        pass

    def load(self, ctx):
        yield from self._data

    def close(self) -> None:
        pass


class FailingSource(ListSource):
    """Test source that raises an exception during load."""

    name = "failing_source"

    def load(self, ctx):
        raise RuntimeError("Source failed intentionally")


class CollectSink:
    """Test sink that collects results in memory."""

    name = "collect"

    def __init__(self) -> None:
        self.results: list[dict] = []

    def on_start(self, ctx) -> None:
        pass

    def on_complete(self, ctx) -> None:
        pass

    def write(self, rows, ctx):
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

    def close(self) -> None:
        pass


class TrackingTransform:
    """Transform that tracks whether close() was called."""

    input_schema = ValueSchema
    output_schema = ValueSchema

    def __init__(self, name: str = "tracking") -> None:
        self.name = name
        self.close_called = False
        self.close_call_count = 0

    def on_start(self, ctx) -> None:
        pass

    def on_complete(self, ctx) -> None:
        pass

    def process(self, row, ctx):
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


class TrackingGate:
    """Gate that tracks whether close() was called."""

    input_schema = ValueSchema
    output_schema = ValueSchema

    def __init__(self, name: str = "tracking_gate") -> None:
        self.name = name
        self.close_called = False
        self.close_call_count = 0

    def on_start(self, ctx) -> None:
        pass

    def on_complete(self, ctx) -> None:
        pass

    def evaluate(self, row, ctx):
        return GateResult(row=row, action=RoutingAction.continue_())

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1


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
            source=source,
            transforms=[transform_1, transform_2],
            sinks={"default": sink},
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
            source=source,
            transforms=[transform_1, transform_2],
            sinks={"default": sink},
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

    def test_gates_closed(self) -> None:
        """Gates should have close() called after run."""
        db = LandscapeDB.in_memory()

        gate = TrackingGate("test_gate")

        source = ListSource([{"value": 1}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=_build_test_graph(config))

        # Verify close() was called on gate
        assert gate.close_called, "gate.close() was not called"
        assert gate.close_call_count == 1, "gate.close() called multiple times"

    def test_cleanup_handles_missing_close_method(self) -> None:
        """Cleanup should handle transforms without close() method gracefully.

        This tests graceful degradation for old plugins that may not have
        the close() method (they implement an older protocol version).
        """
        db = LandscapeDB.in_memory()

        # Create a transform without close() method (simulating old plugin)
        class OldStyleTransform:
            name = "old_style"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def on_start(self, ctx) -> None:
                pass

            def on_complete(self, ctx) -> None:
                pass

            def process(self, row, ctx):
                return TransformResult.success(row)

            # Note: No close() method!

        source = ListSource([{"value": 1}])
        transform = OldStyleTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
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
            source=source,
            transforms=[transform_1, transform_2],
            sinks={"default": sink},
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
