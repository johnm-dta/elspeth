# tests/engine/test_config_gates.py
"""Tests for config-driven gates integration.

Config gates are defined in YAML and evaluated by the engine using ExpressionParser.
They are processed AFTER plugin transforms but BEFORE sinks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode, SourceRow
from elspeth.core.config import GateSettings

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


# ============================================================================
# Test Fixture Base Classes
# ============================================================================


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class providing SourceProtocol required attributes."""

    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass


class _TestSinkBase:
    """Base class providing SinkProtocol required attributes."""

    input_schema = _TestSchema
    idempotent = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"

    def flush(self) -> None:
        pass

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass


def _build_test_graph_with_config_gates(
    config: PipelineConfig,
) -> ExecutionGraph:
    """Build a test graph including config gates.

    Creates a linear graph matching the PipelineConfig structure:
    source -> transforms... -> config_gates... -> sinks
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
        graph.add_node(
            node_id,
            node_type="transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks first (needed for config gate edges)
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Add config gates
    config_gate_ids: dict[str, str] = {}
    route_resolution_map: dict[tuple[str, str], str] = {}

    for gate_config in config.gates:
        node_id = f"config_gate_{gate_config.name}"
        config_gate_ids[gate_config.name] = node_id
        graph.add_node(
            node_id,
            node_type="gate",
            plugin_name=f"config_gate:{gate_config.name}",
            config={
                "condition": gate_config.condition,
                "routes": dict(gate_config.routes),
            },
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)

        # Add route edges and resolution map
        for route_label, target in gate_config.routes.items():
            route_resolution_map[(node_id, route_label)] = target
            if target not in ("continue", "fork") and target in sink_ids:
                graph.add_edge(
                    node_id, sink_ids[target], label=route_label, mode=RoutingMode.MOVE
                )

        prev = node_id

    # Edge to output sink
    output_sink = "default" if "default" in sink_ids else next(iter(sink_ids))
    graph.add_edge(prev, sink_ids[output_sink], label="continue", mode=RoutingMode.MOVE)

    # Populate internal maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._config_gate_id_map = config_gate_ids
    graph._route_resolution_map = route_resolution_map
    graph._output_sink = output_sink

    return graph


class TestConfigGateIntegration:
    """Integration tests for config-driven gates."""

    def test_config_gate_continue(self) -> None:
        """Config gate with 'continue' destination passes rows through."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 20}])
        sink = CollectSink()

        # Config gate that always continues
        gate = GateSettings(
            name="always_pass",
            condition="True",  # Always true
            routes={"true": "continue"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 2
        assert result.rows_succeeded == 2
        assert len(sink.results) == 2

    def test_config_gate_routes_to_sink(self) -> None:
        """Config gate routes rows to different sinks based on condition."""
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

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Rows: 10 (low), 100 (high), 30 (low)
        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        default_sink = CollectSink()
        high_sink = CollectSink()

        # Config gate that routes high values to a different sink
        gate = GateSettings(
            name="threshold_gate",
            condition="row['value'] > 50",
            routes={
                "true": "high",  # High values go to high sink
                "false": "continue",  # Low values continue to default
            },
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink, "high": high_sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 3
        assert result.rows_succeeded == 2  # 10, 30 -> default
        assert result.rows_routed == 1  # 100 -> high

        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1
        assert high_sink.results[0]["value"] == 100

    def test_config_gate_with_string_result(self) -> None:
        """Config gate condition can return a string route label.

        This test uses ExecutionGraph.from_config() for proper edge building.
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            category: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"category": "A"}, {"category": "B"}, {"category": "A"}])
        a_sink = CollectSink()
        b_sink = CollectSink()

        # Build settings to use ExecutionGraph.from_config()
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "a_sink": SinkSettings(plugin="csv"),
                "b_sink": SinkSettings(plugin="csv"),
            },
            output_sink="a_sink",
            gates=[
                GateSettingsConfig(
                    name="category_router",
                    condition="row['category']",  # Returns 'A' or 'B'
                    routes={
                        "A": "a_sink",
                        "B": "b_sink",
                    },
                ),
            ],
        )

        # Build graph from settings (proper edge construction)
        graph = ExecutionGraph.from_config(settings)

        # Build PipelineConfig with actual plugin instances
        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"a_sink": a_sink, "b_sink": b_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # All rows are routed (none go to "continue" default)
        assert result.rows_routed == 3
        assert len(a_sink.results) == 2
        assert len(b_sink.results) == 1

    def test_config_gate_integer_route_label(self) -> None:
        """Config gate condition can return an integer that maps to route labels.

        When an expression returns an integer (e.g., row['priority'] returns 1, 2, 3),
        the executor converts it to a string for route lookup. So routes must use
        string keys like {"1": "priority_1", "2": "priority_2"}.

        This test uses ExecutionGraph.from_config() for proper edge building.
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            priority: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # 3 rows with priorities 1, 2, 1 -> expect 2 go to priority_1, 1 goes to priority_2
        source = ListSource([{"priority": 1}, {"priority": 2}, {"priority": 1}])
        priority_1_sink = CollectSink()
        priority_2_sink = CollectSink()

        # Build settings to use ExecutionGraph.from_config()
        # NOTE: Route keys must be strings because the executor converts
        # non-bool/non-string results to strings via str()
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "priority_1": SinkSettings(plugin="csv"),
                "priority_2": SinkSettings(plugin="csv"),
            },
            output_sink="priority_1",
            gates=[
                GateSettingsConfig(
                    name="priority_router",
                    condition="row['priority']",  # Returns 1 or 2 (integer)
                    routes={
                        "1": "priority_1",  # String key for integer result
                        "2": "priority_2",  # String key for integer result
                    },
                ),
            ],
        )

        # Build graph from settings (proper edge construction)
        graph = ExecutionGraph.from_config(settings)

        # Build PipelineConfig with actual plugin instances
        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"priority_1": priority_1_sink, "priority_2": priority_2_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # All rows are routed (none use "continue")
        assert result.rows_routed == 3
        # 2 rows with priority 1, 1 row with priority 2
        assert len(priority_1_sink.results) == 2
        assert len(priority_2_sink.results) == 1
        # Verify the right rows went to the right sinks
        assert all(row["priority"] == 1 for row in priority_1_sink.results)
        assert all(row["priority"] == 2 for row in priority_2_sink.results)

    def test_config_gate_node_registered_in_landscape(self) -> None:
        """Config gates are registered as nodes in Landscape."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        gate = GateSettings(
            name="my_gate",
            condition="True",
            routes={"true": "continue"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        # Query Landscape for registered nodes
        with db._engine.connect() as conn:
            from sqlalchemy import text

            nodes = conn.execute(
                text("SELECT plugin_name, node_type FROM nodes WHERE run_id = :run_id"),
                {"run_id": result.run_id},
            ).fetchall()

        # Should have source, config gate, and sink
        node_names = [n[0] for n in nodes]
        node_types = [n[1] for n in nodes]

        assert "config_gate:my_gate" in node_names
        assert "gate" in node_types


class TestConfigGateFromSettings:
    """Tests for config gates built via ExecutionGraph.from_config()."""

    def test_from_config_builds_config_gates(self) -> None:
        """ExecutionGraph.from_config() includes config gates."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "output": SinkSettings(plugin="csv"),
                "review": SinkSettings(plugin="csv"),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="quality_check",
                    condition="row['confidence'] >= 0.85",
                    routes={"true": "continue", "false": "review"},
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        # Should have: source, config_gate, output_sink, review_sink
        assert graph.node_count == 4

        # Config gate should be in the graph
        config_gate_map = graph.get_config_gate_id_map()
        assert "quality_check" in config_gate_map

        # Route resolution should include the gate
        route_map = graph.get_route_resolution_map()
        gate_id = config_gate_map["quality_check"]
        assert (gate_id, "true") in route_map
        assert route_map[(gate_id, "true")] == "continue"
        assert (gate_id, "false") in route_map
        assert route_map[(gate_id, "false")] == "review"

    def test_from_config_validates_gate_sink_targets(self) -> None:
        """ExecutionGraph.from_config() validates gate route targets."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
            gates=[
                GateSettings(
                    name="bad_gate",
                    condition="True",
                    routes={"true": "nonexistent_sink"},
                ),
            ],
        )

        with pytest.raises(GraphValidationError) as exc_info:
            ExecutionGraph.from_config(settings)

        assert "nonexistent_sink" in str(exc_info.value)

    def test_config_gates_ordered_after_transforms(self) -> None:
        """Config gates come after plugin transforms in topological order."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
            ],
            gates=[
                GateSettings(
                    name="final_gate",
                    condition="True",
                    routes={"true": "continue"},
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)
        order = graph.topological_order()

        # Find indices
        transform_idx = next(i for i, n in enumerate(order) if "transform_a" in n)
        gate_idx = next(i for i, n in enumerate(order) if "config_gate" in n)
        sink_idx = next(i for i, n in enumerate(order) if "sink" in n)

        # Transform before gate, gate before sink
        assert transform_idx < gate_idx
        assert gate_idx < sink_idx


class TestMultipleConfigGates:
    """Tests for multiple config gates in sequence."""

    def test_multiple_config_gates_processed_in_order(self) -> None:
        """Multiple config gates are processed in definition order."""
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

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Row: value=25 should pass gate1 (>10) but fail gate2 (>50)
        source = ListSource([{"value": 25}])
        default_sink = CollectSink()
        low_sink = CollectSink()
        mid_sink = CollectSink()

        gate1 = GateSettings(
            name="gate1",
            condition="row['value'] > 10",
            routes={"true": "continue", "false": "low"},
        )
        gate2 = GateSettings(
            name="gate2",
            condition="row['value'] > 50",
            routes={"true": "continue", "false": "mid"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink, "low": low_sink, "mid": mid_sink},
            gates=[gate1, gate2],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        # Row passes gate1, routes to mid via gate2
        assert result.rows_routed == 1
        assert len(mid_sink.results) == 1
        assert len(default_sink.results) == 0
        assert len(low_sink.results) == 0
