# tests/core/test_dag.py
"""Tests for DAG validation and operations."""

import pytest


class TestDAGBuilder:
    """Building execution graphs from configuration."""

    def test_empty_dag(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")

        assert graph.node_count == 1
        assert graph.has_node("source")

    def test_add_edge(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="validate")
        graph.add_edge("source", "transform", label="continue")

        assert graph.edge_count == 1

    def test_linear_pipeline(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="enrich")
        graph.add_node("t2", node_type="transform", plugin_name="classify")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        assert graph.node_count == 4
        assert graph.edge_count == 3


class TestDAGValidation:
    """Validation of execution graphs."""

    def test_is_valid_for_acyclic(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="source", plugin_name="csv")
        graph.add_node("b", node_type="transform", plugin_name="x")
        graph.add_node("c", node_type="sink", plugin_name="csv")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "c", label="continue")

        assert graph.is_acyclic() is True

    def test_is_invalid_for_cycle(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="transform", plugin_name="x")
        graph.add_node("b", node_type="transform", plugin_name="y")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="continue")  # Creates cycle!

        assert graph.is_acyclic() is False

    def test_validate_raises_on_cycle(self) -> None:
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("a", node_type="transform", plugin_name="x")
        graph.add_node("b", node_type="transform", plugin_name="y")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="continue")

        with pytest.raises(GraphValidationError, match="cycle"):
            graph.validate()

    def test_topological_order(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="a")
        graph.add_node("t2", node_type="transform", plugin_name="b")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        order = graph.topological_order()

        # Source must come first, sink must come last
        assert order[0] == "source"
        assert order[-1] == "sink"
        # t1 must come before t2
        assert order.index("t1") < order.index("t2")


class TestSourceSinkValidation:
    """Validation of source and sink constraints."""

    def test_validate_requires_exactly_one_source(self) -> None:
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("t1", node_type="transform", plugin_name="x")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("t1", "sink", label="continue")

        with pytest.raises(GraphValidationError, match="exactly one source"):
            graph.validate()

    def test_validate_requires_at_least_one_sink(self) -> None:
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="x")
        graph.add_edge("source", "t1", label="continue")

        with pytest.raises(GraphValidationError, match="at least one sink"):
            graph.validate()

    def test_validate_multiple_sinks_allowed(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="classifier")
        graph.add_node("sink1", node_type="sink", plugin_name="csv")
        graph.add_node("sink2", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink1", label="normal")
        graph.add_edge("gate", "sink2", label="flagged")

        # Should not raise
        graph.validate()

    def test_get_source_node(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("my_source", node_type="source", plugin_name="csv")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("my_source", "sink", label="continue")

        assert graph.get_source() == "my_source"

    def test_get_sink_nodes(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("sink1", node_type="sink", plugin_name="csv")
        graph.add_node("sink2", node_type="sink", plugin_name="json")
        graph.add_edge("source", "sink1", label="continue")
        graph.add_edge("source", "sink2", label="continue")

        sinks = graph.get_sinks()
        assert set(sinks) == {"sink1", "sink2"}


class TestExecutionGraphAccessors:
    """Access node info and edges from graph."""

    def test_get_node_info(self) -> None:
        """Get NodeInfo for a node."""
        from elspeth.core.dag import ExecutionGraph, NodeInfo

        graph = ExecutionGraph()
        graph.add_node(
            "node_1",
            node_type="transform",
            plugin_name="my_plugin",
            config={"key": "value"},
        )

        info = graph.get_node_info("node_1")

        assert isinstance(info, NodeInfo)
        assert info.node_id == "node_1"
        assert info.node_type == "transform"
        assert info.plugin_name == "my_plugin"
        assert info.config == {"key": "value"}

    def test_get_node_info_missing(self) -> None:
        """Get NodeInfo for missing node raises."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        with pytest.raises(KeyError):
            graph.get_node_info("nonexistent")

    def test_get_edges(self) -> None:
        """Get all edges with data."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="source", plugin_name="src")
        graph.add_node("b", node_type="transform", plugin_name="tf")
        graph.add_node("c", node_type="sink", plugin_name="sink")
        graph.add_edge("a", "b", label="continue", mode="move")
        graph.add_edge("b", "c", label="output", mode="copy")

        edges = list(graph.get_edges())

        assert len(edges) == 2
        # Each edge is (from_id, to_id, data_dict)
        assert ("a", "b", {"label": "continue", "mode": "move"}) in edges
        assert ("b", "c", {"label": "output", "mode": "copy"}) in edges

    def test_get_edges_empty_graph(self) -> None:
        """Empty graph returns empty list."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        edges = list(graph.get_edges())

        assert edges == []


class TestExecutionGraphFromConfig:
    """Build ExecutionGraph from ElspethSettings."""

    def test_from_config_minimal(self) -> None:
        """Build graph from minimal config (source -> sink only)."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have: source -> output_sink
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.get_source() is not None
        assert len(graph.get_sinks()) == 1

    def test_from_config_is_valid(self) -> None:
        """Graph from valid config passes validation."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should not raise
        graph.validate()
        assert graph.is_acyclic()

    def test_from_config_with_transforms(self) -> None:
        """Build graph with transform chain."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
                RowPluginSettings(plugin="transform_b"),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have: source -> transform_a -> transform_b -> output_sink
        assert graph.node_count == 4
        assert graph.edge_count == 3

        # Topological order should be correct
        order = graph.topological_order()
        assert len(order) == 4
        # Source should be first (has "source" in node_id)
        assert "source" in order[0]
        # Sink should be last (has "sink" in node_id)
        assert "sink" in order[-1]
        # Verify transform ordering (transform_a before transform_b)
        transform_a_idx = next(i for i, n in enumerate(order) if "transform_a" in n)
        transform_b_idx = next(i for i, n in enumerate(order) if "transform_b" in n)
        assert transform_a_idx < transform_b_idx

    def test_from_config_with_gate_routes(self) -> None:
        """Build graph with gate routing to multiple sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="safety_gate",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have:
        #   source -> safety_gate -> results (via "continue"/"clean")
        #                         -> flagged (via "suspicious")
        assert graph.node_count == 4  # source, gate, results, flagged
        # Edges: source->gate, gate->results (continue), gate->flagged (route)
        assert graph.edge_count == 3

    def test_from_config_validates_route_targets(self) -> None:
        """Gate routes must reference existing sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(
                    plugin="gate",
                    type="gate",
                    routes={"bad": "nonexistent_sink"},
                ),
            ],
            output_sink="output",
        )

        with pytest.raises(GraphValidationError) as exc_info:
            ExecutionGraph.from_config(config)

        assert "nonexistent_sink" in str(exc_info.value)

    def test_get_sink_id_map(self) -> None:
        """Get explicit sink_name -> node_id mapping."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)
        sink_map = graph.get_sink_id_map()

        # Explicit mapping - no substring matching
        assert "results" in sink_map
        assert "flagged" in sink_map
        assert sink_map["results"] != sink_map["flagged"]

    def test_get_transform_id_map(self) -> None:
        """Get explicit sequence -> node_id mapping for transforms."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
                RowPluginSettings(plugin="transform_b"),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)
        transform_map = graph.get_transform_id_map()

        # Explicit mapping by sequence position
        assert 0 in transform_map  # transform_a
        assert 1 in transform_map  # transform_b
        assert transform_map[0] != transform_map[1]

    def test_get_output_sink(self) -> None:
        """Get the output sink name."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        assert graph.get_output_sink() == "results"


class TestExecutionGraphRouteMapping:
    """Test route label <-> sink name mapping for edge lookup."""

    def test_get_route_label_for_sink(self) -> None:
        """Get route label that leads to a sink from a gate."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="classifier",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Get the gate's node_id
        transform_map = graph.get_transform_id_map()
        gate_node_id = transform_map[0]

        # Given gate node and sink name, get the route label
        route_label = graph.get_route_label(gate_node_id, "flagged")

        assert route_label == "suspicious"

    def test_get_route_label_for_continue(self) -> None:
        """Continue routes return 'continue' as label."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"results": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(
                    plugin="gate",
                    type="gate",
                    routes={"pass": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)
        transform_map = graph.get_transform_id_map()
        gate_node_id = transform_map[0]

        # The edge to output sink uses "continue" label
        route_label = graph.get_route_label(gate_node_id, "results")
        assert route_label == "continue"
