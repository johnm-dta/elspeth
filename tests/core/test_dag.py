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

    def test_validate_rejects_duplicate_outgoing_edge_labels(self) -> None:
        """Duplicate outgoing edge labels from same node must be rejected.

        The orchestrator's edge_map keys by (from_node, label), so duplicate
        labels from the same node would cause silent overwrites during
        registration - routing events would be recorded against the wrong
        edge, corrupting the audit trail.
        """
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="config_gate")
        graph.add_node("sink_a", node_type="sink", plugin_name="csv")
        graph.add_node("sink_b", node_type="sink", plugin_name="csv")

        # Gate has one "continue" edge to sink_a
        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink_a", label="continue")
        # Add ANOTHER "continue" edge to a different sink - this is the collision
        graph.add_edge("gate", "sink_b", label="continue")

        with pytest.raises(GraphValidationError, match="duplicate outgoing edge label"):
            graph.validate()

    def test_validate_allows_same_label_from_different_nodes(self) -> None:
        """Same label from different nodes is allowed (no collision).

        The uniqueness constraint is per-node, not global. Multiple nodes
        can each have a 'continue' edge because edge_map keys by (from_node, label).
        """
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="a")
        graph.add_node("t2", node_type="transform", plugin_name="b")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        # Each node has ONE "continue" edge - no collisions
        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        # Should not raise - labels are unique per source node
        graph.validate()


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
        from elspeth.contracts import EdgeInfo, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="source", plugin_name="src")
        graph.add_node("b", node_type="transform", plugin_name="tf")
        graph.add_node("c", node_type="sink", plugin_name="sink")
        graph.add_edge("a", "b", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("b", "c", label="output", mode=RoutingMode.COPY)

        edges = list(graph.get_edges())

        assert len(edges) == 2
        # Each edge is EdgeInfo (not tuple)
        assert (
            EdgeInfo(
                from_node="a", to_node="b", label="continue", mode=RoutingMode.MOVE
            )
            in edges
        )
        assert (
            EdgeInfo(from_node="b", to_node="c", label="output", mode=RoutingMode.COPY)
            in edges
        )

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
        """Build graph with config-driven gate routing to multiple sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            gates=[
                GateSettings(
                    name="safety_gate",
                    condition="row['suspicious'] == True",
                    routes={"true": "flagged", "false": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have:
        #   source -> safety_gate -> results (via "continue")
        #                         -> flagged (via "suspicious")
        assert graph.node_count == 4  # source, config_gate, results, flagged
        # Edges: source->gate, gate->results (continue), gate->flagged (route)
        assert graph.edge_count == 3

    def test_from_config_validates_route_targets(self) -> None:
        """Config gate routes must reference existing sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            gates=[
                GateSettings(
                    name="bad_gate",
                    condition="True",
                    routes={"true": "nonexistent_sink", "false": "continue"},
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
        """Get route label that leads to a sink from a config gate."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            gates=[
                GateSettings(
                    name="classifier",
                    condition="row['suspicious'] == True",
                    routes={"true": "flagged", "false": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Get the config gate's node_id
        gate_node_id = graph.get_config_gate_id_map()["classifier"]

        # Given gate node and sink name, get the route label
        route_label = graph.get_route_label(gate_node_id, "flagged")

        assert route_label == "true"

    def test_get_route_label_for_continue(self) -> None:
        """Continue routes return 'continue' as label."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"results": SinkSettings(plugin="csv")},
            gates=[
                GateSettings(
                    name="gate",
                    condition="True",
                    routes={"true": "continue", "false": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)
        gate_node_id = graph.get_config_gate_id_map()["gate"]

        # The edge to output sink uses "continue" label (both routes resolve to continue)
        route_label = graph.get_route_label(gate_node_id, "results")
        assert route_label == "continue"


class TestMultiEdgeSupport:
    """Tests for MultiDiGraph multi-edge support."""

    def test_multiple_edges_same_node_pair(self) -> None:
        """MultiDiGraph allows multiple labeled edges between same nodes."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("gate", node_type="gate", plugin_name="fork_gate")
        graph.add_node("sink", node_type="sink", plugin_name="output")

        # Add two edges with different labels to SAME destination
        graph.add_edge("gate", "sink", label="path_a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "sink", label="path_b", mode=RoutingMode.COPY)

        # Both edges should exist (DiGraph would show 1, MultiDiGraph shows 2)
        assert graph.edge_count == 2

        edges = graph.get_edges()
        labels = {e.label for e in edges}
        assert labels == {"path_a", "path_b"}

    def test_multi_edge_graph_is_acyclic(self) -> None:
        """Verify is_acyclic() works correctly with MultiDiGraph parallel edges."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="classifier")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        # Multiple parallel edges to same sink - still acyclic
        graph.add_edge("gate", "sink", label="high", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink", label="medium", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink", label="low", mode=RoutingMode.MOVE)

        # Graph with parallel edges should still be detected as acyclic
        assert graph.is_acyclic() is True
        # Full validation should also pass
        graph.validate()


class TestEdgeInfoIntegration:
    """Tests for typed edge returns."""

    def test_get_edges_returns_edge_info(self) -> None:
        """get_edges() returns list of EdgeInfo, not tuples."""
        from elspeth.contracts import EdgeInfo, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source-1", node_type="source", plugin_name="csv")
        graph.add_node("sink-1", node_type="sink", plugin_name="csv")
        graph.add_edge("source-1", "sink-1", label="continue", mode=RoutingMode.MOVE)

        edges = graph.get_edges()

        assert len(edges) == 1
        assert isinstance(edges[0], EdgeInfo)
        assert edges[0].from_node == "source-1"
        assert edges[0].to_node == "sink-1"
        assert edges[0].label == "continue"
        assert edges[0].mode == RoutingMode.MOVE

    def test_add_edge_accepts_routing_mode_enum(self) -> None:
        """add_edge() accepts RoutingMode enum, not string."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("n1", node_type="transform", plugin_name="test")
        graph.add_node("n2", node_type="sink", plugin_name="test")

        # Should accept enum directly
        graph.add_edge("n1", "n2", label="route", mode=RoutingMode.COPY)

        edges = graph.get_edges()
        assert edges[0].mode == RoutingMode.COPY


class TestMultiEdgeScenarios:
    """Tests for scenarios requiring multiple edges between same nodes."""

    def test_fork_gate_config_parses_into_valid_graph(self) -> None:
        """Fork gate configuration parses into valid graph structure.

        Note: This tests config parsing, not the multi-edge bug. Fork routes
        with target="fork" don't create edges to sinks - they create child tokens.
        The multi-edge bug is tested by test_gate_multiple_routes_same_sink.
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always forks
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Validate graph is still valid (DAG, has source and sink)
        graph.validate()

        # The gate should have edges - at minimum the continue edge to output sink
        edges = graph.get_edges()
        gate_edges = [e for e in edges if "config_gate" in e.from_node]

        # Should have at least the continue edge to output sink
        assert len(gate_edges) >= 1

    def test_gate_multiple_routes_same_sink(self) -> None:
        """CRITICAL: Gate with multiple route labels to same sink preserves all labels.

        This is the core bug scenario: {"high": "alerts", "medium": "alerts", "low": "alerts"}
        With DiGraph, only "low" survives. With MultiDiGraph, all three edges exist.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="classifier")
        graph.add_node("alerts", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        # Multiple severity levels all route to same alerts sink
        graph.add_edge("gate", "alerts", label="high", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "alerts", label="medium", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "alerts", label="low", mode=RoutingMode.MOVE)

        # All three edges should exist
        edges = graph.get_edges()
        alert_edges = [e for e in edges if e.to_node == "alerts"]
        assert len(alert_edges) == 3

        labels = {e.label for e in alert_edges}
        assert labels == {"high", "medium", "low"}


class TestCoalesceNodes:
    """Test coalesce node creation in DAG."""

    def test_from_config_creates_coalesce_node(self) -> None:
        """Coalesce config should create a coalesce node in the graph."""
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={
                "output": SinkSettings(plugin="csv", options={"path": "out.csv"}),
            },
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

        graph = ExecutionGraph.from_config(settings)

        # Use proper accessor, not string matching
        coalesce_map = graph.get_coalesce_id_map()
        assert "merge_results" in coalesce_map

        # Verify node type
        node_id = coalesce_map["merge_results"]
        node_info = graph.get_node_info(node_id)
        assert node_info.node_type == "coalesce"
        assert node_info.plugin_name == "coalesce:merge_results"

    def test_from_config_coalesce_edges_from_fork_branches(self) -> None:
        """Coalesce node should have edges from fork gate (via branches)."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import (
            CoalesceSettings,
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
            },
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

        graph = ExecutionGraph.from_config(settings)

        # Get node IDs
        gate_id = graph.get_config_gate_id_map()["forker"]
        coalesce_id = graph.get_coalesce_id_map()["merge_results"]

        # Verify edges from fork gate to coalesce node
        edges = graph.get_edges()
        gate_to_coalesce_edges = [
            e for e in edges if e.from_node == gate_id and e.to_node == coalesce_id
        ]

        # Should have two edges (path_a and path_b) with COPY mode
        assert len(gate_to_coalesce_edges) == 2
        labels = {e.label for e in gate_to_coalesce_edges}
        assert labels == {"path_a", "path_b"}
        assert all(e.mode == RoutingMode.COPY for e in gate_to_coalesce_edges)

    def test_partial_branch_coverage_branches_not_in_coalesce_route_to_sink(
        self,
    ) -> None:
        """Fork branches not in any coalesce should still route to output sink."""
        from elspeth.core.config import (
            CoalesceSettings,
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
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b", "path_c"],  # 3 branches
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],  # Only 2 branches in coalesce
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        # Get node IDs
        gate_id = graph.get_config_gate_id_map()["forker"]
        coalesce_id = graph.get_coalesce_id_map()["merge_results"]
        output_sink_id = graph.get_sink_id_map()["output"]

        # Verify path_c goes to output sink, not coalesce
        edges = graph.get_edges()
        path_c_edges = [
            e for e in edges if e.from_node == gate_id and e.label == "path_c"
        ]

        assert len(path_c_edges) == 1
        assert path_c_edges[0].to_node == output_sink_id

        # Verify path_a and path_b go to coalesce
        coalesce_edges = [
            e for e in edges if e.from_node == gate_id and e.to_node == coalesce_id
        ]
        coalesce_labels = {e.label for e in coalesce_edges}
        assert coalesce_labels == {"path_a", "path_b"}

    def test_get_coalesce_id_map_returns_mapping(self) -> None:
        """get_coalesce_id_map should return coalesce_name -> node_id."""
        from elspeth.core.config import (
            CoalesceSettings,
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
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b", "path_c", "path_d"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_ab",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
                CoalesceSettings(
                    name="merge_cd",
                    branches=["path_c", "path_d"],
                    policy="require_all",
                    merge="nested",
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)
        coalesce_map = graph.get_coalesce_id_map()

        # Should have both coalesce nodes
        assert "merge_ab" in coalesce_map
        assert "merge_cd" in coalesce_map

        # Node IDs should be unique
        assert coalesce_map["merge_ab"] != coalesce_map["merge_cd"]

        # Verify both nodes exist in the graph
        assert graph.has_node(coalesce_map["merge_ab"])
        assert graph.has_node(coalesce_map["merge_cd"])

    def test_get_branch_to_coalesce_map_returns_mapping(self) -> None:
        """get_branch_to_coalesce_map should return branch_name -> coalesce_name."""
        from elspeth.core.config import (
            CoalesceSettings,
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
            },
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

        graph = ExecutionGraph.from_config(settings)
        branch_map = graph.get_branch_to_coalesce_map()

        # Should map branches to coalesce name
        assert branch_map["path_a"] == "merge_results"
        assert branch_map["path_b"] == "merge_results"

    def test_coalesce_node_has_edge_to_output_sink(self) -> None:
        """Coalesce node should have an edge to the output sink."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import (
            CoalesceSettings,
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
            },
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

        graph = ExecutionGraph.from_config(settings)

        coalesce_id = graph.get_coalesce_id_map()["merge_results"]
        output_sink_id = graph.get_sink_id_map()["output"]

        # Verify edge from coalesce to output sink
        edges = graph.get_edges()
        coalesce_to_sink_edges = [
            e
            for e in edges
            if e.from_node == coalesce_id and e.to_node == output_sink_id
        ]

        assert len(coalesce_to_sink_edges) == 1
        assert coalesce_to_sink_edges[0].label == "continue"
        assert coalesce_to_sink_edges[0].mode == RoutingMode.MOVE

    def test_coalesce_node_stores_config(self) -> None:
        """Coalesce node should store configuration for audit trail."""
        from elspeth.core.config import (
            CoalesceSettings,
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
            },
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
                    policy="quorum",
                    quorum_count=1,
                    merge="nested",
                    timeout_seconds=30.0,
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        coalesce_id = graph.get_coalesce_id_map()["merge_results"]
        node_info = graph.get_node_info(coalesce_id)

        # Verify config is stored
        assert node_info.config["branches"] == ["path_a", "path_b"]
        assert node_info.config["policy"] == "quorum"
        assert node_info.config["merge"] == "nested"
        assert node_info.config["timeout_seconds"] == 30.0
        assert node_info.config["quorum_count"] == 1
