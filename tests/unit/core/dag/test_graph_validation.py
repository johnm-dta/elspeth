"""Tests for ExecutionGraph validation error paths and NodeInfo guards.

Exercises rejection paths in graph.py and models.py that are only
implicitly (or never) tested through the builder. Each test constructs
the minimal graph state needed to trigger the specific error path.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.enums import NodeType, RoutingMode
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.core.dag.models import GraphValidationError, NodeInfo

# ---------------------------------------------------------------------------
# Gap 2: _validate_route_resolution_map_complete — all labels missing
# ---------------------------------------------------------------------------


class TestRouteResolutionMapCompleteAllMissing:
    """validate() must reject a gate with MOVE edges to sinks but no route labels.

    The existing test suite covers partial incompleteness (some labels present,
    some missing). This tests the "completely unwired gate" case where NO
    route labels are registered at all.
    """

    def test_gate_with_move_edge_but_no_route_label_raises(self) -> None:
        """Gate has MOVE edge to a sink registered in sink_id_map, but zero route labels."""
        graph = ExecutionGraph()

        # Minimal valid topology: source -> gate -> sink
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate_1", node_type=NodeType.GATE, plugin_name="expression")
        graph.add_node("sink_1", node_type=NodeType.SINK, plugin_name="json")

        graph.add_edge("src", "gate_1", label="continue")
        graph.add_edge("gate_1", "sink_1", label="route_true", mode=RoutingMode.MOVE)

        # Register the sink so the route-label check doesn't early-return
        graph.set_sink_id_map({SinkName("output"): NodeID("sink_1")})

        # Deliberately do NOT add any route label entries
        with pytest.raises(GraphValidationError, match="no registered route label"):
            graph.validate()

    def test_gate_with_multiple_unwired_move_edges_raises(self) -> None:
        """Gate with two MOVE edges to different sinks, neither wired — first triggers error."""
        graph = ExecutionGraph()

        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate_1", node_type=NodeType.GATE, plugin_name="expression")
        graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="json")
        graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="json")

        graph.add_edge("src", "gate_1", label="continue")
        graph.add_edge("gate_1", "sink_a", label="route_true", mode=RoutingMode.MOVE)
        graph.add_edge("gate_1", "sink_b", label="route_false", mode=RoutingMode.MOVE)

        graph.set_sink_id_map(
            {
                SinkName("output_a"): NodeID("sink_a"),
                SinkName("output_b"): NodeID("sink_b"),
            }
        )

        with pytest.raises(GraphValidationError, match="no registered route label"):
            graph.validate()


# ---------------------------------------------------------------------------
# Gap 3: NodeInfo.__post_init__ node_id length validation
# ---------------------------------------------------------------------------


class TestNodeInfoNodeIdLengthValidation:
    """NodeInfo must reject node_id exceeding the column length limit.

    The Pydantic layer validates this too, but __post_init__ is the
    defense-in-depth guard that fires regardless of construction path.
    """

    def test_node_id_at_limit_accepted(self) -> None:
        """64-character node_id is exactly at the limit — must not raise."""
        node_id = "x" * 64
        info = NodeInfo(
            node_id=NodeID(node_id),
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
        )
        assert info.node_id == NodeID(node_id)

    def test_node_id_over_limit_raises(self) -> None:
        """65-character node_id exceeds the column limit — must raise."""
        node_id = "x" * 65
        with pytest.raises(GraphValidationError, match="node_id exceeds"):
            NodeInfo(
                node_id=NodeID(node_id),
                node_type=NodeType.TRANSFORM,
                plugin_name="passthrough",
            )

    def test_node_id_way_over_limit_raises(self) -> None:
        """200-character node_id — error message includes actual length."""
        node_id = "a" * 200
        with pytest.raises(GraphValidationError, match="length=200"):
            NodeInfo(
                node_id=NodeID(node_id),
                node_type=NodeType.TRANSFORM,
                plugin_name="passthrough",
            )


class TestNodeInfoIdentifierValidation:
    """NodeInfo must reject blank identifiers before they reach the graph/audit path."""

    def test_node_info_rejects_empty_node_id(self) -> None:
        with pytest.raises(GraphValidationError, match="node_id must not be empty"):
            NodeInfo(
                node_id=NodeID(""),
                node_type=NodeType.TRANSFORM,
                plugin_name="passthrough",
            )

    def test_node_info_rejects_empty_plugin_name(self) -> None:
        with pytest.raises(GraphValidationError, match="plugin_name must not be empty"):
            NodeInfo(
                node_id=NodeID("node_1"),
                node_type=NodeType.TRANSFORM,
                plugin_name="",
            )

    def test_node_info_rejects_whitespace_only_plugin_name(self) -> None:
        with pytest.raises(GraphValidationError, match="plugin_name must not be empty"):
            NodeInfo(
                node_id=NodeID("node_1"),
                node_type=NodeType.TRANSFORM,
                plugin_name="   ",
            )

    def test_execution_graph_add_node_rejects_blank_source_identifiers(self) -> None:
        graph = ExecutionGraph()

        with pytest.raises(GraphValidationError, match="node_id must not be empty"):
            graph.add_node("", node_type=NodeType.SOURCE, plugin_name="")


# ---------------------------------------------------------------------------
# Gap 3b: NodeInfo.declared_required_fields sink-only invariant
# ---------------------------------------------------------------------------


class TestNodeInfoDeclaredRequiredFieldsSinkOnly:
    """NodeInfo.declared_required_fields is meaningful only for SINK nodes.

    Offensive-programming invariant added during schema-contract reconciliation:
    stray declared_required_fields on a non-sink node would sit unused until a
    future validator widens its scope and produces mysterious errors. Catch the
    misuse at construction time instead.
    """

    def test_node_info_sink_allows_declared_required_fields(self) -> None:
        """SINK nodes are the legitimate consumer of declared_required_fields."""
        info = NodeInfo(
            node_id=NodeID("my_sink"),
            node_type=NodeType.SINK,
            plugin_name="csv",
            declared_required_fields=frozenset({"id", "name"}),
        )
        assert info.declared_required_fields == frozenset({"id", "name"})

    def test_node_info_rejects_declared_required_fields_on_non_sink(self) -> None:
        """Offensive-programming invariant: declared_required_fields is sink-specific.

        Catches the misuse at construction time rather than letting stray data
        sit on a non-sink node until a future validator widens its scope and
        produces mysterious errors.
        """
        for non_sink_type in [
            NodeType.SOURCE,
            NodeType.TRANSFORM,
            NodeType.GATE,
            NodeType.AGGREGATION,
            NodeType.COALESCE,
        ]:
            with pytest.raises(GraphValidationError, match=r"only meaningful for SINK"):
                NodeInfo(
                    node_id=NodeID("bad_node"),
                    node_type=non_sink_type,
                    plugin_name="something",
                    declared_required_fields=frozenset({"x"}),
                )


# ---------------------------------------------------------------------------
# Gap 4: topological_order() cycle detection
# ---------------------------------------------------------------------------


class TestTopologicalOrderCycleDetection:
    """topological_order() must wrap NetworkXUnfeasible into GraphValidationError.

    The builder's validate() also checks for cycles, but topological_order()
    has its own independent guard. This tests it directly.
    """

    def test_cycle_raises_graph_validation_error(self) -> None:
        """Two-node cycle must raise GraphValidationError, not NetworkXUnfeasible."""
        graph = ExecutionGraph()

        graph.add_node("a", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="passthrough")

        graph.add_edge("a", "b", label="forward")
        graph.add_edge("b", "a", label="backward")

        with pytest.raises(GraphValidationError, match="Cannot sort graph"):
            graph.topological_order()

    def test_self_loop_raises_graph_validation_error(self) -> None:
        """Self-loop is a trivial cycle — must still raise GraphValidationError."""
        graph = ExecutionGraph()

        graph.add_node("a", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        graph.add_edge("a", "a", label="loop")

        with pytest.raises(GraphValidationError, match="Cannot sort graph"):
            graph.topological_order()


# ---------------------------------------------------------------------------
# Gap 5: get_source() with zero and multiple sources
# ---------------------------------------------------------------------------


class TestGetSourceErrorPaths:
    """get_source() must raise GraphValidationError when source count != 1.

    The method documents this as a "construction bug" signal. Tests verify
    both the zero-source and multi-source cases.
    """

    def test_no_sources_raises(self) -> None:
        """Graph with only non-source nodes must raise."""
        graph = ExecutionGraph()
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="passthrough")

        with pytest.raises(GraphValidationError, match=r"Expected exactly 1 source.*found 0"):
            graph.get_source()

    def test_multiple_sources_raises(self) -> None:
        """Graph with two source nodes must raise."""
        graph = ExecutionGraph()
        graph.add_node("src_a", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("src_b", node_type=NodeType.SOURCE, plugin_name="csv")

        with pytest.raises(GraphValidationError, match=r"Expected exactly 1 source.*found 2"):
            graph.get_source()

    def test_empty_graph_raises(self) -> None:
        """Completely empty graph has zero sources."""
        graph = ExecutionGraph()

        with pytest.raises(GraphValidationError, match=r"Expected exactly 1 source.*found 0"):
            graph.get_source()
