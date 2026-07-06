"""Tests for ExecutionGraph validation error paths and NodeInfo guards.

Exercises rejection paths in graph.py and models.py that are only
implicitly (or never) tested through the builder. Each test constructs
the minimal graph state needed to trigger the specific error path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from elspeth.contracts import EdgeInfo
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


class TestTypedEdgeContracts:
    """ExecutionGraph edge query APIs must return EdgeInfo contracts."""

    def test_get_edges_and_incoming_edges_preserve_routing_mode_enum(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="expression")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json")

        graph.add_edge("src", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink", label="flagged", mode=RoutingMode.COPY)

        edges = graph.get_edges()
        assert all(isinstance(edge, EdgeInfo) for edge in edges)
        assert {edge.label: edge.mode for edge in edges} == {
            "continue": RoutingMode.MOVE,
            "flagged": RoutingMode.COPY,
        }
        assert all(isinstance(edge.mode, RoutingMode) for edge in edges)

        incoming = graph.get_incoming_edges("sink")
        assert incoming == [
            EdgeInfo(
                from_node=NodeID("gate"),
                to_node=NodeID("sink"),
                label="flagged",
                mode=RoutingMode.COPY,
            )
        ]


class TestMultiProducerFanInValidation:
    """Multi-producer fan-in policy distinguishes terminal sinks from processing nodes."""

    def test_multi_source_direct_sink_fan_in_is_valid(self) -> None:
        """Direct multi-source sink fan-in is terminal write policy, not QUEUE bypass."""
        graph = ExecutionGraph()
        graph.add_node("orders", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("refunds", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("audit_sink", node_type=NodeType.SINK, plugin_name="json")

        graph.add_edge("orders", "audit_sink", label="orders_out", mode=RoutingMode.MOVE)
        graph.add_edge("refunds", "audit_sink", label="refunds_out", mode=RoutingMode.MOVE)

        graph.validate()

    def test_multi_source_processing_node_fan_in_requires_queue(self) -> None:
        """Ordinary processing nodes still require explicit QUEUE fan-in."""
        graph = ExecutionGraph()
        graph.add_node("orders", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("refunds", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("normalize", node_type=NodeType.TRANSFORM, plugin_name="mapper")
        graph.add_node("audit_sink", node_type=NodeType.SINK, plugin_name="json")

        graph.add_edge("orders", "normalize", label="orders_out", mode=RoutingMode.MOVE)
        graph.add_edge("refunds", "normalize", label="refunds_out", mode=RoutingMode.MOVE)
        graph.add_edge("normalize", "audit_sink", label="normalized", mode=RoutingMode.MOVE)

        with pytest.raises(GraphValidationError, match="fan-in from multiple producers without a queue"):
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

    def test_node_id_length_contract_is_owned_by_contracts_not_landscape_schema(self) -> None:
        import elspeth.core.dag.models as dag_models
        from elspeth.contracts import types as contract_types
        from elspeth.core.landscape import schema as landscape_schema

        assert hasattr(contract_types, "NODE_ID_MAX_LENGTH")
        assert dag_models._NODE_ID_MAX_LENGTH == contract_types.NODE_ID_MAX_LENGTH
        assert landscape_schema.NODE_ID_COLUMN_LENGTH == contract_types.NODE_ID_MAX_LENGTH
        assert landscape_schema.nodes_table.c.node_id.type.length == contract_types.NODE_ID_MAX_LENGTH
        node_id_storage_columns = {("nodes", "node_id")}
        for table in landscape_schema.metadata.tables.values():
            for constraint in table.foreign_key_constraints:
                for element in constraint.elements:
                    if element.column.table.name == "nodes" and element.column.name == "node_id":
                        node_id_storage_columns.add((table.name, element.parent.name))
                        assert element.parent.type.length == contract_types.NODE_ID_MAX_LENGTH

        dag_models_source = Path(dag_models.__file__).read_text(encoding="utf-8")
        tree = ast.parse(dag_models_source)
        forbidden_landscape_schema_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden_landscape_schema_imports.extend(
                    alias.name for alias in node.names if alias.name == "elspeth.core.landscape.schema"
                )
            if isinstance(node, ast.ImportFrom):
                if node.module == "elspeth.core.landscape.schema":
                    forbidden_landscape_schema_imports.append(node.module)
                if node.module == "elspeth.core.landscape":
                    forbidden_landscape_schema_imports.extend(alias.name for alias in node.names if alias.name == "schema")
        assert forbidden_landscape_schema_imports == []

        schema_source = Path(landscape_schema.__file__).read_text(encoding="utf-8")
        schema_tree = ast.parse(schema_source)
        active_table: str | None = None
        hardcoded_node_width_columns: list[str] = []
        for node in ast.walk(schema_tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Name) or node.value.func.id != "Table":
                continue
            if not node.value.args or not isinstance(node.value.args[0], ast.Constant):
                continue
            table_name = node.value.args[0].value
            if not isinstance(table_name, str):
                continue
            active_table = table_name
            for arg in node.value.args[2:]:
                if not isinstance(arg, ast.Call) or not isinstance(arg.func, ast.Name) or arg.func.id != "Column":
                    continue
                if len(arg.args) < 2 or not isinstance(arg.args[0], ast.Constant):
                    continue
                column_name = arg.args[0].value
                if (active_table, column_name) not in node_id_storage_columns:
                    continue
                type_arg = arg.args[1]
                if (
                    isinstance(type_arg, ast.Call)
                    and isinstance(type_arg.func, ast.Name)
                    and type_arg.func.id == "String"
                    and type_arg.args
                    and isinstance(type_arg.args[0], ast.Constant)
                    and type_arg.args[0].value == 64
                ):
                    hardcoded_node_width_columns.append(f"{active_table}.{column_name}")
        assert hardcoded_node_width_columns == []

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
# Gap 5: get_sources() with zero and multiple sources
# ---------------------------------------------------------------------------


class TestGetSourceErrorPaths:
    """get_sources() exposes the current multi-source graph contract."""

    def test_no_sources_raises(self) -> None:
        """Graph validation rejects a graph with no source nodes."""
        graph = ExecutionGraph()
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json")
        graph.add_edge("t1", "sink", label="out", mode=RoutingMode.MOVE)

        assert graph.get_sources() == []
        with pytest.raises(GraphValidationError, match="Graph must have at least one source"):
            graph.validate()

    def test_multiple_sources_are_returned(self) -> None:
        """Graph with two source nodes is valid in the multi-source model."""
        graph = ExecutionGraph()
        graph.add_node("src_a", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("src_b", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json")
        graph.add_edge("src_a", "sink", label="src_a_out", mode=RoutingMode.MOVE)
        graph.add_edge("src_b", "sink", label="src_b_out", mode=RoutingMode.MOVE)

        assert graph.get_sources() == [NodeID("src_a"), NodeID("src_b")]
        graph.validate()

    def test_empty_graph_raises(self) -> None:
        """Completely empty graph has zero sources and fails validation."""
        graph = ExecutionGraph()

        assert graph.get_sources() == []
        with pytest.raises(GraphValidationError, match="Graph must have at least one source"):
            graph.validate()
