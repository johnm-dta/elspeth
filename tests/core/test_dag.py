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
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        assert graph.node_count == 1
        assert graph.has_node("source")

    def test_add_edge(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="validate")
        graph.add_edge("source", "transform", label="continue")

        assert graph.edge_count == 1

    def test_linear_pipeline(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="enrich")
        graph.add_node("t2", node_type=NodeType.TRANSFORM, plugin_name="classify")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        assert graph.node_count == 4
        assert graph.edge_count == 3


class TestDAGValidation:
    """Validation of execution graphs."""

    def test_is_valid_for_acyclic(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="x")
        graph.add_node("c", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "c", label="continue")

        assert graph.is_acyclic() is True

    def test_is_invalid_for_cycle(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type=NodeType.TRANSFORM, plugin_name="x")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="y")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="continue")  # Creates cycle!

        assert graph.is_acyclic() is False

    def test_validate_raises_on_cycle(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("a", node_type=NodeType.TRANSFORM, plugin_name="x")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="y")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="continue")

        with pytest.raises(GraphValidationError, match="cycle"):
            graph.validate()

    def test_topological_order(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="a")
        graph.add_node("t2", node_type=NodeType.TRANSFORM, plugin_name="b")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

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
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="config_gate")
        graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="csv")

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
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="a")
        graph.add_node("t2", node_type=NodeType.TRANSFORM, plugin_name="b")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Each node has ONE "continue" edge - no collisions
        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        # Should not raise - labels are unique per source node
        graph.validate()


class TestSourceSinkValidation:
    """Validation of source and sink constraints."""

    def test_validate_requires_exactly_one_source(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="x")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("t1", "sink", label="continue")

        with pytest.raises(GraphValidationError, match="exactly one source"):
            graph.validate()

    def test_validate_requires_at_least_one_sink(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("t1", node_type=NodeType.TRANSFORM, plugin_name="x")
        graph.add_edge("source", "t1", label="continue")

        with pytest.raises(GraphValidationError, match="at least one sink"):
            graph.validate()

    def test_validate_multiple_sinks_allowed(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="classifier")
        graph.add_node("sink1", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_node("sink2", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink1", label="normal")
        graph.add_edge("gate", "sink2", label="flagged")

        # Should not raise
        graph.validate()

    def test_get_source_node(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("my_source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("my_source", "sink", label="continue")

        assert graph.get_source() == "my_source"

    def test_get_sink_nodes(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("sink1", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_node("sink2", node_type=NodeType.SINK, plugin_name="json")
        graph.add_edge("source", "sink1", label="continue")
        graph.add_edge("source", "sink2", label="continue")

        sinks = graph.get_sinks()
        assert set(sinks) == {"sink1", "sink2"}


class TestExecutionGraphAccessors:
    """Access node info and edges from graph."""

    def test_get_node_info(self) -> None:
        """Get NodeInfo for a node."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, NodeInfo

        graph = ExecutionGraph()
        graph.add_node(
            "node_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="my_plugin",
            config={"key": "value"},
        )

        info = graph.get_node_info("node_1")

        assert isinstance(info, NodeInfo)
        assert info.node_id == "node_1"
        assert info.node_type == NodeType.TRANSFORM
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
        from elspeth.contracts import EdgeInfo, NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type=NodeType.SOURCE, plugin_name="src")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="tf")
        graph.add_node("c", node_type=NodeType.SINK, plugin_name="sink")
        graph.add_edge("a", "b", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("b", "c", label="output", mode=RoutingMode.COPY)

        edges = list(graph.get_edges())

        assert len(edges) == 2
        # Each edge is EdgeInfo (not tuple)
        assert EdgeInfo(from_node="a", to_node="b", label="continue", mode=RoutingMode.MOVE) in edges
        assert EdgeInfo(from_node="b", to_node="c", label="output", mode=RoutingMode.COPY) in edges

    def test_get_edges_empty_graph(self) -> None:
        """Empty graph returns empty list."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        edges = list(graph.get_edges())

        assert edges == []

    def test_get_incoming_edges_returns_edges_pointing_to_node(self):
        """get_incoming_edges() returns all edges with to_node matching the given node_id."""
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("A", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("B", node_type=NodeType.TRANSFORM, plugin_name="mapper")
        graph.add_node("C", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("A", "B", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("B", "C", label="continue", mode=RoutingMode.MOVE)

        incoming = graph.get_incoming_edges("B")

        assert len(incoming) == 1
        assert incoming[0].from_node == "A"
        assert incoming[0].to_node == "B"
        assert incoming[0].label == "continue"
        assert incoming[0].mode == RoutingMode.MOVE

    def test_get_incoming_edges_returns_empty_for_source_node(self):
        """get_incoming_edges() returns empty list for nodes with no predecessors."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("A", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("B", node_type=NodeType.SINK, plugin_name="csv")

        incoming = graph.get_incoming_edges("A")

        assert incoming == []

    def test_get_effective_producer_schema_walks_through_gates(self) -> None:
        """_get_effective_producer_schema() recursively finds schema through gate chain."""
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        class OutputSchema(PluginSchema):
            value: int

        graph = ExecutionGraph()

        # Build chain: source -> gate -> sink
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=OutputSchema)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="config_gate:check")  # No schema
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink", label="flagged", mode=RoutingMode.MOVE)

        # Gate's effective producer schema should be source's output schema
        effective_schema = graph._get_effective_producer_schema("gate")

        assert effective_schema == OutputSchema

    def test_get_effective_producer_schema_crashes_on_gate_without_inputs(self):
        """_get_effective_producer_schema() crashes if gate has no incoming edges."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="config_gate:orphan")

        # Gate with no inputs is a bug in our code - should crash
        # ValueError is raised by internal validation during edge compatibility checking
        with pytest.raises(ValueError) as exc_info:
            graph._get_effective_producer_schema("gate")

        assert "no incoming edges" in str(exc_info.value).lower()
        assert "bug in graph construction" in str(exc_info.value).lower()

    def test_get_effective_producer_schema_handles_chained_gates(self) -> None:
        """_get_effective_producer_schema() recursively walks through multiple gates."""
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            id: int
            name: str

        graph = ExecutionGraph()

        # Build chain: source -> gate1 -> gate2 -> sink
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("gate1", node_type=NodeType.GATE, plugin_name="config_gate:first")
        graph.add_node("gate2", node_type=NodeType.GATE, plugin_name="config_gate:second")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "gate1", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate1", "gate2", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate2", "sink", label="approved", mode=RoutingMode.MOVE)

        # gate2's effective schema should trace back to source
        effective_schema = graph._get_effective_producer_schema("gate2")

        assert effective_schema == SourceOutput

    def test_dag_validation_only_checks_structure(self) -> None:
        """DAG validation should only check cycles and connectivity, not schemas."""
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph

        class OutputSchema(PluginSchema):
            value: int

        class DifferentSchema(PluginSchema):
            different: str  # Incompatible!

        graph = ExecutionGraph()

        # Add incompatible schemas
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=OutputSchema)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=DifferentSchema)
        graph.add_edge("source", "sink", label="continue")

        # OLD behavior: Would raise GraphValidationError for schema mismatch
        # NEW behavior: Only checks structural validity (no cycles)
        graph.validate()  # Should NOT raise - no structural problems

    def test_get_effective_producer_schema_returns_direct_schema_for_transform(self) -> None:
        """_get_effective_producer_schema() returns output_schema directly for transform nodes."""
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph

        class TransformOutput(PluginSchema):
            result: str

        graph = ExecutionGraph()
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="field_mapper",
            output_schema=TransformOutput,
        )

        effective_schema = graph._get_effective_producer_schema("transform")

        assert effective_schema == TransformOutput

    def test_validate_edge_schemas_uses_effective_schema_for_gates(self) -> None:
        """validate_edge_compatibility() uses effective producer schema for gate edges."""
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            id: int
            name: str
            # Note: does NOT have 'score' field

        class SinkInput(PluginSchema):
            id: int
            score: float  # Required field not in source output

        graph = ExecutionGraph()

        # Pipeline: source -> gate -> sink
        # Gate has NO schemas (simulates config-driven gate from from_config())
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="config_gate:check")  # NO SCHEMA
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=SinkInput)

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink", label="flagged", mode=RoutingMode.MOVE)

        # Should detect schema incompatibility on gate -> sink edge
        # validate_edge_compatibility() raises ValueError for schema mismatches
        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        # Verify error mentions the missing field
        assert "score" in str(exc_info.value).lower()
        # Verify error includes node IDs (gate -> sink)
        assert "gate" in str(exc_info.value)
        assert "sink" in str(exc_info.value)

    def test_validate_edge_schemas_validates_all_fork_destinations(self) -> None:
        """Fork gates validate all destination edges against effective schema."""
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            id: int
            name: str

        class SinkA(PluginSchema):
            id: int  # Compatible - only requires id

        class SinkB(PluginSchema):
            id: int
            score: float  # Incompatible - requires field not in source

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="config_gate:fork")  # NO SCHEMA
        graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="csv_a", input_schema=SinkA)
        graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="csv_b", input_schema=SinkB)

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink_a", label="branch_a", mode=RoutingMode.COPY)  # Fork: COPY mode
        graph.add_edge("gate", "sink_b", label="branch_b", mode=RoutingMode.COPY)  # Fork: COPY mode

        # Should detect incompatibility on gate -> sink_b edge
        # validate_edge_compatibility() raises ValueError for schema mismatches
        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        assert "score" in str(exc_info.value).lower()
        assert "gate" in str(exc_info.value)


class TestExecutionGraphFromConfig:
    """Build ExecutionGraph from ElspethSettings."""

    def test_from_config_minimal(self, plugin_manager) -> None:
        """Build graph from minimal config (source -> sink only)."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Should have: source -> output_sink
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.get_source() is not None
        assert len(graph.get_sinks()) == 1

    def test_from_config_is_valid(self, plugin_manager) -> None:
        """Graph from valid config passes validation."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Should not raise
        graph.validate()
        assert graph.is_acyclic()

    def test_from_config_with_transforms(self, plugin_manager) -> None:
        """Build graph with transform chain."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            transforms=[
                TransformSettings(plugin="passthrough", options={"schema": {"fields": "dynamic"}}),
                TransformSettings(plugin="field_mapper", options={"schema": {"fields": "dynamic"}}),
            ],
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Should have: source -> passthrough -> field_mapper -> output_sink
        assert graph.node_count == 4
        assert graph.edge_count == 3

        # Topological order should be correct
        order = graph.topological_order()
        assert len(order) == 4
        # Source should be first (has "source" in node_id)
        assert "source" in order[0]
        # Sink should be last (has "sink" in node_id)
        assert "sink" in order[-1]
        # Verify transform ordering (passthrough before field_mapper)
        passthrough_idx = next(i for i, n in enumerate(order) if "passthrough" in n)
        field_mapper_idx = next(i for i, n in enumerate(order) if "field_mapper" in n)
        assert passthrough_idx < field_mapper_idx

    def test_from_config_with_gate_routes(self, plugin_manager) -> None:
        """Build graph with config-driven gate routing to multiple sinks."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"fields": "dynamic"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"fields": "dynamic"}}),
            },
            gates=[
                GateSettings(
                    name="safety_gate",
                    condition="row['suspicious'] == True",
                    routes={"true": "flagged", "false": "continue"},
                ),
            ],
            default_sink="results",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Should have:
        #   source -> safety_gate -> results (via "continue")
        #                         -> flagged (via "suspicious")
        assert graph.node_count == 4  # source, config_gate, results, flagged
        # Edges: source->gate, gate->results (continue), gate->flagged (route)
        assert graph.edge_count == 3

    def test_from_config_validates_route_targets(self, plugin_manager) -> None:
        """Config gate routes must reference existing sinks."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            gates=[
                GateSettings(
                    name="bad_gate",
                    condition="True",
                    routes={"true": "nonexistent_sink", "false": "continue"},
                ),
            ],
            default_sink="output",
        )

        with pytest.raises(GraphValidationError) as exc_info:
            plugins = instantiate_plugins_from_config(config)
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
                default_sink=config.default_sink,
            )

        assert "nonexistent_sink" in str(exc_info.value)

    def test_get_sink_id_map(self, plugin_manager) -> None:
        """Get explicit sink_name -> node_id mapping."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import SinkName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"fields": "dynamic"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="results",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )
        sink_map = graph.get_sink_id_map()

        # Explicit mapping - no substring matching
        assert SinkName("results") in sink_map
        assert SinkName("flagged") in sink_map
        assert sink_map[SinkName("results")] != sink_map[SinkName("flagged")]

    def test_get_transform_id_map(self, plugin_manager) -> None:
        """Get explicit sequence -> node_id mapping for transforms."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            transforms=[
                TransformSettings(plugin="passthrough", options={"schema": {"fields": "dynamic"}}),
                TransformSettings(plugin="field_mapper", options={"schema": {"fields": "dynamic"}}),
            ],
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )
        transform_map = graph.get_transform_id_map()

        # Explicit mapping by sequence position
        assert 0 in transform_map  # passthrough
        assert 1 in transform_map  # field_mapper
        assert transform_map[0] != transform_map[1]

    def test_get_default_sink(self, plugin_manager) -> None:
        """Get the output sink name."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"fields": "dynamic"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="results",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        assert graph.get_default_sink() == "results"


class TestExecutionGraphRouteMapping:
    """Test route label <-> sink name mapping for edge lookup."""

    def test_get_route_label_for_sink(self, plugin_manager) -> None:
        """Get route label that leads to a sink from a config gate."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import GateName
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"fields": "dynamic"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"fields": "dynamic"}}),
            },
            gates=[
                GateSettings(
                    name="classifier",
                    condition="row['suspicious'] == True",
                    routes={"true": "flagged", "false": "continue"},
                ),
            ],
            default_sink="results",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Get the config gate's node_id
        gate_node_id = graph.get_config_gate_id_map()[GateName("classifier")]

        # Given gate node and sink name, get the route label
        route_label = graph.get_route_label(gate_node_id, "flagged")

        assert route_label == "true"

    def test_get_route_label_for_continue(self, plugin_manager) -> None:
        """Continue routes return 'continue' as label."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import GateName
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"fields": "dynamic"}})},
            gates=[
                GateSettings(
                    name="gate",
                    condition="True",
                    routes={"true": "continue", "false": "continue"},
                ),
            ],
            default_sink="results",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )
        gate_node_id = graph.get_config_gate_id_map()[GateName("gate")]

        # The edge to output sink uses "continue" label (both routes resolve to continue)
        route_label = graph.get_route_label(gate_node_id, "results")
        assert route_label == "continue"

    def test_hyphenated_sink_names_work_in_dag(self, plugin_manager) -> None:
        """Gate routing to hyphenated sink names works correctly.

        Regression test for gate-route-destination-name-validation-mismatch bug.
        Sink names don't need to match identifier pattern - they're just dict keys.
        """
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import GateName, SinkName
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output-sink": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "quarantine-bucket": SinkSettings(plugin="json", options={"path": "quarantine.json", "schema": {"fields": "dynamic"}}),
            },
            gates=[
                GateSettings(
                    name="quality_check",
                    condition="row['score'] >= 0.5",
                    routes={"true": "output-sink", "false": "quarantine-bucket"},
                ),
            ],
            default_sink="output-sink",
        )

        # DAG compilation should succeed with hyphenated sink names
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Verify both hyphenated sinks exist
        sink_ids = graph.get_sink_id_map()
        assert SinkName("output-sink") in sink_ids
        assert SinkName("quarantine-bucket") in sink_ids

        # Verify gate routes to the hyphenated sinks
        gate_node_id = graph.get_config_gate_id_map()[GateName("quality_check")]
        assert graph.get_route_label(gate_node_id, "quarantine-bucket") == "false"


class TestMultiEdgeSupport:
    """Tests for MultiDiGraph multi-edge support."""

    def test_multiple_edges_same_node_pair(self) -> None:
        """MultiDiGraph allows multiple labeled edges between same nodes."""
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork_gate")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="output")

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
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="classifier")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

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
        from elspeth.contracts import EdgeInfo, NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source-1", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("sink-1", node_type=NodeType.SINK, plugin_name="csv")
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
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("n1", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("n2", node_type=NodeType.SINK, plugin_name="test")

        # Should accept enum directly
        graph.add_edge("n1", "n2", label="route", mode=RoutingMode.COPY)

        edges = graph.get_edges()
        assert edges[0].mode == RoutingMode.COPY


class TestMultiEdgeScenarios:
    """Tests for scenarios requiring multiple edges between same nodes."""

    def test_fork_gate_config_parses_into_valid_graph(self, plugin_manager) -> None:
        """Fork gate configuration parses into valid graph structure.

        Note: This tests config parsing, not the multi-edge bug. Fork routes
        with target="fork" don't create edges to sinks - they create child tokens.
        The multi-edge bug is tested by test_gate_multiple_routes_same_sink.
        """
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "path_a": SinkSettings(plugin="json", options={"path": "path_a.json", "schema": {"fields": "dynamic"}}),
                "path_b": SinkSettings(plugin="json", options={"path": "path_b.json", "schema": {"fields": "dynamic"}}),
            },
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always forks
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

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
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="classifier")
        graph.add_node("alerts", node_type=NodeType.SINK, plugin_name="csv")

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

    def test_from_config_creates_coalesce_node(self, plugin_manager) -> None:
        """Coalesce config should create a coalesce node in the graph."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName, NodeType
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Use proper accessor, not string matching
        coalesce_map = graph.get_coalesce_id_map()
        assert CoalesceName("merge_results") in coalesce_map

        # Verify node type
        node_id = coalesce_map[CoalesceName("merge_results")]
        node_info = graph.get_node_info(node_id)
        assert node_info.node_type == NodeType.COALESCE
        assert node_info.plugin_name == "coalesce:merge_results"

    def test_from_config_coalesce_edges_from_fork_branches(self, plugin_manager) -> None:
        """Coalesce node should have edges from fork gate (via branches)."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName, GateName, RoutingMode
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Get node IDs
        gate_id = graph.get_config_gate_id_map()[GateName("forker")]
        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]

        # Verify edges from fork gate to coalesce node
        edges = graph.get_edges()
        gate_to_coalesce_edges = [e for e in edges if e.from_node == gate_id and e.to_node == coalesce_id]

        # Should have two edges (path_a and path_b) with COPY mode
        assert len(gate_to_coalesce_edges) == 2
        labels = {e.label for e in gate_to_coalesce_edges}
        assert labels == {"path_a", "path_b"}
        assert all(e.mode == RoutingMode.COPY for e in gate_to_coalesce_edges)

    def test_duplicate_fork_branches_rejected_in_config_gate(self, plugin_manager) -> None:
        """Duplicate branch names in fork_to should be rejected for config gates."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import ElspethSettings, GateSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "path_a": SinkSettings(plugin="json", options={"path": "path_a.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_a"],  # Duplicate branch name
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)

        with pytest.raises(GraphValidationError, match=r"duplicate fork branches"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                default_sink=settings.default_sink,
                coalesce_settings=settings.coalesce,
            )

    def test_duplicate_fork_branches_rejected_in_plugin_gate(self) -> None:
        """Duplicate branch names in fork_to should be rejected for plugin gates."""
        from typing import Any

        from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema, SourceRow
        from elspeth.core.dag import ExecutionGraph, GraphValidationError
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        class DummySchema(PluginSchema):
            pass

        class DummySource:
            name = "dummy_source"
            output_schema = DummySchema
            node_id: str | None = None
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            _on_validation_failure = "discard"

            def __init__(self) -> None:
                self.config = {"schema": {"fields": "dynamic"}}

            def load(self, ctx: PluginContext) -> Any:
                yield SourceRow.valid({"value": 1})

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class DummySink:
            input_schema = DummySchema
            idempotent = True
            node_id: str | None = None
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, name: str) -> None:
                self.name = name
                self.config = {"schema": {"fields": "dynamic"}}

            def write(self, rows: Any, ctx: PluginContext) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(path="memory", content_hash="", size_bytes=0)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class DummyGate(BaseGate):
            name = "fork_gate"
            input_schema = DummySchema
            output_schema = DummySchema

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

        source = DummySource()
        sinks = {
            "output": DummySink("output"),
            "path_a": DummySink("path_a"),
        }
        # Plugin gates must NOT have "fork" as a route target - they use
        # RoutingAction.fork_to_paths() in evaluate() instead.
        # Routes here just define non-fork routing options.
        gate = DummyGate(
            {
                "routes": {"default": "continue"},
                "fork_to": ["path_a", "path_a"],  # Duplicate branch - should be rejected
                "schema": {"fields": "dynamic"},
            }
        )

        with pytest.raises(GraphValidationError, match=r"duplicate fork branches"):
            ExecutionGraph.from_plugin_instances(
                source=source,
                transforms=[gate],
                sinks=sinks,
                aggregations={},
                gates=[],
                default_sink="output",
                coalesce_settings=None,
            )

    def test_partial_branch_coverage_branches_not_in_coalesce_route_to_sink(
        self,
        plugin_manager,
    ) -> None:
        """Fork branches not in any coalesce should still route to output sink."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName, GateName, SinkName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "path_c": SinkSettings(plugin="json", options={"path": "path_c.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Get node IDs
        gate_id = graph.get_config_gate_id_map()[GateName("forker")]
        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]
        path_c_sink_id = graph.get_sink_id_map()[SinkName("path_c")]

        # Verify path_c goes to path_c sink, not coalesce
        edges = graph.get_edges()
        path_c_edges = [e for e in edges if e.from_node == gate_id and e.label == "path_c"]

        assert len(path_c_edges) == 1
        assert path_c_edges[0].to_node == path_c_sink_id

        # Verify path_a and path_b go to coalesce
        coalesce_edges = [e for e in edges if e.from_node == gate_id and e.to_node == coalesce_id]
        coalesce_labels = {e.label for e in coalesce_edges}
        assert coalesce_labels == {"path_a", "path_b"}

    def test_get_coalesce_id_map_returns_mapping(self, plugin_manager) -> None:
        """get_coalesce_id_map should return coalesce_name -> node_id."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )
        coalesce_map = graph.get_coalesce_id_map()

        # Should have both coalesce nodes
        assert CoalesceName("merge_ab") in coalesce_map
        assert CoalesceName("merge_cd") in coalesce_map

        # Node IDs should be unique
        assert coalesce_map[CoalesceName("merge_ab")] != coalesce_map[CoalesceName("merge_cd")]

        # Verify both nodes exist in the graph
        assert graph.has_node(coalesce_map[CoalesceName("merge_ab")])
        assert graph.has_node(coalesce_map[CoalesceName("merge_cd")])

    def test_get_branch_to_coalesce_map_returns_mapping(self, plugin_manager) -> None:
        """get_branch_to_coalesce_map should return branch_name -> coalesce_name."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import BranchName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )
        branch_map = graph.get_branch_to_coalesce_map()

        # Should map branches to coalesce_name (not node_id) for processor step lookup
        assert branch_map[BranchName("path_a")] == "merge_results"
        assert branch_map[BranchName("path_b")] == "merge_results"

    def test_branch_to_coalesce_maps_to_coalesce_name_for_step_lookup(self, plugin_manager) -> None:
        """branch_to_coalesce should map to coalesce_name (not node_id) for use with coalesce_step_map.

        BUG-LINEAGE-01: The processor needs to look up coalesce step position using:
            coalesce_name = branch_to_coalesce[branch_name]
            step = coalesce_step_map[coalesce_name]

        But branch_to_coalesce was mapping branch_name -> node_id, causing KeyError
        when trying to look up in coalesce_step_map which expects coalesce_name keys.
        """
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import BranchName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["analysis_path", "validation_path"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="join_point",
                    branches=["analysis_path", "validation_path"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        branch_map = graph.get_branch_to_coalesce_map()

        # CRITICAL: Must map to coalesce_name (not node_id) for processor step lookup
        # The processor does: coalesce_step_map[branch_to_coalesce[branch_name]]
        # coalesce_step_map has keys like "join_point", NOT node_ids like "coalesce_join_point_abc123"
        assert branch_map[BranchName("analysis_path")] == "join_point", (
            f"Expected coalesce_name 'join_point', got {branch_map[BranchName('analysis_path')]}"
        )
        assert branch_map[BranchName("validation_path")] == "join_point", (
            f"Expected coalesce_name 'join_point', got {branch_map[BranchName('validation_path')]}"
        )

    def test_duplicate_branch_names_across_coalesces_rejected(self, plugin_manager) -> None:
        """Duplicate branch names across coalesce settings should be rejected."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b", "path_x"],
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
                    name="merge_xy",
                    branches=["path_a", "path_x"],  # path_a duplicated!
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)

        with pytest.raises(GraphValidationError, match="Duplicate branch name 'path_a'"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                default_sink=settings.default_sink,
                coalesce_settings=settings.coalesce,
            )

    def test_empty_coalesce_branches_rejected(self, plugin_manager) -> None:
        """Coalesce with empty branches list should be rejected by Pydantic."""
        from pydantic import ValidationError

        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )

        # Pydantic validates min_length=2 for branches field
        with pytest.raises(ValidationError, match="at least 2 items"):
            ElspethSettings(
                source=SourceSettings(
                    plugin="csv",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"fields": "dynamic"},
                    },
                ),
                sinks={
                    "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                },
                default_sink="output",
                coalesce=[
                    CoalesceSettings(
                        name="empty_merge",
                        branches=[],  # Invalid! Pydantic requires min_length=2
                        policy="require_all",
                        merge="union",
                    ),
                ],
            )

    def test_coalesce_branch_not_produced_by_any_gate_rejected(self, plugin_manager) -> None:
        """Coalesce referencing non-existent fork branches should be rejected."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "path_b": SinkSettings(plugin="json", options={"path": "path_b.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],  # path_b goes to sink, path_a goes to coalesce
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_x"],  # path_x not in fork_to!
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)

        with pytest.raises(GraphValidationError, match=r"branch 'path_x'.*no gate produces"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                default_sink=settings.default_sink,
                coalesce_settings=settings.coalesce,
            )

    def test_fork_coalesce_contract_branch_map_compatible_with_step_map(self, plugin_manager) -> None:
        """Contract test: branch_to_coalesce values must be usable as coalesce_step_map keys.

        This is the CRITICAL contract between DAG builder and Processor.
        The processor does: coalesce_step_map[branch_to_coalesce[branch_name]]
        This test ensures the production path (`from_plugin_instances`) produces compatible mappings.
        """
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            transforms=[
                TransformSettings(plugin="passthrough", options={"schema": {"fields": "dynamic"}}),
            ],
            gates=[
                GateSettings(
                    name="analysis_fork",
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
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)

        # Use production path
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Get the mappings that processor would use
        branch_to_coalesce = graph.get_branch_to_coalesce_map()

        # Simulate what orchestrator does (build coalesce_step_map)
        coalesce_step_map: dict[str, int] = {}
        branch_steps: dict[str, int] = {}
        base_step = len(settings.transforms)
        for gate_idx, gate in enumerate(settings.gates):
            if gate.fork_to:
                step = base_step + gate_idx + 1
                for branch in gate.fork_to:
                    existing = branch_steps.get(branch)
                    if existing is None or step > existing:
                        branch_steps[branch] = step
        for cs in settings.coalesce:
            coalesce_step_map[cs.name] = max(branch_steps[branch] for branch in cs.branches)

        # CRITICAL CONTRACT: Every value in branch_to_coalesce must be a key in coalesce_step_map
        # This is what processor relies on at lines 695-696
        for branch_name, coalesce_name in branch_to_coalesce.items():
            assert coalesce_name in coalesce_step_map, (
                f"Contract violation: branch_to_coalesce['{branch_name}'] = '{coalesce_name}', "
                f"but '{coalesce_name}' not in coalesce_step_map keys: {list(coalesce_step_map.keys())}"
            )

            # Also verify it's the coalesce_name, not a node_id
            assert not coalesce_name.startswith("coalesce_"), (
                f"Contract violation: branch_to_coalesce['{branch_name}'] = '{coalesce_name}' "
                f"looks like a node_id (starts with 'coalesce_'), should be coalesce name"
            )

    def test_coalesce_node_has_edge_to_output_sink(self, plugin_manager) -> None:
        """Coalesce node should have an edge to the output sink."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName, RoutingMode, SinkName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]
        output_sink_id = graph.get_sink_id_map()[SinkName("output")]

        # Verify edge from coalesce to output sink
        edges = graph.get_edges()
        coalesce_to_sink_edges = [e for e in edges if e.from_node == coalesce_id and e.to_node == output_sink_id]

        assert len(coalesce_to_sink_edges) == 1
        assert coalesce_to_sink_edges[0].label == "continue"
        assert coalesce_to_sink_edges[0].mode == RoutingMode.MOVE

    def test_coalesce_node_connects_to_next_gate(self, plugin_manager) -> None:
        """Coalesce node should continue to the next gate when one exists."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName, GateName, RoutingMode
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker1",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
                GateSettings(
                    name="gate2",
                    condition="True",
                    routes={"true": "continue", "false": "continue"},
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]
        gate2_id = graph.get_config_gate_id_map()[GateName("gate2")]

        edges = graph.get_edges()
        coalesce_to_gate_edges = [e for e in edges if e.from_node == coalesce_id and e.to_node == gate2_id]

        assert len(coalesce_to_gate_edges) == 1
        assert coalesce_to_gate_edges[0].label == "continue"
        assert coalesce_to_gate_edges[0].mode == RoutingMode.MOVE

    def test_coalesce_node_stores_config(self, plugin_manager) -> None:
        """Coalesce node should store configuration for audit trail."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import CoalesceName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]
        node_info = graph.get_node_info(coalesce_id)

        # Verify config is stored
        assert node_info.config["branches"] == ["path_a", "path_b"]
        assert node_info.config["policy"] == "quorum"
        assert node_info.config["merge"] == "nested"
        assert node_info.config["timeout_seconds"] == 30.0
        assert node_info.config["quorum_count"] == 1


class TestSchemaValidation:
    """Tests for graph-based schema compatibility validation."""

    def test_schema_validation_catches_gate_routing_to_incompatible_sink(self) -> None:
        """Gate routes to sink before required field is added - should fail validation.

        This is the bug scenario: A gate routes rows directly to a sink from an
        intermediate point in the pipeline, but the sink requires a field that
        hasn't been added yet. The old linear validator checked all sinks against
        the "final transform output", missing this incompatibility.
        """
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph

        class SourceSchema(PluginSchema):
            """Source provides: name, quality."""

            name: str
            quality: str

        class AddScoreSchema(PluginSchema):
            """After add_score transform: name, quality, score."""

            name: str
            quality: str
            score: int

        class RawSinkSchema(PluginSchema):
            """Raw sink requires: name, quality, score."""

            name: str
            quality: str
            score: int  # NOT PROVIDED by source or gate!

        graph = ExecutionGraph()

        # Build pipeline: Source -> Gate -> [routes to raw_sink OR continues to add_score]
        graph.add_node(
            "src",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            output_schema=SourceSchema,
        )
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="quality_gate",
            input_schema=SourceSchema,
            output_schema=SourceSchema,  # Gate doesn't modify data
        )
        graph.add_node(
            "add_score",
            node_type=NodeType.TRANSFORM,
            plugin_name="add_score_transform",
            input_schema=SourceSchema,
            output_schema=AddScoreSchema,  # Adds 'score' field
        )
        graph.add_node(
            "raw_sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
            input_schema=RawSinkSchema,  # Requires 'score' field!
        )
        graph.add_node(
            "processed_sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
            input_schema=AddScoreSchema,
        )

        # Edges
        graph.add_edge("src", "gate", label="continue")
        graph.add_edge("gate", "raw_sink", label="raw")  # BUG: Routes BEFORE add_score!
        graph.add_edge("gate", "add_score", label="continue")
        graph.add_edge("add_score", "processed_sink", label="continue")

        # The gate routes to raw_sink with SourceSchema (no 'score' field),
        # but raw_sink requires RawSinkSchema (with 'score' field).
        # validate_edge_compatibility() raises ValueError for schema mismatches
        with pytest.raises(ValueError, match="score"):
            graph.validate_edge_compatibility()

    def test_coalesce_rejects_incompatible_branch_schemas(self) -> None:
        """Coalesce with incompatible branch schemas should fail validation.

        CRITICAL P0 BLOCKER: Coalesce incompatible schema behavior was UNDEFINED.
        Manual graph construction bypasses config schema limitation that
        doesn't support per-branch transforms.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            id: int
            name: str

        class BranchAOutput(PluginSchema):
            """Branch A adds score field."""

            id: int
            name: str
            score: float

        class BranchBOutput(PluginSchema):
            """Branch B adds rank field (different from Branch A)."""

            id: int
            name: str
            rank: int

        graph = ExecutionGraph()

        # Build fork/join DAG with INCOMPATIBLE branch schemas
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")

        # Branch A: adds score field
        graph.add_node(
            "transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="add_score",
            input_schema=SourceOutput,
            output_schema=BranchAOutput,
        )

        # Branch B: adds rank field (incompatible with Branch A)
        graph.add_node(
            "transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="add_rank",
            input_schema=SourceOutput,
            output_schema=BranchBOutput,
        )

        # Coalesce attempts to merge incompatible schemas
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": ["branch_a", "branch_b"],
                "policy": "require_all",
                "merge": "union",
            },
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Build edges
        graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("fork_gate", "transform_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("fork_gate", "transform_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("transform_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("transform_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should crash: coalesce can't merge BranchAOutput and BranchBOutput
        # validate_edge_compatibility() raises ValueError for schema mismatches
        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        # Error should mention incompatible schemas
        error_msg = str(exc_info.value).lower()
        assert "schema" in error_msg or "incompatible" in error_msg

    def test_coalesce_accepts_structurally_identical_schemas(self) -> None:
        """Coalesce should accept branches with structurally identical schemas.

        BUG FIX: P2-2026-01-30-coalesce-schema-identity-check
        Previously, coalesce validation compared schema classes by identity (!=),
        rejecting structurally identical schemas that were distinct class objects.
        This happens when create_schema_from_config() is called multiple times
        with the same field definitions (e.g., per-instance LLM transforms).

        The fix uses check_compatibility() for structural comparison.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create two STRUCTURALLY IDENTICAL schemas from same config
        # These will be distinct class objects (SchemaA is not SchemaB)
        fields = (
            FieldDefinition(name="id", field_type="int"),
            FieldDefinition(name="value", field_type="str"),
        )
        config = SchemaConfig(mode="strict", fields=fields, is_dynamic=False)

        # Each call creates a NEW class object
        SchemaA = create_schema_from_config(config, "BranchASchema")
        SchemaB = create_schema_from_config(config, "BranchBSchema")

        # Verify they are distinct class objects but structurally identical
        assert SchemaA is not SchemaB, "Test requires distinct class objects"
        assert list(SchemaA.model_fields.keys()) == list(SchemaB.model_fields.keys())

        class SourceOutput(PluginSchema):
            id: int

        graph = ExecutionGraph()

        # Build fork/join DAG with COMPATIBLE (structurally identical) schemas
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")

        # Branch A uses SchemaA
        graph.add_node(
            "transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="enrich",
            input_schema=SourceOutput,
            output_schema=SchemaA,
        )

        # Branch B uses SchemaB (structurally identical to SchemaA!)
        graph.add_node(
            "transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="enrich",
            input_schema=SourceOutput,
            output_schema=SchemaB,
        )

        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": ["branch_a", "branch_b"],
                "policy": "require_all",
                "merge": "union",
            },
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Build edges
        graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("fork_gate", "transform_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("fork_gate", "transform_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("transform_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("transform_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should PASS: schemas are structurally identical
        # (This was incorrectly failing before the fix)
        graph.validate_edge_compatibility()

    def test_coalesce_accepts_dynamic_schemas_from_different_instances(self) -> None:
        """Coalesce should accept branches with dynamic schemas from different instances.

        BUG FIX: P2-2026-01-30-coalesce-schema-identity-check
        Dynamic schemas (is_dynamic=True) are compatible with anything.
        Even if they're distinct class objects, they should pass validation.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create two dynamic schemas (accept anything)
        config = SchemaConfig(mode=None, fields=None, is_dynamic=True)

        DynamicA = create_schema_from_config(config, "DynamicA")
        DynamicB = create_schema_from_config(config, "DynamicB")

        # Verify they are distinct class objects
        assert DynamicA is not DynamicB, "Test requires distinct class objects"

        class SourceOutput(PluginSchema):
            id: int

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")

        # Both branches use dynamic schemas (distinct objects)
        graph.add_node(
            "transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            input_schema=SourceOutput,
            output_schema=DynamicA,
        )

        graph.add_node(
            "transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            input_schema=SourceOutput,
            output_schema=DynamicB,
        )

        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": ["branch_a", "branch_b"],
                "policy": "require_all",
                "merge": "union",
            },
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("fork_gate", "transform_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("fork_gate", "transform_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("transform_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("transform_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should PASS: dynamic schemas are compatible with anything
        graph.validate_edge_compatibility()

    def test_coalesce_rejects_mixed_dynamic_explicit_branches(self) -> None:
        """Coalesce must reject mixed dynamic/explicit branch schemas.

        BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected

        When one branch produces a dynamic schema and another produces an explicit
        schema, the coalesce's effective schema becomes the first branch's schema.
        This masks the mismatch: downstream consumers expect explicit fields that
        dynamic-branch rows may not have, causing runtime failures.

        Pre-run validation must detect and reject this mismatch.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create a dynamic schema (no fields, accepts anything)
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create an explicit schema with specific fields
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float", "id: int"]}),
            "ExplicitSchema",
            allow_coercion=False,
        )

        class SourceOutput(PluginSchema):
            id: int

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")

        # Branch A: produces EXPLICIT schema
        graph.add_node(
            "transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="processor",
            input_schema=SourceOutput,
            output_schema=ExplicitSchema,
        )

        # Branch B: produces DYNAMIC schema
        graph.add_node(
            "transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            input_schema=SourceOutput,
            output_schema=DynamicSchema,
        )

        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": ["branch_a", "branch_b"],
                "policy": "require_all",
                "merge": "union",
            },
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("fork_gate", "transform_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("fork_gate", "transform_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("transform_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("transform_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL: mixed dynamic/explicit branches are not allowed
        with pytest.raises(ValueError, match=r"mixed.*dynamic.*explicit|dynamic.*explicit.*mismatch"):
            graph.validate_edge_compatibility()

    def test_coalesce_rejects_mixed_none_explicit_branches(self) -> None:
        """Coalesce must reject mixed None/explicit branch schemas.

        BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected

        None (unspecified output_schema) is treated as dynamic. Mixed with explicit
        schemas, this creates the same mismatch problem as dynamic schema classes.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create an explicit schema with specific fields
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float", "id: int"]}),
            "ExplicitSchema",
            allow_coercion=False,
        )

        class SourceOutput(PluginSchema):
            id: int

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")

        # Branch A: produces EXPLICIT schema
        graph.add_node(
            "transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="processor",
            input_schema=SourceOutput,
            output_schema=ExplicitSchema,
        )

        # Branch B: produces NONE (unspecified = dynamic)
        graph.add_node(
            "transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            input_schema=SourceOutput,
            output_schema=None,
        )

        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": ["branch_a", "branch_b"],
                "policy": "require_all",
                "merge": "union",
            },
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("fork_gate", "transform_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("fork_gate", "transform_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("transform_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("transform_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL: mixed None/explicit branches are not allowed
        with pytest.raises(ValueError, match=r"mixed.*dynamic.*explicit|dynamic.*explicit.*mismatch"):
            graph.validate_edge_compatibility()

    def test_gate_rejects_mixed_dynamic_explicit_incoming_branches(self) -> None:
        """Gate with multiple inputs must reject mixed dynamic/explicit schemas.

        BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected

        Gates can receive inputs from multiple sources (e.g., in complex DAGs).
        When inputs have mixed dynamic/explicit schemas, the same mismatch problem
        occurs as with coalesce nodes.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create a dynamic schema
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create an explicit schema
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float", "id: int"]}),
            "ExplicitSchema",
            allow_coercion=False,
        )

        class SourceOutput(PluginSchema):
            id: int

        graph = ExecutionGraph()

        # Two sources feeding into one gate
        graph.add_node("source1", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("source2", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)

        # Transform 1 produces explicit schema
        graph.add_node(
            "transform1",
            node_type=NodeType.TRANSFORM,
            plugin_name="processor",
            input_schema=SourceOutput,
            output_schema=ExplicitSchema,
        )

        # Transform 2 produces dynamic schema
        graph.add_node(
            "transform2",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            input_schema=SourceOutput,
            output_schema=DynamicSchema,
        )

        # Gate receives both (simulating a join-like pattern)
        graph.add_node("merge_gate", node_type=NodeType.GATE, plugin_name="config_gate")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source1", "transform1", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("source2", "transform2", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform1", "merge_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform2", "merge_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("merge_gate", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL: mixed dynamic/explicit inputs to gate
        with pytest.raises(ValueError, match=r"mixed.*dynamic.*explicit|dynamic.*explicit.*mismatch"):
            graph.validate_edge_compatibility()

    def test_aggregation_schema_transition_in_topology(self) -> None:
        """Aggregation with input_schema and output_schema in single topology.

        Verifies source→agg validates against input_schema,
        and agg→sink validates against output_schema.
        """
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            value: float

        class AggregationOutput(PluginSchema):
            count: int
            sum: float

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node(
            "agg",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            input_schema=SourceOutput,  # Incoming edge validates against this
            output_schema=AggregationOutput,  # Outgoing edge validates against this
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "options": {
                    "schema": {"mode": "strict", "fields": ["value: float"]},
                    "value_field": "value",
                },
                "schema": {"mode": "strict", "fields": ["value: float"]},
            },
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=AggregationOutput)

        graph.add_edge("source", "agg", label="continue")
        graph.add_edge("agg", "sink", label="continue")

        # Should pass - schemas compatible at both edges
        graph.validate()

    def test_aggregation_schema_transition_incompatible_output(self) -> None:
        """Aggregation with incompatible output_schema should fail validation."""
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            value: float

        class AggregationOutput(PluginSchema):
            count: int
            sum: float

        class SinkInput(PluginSchema):
            """Sink requires 'average' field not in aggregation output."""

            count: int
            sum: float
            average: float  # NOT in AggregationOutput!

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node(
            "agg",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            input_schema=SourceOutput,
            output_schema=AggregationOutput,
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "options": {
                    "schema": {"mode": "strict", "fields": ["value: float"]},
                    "value_field": "value",
                },
                "schema": {"mode": "strict", "fields": ["value: float"]},
            },
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=SinkInput)

        graph.add_edge("source", "agg", label="continue")
        graph.add_edge("agg", "sink", label="continue")

        # Should crash - sink requires 'average' field
        # validate_edge_compatibility() raises ValueError for schema mismatches
        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        assert "average" in str(exc_info.value).lower()

    def test_schema_validation_error_includes_diagnostic_details(self) -> None:
        """Schema validation errors include field name, producer node, consumer node."""
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph

        class SourceOutput(PluginSchema):
            id: int
            name: str

        class SinkInput(PluginSchema):
            id: int
            name: str
            score: float  # Missing from source output

        graph = ExecutionGraph()

        graph.add_node("my_source", node_type=NodeType.SOURCE, plugin_name="csv_reader", output_schema=SourceOutput)
        graph.add_node("my_sink", node_type=NodeType.SINK, plugin_name="db_writer", input_schema=SinkInput)

        graph.add_edge("my_source", "my_sink", label="continue")

        # Capture error and verify diagnostic details
        # validate_edge_compatibility() raises ValueError for schema mismatches
        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        error_msg = str(exc_info.value)

        # Should include field name
        assert "score" in error_msg.lower()

        # Should include producer node ID
        assert "my_source" in error_msg

        # Should include consumer node ID
        assert "my_sink" in error_msg


def test_from_plugin_instances_extracts_schemas():
    """Verify from_plugin_instances extracts schemas from instances."""
    import tempfile
    from pathlib import Path

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.contracts import NodeType
    from elspeth.core.config import load_settings
    from elspeth.core.dag import ExecutionGraph

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      mode: strict
      fields:
        - "value: float"
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        mode: strict
        fields:
          - "value: float"

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "value: float"

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )

        # Verify schemas extracted
        source_nodes = [n for n, d in graph._graph.nodes(data=True) if d["info"].node_type == NodeType.SOURCE]
        source_info = graph.get_node_info(source_nodes[0])
        assert source_info.output_schema is not None

    finally:
        config_file.unlink()


def test_validate_aggregation_dual_schema():
    """Verify aggregation edges validate against correct schemas."""
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.schema_factory import create_schema_from_config

    input_schema_config = {"mode": "strict", "fields": ["value: float"]}
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict(input_schema_config),
        "InputSchema",
        allow_coercion=False,
    )

    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"mode": "strict", "fields": ["count: int", "sum: float"]}),
        "OutputSchema",
        allow_coercion=False,
    )

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type=NodeType.AGGREGATION,
        plugin_name="batch_stats",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        config={
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {
                "schema": dict(input_schema_config),
                "value_field": "value",
            },
            "schema": dict(input_schema_config),
        },
    )
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=OutputSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Should pass - validate_edge_compatibility() raises ValueError on failure
    graph.validate_edge_compatibility()  # No exception means success


def test_validate_aggregation_detects_incompatibility():
    """Verify validation detects aggregation output mismatch."""
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.schema_factory import create_schema_from_config

    input_schema_config = {"mode": "strict", "fields": ["value: float"]}
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict(input_schema_config),
        "InputSchema",
        allow_coercion=False,
    )

    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"mode": "strict", "fields": ["count: int"]}),  # Missing 'sum'
        "OutputSchema",
        allow_coercion=False,
    )

    SinkSchema = create_schema_from_config(
        SchemaConfig.from_dict({"mode": "strict", "fields": ["count: int", "sum: float"]}),
        "SinkSchema",
        allow_coercion=False,
    )

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type=NodeType.AGGREGATION,
        plugin_name="batch_stats",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        config={
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {
                "schema": dict(input_schema_config),
                "value_field": "value",
            },
            "schema": dict(input_schema_config),
        },
    )
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=SinkSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Should fail - sink requires 'sum' which aggregation output doesn't provide
    with pytest.raises(ValueError) as exc_info:
        graph.validate_edge_compatibility()

    assert "sum" in str(exc_info.value).lower()


class TestDynamicSchemaDetection:
    """Tests for detecting and handling dynamic schemas."""

    def test_dynamic_source_to_specific_sink_should_skip_validation(self) -> None:
        """Dynamic source → specific sink should PASS (validation skipped).

        Manually constructed graph with dynamic output_schema and specific input_schema.
        Validation should be skipped for dynamic schemas.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create dynamic schema (no fields, extra='allow')
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create specific schema (has fields, extra='forbid')
        SpecificSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float", "name: str"]}),
            "SpecificSchema",
            allow_coercion=False,
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=DynamicSchema)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=SpecificSchema)
        graph.add_edge("source", "sink", label="continue")

        # Should NOT raise - validation is skipped for dynamic schemas
        graph.validate()

    def test_specific_source_to_dynamic_sink_should_skip_validation(self) -> None:
        """Specific source → dynamic sink should PASS (validation skipped).

        Manually constructed graph with specific output_schema and dynamic input_schema.
        Validation should be skipped for dynamic schemas.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create specific schema
        SpecificSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float"]}),
            "SpecificSchema",
            allow_coercion=False,
        )

        # Create dynamic schema
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SpecificSchema)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=DynamicSchema)
        graph.add_edge("source", "sink", label="continue")

        # Should NOT raise - validation is skipped for dynamic schemas
        graph.validate()

    def test_dynamic_schema_detection_in_validation(self) -> None:
        """Dynamic schema detection correctly identifies dynamic vs explicit schemas.

        Dynamic schemas have no fields and extra='allow', matching the detection
        logic in ExecutionGraph._get_missing_required_fields().
        """
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create dynamic schema
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create explicit schema
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float"]}),
            "ExplicitSchema",
            allow_coercion=False,
        )

        # Helper to check if schema is dynamic (matches logic in dag.py)
        def is_dynamic_schema(schema: type[PluginSchema] | None) -> bool:
            if schema is None:
                return True
            return len(schema.model_fields) == 0 and schema.model_config.get("extra") == "allow"

        # Test dynamic schema detection
        assert is_dynamic_schema(DynamicSchema) is True

        # Test explicit schema detection
        assert is_dynamic_schema(ExplicitSchema) is False

        # Test backwards compat (None = dynamic)
        assert is_dynamic_schema(None) is True


class TestDeterministicNodeIDs:
    """Tests for deterministic node ID generation."""

    def test_node_ids_are_deterministic_for_same_config(self) -> None:
        """Node IDs must be deterministic for checkpoint/resume compatibility."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            transforms=[
                TransformSettings(
                    plugin="passthrough",
                    options={"schema": {"fields": "dynamic"}},
                )
            ],
            sinks={"out": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"fields": "dynamic"}})},
            default_sink="out",
        )

        # Build graph twice with same config
        plugins1 = instantiate_plugins_from_config(config)
        graph1 = ExecutionGraph.from_plugin_instances(
            source=plugins1["source"],
            transforms=plugins1["transforms"],
            sinks=plugins1["sinks"],
            aggregations=plugins1["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        plugins2 = instantiate_plugins_from_config(config)
        graph2 = ExecutionGraph.from_plugin_instances(
            source=plugins2["source"],
            transforms=plugins2["transforms"],
            sinks=plugins2["sinks"],
            aggregations=plugins2["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )

        # Node IDs must be identical
        nodes1 = sorted(graph1._graph.nodes())
        nodes2 = sorted(graph2._graph.nodes())

        assert nodes1 == nodes2, "Node IDs must be deterministic for checkpoint compatibility"

    def test_node_ids_change_when_config_changes(self) -> None:
        """Node IDs should change if plugin config changes."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config1 = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            transforms=[],
            sinks={"out": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"fields": "dynamic"}})},
            default_sink="out",
        )

        config2 = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "strict", "fields": ["id: int"]},  # Different!
                },
            ),
            transforms=[],
            sinks={"out": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"fields": "dynamic"}})},
            default_sink="out",
        )

        plugins1 = instantiate_plugins_from_config(config1)
        graph1 = ExecutionGraph.from_plugin_instances(
            source=plugins1["source"],
            transforms=plugins1["transforms"],
            sinks=plugins1["sinks"],
            aggregations=plugins1["aggregations"],
            gates=list(config1.gates),
            default_sink=config1.default_sink,
        )

        plugins2 = instantiate_plugins_from_config(config2)
        graph2 = ExecutionGraph.from_plugin_instances(
            source=plugins2["source"],
            transforms=plugins2["transforms"],
            sinks=plugins2["sinks"],
            aggregations=plugins2["aggregations"],
            gates=list(config2.gates),
            default_sink=config2.default_sink,
        )

        # Source node IDs should differ (different config)
        source_id_1 = next(n for n in graph1._graph.nodes() if n.startswith("source_"))
        source_id_2 = next(n for n in graph2._graph.nodes() if n.startswith("source_"))

        assert source_id_1 != source_id_2


class TestCoalesceGateIndex:
    """Test coalesce_gate_index exposure from ExecutionGraph."""

    def test_get_coalesce_gate_index_returns_copy(self) -> None:
        """Getter should return a copy to prevent external mutation."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts.types import CoalesceName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                ),
            ],
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json", "schema": {"fields": "dynamic"}})},
            coalesce=[
                CoalesceSettings(
                    name="merge_branches",
                    branches=["branch_a", "branch_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
            default_sink=settings.default_sink,
        )

        # Get the index
        index = graph.get_coalesce_gate_index()

        # Verify it contains expected mapping
        assert CoalesceName("merge_branches") in index
        assert isinstance(index[CoalesceName("merge_branches")], int)

        # Verify it's a copy (mutation doesn't affect internal state)
        original_value = index[CoalesceName("merge_branches")]
        index[CoalesceName("merge_branches")] = 999

        fresh_index = graph.get_coalesce_gate_index()
        assert fresh_index[CoalesceName("merge_branches")] == original_value

    def test_get_coalesce_gate_index_empty_when_no_coalesce(self) -> None:
        """Getter returns empty dict when no coalesce configured."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
            default_sink=settings.default_sink,
        )

        index = graph.get_coalesce_gate_index()
        assert index == {}
