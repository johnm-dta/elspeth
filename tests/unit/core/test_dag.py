# tests/core/test_dag.py
"""Tests for DAG validation and operations."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import pytest

from elspeth.core.config import AggregationSettings, ElspethSettings, GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph, WiredTransform
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol


class _PluginInstances(TypedDict):
    source: SourceProtocol
    source_settings: SourceSettings
    transforms: list[WiredTransform]
    sinks: dict[str, SinkProtocol]
    aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]]


_AUTO_SOURCE_ON_SUCCESS = "_auto_source_on_success"
_AUTO_GATE_INPUT = "_auto_gate_input"
_AUTO_AGG_INPUT = "_auto_agg_input"
_AUTO_TRANSFORM_INPUT_PREFIX = "_auto_transform_input_"


def _source_settings(cls: Any, /, **kwargs: Any) -> Any:
    """Build SourceSettings with backward-compatible defaults for legacy tests."""
    options = dict(kwargs.pop("options", {}) or {})
    on_success = kwargs.pop("on_success", None)
    if on_success is None:
        on_success = options.pop("on_success", _AUTO_SOURCE_ON_SUCCESS)
    return cls(on_success=on_success, options=options, **kwargs)


def _transform_settings(cls: Any, /, **kwargs: Any) -> Any:
    """Build TransformSettings with explicit on_success and on_error.

    Callers MUST provide on_success explicitly — no defaults are inferred.
    on_error defaults to "discard" (a concrete valid value, not a sentinel).
    """
    options = dict(kwargs.pop("options", {}) or {})
    on_success = kwargs.pop("on_success", None)
    if "on_success" in options:
        on_success = options.pop("on_success")
    on_error = kwargs.pop("on_error", "discard")
    if "on_error" in options:
        on_error = options.pop("on_error")
    name = kwargs.pop("name", None)
    plugin = kwargs.get("plugin", "transform")
    if name is None:
        name = f"{plugin}_{abs(hash((plugin, repr(sorted(options.items()))))) % 10_000}"
    if on_success is None:
        msg = f"_transform_settings: on_success is required for transform '{name}' — no defaults inferred"
        raise ValueError(msg)
    input_connection = kwargs.pop("input", None)
    if input_connection is None:
        input_connection = f"{_AUTO_TRANSFORM_INPUT_PREFIX}{name}"
    return cls(
        name=name,
        input=input_connection,
        on_success=on_success,
        on_error=on_error,
        options=options,
        **kwargs,
    )


def _gate_settings(cls: Any, /, **kwargs: Any) -> Any:
    """Build GateSettings with default input for legacy tests."""
    input_connection = kwargs.pop("input", _AUTO_GATE_INPUT)
    return cls(input=input_connection, **kwargs)


def _aggregation_settings(cls: Any, /, **kwargs: Any) -> Any:
    """Build AggregationSettings from legacy call sites."""
    options = dict(kwargs.pop("options", {}) or {})
    on_success = kwargs.pop("on_success", None)
    if on_success is None and "on_success" in options:
        on_success = options.pop("on_success")
    input_connection = kwargs.pop("input", _AUTO_AGG_INPUT)
    return cls(input=input_connection, on_success=on_success, options=options, **kwargs)


def _apply_explicit_success_routing(settings: Any) -> Any:
    """Inject explicit on_success routing for legacy test fixtures.

    These tests historically relied on `default_sink`. This helper migrates
    fixture settings to explicit sink routing before plugin instantiation.
    The mutation is applied to a deep copy to avoid test-to-test bleed.
    """
    routed_settings = settings.model_copy(deep=True)

    sinks = routed_settings.sinks
    if not sinks:
        return routed_settings
    sink_name = next(iter(sinks.keys()))

    transforms = list(routed_settings.transforms)
    gates = list(routed_settings.gates)
    aggregations = list(routed_settings.aggregations)

    source_on_success = routed_settings.source.on_success
    if source_on_success == _AUTO_SOURCE_ON_SUCCESS:
        if transforms or gates or aggregations:
            source_on_success = "source_out"
        else:
            source_on_success = sink_name

    updated_transforms = []
    previous_connection = source_on_success
    for _index, transform in enumerate(transforms):
        input_connection = transform.input
        if input_connection.startswith(_AUTO_TRANSFORM_INPUT_PREFIX):
            input_connection = previous_connection

        # on_success is always explicit — no inference needed.
        updated_transform = transform.model_copy(update={"input": input_connection})
        updated_transforms.append(updated_transform)
        previous_connection = transform.on_success

    updated_aggregations: list[Any] = []
    for index, aggregation in enumerate(aggregations):
        input_connection = aggregation.input
        if input_connection == _AUTO_AGG_INPUT:
            if index == 0:
                if updated_transforms:
                    input_connection = updated_transforms[-1].on_success
                else:
                    input_connection = source_on_success
            else:
                input_connection = updated_aggregations[index - 1].name

        on_success = aggregation.on_success
        if on_success is None:
            on_success = sink_name

        updated_aggregations.append(
            aggregation.model_copy(
                update={
                    "input": input_connection,
                    "on_success": on_success,
                }
            )
        )

    upstream_for_gates = source_on_success
    if updated_transforms:
        upstream_for_gates = updated_transforms[-1].on_success
    elif updated_aggregations:
        upstream_for_gates = updated_aggregations[-1].on_success

    updated_gates: list[Any] = []
    for index, gate in enumerate(gates):
        input_connection = gate.input
        if input_connection == _AUTO_GATE_INPUT:
            if index == 0:
                input_connection = upstream_for_gates
            else:
                prev_gate = updated_gates[index - 1]
                # Check if the previous gate has a named output connection
                # (i.e., at least one route was renamed from "continue" to a
                # connection name). If all routes are "continue", the DAG model
                # resolves them at runtime; the next gate shares the same input.
                prev_named_outputs = [
                    d for d in prev_gate.routes.values() if d != "continue" and d not in {str(s) for s in sinks} and d != "fork"
                ]
                if prev_named_outputs:
                    input_connection = prev_named_outputs[0]
                else:
                    input_connection = prev_gate.input

        routes = dict(gate.routes)
        if index < len(gates) - 1:
            # Rename lone "continue" routes to named connections so the next gate
            # can consume them. When multiple routes all say "continue", keep them
            # as-is — the DAG resolves "continue" at runtime.
            continue_count = sum(1 for d in routes.values() if d == "continue")
            if continue_count <= 1:
                routes = {
                    label: (f"{gate.name}_out" if destination == "continue" else destination) for label, destination in routes.items()
                }

        updated_gates.append(
            gate.model_copy(
                update={
                    "input": input_connection,
                    "routes": routes,
                }
            )
        )

    updated_source = routed_settings.source.model_copy(update={"on_success": source_on_success})

    routed_settings = routed_settings.model_copy(
        update={
            "source": updated_source,
            "transforms": updated_transforms,
            "aggregations": updated_aggregations,
            "gates": updated_gates,
        }
    )

    if routed_settings.coalesce and routed_settings.gates:
        branch_to_gate_idx: dict[str, int] = {}
        for gate_idx, gate in enumerate(routed_settings.gates):
            if gate.fork_to:
                for branch in gate.fork_to:
                    branch_to_gate_idx[branch] = gate_idx

        updated_coalesce = []
        for coalesce_cfg in routed_settings.coalesce:
            produced_idxs = [branch_to_gate_idx[b] for b in coalesce_cfg.branches if b in branch_to_gate_idx]
            if produced_idxs and max(produced_idxs) == len(routed_settings.gates) - 1 and coalesce_cfg.on_success is None:
                updated_coalesce.append(coalesce_cfg.model_copy(update={"on_success": sink_name}))
            else:
                updated_coalesce.append(coalesce_cfg)

        routed_settings = routed_settings.model_copy(update={"coalesce": updated_coalesce})

    return routed_settings


def instantiate_plugins_from_config_raw(settings: Any) -> _PluginInstances:
    """Instantiate plugins without any test-time routing injection."""
    from elspeth.cli_helpers import instantiate_plugins_from_config as _real_instantiate

    return cast(_PluginInstances, _real_instantiate(settings))


def instantiate_plugins_from_config(settings: Any) -> _PluginInstances:
    """Centralized test factory wrapper for plugin instantiation."""
    routed_settings = _apply_explicit_success_routing(settings)
    # Back-patch routed settings onto the original settings object so callers
    # that pass settings.coalesce or settings.gates to from_plugin_instances()
    # get the routed values (with auto-inputs resolved).
    if routed_settings.coalesce != settings.coalesce:
        object.__setattr__(settings, "coalesce", routed_settings.coalesce)
    if list(routed_settings.gates) != list(settings.gates):
        object.__setattr__(settings, "gates", routed_settings.gates)
    return instantiate_plugins_from_config_raw(routed_settings)


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

    def test_pipeline_sequence_includes_route_labeled_processing_targets(self) -> None:
        """Fallback sequence discovery must include MOVE edges beyond 'continue'."""
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.contracts.types import NodeID
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("router", node_type=NodeType.GATE, plugin_name="router")
        graph.add_node("branch_t", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json")

        graph.add_edge("source", "router", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("router", "branch_t", label="high", mode=RoutingMode.MOVE)
        graph.add_edge("branch_t", "sink", label="on_success", mode=RoutingMode.MOVE)

        sequence = graph.get_pipeline_node_sequence()
        assert sequence == [NodeID("router"), NodeID("branch_t")]

        step_map = graph.build_step_map()
        assert step_map[NodeID("source")] == 0
        assert step_map[NodeID("router")] == 1
        assert step_map[NodeID("branch_t")] == 2


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

    def test_validate_rejects_disconnected_graph(self) -> None:
        """Graphs with unreachable nodes must be rejected.

        If a node cannot be reached from the source, it will never execute.
        This is either a configuration error or indicates orphaned nodes.
        """
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="process")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_node("orphan", node_type=NodeType.TRANSFORM, plugin_name="abandoned")  # Not connected!

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")
        # orphan has no incoming edges - unreachable

        with pytest.raises(GraphValidationError, match=r"unreachable|disconnected"):
            graph.validate()

    def test_validate_rejects_self_loop(self) -> None:
        """Self-loops (node with edge to itself) must be rejected.

        A node cannot route to itself - this would create infinite recursion.
        Should be caught as a special case of cycle detection.
        """
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="loop")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "transform", label="retry")  # Self-loop!
        graph.add_edge("transform", "sink", label="continue")

        with pytest.raises(GraphValidationError, match=r"cycle|self-loop"):
            graph.validate()

    def test_validate_allows_diamond_merge(self) -> None:
        """Diamond patterns (node with multiple incoming edges) should be allowed.

        A node can have multiple incoming edges as long as the graph is acyclic.
        Common pattern: fork → parallel processing → merge.
        """
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork_gate")
        graph.add_node("path_a", node_type=NodeType.TRANSFORM, plugin_name="fast")
        graph.add_node("path_b", node_type=NodeType.TRANSFORM, plugin_name="slow")
        graph.add_node("merge", node_type=NodeType.COALESCE, plugin_name="merge_results")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "path_a", label="path_a")
        graph.add_edge("gate", "path_b", label="path_b")
        graph.add_edge("path_a", "merge", label="continue")  # First incoming edge
        graph.add_edge("path_b", "merge", label="continue")  # Second incoming edge (diamond!)
        graph.add_edge("merge", "sink", label="continue")

        # Should not raise - diamond merge is valid
        graph.validate()


class TestSchemaContractValidation:
    """Schema contract validation during DAG construction."""

    def test_unsatisfiable_contract_dependencies_detected(self) -> None:
        """Impossible contract dependencies should be detected during validation.

        Scenario: Transform B requires field from Transform A's output, but A comes
        AFTER B in the pipeline (impossible to satisfy).

        This documents the "circular contract dependencies" edge case: when contract
        requirements create a logical impossibility in a linear pipeline.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Build a linear pipeline: source → transform_b → transform_a → sink
        # But transform_b requires a field that only transform_a produces (impossible!)

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="requires_a_output",
            config={"required_input_fields": ["field_from_a"]},  # Requires A's output
        )
        graph.add_node(
            "transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="produces_output",
            output_schema_config=SchemaConfig.from_dict({"mode": "observed", "guaranteed_fields": ["field_from_a"]}),  # Produces the field
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Pipeline order means B can't see A's output
        graph.add_edge("source", "transform_b", label="continue")
        graph.add_edge("transform_b", "transform_a", label="continue")
        graph.add_edge("transform_a", "sink", label="continue")

        # Should raise during edge validation - B requires field that isn't available
        with pytest.raises(ValueError, match=r"required field|missing field|unsatisfied"):
            graph.validate_edge_compatibility()


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
        effective_schema = graph.get_effective_producer_schema("gate")

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
            graph.get_effective_producer_schema("gate")

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
        effective_schema = graph.get_effective_producer_schema("gate2")

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

        effective_schema = graph.get_effective_producer_schema("transform")

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
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        # Should have: source -> output_sink
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.get_source() is not None
        assert len(graph.get_sinks()) == 1

    def test_from_config_is_valid(self, plugin_manager) -> None:
        """Graph from valid config passes validation."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        # Should not raise
        graph.validate()
        assert graph.is_acyclic()

    def test_from_config_with_transforms(self, plugin_manager) -> None:
        """Build graph with transform chain."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="passthrough_0",
                    plugin="passthrough",
                    on_success="to_field_mapper",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="field_mapper_0",
                    plugin="field_mapper",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
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

    def test_pydantic_rejects_missing_on_success_on_transform(self, plugin_manager) -> None:
        """on_success is required on TransformSettings — omitting it is a Pydantic error."""
        from pydantic import ValidationError

        from elspeth.core.config import TransformSettings

        with pytest.raises(ValidationError, match="on_success"):
            TransformSettings(
                name="passthrough_0",
                plugin="passthrough",
                input="source_out",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
            )

    def test_non_terminal_transform_with_on_success_builds_when_properly_wired(self, plugin_manager) -> None:
        """Non-terminal transforms with explicit wired connections build successfully.

        In the new WiredTransform system, every transform declares input/on_success
        connections. This is not an error - it's how connection matching works.
        """
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings, TransformSettings
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="passthrough_0",
                    plugin="passthrough",
                    input="source_out",
                    on_success="conn_0_1",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="passthrough_1",
                    plugin="passthrough",
                    input="conn_0_1",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        graph.validate()

    def test_terminal_transform_on_success_unknown_sink_raises(self, plugin_manager) -> None:
        """Terminal transform on_success must reference a configured sink."""
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings, TransformSettings
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "on_success": "source_out",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            transforms=[
                _transform_settings(
                    TransformSettings,
                    plugin="passthrough",
                    input="source_out",
                    on_success="nowhere",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"on_success 'nowhere' is neither a sink nor a known connection"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
            )

    def test_terminal_transform_with_valid_on_success_passes(self, plugin_manager) -> None:
        """Terminal transform with valid on_success builds successfully."""
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings, TransformSettings
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "on_success": "source_out",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            transforms=[
                _transform_settings(
                    TransformSettings,
                    plugin="passthrough",
                    input="source_out",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        graph.validate()

    def test_non_terminal_without_on_success_and_terminal_with_on_success_passes(self, plugin_manager) -> None:
        """Non-terminal transform omitted on_success, terminal declares it."""
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings, TransformSettings
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "on_success": "source_out",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="passthrough_0",
                    plugin="passthrough",
                    input="source_out",
                    on_success="conn_pt_fm",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="field_mapper_0",
                    plugin="field_mapper",
                    input="conn_pt_fm",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        graph.validate()

    def test_transform_before_aggregation_not_treated_as_terminal(self, plugin_manager) -> None:
        """A transform before an aggregation is non-terminal and must not require on_success."""
        from elspeth.core.config import (
            AggregationSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
            TriggerConfig,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
            transforms=[
                _transform_settings(
                    TransformSettings,
                    plugin="passthrough",
                    on_success="to_agg",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            aggregations=[
                _aggregation_settings(
                    AggregationSettings,
                    name="batch_stats",
                    plugin="batch_stats",
                    trigger=TriggerConfig(count=10),
                    output_mode="transform",
                    options={
                        "value_field": "value",
                        "schema": {"mode": "observed"},
                        "on_success": "output",
                    },
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        graph.validate()

    def test_from_config_with_gate_routes(self, plugin_manager) -> None:
        """Build graph with config-driven gate routing to multiple sinks."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="safety_gate",
                    condition="row['suspicious'] == True",
                    routes={"true": "flagged", "false": "results"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        # Should have:
        #   source -> safety_gate -> results (via "continue")
        #                         -> flagged (via "suspicious")
        assert graph.node_count == 4  # source, config_gate, results, flagged
        # Edges: source->gate, gate->results (continue), gate->flagged (route)
        assert graph.edge_count == 3

    def test_from_config_validates_route_targets(self, plugin_manager) -> None:
        """Config gate routes must reference existing sinks."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
            gates=[
                _gate_settings(
                    GateSettings,
                    name="bad_gate",
                    condition="True",
                    routes={"true": "nonexistent_sink", "false": "output"},
                ),
            ],
        )

        with pytest.raises(GraphValidationError) as exc_info:
            plugins = instantiate_plugins_from_config(config)
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
            )

        assert "nonexistent_sink" in str(exc_info.value)

    def test_get_sink_id_map(self, plugin_manager) -> None:
        """Get explicit sink_name -> node_id mapping."""
        from elspeth.contracts import SinkName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        sink_map = graph.get_sink_id_map()

        # Explicit mapping - no substring matching
        assert SinkName("results") in sink_map
        assert SinkName("flagged") in sink_map
        assert sink_map[SinkName("results")] != sink_map[SinkName("flagged")]

    def test_get_transform_id_map(self, plugin_manager) -> None:
        """Get explicit sequence -> node_id mapping for transforms."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="passthrough_0",
                    plugin="passthrough",
                    on_success="to_field_mapper",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="field_mapper_0",
                    plugin="field_mapper",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        transform_map = graph.get_transform_id_map()

        # Explicit mapping by sequence position
        assert 0 in transform_map  # passthrough
        assert 1 in transform_map  # field_mapper
        assert transform_map[0] != transform_map[1]

    def test_pipeline_traversal_methods_linear(self, plugin_manager) -> None:
        """Traversal helpers should follow continue edges over processing nodes."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="passthrough_0",
                    plugin="passthrough",
                    on_success="to_field_mapper",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="field_mapper_0",
                    plugin="field_mapper",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        sequence = graph.get_pipeline_node_sequence()
        assert len(sequence) == 2
        assert graph.get_first_transform_node() == sequence[0]
        assert graph.get_next_node(sequence[0]) == sequence[1]
        assert graph.get_next_node(sequence[1]) is None

    def test_build_step_map_matches_schema_contract(self, plugin_manager) -> None:
        """Step numbering must satisfy Landscape DB schema contract (UniqueConstraint on token_id, step_index, attempt)."""
        from elspeth.contracts.types import GateName, SinkName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="passthrough_0",
                    plugin="passthrough",
                    on_success="pt0_out",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="passthrough_1",
                    plugin="passthrough",
                    on_success="pt1_out",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="field_mapper_0",
                    plugin="field_mapper",
                    on_success="to_gates",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            gates=[
                _gate_settings(GateSettings, name="g1", condition="True", routes={"true": "g2_in", "false": "flagged"}),
                _gate_settings(GateSettings, name="g2", input="g2_in", condition="True", routes={"true": "output", "false": "output"}),
            ],
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        step_map = graph.build_step_map()
        source_id = graph.get_source()
        assert source_id is not None
        assert step_map[source_id] == 0

        transform_map = graph.get_transform_id_map()
        assert step_map[transform_map[0]] == 1
        assert step_map[transform_map[1]] == 2
        assert step_map[transform_map[2]] == 3

        config_gate_map = graph.get_config_gate_id_map()
        assert step_map[config_gate_map[GateName("g1")]] == 4
        assert step_map[config_gate_map[GateName("g2")]] == 5

        expected_step_map = {
            source_id: 0,
            transform_map[0]: 1,
            transform_map[1]: 2,
            transform_map[2]: 3,
            config_gate_map[GateName("g1")]: 4,
            config_gate_map[GateName("g2")]: 5,
        }
        assert step_map == expected_step_map

        sink_id = graph.get_sink_id_map()[SinkName("output")]
        assert graph.is_sink_node(sink_id) is True
        assert graph.is_sink_node(transform_map[0]) is False
        from elspeth.contracts.types import NodeID

        with pytest.raises(KeyError, match="Node not found"):
            graph.is_sink_node(NodeID("does_not_exist"))

    def test_get_terminal_sink_map_for_source_only(self, plugin_manager) -> None:
        """Source-only graph exposes terminal sink mapping (no test-time injection)."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "on_success": "results",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        terminal_map = graph.get_terminal_sink_map()
        source_node = graph.get_source()
        assert source_node is not None
        assert terminal_map[source_node] == "results"

        bad_config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "on_success": "missing_sink",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"mode": "observed"}}),
            },
        )
        bad_plugins = instantiate_plugins_from_config_raw(bad_config)
        with pytest.raises(GraphValidationError, match=r"Source 'csv' on_success 'missing_sink' is neither a sink nor a known connection"):
            ExecutionGraph.from_plugin_instances(
                source=bad_plugins["source"],
                source_settings=bad_plugins["source_settings"],
                transforms=bad_plugins["transforms"],
                sinks=bad_plugins["sinks"],
                aggregations=bad_plugins["aggregations"],
                gates=list(bad_config.gates),
            )


class TestGateConnectionRouteMaterialization:
    """Gate routes to named connections should wire explicit labeled edges."""

    def test_gate_connection_route_creates_labeled_edge(self, plugin_manager) -> None:
        """A gate route targeting a named connection creates a labeled MOVE edge."""
        from elspeth.contracts.types import GateName, NodeID
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsModel,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            gates=[
                GateSettingsModel(
                    name="router",
                    input="source_out",
                    condition="True",
                    routes={"true": "checker_in", "false": "flagged"},
                ),
                GateSettingsModel(
                    name="checker",
                    input="checker_in",
                    condition="True",
                    routes={"true": "output", "false": "flagged"},
                ),
            ],
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        router_id = graph.get_config_gate_id_map()[GateName("router")]
        checker_id = graph.get_config_gate_id_map()[GateName("checker")]

        edges = graph.get_edges()
        true_edges = [e for e in edges if NodeID(e.from_node) == router_id and e.to_node == checker_id and e.label == "true"]
        assert len(true_edges) == 1, f"Expected 1 'true' edge from router to checker, got {len(true_edges)}"
        continue_edges = [e for e in edges if NodeID(e.from_node) == router_id and e.to_node == checker_id and e.label == "continue"]
        assert len(continue_edges) == 1, f"Expected 1 'continue' edge from router to checker, got {len(continue_edges)}"

    def test_gate_converging_connection_routes_preserve_all_labels(self, plugin_manager) -> None:
        """Converging gate labels to one connection should materialize all route edges."""
        from elspeth.contracts import RouteDestinationKind
        from elspeth.contracts.types import GateName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsModel,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            gates=[
                GateSettingsModel(
                    name="router",
                    input="source_out",
                    condition="True",
                    routes={"true": "checker_in", "false": "checker_in"},
                ),
                GateSettingsModel(
                    name="checker",
                    input="checker_in",
                    condition="True",
                    routes={"true": "output", "false": "output"},
                ),
            ],
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        gate_ids = graph.get_config_gate_id_map()
        router_id = gate_ids[GateName("router")]
        checker_id = gate_ids[GateName("checker")]

        edges = [edge for edge in graph.get_edges() if edge.from_node == router_id and edge.to_node == checker_id]
        assert len(edges) == 3
        assert {edge.label for edge in edges} == {"true", "false", "continue"}

        route_map = graph.get_route_resolution_map()
        for route_label in ("true", "false"):
            destination = route_map[(router_id, route_label)]
            assert destination.kind == RouteDestinationKind.PROCESSING_NODE
            assert destination.next_node_id == checker_id

    def test_gate_connection_route_preserves_route_resolution(self, plugin_manager) -> None:
        """Named-connection routes resolve to processing nodes; sink routes resolve to sinks."""
        from elspeth.contracts import RouteDestination, RouteDestinationKind
        from elspeth.contracts.types import GateName, SinkName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsModel,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            gates=[
                GateSettingsModel(
                    name="router",
                    input="source_out",
                    condition="True",
                    routes={"true": "checker_in", "false": "flagged"},
                ),
                GateSettingsModel(
                    name="checker",
                    input="checker_in",
                    condition="True",
                    routes={"true": "output", "false": "flagged"},
                ),
            ],
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        router_id = graph.get_config_gate_id_map()[GateName("router")]
        resolution_map = graph.get_route_resolution_map()

        # "false" -> sink
        assert (router_id, "false") in resolution_map
        assert resolution_map[(router_id, "false")] == RouteDestination.sink(SinkName("flagged"))

        # "true" -> processing_node (the checker gate)
        assert (router_id, "true") in resolution_map
        assert resolution_map[(router_id, "true")].kind == RouteDestinationKind.PROCESSING_NODE

    def test_terminal_gate_unconsumed_connection_rejected(self, plugin_manager) -> None:
        """A terminal gate route to an unconsumed connection fails validation."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsModel,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            gates=[
                GateSettingsModel(
                    name="terminal_gate",
                    input="source_out",
                    condition="True",
                    routes={"true": "output", "false": "orphan_conn"},
                ),
            ],
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"neither a sink nor a known connection name"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
            )


class TestExecutionGraphRouteMapping:
    """Test route label <-> sink name mapping for edge lookup."""

    def test_get_route_label_for_sink(self, plugin_manager) -> None:
        """Get route label that leads to a sink from a config gate."""
        from elspeth.contracts import GateName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="classifier",
                    condition="row['suspicious'] == True",
                    routes={"true": "flagged", "false": "results"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        # Get the config gate's node_id
        gate_node_id = graph.get_config_gate_id_map()[GateName("classifier")]

        # Given gate node and sink name, get the route label
        route_label = graph.get_route_label(gate_node_id, "flagged")

        assert route_label == "true"

    def test_get_route_label_for_continue(self, plugin_manager) -> None:
        """Non-terminal continue routes return 'continue' as label."""
        from elspeth.contracts import GateName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "results": SinkSettings(plugin="json", options={"path": "results.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="gate1",
                    condition="True",
                    routes={"true": "gate1_pass", "false": "flagged"},
                ),
                _gate_settings(
                    GateSettings,
                    input="gate1_pass",
                    name="gate2",
                    condition="True",
                    routes={"true": "results", "false": "results"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        gate_node_id = graph.get_config_gate_id_map()[GateName("gate1")]

        # gate1 reaches results via a continue edge to gate2
        route_label = graph.get_route_label(gate_node_id, "results")
        assert route_label == "continue"

    def test_hyphenated_sink_names_work_in_dag(self, plugin_manager) -> None:
        """Gate routing to hyphenated sink names works correctly.

        Regression test for gate-route-destination-name-validation-mismatch bug.
        Sink names don't need to match identifier pattern - they're just dict keys.
        """
        from elspeth.contracts import GateName, SinkName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output-sink": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "quarantine-bucket": SinkSettings(plugin="json", options={"path": "quarantine.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="quality_check",
                    condition="row['score'] >= 0.5",
                    routes={"true": "output-sink", "false": "quarantine-bucket"},
                ),
            ],
        )

        # DAG compilation should succeed with hyphenated sink names
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
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
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "path_a": SinkSettings(plugin="json", options={"path": "path_a.json", "schema": {"mode": "observed"}}),
                "path_b": SinkSettings(plugin="json", options={"path": "path_b.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="fork_gate",
                    condition="True",  # Always forks
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
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
        from elspeth.contracts import CoalesceName, NodeType
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
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
        from elspeth.contracts import CoalesceName, GateName, RoutingMode
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
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
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "path_a": SinkSettings(plugin="json", options={"path": "path_a.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_a"],  # Duplicate branch name
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)

        with pytest.raises(GraphValidationError, match=r"duplicate fork branches"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                coalesce_settings=settings.coalesce,
            )

    def test_partial_branch_coverage_branches_not_in_coalesce_route_to_sink(
        self,
        plugin_manager,
    ) -> None:
        """Fork branches not in any coalesce should still route to output sink."""
        from elspeth.contracts import CoalesceName, GateName, SinkName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "path_c": SinkSettings(plugin="json", options={"path": "path_c.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
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
        from elspeth.contracts import CoalesceName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
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
        from elspeth.contracts import BranchName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )
        branch_map = graph.get_branch_to_coalesce_map()

        # Should map branches to coalesce_name (not node_id) for processor node-map lookup
        assert branch_map[BranchName("path_a")] == "merge_results"
        assert branch_map[BranchName("path_b")] == "merge_results"

    def test_branch_to_coalesce_maps_to_coalesce_name_for_node_lookup(self, plugin_manager) -> None:
        """branch_to_coalesce should map to coalesce_name (not node_id) for coalesce node lookup.

        BUG-LINEAGE-01: The processor needs to look up coalesce node position using:
            coalesce_name = branch_to_coalesce[branch_name]
            coalesce_node_id = coalesce_node_map[coalesce_name]

        But branch_to_coalesce was mapping branch_name -> node_id, causing KeyError
        when trying to look up in coalesce_node_map which expects coalesce_name keys.
        """
        from elspeth.contracts import BranchName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        branch_map = graph.get_branch_to_coalesce_map()

        # CRITICAL: Must map to coalesce_name (not node_id) for processor node lookup
        # The processor does: coalesce_node_map[branch_to_coalesce[branch_name]]
        # coalesce_node_map has keys like "join_point", NOT node_ids like "coalesce_join_point_abc123"
        assert branch_map[BranchName("analysis_path")] == "join_point", (
            f"Expected coalesce_name 'join_point', got {branch_map[BranchName('analysis_path')]}"
        )
        assert branch_map[BranchName("validation_path")] == "join_point", (
            f"Expected coalesce_name 'join_point', got {branch_map[BranchName('validation_path')]}"
        )

    def test_duplicate_branch_names_across_coalesces_rejected(self, plugin_manager) -> None:
        """Duplicate branch names across coalesce settings should be rejected."""
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
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
                source=_source_settings(
                    SourceSettings,
                    plugin="csv",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                ),
                sinks={
                    "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                },
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
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "path_b": SinkSettings(plugin="json", options={"path": "path_b.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                coalesce_settings=settings.coalesce,
            )

    def test_fork_coalesce_contract_branch_map_compatible_with_node_map(self, plugin_manager) -> None:
        """Contract test: branch_to_coalesce values must be usable as coalesce_node_map keys.

        This is the CRITICAL contract between DAG builder and Processor.
        The processor does: coalesce_node_map[branch_to_coalesce[branch_name]]
        This test ensures the production path (`from_plugin_instances`) produces compatible mappings.
        """
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            transforms=[
                _transform_settings(
                    TransformSettings, plugin="passthrough", on_success="to_gate", options={"schema": {"mode": "observed"}}
                ),
            ],
            gates=[
                _gate_settings(
                    GateSettings,
                    name="analysis_fork",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        # Get the mappings that processor would use
        branch_to_coalesce = graph.get_branch_to_coalesce_map()
        coalesce_node_map = graph.get_coalesce_id_map()

        # CRITICAL CONTRACT: Every value in branch_to_coalesce must be a key in coalesce_node_map.
        for branch_name, coalesce_name in branch_to_coalesce.items():
            assert coalesce_name in coalesce_node_map, (
                f"Contract violation: branch_to_coalesce['{branch_name}'] = '{coalesce_name}', "
                f"but '{coalesce_name}' not in coalesce_node_map keys: {list(coalesce_node_map.keys())}"
            )

            # Also verify it's the coalesce_name, not a node_id
            assert not coalesce_name.startswith("coalesce_"), (
                f"Contract violation: branch_to_coalesce['{branch_name}'] = '{coalesce_name}' "
                f"looks like a node_id (starts with 'coalesce_'), should be coalesce name"
            )

    def test_coalesce_node_has_edge_to_output_sink(self, plugin_manager) -> None:
        """Coalesce node should have an edge to the output sink."""
        from elspeth.contracts import CoalesceName, RoutingMode, SinkName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]
        output_sink_id = graph.get_sink_id_map()[SinkName("output")]

        # Verify edge from coalesce to output sink
        edges = graph.get_edges()
        coalesce_to_sink_edges = [e for e in edges if e.from_node == coalesce_id and e.to_node == output_sink_id]

        assert len(coalesce_to_sink_edges) == 1
        assert coalesce_to_sink_edges[0].label == "on_success"
        assert coalesce_to_sink_edges[0].mode == RoutingMode.MOVE

    def test_coalesce_node_connects_to_next_gate(self, plugin_manager) -> None:
        """Coalesce node should continue to the next gate when one exists."""
        from elspeth.contracts import CoalesceName, GateName, RoutingMode
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker1",
                    input="source_out",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_b"],
                ),
                _gate_settings(
                    GateSettings,
                    name="gate2",
                    input="merge_results",
                    condition="True",
                    routes={"true": "output", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
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
        from elspeth.contracts import CoalesceName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
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
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        coalesce_id = graph.get_coalesce_id_map()[CoalesceName("merge_results")]
        node_info = graph.get_node_info(coalesce_id)

        # Verify config is stored
        assert node_info.config["branches"] == {"path_a": "path_a", "path_b": "path_b"}
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

    def test_gate_routes_to_multiple_processing_connections(self, plugin_manager) -> None:
        """Gate route labels can fan out to different downstream processing nodes."""
        from elspeth.contracts import GateName, RouteDestinationKind
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_router",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "flagged": SinkSettings(plugin="json", options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="router",
                    input="to_router",
                    condition="row['score'] > 0.5",
                    routes={"true": "high_path", "false": "low_path"},
                ),
                _gate_settings(
                    GateSettings,
                    name="high_gate",
                    input="high_path",
                    condition="True",
                    routes={"true": "output", "false": "output"},
                ),
                _gate_settings(
                    GateSettings,
                    name="low_gate",
                    input="low_path",
                    condition="True",
                    routes={"true": "flagged", "false": "flagged"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        gate_ids = graph.get_config_gate_id_map()
        router_id = gate_ids[GateName("router")]
        high_gate_id = gate_ids[GateName("high_gate")]
        low_gate_id = gate_ids[GateName("low_gate")]

        router_edges = [edge for edge in graph.get_edges() if edge.from_node == router_id]
        router_targets = {(edge.label, edge.to_node) for edge in router_edges}
        assert ("true", high_gate_id) in router_targets
        assert ("false", low_gate_id) in router_targets

        route_map = graph.get_route_resolution_map()
        true_dest = route_map[(router_id, "true")]
        false_dest = route_map[(router_id, "false")]

        assert true_dest.kind == RouteDestinationKind.PROCESSING_NODE
        assert true_dest.next_node_id == high_gate_id
        assert false_dest.kind == RouteDestinationKind.PROCESSING_NODE
        assert false_dest.next_node_id == low_gate_id

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
        config = SchemaConfig(mode="fixed", fields=fields)

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
        """Coalesce should accept branches with observed schemas from different instances.

        BUG FIX: P2-2026-01-30-coalesce-schema-identity-check
        Observed schemas (mode="observed") are compatible with anything.
        Even if they're distinct class objects, they should pass validation.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create two observed schemas (accept anything)
        config = SchemaConfig(mode="observed", fields=None)

        DynamicA = create_schema_from_config(config, "DynamicA")
        DynamicB = create_schema_from_config(config, "DynamicB")

        # Verify they are distinct class objects
        assert DynamicA is not DynamicB, "Test requires distinct class objects"

        class SourceOutput(PluginSchema):
            id: int

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")

        # Both branches use observed schemas (distinct objects)
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

        # Should PASS: observed schemas are compatible with anything
        graph.validate_edge_compatibility()

    def test_coalesce_rejects_mixed_observed_explicit_branches(self) -> None:
        """Coalesce must reject mixed observed/explicit branch schemas.

        BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected

        When one branch produces a observed schema and another produces an explicit
        schema, the coalesce's effective schema becomes the first branch's schema.
        This masks the mismatch: downstream consumers expect explicit fields that
        dynamic-branch rows may not have, causing runtime failures.

        Pre-run validation must detect and reject this mismatch.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create a observed schema (no fields, accepts anything)
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create an explicit schema with specific fields
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "fixed", "fields": ["value: float", "id: int"]}),
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

        # Should FAIL: mixed observed/explicit branches are not allowed
        with pytest.raises(ValueError, match=r"mixed.*observed.*explicit|observed.*explicit.*mismatch"):
            graph.validate_edge_compatibility()

    def test_coalesce_rejects_mixed_none_explicit_branches(self) -> None:
        """Coalesce must reject mixed None/explicit branch schemas.

        BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected

        None (unspecified output_schema) is treated as dynamic. Mixed with explicit
        schemas, this creates the same mismatch problem as observed schema classes.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create an explicit schema with specific fields
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "fixed", "fields": ["value: float", "id: int"]}),
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

        # Branch B: produces NONE (unspecified = observed schema)
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
        with pytest.raises(ValueError, match=r"mixed.*observed.*explicit|observed.*explicit.*mismatch"):
            graph.validate_edge_compatibility()

    def test_gate_rejects_mixed_observed_explicit_incoming_branches(self) -> None:
        """Gate with multiple inputs must reject mixed observed/explicit schemas.

        BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected

        Gates can receive inputs from multiple sources (e.g., in complex DAGs).
        When inputs have mixed observed/explicit schemas, the same mismatch problem
        occurs as with coalesce nodes.
        """
        from elspeth.contracts import NodeType, PluginSchema, RoutingMode
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create a observed schema
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create an explicit schema
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "fixed", "fields": ["value: float", "id: int"]}),
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

        # Transform 2 produces observed schema
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

        # Should FAIL: mixed observed/explicit inputs to gate
        with pytest.raises(ValueError, match=r"mixed.*observed.*explicit|observed.*explicit.*mismatch"):
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
                    "schema": {"mode": "fixed", "fields": ["value: float"]},
                    "value_field": "value",
                },
                "schema": {"mode": "fixed", "fields": ["value: float"]},
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
                    "schema": {"mode": "fixed", "fields": ["value: float"]},
                    "value_field": "value",
                },
                "schema": {"mode": "fixed", "fields": ["value: float"]},
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

    from elspeth.core.config import load_settings
    from elspeth.core.dag import ExecutionGraph

    config_yaml = """
source:
  plugin: csv
  on_success: source_out
  options:
    path: test.csv
    schema:
      mode: fixed
      fields:
        - "value: float"
    on_validation_failure: discard

transforms:
  - name: passthrough_0
    plugin: passthrough
    input: source_out
    on_success: output
    on_error: discard
    options:
      schema:
        mode: fixed
        fields:
          - "value: float"

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "value: float"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )

        # Verify schemas extracted
        source_id = graph.get_source()
        assert source_id is not None
        source_info = graph.get_node_info(source_id)
        assert source_info.output_schema is not None

    finally:
        config_file.unlink()


def test_from_plugin_instances_cycle_raises_graph_validation_error(monkeypatch: pytest.MonkeyPatch):
    """NetworkXUnfeasible during topo sort in from_plugin_instances yields GraphValidationError.

    Regression: nx.topological_sort at the pipeline-ordering step was unguarded,
    so a cycle raised raw NetworkXUnfeasible instead of GraphValidationError.
    The single-producer invariant prevents cycles through normal config, but
    this wrapping is defense-in-depth for future edge-building changes.
    """
    import tempfile
    from pathlib import Path

    import networkx as nx

    from elspeth.core.config import load_settings
    from elspeth.core.dag import ExecutionGraph, GraphValidationError

    config_yaml = """
source:
  plugin: csv
  on_success: step_a
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

transforms:
  - name: passthrough_0
    plugin: passthrough
    input: step_a
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: observed
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        def raise_unfeasible(g):
            raise nx.NetworkXUnfeasible("graph contains a cycle")

        monkeypatch.setattr(nx, "topological_sort", raise_unfeasible)

        with pytest.raises(GraphValidationError, match="cycle"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
    finally:
        config_file.unlink()


def test_validate_aggregation_dual_schema():
    """Verify aggregation edges validate against correct schemas."""
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.schema_factory import create_schema_from_config

    input_schema_config = {"mode": "fixed", "fields": ["value: float"]}
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict(input_schema_config),
        "InputSchema",
        allow_coercion=False,
    )

    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"mode": "fixed", "fields": ["count: int", "sum: float"]}),
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

    input_schema_config = {"mode": "fixed", "fields": ["value: float"]}
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict(input_schema_config),
        "InputSchema",
        allow_coercion=False,
    )

    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"mode": "fixed", "fields": ["count: int"]}),  # Missing 'sum'
        "OutputSchema",
        allow_coercion=False,
    )

    SinkSchema = create_schema_from_config(
        SchemaConfig.from_dict({"mode": "fixed", "fields": ["count: int", "sum: float"]}),
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
    """Tests for detecting and handling observed schemas."""

    def test_dynamic_source_to_specific_sink_should_skip_validation(self) -> None:
        """Dynamic source → specific sink should PASS (validation skipped).

        Manually constructed graph with dynamic output_schema and specific input_schema.
        Validation should be skipped for observed schemas.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create observed schema (no fields, extra='allow')
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create specific schema (has fields, extra='forbid')
        SpecificSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "fixed", "fields": ["value: float", "name: str"]}),
            "SpecificSchema",
            allow_coercion=False,
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=DynamicSchema)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=SpecificSchema)
        graph.add_edge("source", "sink", label="continue")

        # Should NOT raise - validation is skipped for observed schemas
        graph.validate()

    def test_specific_source_to_dynamic_sink_should_skip_validation(self) -> None:
        """Specific source → dynamic sink should PASS (validation skipped).

        Manually constructed graph with specific output_schema and dynamic input_schema.
        Validation should be skipped for observed schemas.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create specific schema
        SpecificSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "fixed", "fields": ["value: float"]}),
            "SpecificSchema",
            allow_coercion=False,
        )

        # Create observed schema
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SpecificSchema)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=DynamicSchema)
        graph.add_edge("source", "sink", label="continue")

        # Should NOT raise - validation is skipped for observed schemas
        graph.validate()

    def test_dynamic_schema_detection_in_validation(self) -> None:
        """Observed schema detection correctly identifies dynamic vs explicit schemas.

        Observed schemas have no fields and extra='allow', matching the detection
        logic in ExecutionGraph._get_missing_required_fields().
        """
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        # Create observed schema
        DynamicSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "DynamicSchema",
            allow_coercion=False,
        )

        # Create explicit schema
        ExplicitSchema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "fixed", "fields": ["value: float"]}),
            "ExplicitSchema",
            allow_coercion=False,
        )

        # Helper to check if schema is observed (matches logic in dag.py)
        def is_observed_schema(schema: type[PluginSchema] | None) -> bool:
            if schema is None:
                return True
            return len(schema.model_fields) == 0 and schema.model_config.get("extra") == "allow"

        # Test observed schema detection
        assert is_observed_schema(DynamicSchema) is True

        # Test explicit schema detection
        assert is_observed_schema(ExplicitSchema) is False

        # Test backwards compat (None = observed)
        assert is_observed_schema(None) is True


class TestDeterministicNodeIDs:
    """Tests for deterministic node ID generation."""

    def test_node_ids_are_deterministic_for_same_config(self) -> None:
        """Node IDs must be deterministic for checkpoint/resume compatibility."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                _transform_settings(
                    TransformSettings,
                    plugin="passthrough",
                    on_success="out",
                    options={"schema": {"mode": "observed"}},
                )
            ],
            sinks={"out": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}})},
        )

        # Build graph twice with same config
        plugins1 = instantiate_plugins_from_config(config)
        graph1 = ExecutionGraph.from_plugin_instances(
            source=plugins1["source"],
            source_settings=plugins1["source_settings"],
            transforms=plugins1["transforms"],
            sinks=plugins1["sinks"],
            aggregations=plugins1["aggregations"],
            gates=list(config.gates),
        )

        plugins2 = instantiate_plugins_from_config(config)
        graph2 = ExecutionGraph.from_plugin_instances(
            source=plugins2["source"],
            source_settings=plugins2["source_settings"],
            transforms=plugins2["transforms"],
            sinks=plugins2["sinks"],
            aggregations=plugins2["aggregations"],
            gates=list(config.gates),
        )

        # Node IDs must be identical
        nodes1 = sorted(graph1._graph.nodes())
        nodes2 = sorted(graph2._graph.nodes())

        assert nodes1 == nodes2, "Node IDs must be deterministic for checkpoint compatibility"

    def test_node_ids_change_when_config_changes(self) -> None:
        """Node IDs should change if plugin config changes."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config1 = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[],
            sinks={"out": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}})},
        )

        config2 = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "fixed", "fields": ["id: int"]},  # Different!
                },
            ),
            transforms=[],
            sinks={"out": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}})},
        )

        plugins1 = instantiate_plugins_from_config(config1)
        graph1 = ExecutionGraph.from_plugin_instances(
            source=plugins1["source"],
            source_settings=plugins1["source_settings"],
            transforms=plugins1["transforms"],
            sinks=plugins1["sinks"],
            aggregations=plugins1["aggregations"],
            gates=list(config1.gates),
        )

        plugins2 = instantiate_plugins_from_config(config2)
        graph2 = ExecutionGraph.from_plugin_instances(
            source=plugins2["source"],
            source_settings=plugins2["source_settings"],
            transforms=plugins2["transforms"],
            sinks=plugins2["sinks"],
            aggregations=plugins2["aggregations"],
            gates=list(config2.gates),
        )

        # Source node IDs should differ (different config)
        source_id_1 = next(n for n in graph1._graph.nodes() if n.startswith("source_"))
        source_id_2 = next(n for n in graph2._graph.nodes() if n.startswith("source_"))

        assert source_id_1 != source_id_2

    def test_overlong_node_name_is_rejected_at_settings_level(self) -> None:
        """Overlong transform names fail fast at settings validation.

        The _MAX_NODE_NAME_LENGTH=38 constraint on TransformSettings ensures
        generated node IDs stay within the 64-char landscape column limit.
        This is the primary protection layer; the DAG-level node_id length
        check is defense-in-depth that is unreachable with valid settings.
        """
        from pydantic import ValidationError

        from elspeth.core.config import TransformSettings

        long_name = "x" * 39  # Exceeds _MAX_NODE_NAME_LENGTH (38)
        with pytest.raises(ValidationError, match=r"exceeds max length 38"):
            TransformSettings(
                name=long_name,
                plugin="passthrough",
                input="source_out",
                on_success="out",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
            )


class TestBranchGateMap:
    """Test branch_gate_map exposure from ExecutionGraph."""

    def test_get_branch_gate_map_returns_copy(self) -> None:
        """Getter should return a copy to prevent external mutation."""
        from elspeth.contracts.types import BranchName, NodeID
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(SourceSettings, plugin="null"),
            gates=[
                _gate_settings(
                    GateSettings,
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["branch_a", "branch_b"],
                ),
            ],
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json", "schema": {"mode": "observed"}})},
            coalesce=[
                CoalesceSettings(
                    name="merge_branches",
                    branches=["branch_a", "branch_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        # Get the map
        branch_map = graph.get_branch_gate_map()

        # Verify it contains expected mappings (each branch maps to a NodeID)
        assert BranchName("branch_a") in branch_map
        assert BranchName("branch_b") in branch_map

        # Verify it's a copy (mutation doesn't affect internal state)
        original_value = branch_map[BranchName("branch_a")]
        branch_map[BranchName("branch_a")] = NodeID("fake")

        fresh_map = graph.get_branch_gate_map()
        assert fresh_map[BranchName("branch_a")] == original_value

    def test_get_branch_gate_map_empty_when_no_coalesce(self) -> None:
        """Getter returns empty dict when no coalesce configured."""
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=_source_settings(SourceSettings, plugin="null"),
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json", "schema": {"mode": "observed"}})},
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        branch_map = graph.get_branch_gate_map()
        assert branch_map == {}


class TestDivertEdges:
    """Tests for quarantine and error divert edges in graph construction.

    These tests use the production factory path (instantiate_plugins_from_config)
    per the Test Path Integrity rule. Divert edges make quarantine/error sinks
    reachable in the graph without participating in normal DAG traversal.
    """

    def _build_graph(self, settings: ElspethSettings) -> ExecutionGraph:
        """Build ExecutionGraph via production factory path."""
        from elspeth.core.dag import ExecutionGraph

        plugins = instantiate_plugins_from_config(settings)
        return ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
        )

    def test_source_quarantine_edge_created(self, plugin_manager) -> None:
        """Source with on_validation_failure creates a divert edge to quarantine sink."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "quarantine",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "default": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}}),
                "quarantine": SinkSettings(plugin="json", options={"path": "quar.json", "schema": {"mode": "observed"}}),
            },
        )
        graph = self._build_graph(settings)
        graph.validate()

        edges = graph.get_edges()
        divert_edges = [e for e in edges if e.mode == RoutingMode.DIVERT]
        assert len(divert_edges) == 1
        assert divert_edges[0].label == "__quarantine__"

    def test_source_discard_no_divert_edge(self, plugin_manager) -> None:
        """Source with on_validation_failure='discard' creates no divert edge."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"default": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}})},
        )
        graph = self._build_graph(settings)
        graph.validate()

        edges = graph.get_edges()
        divert_edges = [e for e in edges if e.mode == RoutingMode.DIVERT]
        assert len(divert_edges) == 0

    def test_transform_error_edge_created(self, plugin_manager) -> None:
        """Transform with on_error creates a divert edge to error sink."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                _transform_settings(
                    TransformSettings,
                    plugin="passthrough",
                    on_success="default",
                    on_error="errors",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "default": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}}),
                "errors": SinkSettings(plugin="json", options={"path": "err.json", "schema": {"mode": "observed"}}),
            },
        )
        graph = self._build_graph(settings)
        graph.validate()

        edges = graph.get_edges()
        divert_edges = [e for e in edges if e.mode == RoutingMode.DIVERT]
        assert len(divert_edges) == 1
        assert divert_edges[0].label.startswith("__error_") and divert_edges[0].label.endswith("__")

    def test_quarantine_and_error_both_present(self, plugin_manager) -> None:
        """Both quarantine and error divert edges coexist."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "quarantine",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                _transform_settings(
                    TransformSettings,
                    plugin="passthrough",
                    on_success="default",
                    on_error="errors",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "default": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}}),
                "quarantine": SinkSettings(plugin="json", options={"path": "quar.json", "schema": {"mode": "observed"}}),
                "errors": SinkSettings(plugin="json", options={"path": "err.json", "schema": {"mode": "observed"}}),
            },
        )
        graph = self._build_graph(settings)
        graph.validate()

        edges = graph.get_edges()
        divert_edges = [e for e in edges if e.mode == RoutingMode.DIVERT]
        assert len(divert_edges) == 2

    def test_quarantine_to_default_sink_creates_divert_edge(self, plugin_manager) -> None:
        """If quarantine destination is default sink, divert edge still created.

        Verifies that schema validation still runs for the normal continue
        edge to default sink (the DIVERT skip is per-edge, not per-node-pair).
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "default",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"default": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}})},
        )
        graph = self._build_graph(settings)
        graph.validate()

        edges = graph.get_edges()
        divert_edges = [e for e in edges if e.mode == RoutingMode.DIVERT]
        assert len(divert_edges) == 1
        # Normal continue edge should also exist
        normal_edges = [e for e in edges if e.mode != RoutingMode.DIVERT]
        assert any(e.label == "on_success" for e in normal_edges)

    def test_multiple_transforms_share_error_sink(self, plugin_manager) -> None:
        """Multiple transforms can route errors to the same sink."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )

        settings = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="pt_a",
                    plugin="passthrough",
                    on_success="pt_a_out",
                    on_error="errors",
                    options={"schema": {"mode": "observed"}},
                ),
                _transform_settings(
                    TransformSettings,
                    name="pt_b",
                    plugin="passthrough",
                    on_success="default",
                    on_error="errors",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "default": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}}),
                "errors": SinkSettings(plugin="json", options={"path": "err.json", "schema": {"mode": "observed"}}),
            },
        )
        graph = self._build_graph(settings)
        graph.validate()

        edges = graph.get_edges()
        divert_edges = [e for e in edges if e.mode == RoutingMode.DIVERT]
        error_edges = [e for e in divert_edges if e.label.startswith("__error_")]
        assert len(error_edges) == 2
        # With auto-generated names, error labels include the transform name hash
        assert len({e.label for e in error_edges}) == 2
        assert all(e.label.startswith("__error_") and e.label.endswith("__") for e in error_edges)


class TestTerminalGateRouteValidation:
    """Terminal gates must not emit unconsumed connection outputs."""

    def test_terminal_gate_with_unconsumed_connection_raises(self, plugin_manager) -> None:
        """Terminal gate route to an orphan connection raises GraphValidationError."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_gate",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="threshold",
                    input="to_gate",
                    condition="row['score'] > 0.5",
                    routes={"true": "output", "false": "orphan_conn"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"neither a sink nor a known connection name"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
            )

    def test_non_terminal_gate_with_connection_route_passes(self, plugin_manager) -> None:
        """Non-terminal gate with explicit connection route builds successfully.

        Config gates are positioned AFTER transforms in the pipeline. A gate is
        non-terminal when another config gate follows it. The first gate routes
        to the second through a named connection.
        """
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_gates",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
                "flagged": SinkSettings(
                    plugin="json",
                    options={"path": "flagged.json", "schema": {"mode": "observed"}},
                ),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="first_gate",
                    input="to_gates",
                    condition="row['score'] > 0.5",
                    routes={"true": "flagged", "false": "to_second"},
                ),
                _gate_settings(
                    GateSettings,
                    name="second_gate",
                    input="to_second",
                    condition="True",
                    routes={"true": "output", "false": "output"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        graph.validate()

    def test_gate_route_to_transform_connection_passes(self, plugin_manager) -> None:
        """Gate routes can target downstream transforms via connection names."""
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.types import GateName
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_gate",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                _transform_settings(
                    TransformSettings,
                    name="after_gate_transform",
                    plugin="passthrough",
                    input="after_gate",
                    on_success="output",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="router",
                    input="to_gate",
                    condition="row['score'] > 0.5",
                    routes={"true": "output", "false": "after_gate"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        graph.validate()

        gate_id = graph.get_config_gate_id_map()[GateName("router")]
        transform_id = graph.get_transform_id_map()[0]
        assert any(
            edge.from_node == gate_id and edge.to_node == transform_id and edge.label == "false" and edge.mode == RoutingMode.MOVE
            for edge in graph.get_edges()
        )


class TestAggregationOnSuccessValidation:
    """wp68: Aggregation on_success routing validation (3 error paths).

    Three error paths in _validate_aggregation_on_success_routing:
    - dag.py:210: Terminal aggregation missing on_success
    - dag.py:217: Aggregation on_success references unknown sink
    - dag.py:228: Non-terminal aggregation with on_success set
    """

    def test_terminal_aggregation_missing_on_success_raises(self, plugin_manager) -> None:
        """Terminal aggregation without on_success raises GraphValidationError."""
        from elspeth.core.config import (
            AggregationSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TriggerConfig,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_agg",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            aggregations=[
                _aggregation_settings(
                    AggregationSettings,
                    name="batch_stats",
                    plugin="batch_stats",
                    input="to_agg",
                    trigger=TriggerConfig(count=10),
                    output_mode="transform",
                    options={
                        "value_field": "value",
                        "schema": {"mode": "observed"},
                        # No on_success — terminal aggregation must have it
                    },
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"Dangling output connections"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
            )

    def test_terminal_aggregation_unknown_sink_raises(self, plugin_manager) -> None:
        """Terminal aggregation on_success referencing unknown sink raises GraphValidationError."""
        from elspeth.core.config import (
            AggregationSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TriggerConfig,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_agg",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            aggregations=[
                _aggregation_settings(
                    AggregationSettings,
                    name="batch_stats",
                    plugin="batch_stats",
                    input="to_agg",
                    trigger=TriggerConfig(count=10),
                    output_mode="transform",
                    options={
                        "value_field": "value",
                        "schema": {"mode": "observed"},
                        "on_success": "nonexistent_sink",
                    },
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"on_success 'nonexistent_sink' is neither a sink nor a known connection"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
            )

    def test_non_terminal_aggregation_with_on_success_routes_to_sink(self, plugin_manager) -> None:
        """Aggregation with on_success pointing to a sink routes correctly.

        In the new WiredTransform system, aggregation on_success creates a terminal
        edge to the sink. The gate separately consumes from its own input connection.
        """
        from elspeth.core.config import (
            AggregationSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TriggerConfig,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_agg",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            aggregations=[
                _aggregation_settings(
                    AggregationSettings,
                    name="batch_stats",
                    plugin="batch_stats",
                    input="to_agg",
                    trigger=TriggerConfig(count=10),
                    output_mode="transform",
                    options={
                        "value_field": "value",
                        "schema": {"mode": "observed"},
                        "on_success": "output",
                    },
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )
        graph.validate()


class TestConnectionNamespaceValidation:
    """Unit tests for _validate_connection_namespaces invariants."""

    def test_sink_and_connection_name_collision_raises(self) -> None:
        """Connection namespace must be disjoint from sink namespace."""
        from elspeth.contracts.types import NodeID
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        with pytest.raises(GraphValidationError, match="overlap with sink names"):
            ExecutionGraph._validate_connection_namespaces(
                producers={"shared_name": (NodeID("transform-1"), "continue")},
                consumers={"shared_name": NodeID("transform-2")},
                consumer_claims=[("shared_name", NodeID("transform-2"), "transform 'downstream'")],
                sink_names={"shared_name"},
                check_dangling=False,
            )


class TestCoalesceOnSuccessValidation:
    """xe91: Coalesce on_success routing validation (3 error paths).

    Three error paths in coalesce wiring (dag.py:1081-1103):
    - dag.py:1091: Terminal coalesce missing on_success
    - dag.py:1082: Non-terminal coalesce with on_success set
    - dag.py:1099: Coalesce on_success references unknown sink
    """

    def test_terminal_coalesce_missing_on_success_raises(self, plugin_manager) -> None:
        """Terminal coalesce without on_success raises GraphValidationError."""
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_gate",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    input="to_gate",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                    # No on_success — terminal coalesce must have it
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"Dangling output|no incoming branches"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
                coalesce_settings=config.coalesce,
            )

    def test_non_terminal_coalesce_with_on_success_to_sink_builds(self, plugin_manager) -> None:
        """Coalesce with on_success pointing to a sink builds successfully.

        In the new WiredTransform system, coalesce on_success creates a terminal
        edge to the named sink. This is valid wiring.
        """
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_gate",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    input="to_gate",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                    on_success="output",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        # Coalesce on_success to a valid sink should build successfully
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            coalesce_settings=config.coalesce,
        )
        assert graph is not None

    def test_terminal_coalesce_unknown_sink_raises(self, plugin_manager) -> None:
        """Coalesce on_success referencing unknown sink raises GraphValidationError."""
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                on_success="to_gate",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            gates=[
                _gate_settings(
                    GateSettings,
                    name="forker",
                    input="to_gate",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                    on_success="nonexistent_sink",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config_raw(config)
        with pytest.raises(GraphValidationError, match=r"unknown sink 'nonexistent_sink'"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(config.gates),
                coalesce_settings=config.coalesce,
            )


class TestNodeInfoImmutability:
    """Tests for NodeInfo config immutability after graph construction."""

    def test_config_frozen_after_from_plugin_instances(self, plugin_manager) -> None:
        """NodeInfo.config should be MappingProxyType after from_plugin_instances.

        The DAG builder freezes all NodeInfo configs to MappingProxyType after
        construction to prevent accidental mutation of node configs. Attempting
        to set a key should raise TypeError.
        """
        from types import MappingProxyType

        from elspeth.core.config import SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            source=_source_settings(
                SourceSettings,
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}}),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            coalesce_settings=config.coalesce,
        )
        frozen_count = 0
        for info in graph.get_nodes():
            if info.config:
                assert isinstance(info.config, MappingProxyType), (
                    f"Node '{info.node_id}' config should be MappingProxyType after construction, got {type(info.config).__name__}"
                )
                with pytest.raises(TypeError):
                    info.config["injected_key"] = "should_fail"  # type: ignore[index]
                frozen_count += 1
        assert frozen_count > 0, "Expected at least one node with non-empty config"
