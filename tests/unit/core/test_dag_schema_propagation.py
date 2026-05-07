# tests/core/test_dag_schema_propagation.py
"""Tests for DAG schema config propagation from transforms to NodeInfo.

These tests verify that transforms with computed _output_schema_config attributes
have their schema configs correctly propagated through from_plugin_instances()
to NodeInfo, and that get_schema_config_from_node() prioritizes these computed
configs over raw config dict parsing.

Also tests that pass-through nodes (gates, coalesce) inherit computed schema
contracts from upstream transforms, so audit records reflect actual data contracts.
(P1-2026-02-05: pass-through nodes drop computed schema contracts)
"""

from typing import Any, ClassVar

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.config import (
    AggregationSettings,
    CoalesceSettings,
    GateSettings,
    SourceSettings,
    TransformSettings,
    TriggerConfig,
)
from elspeth.core.dag import ExecutionGraph, WiredTransform


class MockTransformWithSchemaConfig:
    """Mock transform with computed _output_schema_config attribute."""

    name = "mock_transform_with_schema"
    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = False

    def __init__(self) -> None:
        # Computed schema config with guaranteed and audit fields
        self._output_schema_config = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("field_a", "field_b"),
            audit_fields=("field_c", "field_d"),
        )


class MockTransformWithoutSchemaConfig:
    """Mock transform without _output_schema_config attribute."""

    name = "mock_transform_no_schema"
    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed", "guaranteed_fields": ["config_field"]}}
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = False
    _output_schema_config: SchemaConfig | None = None


class MockSource:
    """Mock source plugin."""

    name = "mock_source"
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed", "guaranteed_fields": ["source_field"]}}
    _on_validation_failure = "discard"
    on_success = "output"


class MockSink:
    """Mock sink plugin."""

    name = "mock_sink"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {}
    _on_write_failure: str = "discard"
    declared_required_fields: ClassVar[frozenset[str]] = frozenset()

    def _reset_diversion_log(self) -> None:
        pass


class MockSinkWithSchema:
    """Mock sink plugin with schema config."""

    name = "mock_sink_schema"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    _on_write_failure: str = "discard"
    declared_required_fields: ClassVar[frozenset[str]] = frozenset()

    def _reset_diversion_log(self) -> None:
        pass


class TestOutputSchemaConfigPropagation:
    """Tests for _output_schema_config propagation to NodeInfo."""

    def test_output_schema_config_propagates_to_nodeinfo(self) -> None:
        """Verify _output_schema_config is extracted and stored in NodeInfo."""
        transform = MockTransformWithSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="transform_0",
                plugin=transform.name,
                input="source_out",
                on_success="output",
                on_error="discard",
                options={},
            ),
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
        )

        # Find the transform node
        transform_nodes = [n for n in graph.get_nodes() if n.plugin_name == "mock_transform_with_schema"]
        assert len(transform_nodes) == 1

        node_info = transform_nodes[0]

        # Verify schema config was propagated
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.guaranteed_fields == ("field_a", "field_b")
        assert node_info.output_schema_config.audit_fields == ("field_c", "field_d")
        assert node_info.output_schema_config.is_observed is True

    def test_shape_preserving_transform_has_output_schema_config(self) -> None:
        """Transforms without _output_schema_config should still get output_schema_config
        populated from config['schema'] at construction time."""
        transform = MockTransformWithoutSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="transform_0",
                plugin=transform.name,
                input="source_out",
                on_success="output",
                on_error="discard",
                options={},
            ),
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
        )

        transform_nodes = [n for n in graph.get_nodes() if n.plugin_name == "mock_transform_no_schema"]
        assert len(transform_nodes) == 1

        node_info = transform_nodes[0]
        # Previously None — now populated from config["schema"]
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.mode == "observed"
        assert node_info.output_schema_config.guaranteed_fields == ("config_field",)

    def test_source_node_has_output_schema_config(self) -> None:
        """Source nodes should have output_schema_config populated from config['schema']."""
        source = MockSource()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="output", options={}),
            transforms=[],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
        )

        source_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.SOURCE]
        assert len(source_nodes) == 1

        node_info = source_nodes[0]
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.mode == "observed"
        assert node_info.output_schema_config.guaranteed_fields == ("source_field",)

    def test_sink_node_has_output_schema_config(self) -> None:
        """Sink nodes should have output_schema_config populated from config['schema']."""
        source = MockSource()

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="output", options={}),
            transforms=[],
            sinks={"output": MockSinkWithSchema()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
        )

        sink_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.SINK]
        assert len(sink_nodes) == 1

        node_info = sink_nodes[0]
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.mode == "observed"


class TestGetSchemaConfigFromNodePriority:
    """Tests for get_schema_config_from_node() prioritization."""

    def test_prioritizes_nodeinfo_schema_config_over_config_dict(self) -> None:
        """NodeInfo.output_schema_config takes precedence over config dict parsing."""
        graph = ExecutionGraph()

        # Create schema config to put in NodeInfo
        nodeinfo_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("from_nodeinfo",),
            audit_fields=("audit_from_nodeinfo",),
        )

        # Add node with BOTH output_schema_config AND schema in config dict
        # The config dict has different guaranteed_fields
        graph.add_node(
            "test_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["from_config_dict"]}},
            output_schema_config=nodeinfo_schema,  # This should win
        )

        # Get schema config from node
        result = graph.get_schema_config_from_node("test_node")

        # Should return the NodeInfo schema config, not parse from config dict
        assert result is not None
        assert result.guaranteed_fields == ("from_nodeinfo",)
        assert result.audit_fields == ("audit_from_nodeinfo",)

    def test_returns_output_schema_config_directly(self) -> None:
        """get_schema_config_from_node returns output_schema_config without parsing config dict."""
        graph = ExecutionGraph()

        schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("field_a",),
        )

        graph.add_node(
            "test_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["different_field"]}},
            output_schema_config=schema,
        )

        result = graph.get_schema_config_from_node("test_node")
        # Returns the typed object, ignores config dict
        assert result is schema
        assert result.guaranteed_fields == ("field_a",)

    def test_returns_none_when_no_schema_anywhere(self) -> None:
        """Returns None when neither NodeInfo nor config dict has schema."""
        graph = ExecutionGraph()

        # Add node with no schema info at all
        graph.add_node(
            "test_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config={},  # No schema in config
            # No output_schema_config
        )

        # Get schema config from node
        result = graph.get_schema_config_from_node("test_node")

        # Should return None
        assert result is None


class TestGuaranteedFieldsWithSchemaConfig:
    """Tests for get_guaranteed_fields() using output_schema_config."""

    def test_guaranteed_fields_from_nodeinfo_schema_config(self) -> None:
        """get_guaranteed_fields returns fields from NodeInfo.output_schema_config."""
        graph = ExecutionGraph()

        nodeinfo_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("response", "response_usage", "response_model"),
            audit_fields=("response_template_hash",),  # audit fields should NOT be included
        )

        graph.add_node(
            "llm_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"mode": "observed"}},
            output_schema_config=nodeinfo_schema,
        )

        result = graph.get_guaranteed_fields("llm_node")

        # Should include guaranteed_fields from schema config
        assert "response" in result
        assert "response_usage" in result
        assert "response_model" in result

        # Should NOT include audit_fields
        assert "response_template_hash" not in result

    def test_contract_validation_uses_nodeinfo_schema_config(self) -> None:
        """Contract validation respects guaranteed_fields from NodeInfo.output_schema_config."""
        graph = ExecutionGraph()

        # Source has no guarantees
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        # Transform guarantees fields via output_schema_config
        transform_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("result", "result_usage"),
            audit_fields=("result_hash",),
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"mode": "observed"}},
            output_schema_config=transform_schema,
        )

        # Sink requires result_usage (guaranteed by transform)
        sink_schema = SchemaConfig(
            mode="observed",
            fields=None,
            required_fields=("result_usage",),
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={},
            output_schema_config=sink_schema,
        )

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Should not raise - transform guarantees what sink requires
        graph.validate_edge_compatibility()

    def test_contract_validation_rejects_dependency_on_audit_fields(self) -> None:
        """Contract validation rejects downstream dependency on audit-only fields."""
        graph = ExecutionGraph()

        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        # Transform has audit_fields that are NOT in guaranteed_fields
        transform_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("result",),
            audit_fields=("result_template_hash",),  # audit only
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"mode": "observed"}},
            output_schema_config=transform_schema,
        )

        # Sink requires result_template_hash (audit field - NOT guaranteed)
        sink_schema = SchemaConfig(
            mode="observed",
            fields=None,
            required_fields=("result_template_hash",),
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "required_fields": ["result_template_hash"]}},
            output_schema_config=sink_schema,
        )

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Should raise - result_template_hash is audit-only, not guaranteed
        with pytest.raises(ValueError, match="result_template_hash"):
            graph.validate_edge_compatibility()


class MockAggregationTransform:
    """Mock transform for aggregation with _output_schema_config."""

    name = "mock_agg_transform"
    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = False

    def __init__(self) -> None:
        self._output_schema_config = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("batch_result",),
            audit_fields=("batch_hash",),
        )


class TestAggregationSchemaConfigPropagation:
    """Tests for aggregation node schema handling.

    Aggregations have dynamic output by design (e.g., BatchStats produces
    count/sum/mean rather than the input fields). The builder propagates
    the aggregation's _output_schema_config to NodeInfo so:
    1. Downstream pass-through nodes (gates, coalesce branches) can inherit
       the schema via _best_schema_config()
    2. Derived input requirements (like group_by → required_fields) are
       preserved for DAG validation

    The _output_schema_config IS correct for output: _build_output_schema_config()
    merges declared_output_fields into guaranteed_fields, not input fields.
    """

    def test_aggregation_output_schema_config_propagates(self) -> None:
        """Verify aggregation nodes have output_schema_config from transform."""
        transform = MockAggregationTransform()
        trigger = TriggerConfig(count=10)
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="mock_agg_transform",
            input="agg_in",
            on_success="output",
            on_error="discard",
            trigger=trigger,
            output_mode="transform",
            options={"schema": {"mode": "observed"}},
        )

        source = MockSource()
        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="agg_in", options={}),
            transforms=[],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={"test_agg": (transform, agg_settings)},  # type: ignore[dict-item]
            gates=[],
        )

        # Find the aggregation node
        agg_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.AGGREGATION]
        assert len(agg_nodes) == 1

        node_info = agg_nodes[0]

        # Aggregation's output_schema_config is propagated from the transform.
        # This enables downstream pass-through nodes to inherit via _best_schema_config()
        # and preserves derived input requirements for DAG validation.
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.guaranteed_fields == ("batch_result",)
        assert node_info.output_schema_config.audit_fields == ("batch_hash",)


class TestGateSchemaConfigInheritance:
    """Tests for gate nodes inheriting computed output_schema_config from upstream.

    P1-2026-01-31: Gate nodes were short-circuiting on raw schema guarantees
    instead of walking upstream to find computed guarantees from output_schema_config.
    """

    def test_gate_uses_full_propagated_schema_from_upstream(self) -> None:
        """Gate uses full computed schema propagated by builder from upstream.

        In production, the builder calls _best_schema_dict() which prefers
        output_schema_config over raw config["schema"]. This means gates get
        the FULL computed guarantees (including dynamically added fields like
        LLM usage/model fields), not just the raw config subset.
        """
        graph = ExecutionGraph()

        # Source with some fields
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["input_field"]}},
        )

        # Transform with COMPUTED output_schema_config
        transform_computed_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("result", "result_usage", "result_model"),
            audit_fields=None,
        )
        graph.add_node(
            "llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["result"]}},
            output_schema_config=transform_computed_schema,
        )

        # Gate with full computed schema (as the builder would propagate via _assign_schema)
        gate_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("result", "result_usage", "result_model"),
            audit_fields=None,
        )
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={},
            output_schema_config=gate_schema,
        )

        graph.add_edge("source", "llm_transform", label="continue")
        graph.add_edge("llm_transform", "gate", label="continue")

        # Verify transform has computed guarantees
        transform_guarantees = graph.get_effective_guaranteed_fields("llm_transform")
        assert "result_usage" in transform_guarantees

        # Gate has the full propagated schema
        gate_guarantees = graph.get_effective_guaranteed_fields("gate")
        assert "result_usage" in gate_guarantees
        assert "result_model" in gate_guarantees
        assert "result" in gate_guarantees

    def test_chained_gates_use_propagated_schema(self) -> None:
        """Multiple chained gates each use their propagated schema.

        In production, the builder propagates the full computed schema to
        each gate via _best_schema_dict(). Both gates get the same schema.
        """
        graph = ExecutionGraph()

        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        transform_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("computed_a", "computed_b"),
            audit_fields=None,
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"mode": "observed"}},
            output_schema_config=transform_schema,
        )

        # Two gates in sequence — builder propagates computed schema to each
        gate_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("computed_a", "computed_b"),
            audit_fields=None,
        )
        graph.add_node(
            "gate1",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={},
            output_schema_config=gate_schema,
        )
        graph.add_node(
            "gate2",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={},
            output_schema_config=gate_schema,
        )

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "gate1", label="continue")
        graph.add_edge("gate1", "gate2", label="continue")

        # Both gates have computed guarantees from propagation
        gate1_guarantees = graph.get_effective_guaranteed_fields("gate1")
        gate2_guarantees = graph.get_effective_guaranteed_fields("gate2")

        assert gate1_guarantees == frozenset({"computed_a", "computed_b"})
        assert gate2_guarantees == frozenset({"computed_a", "computed_b"})


class TestPassThroughNodesInheritComputedSchema:
    """P1-2026-02-05: Gate and coalesce nodes must propagate computed
    output_schema_config (not just raw config["schema"]) so audit metadata
    reflects actual data contracts including guaranteed/audit fields.

    These tests exercise from_plugin_instances() — the production code path.
    """

    def test_gate_inherits_computed_schema_config(self) -> None:
        """Gate inherits output_schema_config with guaranteed_fields and
        audit_fields from upstream transform's computed schema.
        """
        transform = MockTransformWithSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="gate_in",
                on_error="discard",
                options={},
            ),
        )

        gate = GateSettings(
            name="quality_gate",
            input="gate_in",
            condition="True",
            routes={"true": "output", "false": "output"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[gate],
        )

        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1

        gate_schema = gate_nodes[0].output_schema_config
        assert gate_schema is not None
        assert gate_schema.guaranteed_fields is not None
        assert set(gate_schema.guaranteed_fields) == {"field_a", "field_b"}
        assert gate_schema.audit_fields is not None
        assert set(gate_schema.audit_fields) == {"field_c", "field_d"}

    def test_gate_inherits_raw_schema_when_no_computed(self) -> None:
        """Gate inherits output_schema_config from upstream even when
        upstream has no computed _output_schema_config (parsed from config).
        """
        transform = MockTransformWithoutSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="basic_step",
                plugin=transform.name,
                input="source_out",
                on_success="gate_in",
                on_error="discard",
                options={},
            ),
        )

        gate = GateSettings(
            name="quality_gate",
            input="gate_in",
            condition="True",
            routes={"true": "output", "false": "output"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[gate],
        )

        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1

        gate_schema = gate_nodes[0].output_schema_config
        assert gate_schema is not None
        assert gate_schema.guaranteed_fields == ("config_field",)

    def test_coalesce_inherits_computed_schema_config(self) -> None:
        """Coalesce node inherits output_schema_config with computed
        guaranteed/audit fields from upstream fork branches.
        """
        transform = MockTransformWithSchemaConfig()
        source = MockSource()

        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="fork_in",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="fork_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
            on_success="output",
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1

        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        assert coal_schema.guaranteed_fields is not None
        assert set(coal_schema.guaranteed_fields) == {"field_a", "field_b"}
        assert coal_schema.audit_fields is not None
        assert set(coal_schema.audit_fields) == {"field_c", "field_d"}

    def test_deferred_gate_after_coalesce_inherits_computed_schema(self) -> None:
        """A gate downstream of a coalesce node (deferred to pass 2 in builder)
        should also inherit computed schema fields.
        """
        transform = MockTransformWithSchemaConfig()
        source = MockSource()

        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="fork_in",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="fork_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
        )

        # This gate is downstream of coalesce — resolved in pass 2
        post_coalesce_gate = GateSettings(
            name="final_check",
            input="merger",
            condition="True",
            routes={"true": "output", "false": "output"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate, post_coalesce_gate],
            coalesce_settings=[coalesce],
        )

        # Find the post-coalesce gate
        gate_nodes = [n for n in graph.get_nodes() if n.plugin_name == "config_gate:final_check"]
        assert len(gate_nodes) == 1

        gate_schema = gate_nodes[0].output_schema_config
        assert gate_schema is not None
        assert gate_schema.guaranteed_fields is not None
        assert set(gate_schema.guaranteed_fields) == {"field_a", "field_b"}
        assert gate_schema.audit_fields is not None
        assert set(gate_schema.audit_fields) == {"field_c", "field_d"}


class TestPassThroughNodesUseTypedSchema:
    """After single-source-of-truth refactor, pass-through nodes (gates, coalesce)
    should have output_schema_config populated but should NOT have config['schema'].
    """

    def test_gate_has_output_schema_config_not_dict(self) -> None:
        """Gate should have output_schema_config but no config['schema']."""
        transform = MockTransformWithSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="gate_in",
                on_error="discard",
                options={},
            ),
        )

        gate = GateSettings(
            name="quality_gate",
            input="gate_in",
            condition="True",
            routes={"true": "output", "false": "output"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[gate],
        )

        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1

        gate_info = gate_nodes[0]
        # Typed schema is populated
        assert gate_info.output_schema_config is not None
        assert gate_info.output_schema_config.guaranteed_fields == ("field_a", "field_b")
        assert gate_info.output_schema_config.audit_fields == ("field_c", "field_d")
        # Dict form is NOT written to pass-through nodes
        assert "schema" not in gate_info.config

    def test_coalesce_has_output_schema_config_not_dict(self) -> None:
        """Coalesce should have output_schema_config but no config['schema']."""
        transform = MockTransformWithSchemaConfig()
        source = MockSource()

        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="fork_in",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="fork_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
            on_success="output",
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1

        coal_info = coalesce_nodes[0]
        assert coal_info.output_schema_config is not None
        assert coal_info.output_schema_config.guaranteed_fields == ("field_a", "field_b")
        assert "schema" not in coal_info.config


class _ConfigurableTransform:
    """Mock transform with per-instance guaranteed_fields for schema tests."""

    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = False

    def __init__(self, name: str, guaranteed_fields: tuple[str, ...] | None) -> None:
        self.name = name
        self._output_schema_config = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=guaranteed_fields,
        )


class TestCoalesceMaterializedSchemaFromBuilder:
    """Integration tests: builder coalesce schema materialization via from_plugin_instances().

    These verify the materialized output_schema_config on the coalesce node
    preserves the None-vs-empty-tuple contract when branches have different
    guaranteed_fields declarations.

    The unit tests in test_dag_contract_validation.py exercise
    get_effective_guaranteed_fields() on manually-built graphs. These tests
    exercise the builder's coalesce path that COMPUTES and MATERIALIZES
    the intersection during from_plugin_instances().
    """

    def _build_fork_coalesce_with_branch_transforms(
        self,
        transform_a_guaranteed: tuple[str, ...] | None,
        transform_b_guaranteed: tuple[str, ...] | None,
        *,
        policy: str = "best_effort",
    ) -> ExecutionGraph:
        """Build: source → fork → [transform_a, transform_b] → coalesce → sink.

        Default policy is best_effort because most tests in this class assert
        the INTERSECTION contract (None-vs-empty-tuple distinction under AND
        semantics). Under require_all, the builder uses UNION instead — those
        assertions would not hold. Tests that specifically cover the
        require_all union materialization live in
        test_dag_coalesce_optionality.py::TestBuilderBranchExclusiveFieldDowngrade.
        """
        source = MockSource()

        t_a = _ConfigurableTransform("branch_transform_a", transform_a_guaranteed)
        t_b = _ConfigurableTransform("branch_transform_b", transform_b_guaranteed)

        wired_a = WiredTransform(
            plugin=t_a,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_a",
                plugin=t_a.name,
                input="branch_a",
                on_success="t_a_out",
                on_error="discard",
                options={},
            ),
        )
        wired_b = WiredTransform(
            plugin=t_b,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_b",
                plugin=t_b.name,
                input="branch_b",
                on_success="t_b_out",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        # best_effort requires timeout_seconds; require_all ignores it.
        # Pass unconditionally so the helper supports either policy.
        coalesce = CoalesceSettings(
            name="merger",
            branches={"branch_a": "t_a_out", "branch_b": "t_b_out"},
            policy=policy,
            merge="union",
            on_success="output",
            timeout_seconds=60.0,
        )

        return ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired_a, wired_b],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

    def test_mixed_none_and_explicit_materializes_abstain(self) -> None:
        """Branch with None guaranteed_fields abstains — doesn't kill materialized intersection."""
        graph = self._build_fork_coalesce_with_branch_transforms(
            transform_a_guaranteed=("x", "y"),
            transform_b_guaranteed=None,
        )
        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        # branch_b abstains, branch_a's guarantees survive
        assert coal_schema.guaranteed_fields is not None
        assert set(coal_schema.guaranteed_fields) == {"x", "y"}

    def test_empty_intersection_materializes_empty_tuple_not_none(self) -> None:
        """Branches with disjoint fields → guaranteed_fields is (), not None.

        () means "explicitly guarantees nothing" (branches declared but share
        no fields). None means "abstains" (no branch made any declaration).
        The audit trail must distinguish these for IRAP traceability.

        Tests the INTERSECTION materialization path (best_effort). Under
        require_all, the same setup would materialize union = ('x', 'y')
        because every branch always arrives.
        """
        graph = self._build_fork_coalesce_with_branch_transforms(
            transform_a_guaranteed=("x",),
            transform_b_guaranteed=("y",),
        )
        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        assert coal_schema.guaranteed_fields is not None, "guaranteed_fields should be () (explicitly empty), not None (abstain)"
        assert coal_schema.guaranteed_fields == ()

    def test_all_none_materializes_none(self) -> None:
        """Both branches with None guaranteed_fields → materialized as None (abstain)."""
        graph = self._build_fork_coalesce_with_branch_transforms(
            transform_a_guaranteed=None,
            transform_b_guaranteed=None,
        )
        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        assert coal_schema.guaranteed_fields is None

    def test_explicit_empty_tuple_kills_materialized_intersection(self) -> None:
        """Branch with guaranteed_fields=() participates and collapses materialized intersection.

        This tests the Python API boundary: a transform that constructs
        SchemaConfig(guaranteed_fields=()) is saying "I guarantee zero fields."
        Under best_effort, the coalesce should materialize () (explicitly
        empty), not None (abstain) — the None-vs-empty distinction must be
        preserved for IRAP traceability.

        Under require_all, the same setup would materialize union = ('x', 'y')
        because branch_a's guarantees survive (branch_a always arrives).
        """
        graph = self._build_fork_coalesce_with_branch_transforms(
            transform_a_guaranteed=("x", "y"),
            transform_b_guaranteed=(),
        )
        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        # branch_b explicitly guarantees nothing → intersection collapses to ()
        assert coal_schema.guaranteed_fields is not None, "guaranteed_fields should be () (explicitly empty), not None (abstain)"
        assert coal_schema.guaranteed_fields == ()

    def test_require_all_materializes_union_preserving_explicit_empty_semantics(self) -> None:
        """Under require_all, materialized guarantees are UNION (not intersection).

        Symmetric counterpart to test_empty_intersection_materializes_empty_tuple_not_none:
        the same disjoint branch guarantees that produce () under best_effort
        produce the union under require_all because every branch always
        arrives and contributes its guarantees to the merged row.
        """
        graph = self._build_fork_coalesce_with_branch_transforms(
            transform_a_guaranteed=("x",),
            transform_b_guaranteed=("y",),
            policy="require_all",
        )
        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        # Union under require_all: both branches' guarantees survive.
        assert coal_schema.guaranteed_fields is not None
        assert set(coal_schema.guaranteed_fields) == {"x", "y"}
