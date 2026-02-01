# tests/core/test_dag_schema_propagation.py
"""Tests for DAG schema config propagation from transforms to NodeInfo.

These tests verify that transforms with computed _output_schema_config attributes
have their schema configs correctly propagated through from_plugin_instances()
to NodeInfo, and that _get_schema_config_from_node() prioritizes these computed
configs over raw config dict parsing.
"""

from typing import Any, ClassVar

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.dag import ExecutionGraph


class MockTransformWithSchemaConfig:
    """Mock transform with computed _output_schema_config attribute."""

    name = "mock_transform_with_schema"
    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}

    def __init__(self) -> None:
        # Computed schema config with guaranteed and audit fields
        self._output_schema_config = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("field_a", "field_b"),
            audit_fields=("field_c", "field_d"),
        )


class MockTransformWithoutSchemaConfig:
    """Mock transform without _output_schema_config attribute."""

    name = "mock_transform_no_schema"
    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic", "guaranteed_fields": ["config_field"]}}


class MockSource:
    """Mock source plugin."""

    name = "mock_source"
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}


class MockSink:
    """Mock sink plugin."""

    name = "mock_sink"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {}


class TestOutputSchemaConfigPropagation:
    """Tests for _output_schema_config propagation to NodeInfo."""

    def test_output_schema_config_propagates_to_nodeinfo(self) -> None:
        """Verify _output_schema_config is extracted and stored in NodeInfo."""
        transform = MockTransformWithSchemaConfig()

        graph = ExecutionGraph.from_plugin_instances(
            source=MockSource(),
            transforms=[transform],  # type: ignore[list-item]
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
            default_sink="output",
        )

        # Find the transform node
        transform_nodes = [n for n in graph.get_nodes() if n.plugin_name == "mock_transform_with_schema"]
        assert len(transform_nodes) == 1

        node_info = transform_nodes[0]

        # Verify schema config was propagated
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.guaranteed_fields == ("field_a", "field_b")
        assert node_info.output_schema_config.audit_fields == ("field_c", "field_d")
        assert node_info.output_schema_config.is_dynamic is True

    def test_transform_without_schema_config_has_none(self) -> None:
        """Verify transforms without _output_schema_config have None in NodeInfo."""
        transform = MockTransformWithoutSchemaConfig()

        graph = ExecutionGraph.from_plugin_instances(
            source=MockSource(),
            transforms=[transform],  # type: ignore[list-item]
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
            default_sink="output",
        )

        # Find the transform node
        transform_nodes = [n for n in graph.get_nodes() if n.plugin_name == "mock_transform_no_schema"]
        assert len(transform_nodes) == 1

        node_info = transform_nodes[0]

        # output_schema_config should be None since transform doesn't have the attribute
        assert node_info.output_schema_config is None


class TestGetSchemaConfigFromNodePriority:
    """Tests for _get_schema_config_from_node() prioritization."""

    def test_prioritizes_nodeinfo_schema_config_over_config_dict(self) -> None:
        """NodeInfo.output_schema_config takes precedence over config dict parsing."""
        graph = ExecutionGraph()

        # Create schema config to put in NodeInfo
        nodeinfo_schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("from_nodeinfo",),
            audit_fields=("audit_from_nodeinfo",),
        )

        # Add node with BOTH output_schema_config AND schema in config dict
        # The config dict has different guaranteed_fields
        graph.add_node(
            "test_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["from_config_dict"]}},
            output_schema_config=nodeinfo_schema,  # This should win
        )

        # Get schema config from node
        result = graph._get_schema_config_from_node("test_node")

        # Should return the NodeInfo schema config, not parse from config dict
        assert result is not None
        assert result.guaranteed_fields == ("from_nodeinfo",)
        assert result.audit_fields == ("audit_from_nodeinfo",)

    def test_falls_back_to_config_dict_when_no_nodeinfo_schema(self) -> None:
        """Falls back to config dict when NodeInfo.output_schema_config is None."""
        graph = ExecutionGraph()

        # Add node with only config dict schema (no output_schema_config)
        graph.add_node(
            "test_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["from_config"]}},
            # No output_schema_config - should fall back to config dict
        )

        # Get schema config from node
        result = graph._get_schema_config_from_node("test_node")

        # Should parse from config dict
        assert result is not None
        assert result.guaranteed_fields == ("from_config",)
        assert result.audit_fields is None

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
        result = graph._get_schema_config_from_node("test_node")

        # Should return None
        assert result is None


class TestGuaranteedFieldsWithSchemaConfig:
    """Tests for _get_guaranteed_fields() using output_schema_config."""

    def test_guaranteed_fields_from_nodeinfo_schema_config(self) -> None:
        """_get_guaranteed_fields returns fields from NodeInfo.output_schema_config."""
        graph = ExecutionGraph()

        nodeinfo_schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("response", "response_usage", "response_model"),
            audit_fields=("response_template_hash",),  # audit fields should NOT be included
        )

        graph.add_node(
            "llm_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={"schema": {"fields": "dynamic"}},
            output_schema_config=nodeinfo_schema,
        )

        result = graph._get_guaranteed_fields("llm_node")

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
            config={"schema": {"fields": "dynamic"}},
        )

        # Transform guarantees fields via output_schema_config
        transform_schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("result", "result_usage"),
            audit_fields=("result_hash",),
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"fields": "dynamic"}},
            output_schema_config=transform_schema,
        )

        # Sink requires result_usage (guaranteed by transform)
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "required_fields": ["result_usage"]}},
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
            config={"schema": {"fields": "dynamic"}},
        )

        # Transform has audit_fields that are NOT in guaranteed_fields
        transform_schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("result",),
            audit_fields=("result_template_hash",),  # audit only
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"fields": "dynamic"}},
            output_schema_config=transform_schema,
        )

        # Sink requires result_template_hash (audit field - NOT guaranteed)
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "required_fields": ["result_template_hash"]}},
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
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}

    def __init__(self) -> None:
        self._output_schema_config = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("batch_result",),
            audit_fields=("batch_hash",),
        )


class TestAggregationSchemaConfigPropagation:
    """Tests for _output_schema_config propagation from aggregation transforms."""

    def test_aggregation_schema_config_propagates(self) -> None:
        """Verify aggregation transform's _output_schema_config is stored in NodeInfo."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        transform = MockAggregationTransform()
        trigger = TriggerConfig(count=10)
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="mock_agg_transform",
            trigger=trigger,
            output_mode="transform",
            options={"schema": {"fields": "dynamic"}},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=MockSource(),
            transforms=[],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={"test_agg": (transform, agg_settings)},  # type: ignore[dict-item]
            gates=[],
            default_sink="output",
        )

        # Find the aggregation node
        agg_nodes = [n for n in graph.get_nodes() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1

        node_info = agg_nodes[0]

        # Verify schema config was propagated
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.guaranteed_fields == ("batch_result",)
        assert node_info.output_schema_config.audit_fields == ("batch_hash",)


class TestGateSchemaConfigInheritance:
    """Tests for gate nodes inheriting computed output_schema_config from upstream.

    P1-2026-01-31: Gate nodes were short-circuiting on raw schema guarantees
    instead of walking upstream to find computed guarantees from output_schema_config.
    """

    def test_gate_inherits_output_schema_config_from_upstream(self) -> None:
        """Gate should inherit computed output_schema_config guarantees from upstream.

        Scenario:
        - Transform has output_schema_config with computed guaranteed_fields
          (e.g., LLM transforms with ["result", "result_usage", "result_model"])
        - Transform's raw config["schema"] only has a SUBSET of these fields
          (e.g., ["result"]) because others are computed dynamically
        - Gate copies raw config["schema"] from transform (only gets ["result"])
        - Gate should still inherit ALL guarantees from upstream's output_schema_config
        """
        graph = ExecutionGraph()

        # Source with some fields
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["input_field"]}},
        )

        # Transform with COMPUTED output_schema_config
        # Raw config only declares "result", but computed schema adds more
        transform_computed_schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("result", "result_usage", "result_model"),
            audit_fields=None,
        )
        graph.add_node(
            "llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["result"]}},
            output_schema_config=transform_computed_schema,
        )

        # Gate that would copy raw schema from transform (simulates from_plugin_instances)
        # The gate gets raw schema: {"fields": "dynamic", "guaranteed_fields": ["result"]}
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["result"]}},
            # NO output_schema_config - gate doesn't compute schema
        )

        graph.add_edge("source", "llm_transform", label="continue")
        graph.add_edge("llm_transform", "gate", label="continue")

        # Verify transform has computed guarantees
        transform_guarantees = graph._get_effective_guaranteed_fields("llm_transform")
        assert "result_usage" in transform_guarantees

        # Gate should inherit computed guarantees from upstream
        gate_guarantees = graph._get_effective_guaranteed_fields("gate")
        assert "result_usage" in gate_guarantees, f"Gate should inherit result_usage from upstream transform, has: {gate_guarantees}"
        assert "result_model" in gate_guarantees
        assert "result" in gate_guarantees

    def test_chained_gates_inherit_through_all(self) -> None:
        """Multiple chained gates should all inherit from original transform."""
        graph = ExecutionGraph()

        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        transform_schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("computed_a", "computed_b"),
            audit_fields=None,
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={"schema": {"fields": "dynamic"}},
            output_schema_config=transform_schema,
        )

        # Two gates in sequence
        graph.add_node(
            "gate1",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={"schema": {"fields": "dynamic"}},
        )
        graph.add_node(
            "gate2",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={"schema": {"fields": "dynamic"}},
        )

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "gate1", label="continue")
        graph.add_edge("gate1", "gate2", label="continue")

        # Both gates should inherit computed guarantees
        gate1_guarantees = graph._get_effective_guaranteed_fields("gate1")
        gate2_guarantees = graph._get_effective_guaranteed_fields("gate2")

        assert gate1_guarantees == frozenset({"computed_a", "computed_b"})
        assert gate2_guarantees == frozenset({"computed_a", "computed_b"})
