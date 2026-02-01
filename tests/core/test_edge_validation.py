"""Test edge compatibility validation during graph construction."""

from typing import Any, ClassVar

import pytest

from elspeth.contracts import NodeType, PluginSchema
from elspeth.core.dag import ExecutionGraph
from tests.conftest import as_sink, as_source


class ProducerSchema(PluginSchema):
    """Producer output schema."""

    id: int
    name: str


class ConsumerSchema(PluginSchema):
    """Consumer input schema."""

    id: int
    name: str
    email: str  # Required field NOT in producer!


def test_edge_validation_detects_missing_fields() -> None:
    """Edges should fail if producer missing required fields.

    Validation happens in validate_edge_compatibility() called from
    from_plugin_instances(), NOT during add_edge() (which is a dumb primitive).
    """
    graph = ExecutionGraph()

    # Add source with ProducerSchema
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)

    # Add sink requiring ConsumerSchema (has 'email' field)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerSchema)

    # Wire them together (add_edge does NOT validate)
    graph.add_edge("source", "sink", label="continue")

    # Validation happens when we explicitly call it
    with pytest.raises(ValueError, match=r"Missing fields.*email"):
        graph.validate_edge_compatibility()


def test_edge_validation_allows_dynamic_schemas() -> None:
    """Dynamic schemas should be compatible with anything."""
    from elspeth.plugins.schema_factory import _create_dynamic_schema

    graph = ExecutionGraph()

    dynamic_schema = _create_dynamic_schema("DynamicSchema")

    # Dynamic producer → strict consumer: OK
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=dynamic_schema)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerSchema)
    graph.add_edge("source", "sink", label="continue")  # Should succeed

    # Should not raise
    graph.validate_edge_compatibility()

    # Strict producer → dynamic consumer: OK
    graph2 = ExecutionGraph()
    graph2.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)
    graph2.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=dynamic_schema)
    graph2.add_edge("source", "sink", label="continue")  # Should succeed

    # Should not raise
    graph2.validate_edge_compatibility()


def test_gate_passthrough_validation() -> None:
    """Gates must preserve schema (input == output)."""
    graph = ExecutionGraph()

    # Source produces ProducerSchema
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)

    # Gate claims to pass through but has DIFFERENT output schema
    graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold", input_schema=ProducerSchema, output_schema=ConsumerSchema)

    # Wire them (add_edge does NOT validate)
    graph.add_edge("source", "gate", label="continue")

    # Validation happens when we explicitly call it
    with pytest.raises(ValueError, match=r"Gate.*must preserve schema"):
        graph.validate_edge_compatibility()


def test_coalesce_branch_compatibility() -> None:
    """Coalesce must receive compatible schemas from all branches."""
    graph = ExecutionGraph()

    # Source forks to two paths
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)

    # Path 1: Transform to Schema A
    class SchemaA(PluginSchema):
        value: int

    graph.add_node(
        "transform1", node_type=NodeType.TRANSFORM, plugin_name="field_mapper", input_schema=ProducerSchema, output_schema=SchemaA
    )

    # Path 2: Transform to Schema B (INCOMPATIBLE!)
    class SchemaB(PluginSchema):
        different: str

    graph.add_node(
        "transform2", node_type=NodeType.TRANSFORM, plugin_name="field_mapper", input_schema=ProducerSchema, output_schema=SchemaB
    )

    # Coalesce node
    graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="merge", input_schema=SchemaA)  # Expects SchemaA from both branches

    # Wire up
    graph.add_edge("source", "transform1", label="fork_path_1")
    graph.add_edge("source", "transform2", label="fork_path_2")
    graph.add_edge("transform1", "coalesce", label="continue")  # OK
    graph.add_edge("transform2", "coalesce", label="continue")  # Add second edge

    # This should fail during graph validation (AFTER edges added)
    with pytest.raises(ValueError, match=r"Coalesce.*incompatible schemas"):
        graph.validate_edge_compatibility()  # Explicit validation call


def test_gate_walk_through_for_effective_schema() -> None:
    """Edge validation must walk through gates to find effective producer schema."""
    graph = ExecutionGraph()

    # Source produces ProducerSchema
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)

    # Gate has NO schema (config-driven gate)
    graph.add_node(
        "gate", node_type=NodeType.GATE, plugin_name="config_gate", input_schema=None, output_schema=None
    )  # Inherits from upstream!

    # Sink requires ConsumerSchema
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerSchema)

    # Wire: source → gate → sink
    graph.add_edge("source", "gate", label="continue")
    graph.add_edge("gate", "sink", label="continue")

    # Should walk through gate to find ProducerSchema
    # Then check if ProducerSchema has fields required by ConsumerSchema
    with pytest.raises(ValueError, match=r"Missing fields.*email"):
        graph.validate_edge_compatibility()


def test_chained_gates() -> None:
    """Validation must handle multiple chained gates."""
    graph = ExecutionGraph()

    # Source → Gate1 → Gate2 → Sink
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)
    graph.add_node("gate1", node_type=NodeType.GATE, plugin_name="config_gate", input_schema=None, output_schema=None)
    graph.add_node("gate2", node_type=NodeType.GATE, plugin_name="config_gate", input_schema=None, output_schema=None)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerSchema)

    graph.add_edge("source", "gate1", label="continue")
    graph.add_edge("gate1", "gate2", label="continue")
    graph.add_edge("gate2", "sink", label="continue")

    # Should walk gate1 → gate2 → source, find ProducerSchema missing 'email'
    with pytest.raises(ValueError, match=r"Missing fields.*email"):
        graph.validate_edge_compatibility()


def test_none_schema_handling() -> None:
    """None schemas (dynamic by convention) should be compatible with anything."""
    graph = ExecutionGraph()

    # Source with None schema (dynamic)
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=None)

    # Sink with strict schema
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerSchema)

    graph.add_edge("source", "sink", label="continue")

    # Should pass - None is compatible with anything
    graph.validate_edge_compatibility()  # No exception


def test_edge_validation_timing_from_plugin_instances() -> None:
    """CRITICAL: Validation must happen during from_plugin_instances(), not in validate().

    This test verifies the core architectural change - that schema validation
    has been moved from DAG.validate() to graph construction time.
    """

    # Create mock plugins with incompatible schemas
    class MockSource:
        name: ClassVar[str] = "test_source"
        config: ClassVar[dict[str, Any]] = {}
        output_schema: ClassVar[type[PluginSchema]] = ProducerSchema  # Has: id, name

    class MockSink:
        name: ClassVar[str] = "test_sink"
        config: ClassVar[dict[str, Any]] = {}
        input_schema: ClassVar[type[PluginSchema]] = ConsumerSchema  # Needs: id, name, email

    # Should fail DURING from_plugin_instances (PHASE 2 validation)
    with pytest.raises(ValueError, match=r"Missing fields.*email"):
        ExecutionGraph.from_plugin_instances(
            source=as_source(MockSource()),
            transforms=[],
            sinks={"out": as_sink(MockSink())},
            aggregations={},
            gates=[],
            default_sink="out",
        )


def test_aggregation_dual_schema_both_edges_validated() -> None:
    """Aggregations have both input_schema and output_schema - validate both edges."""

    class SourceOutput(PluginSchema):
        value: float

    class AggInput(PluginSchema):
        value: float
        label: str  # Required, not in source!

    class AggOutput(PluginSchema):
        count: int
        sum: float

    class SinkInput(PluginSchema):
        count: int
        sum: float
        average: float  # Required, not in agg output!

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=SourceOutput)
    graph.add_node(
        "agg",
        node_type=NodeType.AGGREGATION,
        plugin_name="batch_stats",
        input_schema=AggInput,
        output_schema=AggOutput,
        config={
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {
                "schema": {"mode": "strict", "fields": ["value: float", "label: str"]},
                "value_field": "value",
            },
            "schema": {"mode": "strict", "fields": ["value: float", "label: str"]},
        },
    )
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=SinkInput)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Should detect BOTH mismatches (source→agg has 'label' missing, agg→sink has 'average' missing)
    with pytest.raises(ValueError, match=r"label|average"):
        graph.validate_edge_compatibility()


def test_orphaned_config_gate_crashes_with_diagnostic() -> None:
    """Config gate with no incoming edges is a graph construction bug - should crash with clear error."""

    graph = ExecutionGraph()
    graph.add_node("gate", node_type=NodeType.GATE, plugin_name="config_gate", input_schema=None, output_schema=None)  # Config gate
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerSchema)

    # Accidentally wired gate→sink without source→gate
    graph.add_edge("gate", "sink", label="continue")

    # Should crash with diagnostic error (not silent failure)
    with pytest.raises(ValueError, match=r"no incoming edges"):
        graph.validate_edge_compatibility()


def test_schema_mismatch_error_includes_field_name_and_nodes() -> None:
    """Error messages must be actionable - include field names and node IDs."""

    graph = ExecutionGraph()
    graph.add_node("csv_reader", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerSchema)  # Has: id, name
    graph.add_node("db_writer", node_type=NodeType.SINK, plugin_name="database", input_schema=ConsumerSchema)  # Needs: id, name, email

    graph.add_edge("csv_reader", "db_writer", label="continue")

    try:
        graph.validate_edge_compatibility()
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error = str(e)
        # Must include both node names
        assert "csv_reader" in error, "Error must name producer node"
        assert "db_writer" in error, "Error must name consumer node"
        # Must include missing field name
        assert "email" in error.lower(), "Error must name missing field"


def test_edge_validation_detects_type_mismatch() -> None:
    """Edges should fail if producer type doesn't match consumer expected type.

    Bug: P2-2026-01-21-type-mismatches - Previously only field names were checked,
    not field types. Producer `value: str` should fail when consumer expects `value: int`.
    """
    from pydantic import ConfigDict

    class ProducerWithString(PluginSchema):
        """Producer outputs string."""

        value: str

    class ConsumerExpectsInt(PluginSchema):
        """Consumer expects int."""

        model_config = ConfigDict(extra="ignore")
        value: int

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerWithString)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerExpectsInt)
    graph.add_edge("source", "sink", label="continue")

    with pytest.raises(ValueError, match=r"Type mismatch.*value.*expected int.*got str"):
        graph.validate_edge_compatibility()


def test_edge_validation_allows_numeric_coercion() -> None:
    """int -> float coercion should be allowed (Pydantic default)."""

    class ProducerWithInt(PluginSchema):
        """Producer outputs int."""

        value: int

    class ConsumerExpectsFloat(PluginSchema):
        """Consumer expects float - int is coercible."""

        value: float

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerWithInt)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=ConsumerExpectsFloat)
    graph.add_edge("source", "sink", label="continue")

    # Should NOT raise - int is coercible to float
    graph.validate_edge_compatibility()


def test_strict_consumer_rejects_extra_fields() -> None:
    """Consumer with extra='forbid' should reject producers with extra fields.

    Bug: P2-2026-01-21-strict-extra-fields - Previously extra fields were never
    checked, allowing schema mismatch when consumer has strict validation.
    """
    from pydantic import ConfigDict

    class ProducerWithExtras(PluginSchema):
        """Producer has more fields than consumer expects."""

        id: int
        name: str
        extra_field: str  # This field is NOT in consumer

    class StrictConsumer(PluginSchema):
        """Consumer forbids extra fields."""

        model_config = ConfigDict(extra="forbid")
        id: int
        name: str

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerWithExtras)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=StrictConsumer)
    graph.add_edge("source", "sink", label="continue")

    with pytest.raises(ValueError, match=r"Extra fields forbidden.*extra_field"):
        graph.validate_edge_compatibility()


def test_permissive_consumer_allows_extra_fields() -> None:
    """Consumer with extra='ignore' (default) should accept producers with extra fields."""

    class ProducerWithExtras(PluginSchema):
        """Producer has more fields than consumer needs."""

        id: int
        name: str
        extra_field: str

    class PermissiveConsumer(PluginSchema):
        """Consumer ignores extra fields (default PluginSchema behavior)."""

        id: int
        name: str
        # extra='ignore' is the default from PluginSchema base class

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=ProducerWithExtras)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=PermissiveConsumer)
    graph.add_edge("source", "sink", label="continue")

    # Should NOT raise - extra fields are ignored
    graph.validate_edge_compatibility()


def test_dynamic_producer_with_strict_consumer_passes() -> None:
    """Dynamic producer should pass validation even with strict consumer.

    Dynamic schemas (no fields + extra='allow') represent runtime-determined
    schemas that cannot be validated statically. They bypass all validation.
    """
    from pydantic import ConfigDict

    from elspeth.plugins.schema_factory import _create_dynamic_schema

    class StrictConsumer(PluginSchema):
        """Consumer forbids extra fields."""

        model_config = ConfigDict(extra="forbid")
        id: int
        name: str

    dynamic_schema = _create_dynamic_schema("DynamicOutput")

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=dynamic_schema)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", input_schema=StrictConsumer)
    graph.add_edge("source", "sink", label="continue")

    # Should NOT raise - dynamic schemas bypass static validation
    graph.validate_edge_compatibility()
