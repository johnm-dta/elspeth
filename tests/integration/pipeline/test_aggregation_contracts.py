# tests/integration/pipeline/test_aggregation_contracts.py
"""Integration tests for aggregation schema contract validation.

Aggregations are the ONLY node type with distinct input_schema vs output_schema:
- Input schema: What fields the aggregation REQUIRES from upstream
- Output schema: What fields the aggregation GUARANTEES to downstream

These tests verify that contract validation works correctly for both edges.

Migrated from tests/integration/test_aggregation_contracts.py.

NOTE: Manual ExecutionGraph construction is intentional here. These tests
verify DAG edge compatibility validation logic, not pipeline assembly. They
need specific node configurations to test schema contract edge cases, which
makes from_plugin_instances() inappropriate (it would require real plugin
instances with matching schemas).
"""

import pytest

from elspeth.contracts import NodeType
from elspeth.core.dag import ExecutionGraph


class TestAggregationInputContracts:
    """Tests for aggregation input contract validation (incoming edge)."""

    def test_aggregation_input_requires_field_source_provides(self) -> None:
        """Aggregation passes when source guarantees required input fields."""
        graph = ExecutionGraph()

        # Source guarantees the field aggregation needs
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["value", "timestamp"]}},
        )

        # Aggregation requires 'value' field for input
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count", "sum"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],
                },
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        # Should pass - source provides what aggregation needs
        graph.validate_edge_compatibility()

    def test_aggregation_input_missing_required_field_fails(self) -> None:
        """Aggregation fails when source doesn't guarantee required input field."""
        graph = ExecutionGraph()

        # Source only guarantees 'id', not 'value'
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id"]}},
        )

        # Aggregation requires 'value' field that source doesn't provide
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count", "sum"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],  # Source doesn't provide this!
                },
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        error = str(exc_info.value)
        assert "value" in error
        assert "Missing fields" in error

    def test_aggregation_input_with_multiple_required_fields(self) -> None:
        """Aggregation validation checks all required input fields."""
        graph = ExecutionGraph()

        # Source only provides some fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "amount"]}},
        )

        # Aggregation requires multiple fields, one is missing
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="grouped_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed"},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["id", "amount", "category"],  # 'category' missing!
                },
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="category"):
            graph.validate_edge_compatibility()


class TestAggregationOutputContracts:
    """Tests for aggregation output contract validation (outgoing edge)."""

    def test_aggregation_output_satisfies_sink_requirements(self) -> None:
        """Aggregation passes when it guarantees what sink requires."""
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["value"]}},
        )

        # Aggregation guarantees the fields sink needs
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count", "sum", "mean"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],
                },
            },
        )

        # Sink requires subset of what aggregation guarantees. Sink
        # requirements are enforced from declared_required_fields (the
        # builder invariant: populated from SinkProtocol), not the raw
        # config schema — mirror both, as the builder does.
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["count", "sum"]},
            },
            declared_required_fields=frozenset({"count", "sum"}),
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        # Should pass - aggregation provides what sink needs
        graph.validate_edge_compatibility()

    def test_aggregation_output_missing_sink_required_field_fails(self) -> None:
        """Aggregation fails when it doesn't guarantee field sink requires."""
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["value"]}},
        )

        # Aggregation only guarantees count and sum
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count", "sum"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],
                },
            },
        )

        # Sink requires 'median' which aggregation doesn't guarantee. As in
        # the positive test above, the enforced requirement set is
        # declared_required_fields (builder invariant), and the rejection
        # comes from validate_sink_required_fields — the single owner of
        # sink required-field checks (elspeth-3283f2eaec).
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["count", "median"]},
            },
            declared_required_fields=frozenset({"count", "median"}),
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        error = str(exc_info.value)
        assert "median" in error
        assert "does not guarantee" in error

    def test_direct_sink_config_required_fields_without_declared_argument_still_fail(self) -> None:
        """Direct graph sink config requirements are still enforced."""
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["value"]}},
        )

        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count", "sum"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],
                },
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["count", "median"]},
            },
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        error = str(exc_info.value)
        assert "median" in error
        assert "does not guarantee" in error


class TestAggregationChainValidation:
    """Tests for aggregation in multi-node chains (both edges validated)."""

    def test_aggregation_chain_both_edges_validated(self) -> None:
        """Aggregation in chain validates both input and output contracts."""
        graph = ExecutionGraph()

        # Source guarantees raw input fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["raw_value", "timestamp"]}},
        )

        # Aggregation: requires raw_value, guarantees stats
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["batch_id", "count", "sum"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["raw_value"],
                },
            },
        )

        # Transform after aggregation requires aggregation outputs
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="calculator",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["batch_id", "count", "sum", "average"]},
                "required_input_fields": ["count", "sum"],
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        # Should pass - contracts satisfied at both edges
        graph.validate_edge_compatibility()

    def test_aggregation_chain_fails_on_input_gap(self) -> None:
        """Chain fails when aggregation input requirements not met."""
        graph = ExecutionGraph()

        # Source doesn't provide what aggregation needs
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["wrong_field"]}},
        )

        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count", "sum"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],  # Source doesn't have this
                },
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="value"):
            graph.validate_edge_compatibility()

    def test_aggregation_chain_fails_on_output_gap(self) -> None:
        """Chain fails when aggregation output doesn't satisfy downstream."""
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["value"]}},
        )

        # Aggregation provides some stats but not all
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed", "guaranteed_fields": ["count"]},
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],
                },
            },
        )

        # Transform needs field aggregation doesn't provide
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="calculator",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["count", "sum"],  # 'sum' not guaranteed by agg
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="sum"):
            graph.validate_edge_compatibility()


class TestAggregationDynamicSchemas:
    """Tests for aggregation with dynamic schemas."""

    def test_aggregation_pure_dynamic_skips_contract_validation(self) -> None:
        """Pure dynamic schemas (no contracts) skip validation."""
        graph = ExecutionGraph()

        # All dynamic, no contracts declared
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed"},  # No contracts
                "options": {"schema": {"mode": "observed"}},
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        # Should pass - no contracts to validate
        graph.validate_edge_compatibility()

    def test_aggregation_with_partial_contracts(self) -> None:
        """Aggregation with contracts only on one side validates that side."""
        graph = ExecutionGraph()

        # Source with guarantees
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["value"]}},
        )

        # Aggregation with input requirements but dynamic output
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "trigger": {"count": 1},
                "output_mode": "transform",
                "schema": {"mode": "observed"},  # No guaranteed_fields - dynamic output
                "options": {
                    "schema": {"mode": "observed"},
                    "required_input_fields": ["value"],
                },
            },
        )

        # Sink with no requirements (accepts anything)
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "agg_1", label="continue")
        graph.add_edge("agg_1", "sink_1", label="continue")

        # Should pass - input contract satisfied, output is dynamic
        graph.validate_edge_compatibility()
