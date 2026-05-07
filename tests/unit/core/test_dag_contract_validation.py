# tests/core/test_dag_contract_validation.py
"""Tests for DAG schema contract validation (guaranteed/required fields)."""

from typing import Any

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.dag import ExecutionGraph
from tests.helpers.coalesce import (
    _add_coalesce_with_computed_schema,
    _compute_coalesce_schema,
)


class TestContractHelpers:
    """Tests for contract extraction helper methods."""

    def test_get_guaranteed_fields_from_dynamic_with_explicit(self) -> None:
        """Extract explicit guaranteed_fields from dynamic schema."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "name"]}},
        )

        result = graph.get_guaranteed_fields("source_1")
        assert result == frozenset({"id", "name"})

    def test_get_guaranteed_fields_from_schema_config_alias(self) -> None:
        """Alias-form schema_config is honored by direct graph construction."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema_config": {"mode": "observed", "guaranteed_fields": ["id", "name"]}},
        )

        result = graph.get_guaranteed_fields("source_1")
        assert result == frozenset({"id", "name"})

    def test_get_guaranteed_fields_from_explicit_schema(self) -> None:
        """Extract guaranteed fields from free mode schema (implicit from declared fields)."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "flexible", "fields": ["a: int", "b: str"]}},
        )

        result = graph.get_guaranteed_fields("source_1")
        assert result == frozenset({"a", "b"})

    def test_get_guaranteed_fields_empty_for_pure_dynamic(self) -> None:
        """Pure dynamic schema without explicit guarantees returns empty set."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        result = graph.get_guaranteed_fields("source_1")
        assert result == frozenset()

    def test_get_required_fields_from_required_input_fields(self) -> None:
        """Required fields from explicit required_input_fields config."""
        graph = ExecutionGraph()
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["customer_id", "amount"],
            },
        )

        result = graph.get_required_fields("transform_1")
        assert result == frozenset({"customer_id", "amount"})

    def test_get_required_fields_from_schema_required_fields(self) -> None:
        """Required fields from schema's required_fields (fallback)."""
        graph = ExecutionGraph()
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed", "required_fields": ["x", "y"]},
            },
        )

        result = graph.get_required_fields("transform_1")
        assert result == frozenset({"x", "y"})

    def test_get_required_fields_from_aggregation_wrapper_schema_required_fields(self) -> None:
        """Aggregation wrapper config honors nested schema.required_fields."""
        graph = ExecutionGraph()
        graph.add_node(
            "agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={
                "input_schema": {"mode": "observed"},
                "options": {
                    "schema": {"mode": "observed", "required_fields": ["value"]},
                },
                "trigger": {"count": 1},
                "output_mode": "on_trigger",
            },
        )

        result = graph.get_required_fields("agg_1")
        assert result == frozenset({"value"})

    def test_get_required_fields_from_explicit_schema(self) -> None:
        """Implicit requirements from strict schema are NOT returned.

        Contract validation only considers explicit declarations
        (required_input_fields or required_fields). Type validation
        handles implicit requirements from typed schemas.
        """
        graph = ExecutionGraph()
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "fixed", "fields": ["id: int", "value: str"]}},
        )

        # Implicit requirements are NOT returned - type validation handles these
        result = graph.get_required_fields("sink_1")
        assert result == frozenset()

    def test_get_required_fields_from_explicit_required_fields(self) -> None:
        """Explicit required_fields in schema config ARE returned."""
        graph = ExecutionGraph()
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {
                    "mode": "observed",
                    "required_fields": ["customer_id", "amount"],
                },
            },
        )

        result = graph.get_required_fields("sink_1")
        assert result == frozenset({"customer_id", "amount"})

    def test_required_input_fields_takes_priority(self) -> None:
        """Explicit required_input_fields overrides schema's required_fields."""
        graph = ExecutionGraph()
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed", "required_fields": ["schema_req"]},
                "required_input_fields": ["config_req"],  # This wins
            },
        )

        result = graph.get_required_fields("transform_1")
        assert result == frozenset({"config_req"})


class TestEffectiveGuaranteedFields:
    """Tests for _get_effective_guaranteed_fields with pass-through nodes."""

    def test_source_guarantees_directly(self) -> None:
        """Source node's guarantees are returned directly."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a", "b"]}},
        )

        result = graph.get_effective_guaranteed_fields("source_1")
        assert result == frozenset({"a", "b"})

    def test_gate_uses_propagated_schema(self) -> None:
        """Gate node uses its own schema (propagated by builder from upstream).

        In production, the builder copies the upstream schema to gates.
        This test simulates that propagation — the gate's config already
        contains the upstream's guaranteed_fields.
        """
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["x", "y"]}},
        )
        graph.add_node(
            "gate_1",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["x", "y"]}},
        )
        graph.add_edge("source_1", "gate_1", label="continue")

        result = graph.get_effective_guaranteed_fields("gate_1")
        assert result == frozenset({"x", "y"})

    def test_coalesce_returns_intersection(self) -> None:
        """Coalesce node returns intersection of branch guarantees under best_effort.

        Under best_effort policy, branches may not arrive, so the merged
        guarantees are the intersection of branches' guaranteed_fields.
        Under require_all this would be the union — see
        test_dag_coalesce_optionality.py::test_require_all_* for that path.
        """
        graph = ExecutionGraph()

        # Fork gate with two branches
        graph.add_node(
            "gate_1",
            node_type=NodeType.GATE,
            plugin_name="fork_gate",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "branch_a"]}},
        )

        # Branch A source
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform_a",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only"]}},
        )

        # Branch B source
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform_b",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "b_only"]}},
        )

        # Compute merged schema BEFORE adding coalesce node (NodeInfo is frozen)
        branch_schemas = {
            "branch_a": graph.get_schema_config_from_node("branch_a"),
            "branch_b": graph.get_schema_config_from_node("branch_b"),
        }
        coalesce_schema = _compute_coalesce_schema(branch_schemas, policy="best_effort")
        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"merge": "union", "branches": {}, "policy": "best_effort"},
            output_schema_config=coalesce_schema,
        )

        graph.add_edge("branch_a", "coalesce_1", label="path_a")
        graph.add_edge("branch_b", "coalesce_1", label="path_b")

        # Coalesce guarantees only the intersection (best_effort semantics)
        result = graph.get_effective_guaranteed_fields("coalesce_1")
        assert result == frozenset({"common"})


class TestContractValidation:
    """Tests for contract validation in _validate_single_edge."""

    def test_producer_satisfies_consumer(self) -> None:
        """Producer with guaranteed fields satisfies consumer requirements."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a", "b"]}},
        )
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["a"],  # Requires only 'a'
            },
        )
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        # Should not raise - producer guarantees what consumer needs
        graph.validate_edge_compatibility()

    def test_producer_missing_required_fails(self) -> None:
        """Producer without required field raises clear error."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a"]}},
        )
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["a", "b"],  # Requires 'b' which source doesn't have
            },
        )
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="Schema contract violation"):
            graph.validate_edge_compatibility()

    def test_error_message_contains_missing_fields(self) -> None:
        """Error message shows exactly which fields are missing."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["customer_id", "order_amount"],
            },
        )
        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("transform_1", "sink_1", label="continue")

        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        error_message = str(exc_info.value)
        assert "customer_id" in error_message
        assert "order_amount" in error_message
        assert "Missing fields" in error_message

    def test_dynamic_producer_fails_requirements(self) -> None:
        """Dynamic producer cannot satisfy consumer requirements."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},  # No guarantees
        )
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["required_field"],
            },
        )
        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("transform_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="none - dynamic schema"):
            graph.validate_edge_compatibility()

    def test_consumer_without_requirements_passes(self) -> None:
        """Consumer without required fields always passes contract check."""
        graph = ExecutionGraph()
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},  # No requirements
        )
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        # Should not raise - no requirements to satisfy
        graph.validate_edge_compatibility()


class TestChainValidation:
    """Tests for contract validation across multi-node chains."""

    def test_three_node_chain_catches_middle_gap(self) -> None:
        """Chain validation catches missing field in middle of chain."""
        graph = ExecutionGraph()

        # Source guarantees [a, b]
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a", "b"]}},
        )

        # Transform consumes [a], produces [a, c] (drops b, adds c)
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="mapper",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "c"]},
                "required_input_fields": ["a"],
            },
        )

        # Sink requires [a, b] - should fail because transform dropped b
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["a", "b"]},
            },
        )

        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match=r"Missing fields.*'b'"):
            graph.validate_edge_compatibility()

    def test_explicit_schema_combined_with_contract(self) -> None:
        """Explicit schema fields combine with contract fields for validation."""
        graph = ExecutionGraph()

        # Source with explicit free mode schema
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "flexible", "fields": ["id: int", "name: str"]}},
        )

        # Transform requires id (which source guarantees)
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"mode": "observed"},
                "required_input_fields": ["id"],
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},
        )

        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "sink_1", label="continue")

        # Should pass - source's explicit schema guarantees 'id'
        graph.validate_edge_compatibility()

    def test_five_node_chain_detects_late_gap(self) -> None:
        """Long chain where field dropped early is only required late."""
        graph = ExecutionGraph()

        # Source guarantees [a, b, c]
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c"]}},
        )

        # Transform 1: consumes [a], produces [a, b, c] (pass-through)
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="t1",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c"]},
                "required_input_fields": ["a"],
            },
        )

        # Transform 2: consumes [b], produces [a, b, c] (pass-through)
        graph.add_node(
            "transform_2",
            node_type=NodeType.TRANSFORM,
            plugin_name="t2",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c"]},
                "required_input_fields": ["b"],
            },
        )

        # Transform 3: consumes [c], produces [a, b, d] (DROPS c, adds d)
        graph.add_node(
            "transform_3",
            node_type=NodeType.TRANSFORM,
            plugin_name="t3",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "d"]},
                "required_input_fields": ["c"],
            },
        )

        # Sink requires [a, b, c] - should FAIL because c was dropped by transform_3
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["a", "b", "c"]},
            },
        )

        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "transform_2", label="continue")
        graph.add_edge("transform_2", "transform_3", label="continue")
        graph.add_edge("transform_3", "sink_1", label="continue")

        with pytest.raises(ValueError, match="'c'"):
            graph.validate_edge_compatibility()

    def test_five_node_chain_passes_when_contracts_satisfied(self) -> None:
        """Long chain where all contracts are satisfied passes validation."""
        graph = ExecutionGraph()

        # Source guarantees [a, b, c]
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c"]}},
        )

        # Transform 1: consumes [a], adds [d], keeps all
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="t1",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c", "d"]},
                "required_input_fields": ["a"],
            },
        )

        # Transform 2: consumes [b], adds [e], keeps all
        graph.add_node(
            "transform_2",
            node_type=NodeType.TRANSFORM,
            plugin_name="t2",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c", "d", "e"]},
                "required_input_fields": ["b"],
            },
        )

        # Transform 3: consumes [c, d], adds [f], keeps all
        graph.add_node(
            "transform_3",
            node_type=NodeType.TRANSFORM,
            plugin_name="t3",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c", "d", "e", "f"]},
                "required_input_fields": ["c", "d"],
            },
        )

        # Sink requires [a, b, c, f] - all should be guaranteed
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["a", "b", "c", "f"]},
            },
        )

        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "transform_2", label="continue")
        graph.add_edge("transform_2", "transform_3", label="continue")
        graph.add_edge("transform_3", "sink_1", label="continue")

        # Should pass - all contracts satisfied
        graph.validate_edge_compatibility()


class TestGatePassthrough:
    """Tests for gates preserving upstream guarantees."""

    def test_gate_passes_through_guarantees(self) -> None:
        """Gate node passes upstream guarantees to downstream consumer.

        In production, the builder propagates the upstream schema to the gate.
        This test simulates that propagation.
        """
        graph = ExecutionGraph()

        # Source with guarantees
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["field_a"]}},
        )

        # Gate (passthrough — builder would copy source schema here)
        graph.add_node(
            "gate_1",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["field_a"]}},
        )

        # Sink requires field_a
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["field_a"]},
            },
        )

        graph.add_edge("source_1", "gate_1", label="continue")
        graph.add_edge("gate_1", "sink_1", label="continue")

        # Should pass - gate passes through source's guarantees
        graph.validate_edge_compatibility()


class TestForkCoalesceContracts:
    """Tests for fork → multiple branches → coalesce contract flows."""

    def test_fork_with_different_branch_transforms(self) -> None:
        """Fork gate to two branches with different transforms, coalesce guarantees union under require_all."""
        graph = ExecutionGraph()

        # Source guarantees base fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "raw_data"]}},
        )

        # Fork gate (builder would propagate source schema here)
        graph.add_node(
            "fork_gate",
            node_type=NodeType.GATE,
            plugin_name="fork_gate",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "raw_data"]}},
        )

        # Branch A: adds 'classification' field
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="classifier",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["id", "raw_data", "classification"]},
                "required_input_fields": ["raw_data"],
            },
        )

        # Branch B: adds 'sentiment' field
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="sentiment_analyzer",
            config={
                "schema": {"mode": "observed", "guaranteed_fields": ["id", "raw_data", "sentiment"]},
                "required_input_fields": ["raw_data"],
            },
        )

        # Coalesce merges branches - guarantees UNION under require_all
        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"schema": {"mode": "observed"}, "merge": "union", "branches": {}, "policy": "require_all"},
        )

        # Sink requires only 'id' (which is in union)
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["id"]},
            },
        )

        graph.add_edge("source_1", "fork_gate", label="continue")
        graph.add_edge("fork_gate", "branch_a", label="path_a")
        graph.add_edge("fork_gate", "branch_b", label="path_b")

        # Add coalesce with computed schema (must be done before adding edges TO it)
        _add_coalesce_with_computed_schema(graph, "coalesce_1", ["branch_a", "branch_b"], policy="require_all")

        graph.add_edge("branch_a", "coalesce_1", label="merge")
        graph.add_edge("branch_b", "coalesce_1", label="merge")
        graph.add_edge("coalesce_1", "sink_1", label="continue")

        # Should pass - sink requires 'id' which is in union
        graph.validate_edge_compatibility()

    def test_coalesce_intersection_rejects_branch_specific_requirement(self) -> None:
        """Under best_effort, a consumer after coalesce cannot require branch-specific fields.

        Under best_effort, the branch producing 'a_only' may be lost — the
        merged row may be missing it, so a sink requiring 'a_only' must be
        rejected at build time. Under require_all the same configuration
        would be valid — see test_dag_coalesce_optionality.py for that path.
        """
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common"]}},
        )

        # Branch A: guarantees common + a_only
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform_a",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only"]}},
        )

        # Branch B: guarantees common + b_only
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform_b",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "b_only"]}},
        )

        graph.add_edge("source_1", "branch_a", label="path_a")
        graph.add_edge("source_1", "branch_b", label="path_b")

        # Add coalesce with computed schema (must be done before adding edges TO it)
        _add_coalesce_with_computed_schema(graph, "coalesce_1", ["branch_a", "branch_b"], policy="best_effort")

        # Sink requires 'a_only' - NOT guaranteed by coalesce (only 'common' is)
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["a_only"]},
            },
        )

        graph.add_edge("branch_a", "coalesce_1", label="merge")
        graph.add_edge("branch_b", "coalesce_1", label="merge")
        graph.add_edge("coalesce_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="a_only"):
            graph.validate_edge_compatibility()

    def test_coalesce_three_branches_intersection(self) -> None:
        """Coalesce with three branches takes intersection of all three."""
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id"]}},
        )

        # Three branches with overlapping guarantees
        # Branch A: id, x, y
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "x", "y"]}},
        )

        # Branch B: id, x, z
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "x", "z"]}},
        )

        # Branch C: id, y, z
        graph.add_node(
            "branch_c",
            node_type=NodeType.TRANSFORM,
            plugin_name="c",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "y", "z"]}},
        )

        # Add coalesce with computed schema (simulates builder)
        # Under require_all: union = {id, x, y, z}
        _add_coalesce_with_computed_schema(graph, "coalesce_1", ["branch_a", "branch_b", "branch_c"], policy="require_all")

        # Sink requires only 'id' (the only field guaranteed by ALL branches)
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["id"]},
            },
        )

        graph.add_edge("source_1", "branch_a", label="a")
        graph.add_edge("source_1", "branch_b", label="b")
        graph.add_edge("source_1", "branch_c", label="c")
        graph.add_edge("branch_a", "coalesce_1", label="merge")
        graph.add_edge("branch_b", "coalesce_1", label="merge")
        graph.add_edge("branch_c", "coalesce_1", label="merge")
        graph.add_edge("coalesce_1", "sink_1", label="continue")

        # Should pass - 'id' is in the union of all three branches
        graph.validate_edge_compatibility()

    def test_coalesce_empty_intersection_fails_requirements(self) -> None:
        """Coalesce with no common fields fails any downstream requirements."""
        graph = ExecutionGraph()

        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},  # No guarantees
        )

        # Branch A: only guarantees 'a'
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["a"]}},
        )

        # Branch B: only guarantees 'b'
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["b"]}},
        )

        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"schema": {"mode": "observed"}, "merge": "union", "branches": {}, "policy": "require_all"},
        )

        # Sink requires any field - will fail since intersection is empty
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["any_field"]},
            },
        )

        graph.add_edge("source_1", "branch_a", label="a")
        graph.add_edge("source_1", "branch_b", label="b")
        graph.add_edge("branch_a", "coalesce_1", label="merge")
        graph.add_edge("branch_b", "coalesce_1", label="merge")
        graph.add_edge("coalesce_1", "sink_1", label="continue")

        with pytest.raises(ValueError, match="any_field"):
            graph.validate_edge_compatibility()


class TestCoalesceGuaranteedFieldsSemantics:
    """Tests for None vs empty-tuple distinction in coalesce guaranteed_fields.

    The intersection of guaranteed_fields across coalesce branches must
    distinguish between:
      None  = "observed schema, unknown fields" → abstain from intersection
      ()    = "explicitly guarantee zero fields" → participates, kills intersection

    Uses best_effort policy because intersection semantics is the correct
    merge for policies where branches can be lost (best_effort/quorum/first).
    Under require_all every branch always arrives, so the merged guarantees
    are the UNION (any branch's guarantee is in the merged row) — see
    test_dag_coalesce_optionality.py::test_require_all_* for that path.
    """

    def _build_coalesce_graph(
        self,
        branch_configs: dict[str, dict[str, Any]],
        *,
        policy: str = "best_effort",
    ) -> ExecutionGraph:
        """Build a coalesce graph with given schema configs.

        Simulates what the builder does: computes merged guarantees from
        branch schemas and sets output_schema_config on the coalesce node.
        """
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # First pass: add branch nodes and collect their schemas
        branch_schemas: dict[str, SchemaConfig | None] = {}
        for branch_name, config in branch_configs.items():
            graph.add_node(
                branch_name,
                node_type=NodeType.TRANSFORM,
                plugin_name="transform",
                config=config,
            )
            graph.add_edge("source", branch_name, label=branch_name)
            # Get the parsed schema from the node we just added
            branch_schemas[branch_name] = graph.get_schema_config_from_node(branch_name)

        # Compute merged schema BEFORE adding coalesce node (NodeInfo is frozen)
        coalesce_schema = _compute_coalesce_schema(branch_schemas, policy=policy)

        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"merge": "union", "branches": {}, "policy": policy},
            output_schema_config=coalesce_schema,
        )
        for branch_name in branch_configs:
            graph.add_edge(branch_name, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")
        return graph

    def test_both_branches_with_guarantees_intersects(self) -> None:
        """Two branches with explicit guaranteed_fields → intersection."""
        graph = self._build_coalesce_graph(
            {
                "branch_a": {"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only"]}},
                "branch_b": {"schema": {"mode": "observed", "guaranteed_fields": ["common", "b_only"]}},
            }
        )
        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset({"common"})

    def test_one_branch_none_guarantees_abstains(self) -> None:
        """Branch with None guaranteed_fields abstains — doesn't kill intersection."""
        graph = self._build_coalesce_graph(
            {
                "branch_a": {"schema": {"mode": "observed", "guaranteed_fields": ["x", "y"]}},
                "branch_b": {"schema": {"mode": "observed"}},  # guaranteed_fields=None
            }
        )
        result = graph.get_effective_guaranteed_fields("coalesce")
        # branch_b abstains, so branch_a's guarantees survive
        assert result == frozenset({"x", "y"})

    def test_all_branches_none_guarantees_returns_empty(self) -> None:
        """All branches with None guaranteed_fields → no guarantees (empty set)."""
        graph = self._build_coalesce_graph(
            {
                "branch_a": {"schema": {"mode": "observed"}},
                "branch_b": {"schema": {"mode": "observed"}},
            }
        )
        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset()

    def test_empty_list_guarantees_treated_as_none(self) -> None:
        """Empty list in config is normalized to None by _parse_field_names_list → abstains."""
        graph = self._build_coalesce_graph(
            {
                "branch_a": {"schema": {"mode": "observed", "guaranteed_fields": ["x", "y"]}},
                "branch_b": {"schema": {"mode": "observed", "guaranteed_fields": []}},  # [] → None
            }
        )
        result = graph.get_effective_guaranteed_fields("coalesce")
        # Empty list is parsed as None (unspecified), so branch_b abstains
        assert result == frozenset({"x", "y"})

    def test_explicit_empty_tuple_kills_intersection(self) -> None:
        """SchemaConfig with guaranteed_fields=() (Python API) participates and kills intersection.

        This distinction only exists at the Python API level — config parsing
        normalizes [] to None. A transform that explicitly constructs
        SchemaConfig(guaranteed_fields=()) is saying "I guarantee zero fields."
        """
        from elspeth.contracts.schema import SchemaConfig

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["x", "y"]}},
        )
        # Manually set output_schema_config with explicit empty tuple
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=SchemaConfig(mode="observed", fields=None, guaranteed_fields=()),
        )

        # Add coalesce with computed schema (simulates builder)
        # branch_b explicitly guarantees nothing → intersection is ∅ (best_effort)
        _add_coalesce_with_computed_schema(graph, "coalesce", ["branch_a", "branch_b"], policy="best_effort")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "branch_a", label="a")
        graph.add_edge("source", "branch_b", label="b")
        graph.add_edge("branch_a", "coalesce", label="continue")
        graph.add_edge("branch_b", "coalesce", label="continue")
        graph.add_edge("coalesce", "sink", label="continue")

        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset()

    def test_three_branches_mixed_none_and_explicit(self) -> None:
        """Three branches: two explicit, one None → intersection of explicit only."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["x", "y", "z"]}},
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["x", "z"]}},
        )
        graph.add_node(
            "branch_c",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed"}},  # None → abstains
        )
        # Add coalesce with computed schema (simulates builder)
        # branch_c abstains, intersection of a ∩ b = {"x", "z"}
        _add_coalesce_with_computed_schema(graph, "coalesce", ["branch_a", "branch_b", "branch_c"], policy="best_effort")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        for b in ("branch_a", "branch_b", "branch_c"):
            graph.add_edge("source", b, label=b)
            graph.add_edge(b, "coalesce", label="continue")
        graph.add_edge("coalesce", "sink", label="continue")

        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset({"x", "z"})

    def test_typed_required_fields_participate_without_explicit_guaranteed_fields(
        self,
    ) -> None:
        """Typed required fields contribute guarantees even when guaranteed_fields=None.

        Bug reproduction: A branch with mode="fixed", fields=(id, x) required=True,
        but guaranteed_fields=None should still contribute {id, x} to the coalesce
        merge. The declares_guaranteed_fields property only checks explicit
        guaranteed_fields, but get_effective_guaranteed_fields() includes typed
        required fields — the coalesce merge loop must consider both sources.

        Under require_all, both branches always arrive, so:
        - branch_a contributes {id, x} from typed required fields
        - branch_b contributes {id} from typed required fields
        - Merged guarantees = {id, x} (union under require_all)
        """
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Branch A: typed fields (id, x) required, NO explicit guaranteed_fields
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=SchemaConfig(
                mode="fixed",
                fields=(
                    FieldDefinition("id", "int", required=True),
                    FieldDefinition("x", "str", required=True),
                ),
                guaranteed_fields=None,  # Abstains from explicit declaration
            ),
        )

        # Branch B: typed field (id) required, NO explicit guaranteed_fields
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=SchemaConfig(
                mode="fixed",
                fields=(FieldDefinition("id", "int", required=True),),
                guaranteed_fields=None,  # Abstains from explicit declaration
            ),
        )

        # Add coalesce with computed schema (simulates builder)
        # Both branches have typed required fields that ARE effective guarantees
        # Under require_all, merged guarantees = union = {id, x}
        _add_coalesce_with_computed_schema(graph, "coalesce", ["branch_a", "branch_b"], policy="require_all")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)
            graph.add_edge(b, "coalesce", label="continue")
        graph.add_edge("coalesce", "sink", label="continue")

        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset({"id", "x"})

    def test_quorum_equal_to_branch_count_uses_union_semantics(self) -> None:
        """quorum_count == len(branches) is semantically identical to require_all.

        Bug reproduction: policy='quorum' with quorum_count == 2 and 2 branches
        is runtime-equivalent to require_all — _should_merge() waits for all,
        _evaluate_after_loss() fails on any loss. But the builder was using
        intersection semantics, rejecting valid branch-exclusive field requirements.

        Under quorum=N (where N == branch_count), every branch always arrives,
        so the union of branch guarantees should be the merged guarantees.
        """
        # Build with quorum policy directly (quorum_count == len(branches) uses union semantics)
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only"]}},
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "b_only"]}},
        )

        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)

        # Add coalesce with computed schema (simulates builder)
        # quorum=2 with 2 branches == require_all semantics → UNION
        # Note: we pass policy="require_all" to get union semantics,
        # but the node config still says "quorum" with quorum_count=2
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce",
            ["branch_a", "branch_b"],
            policy="require_all",  # quorum=N uses union like require_all
            extra_config={
                "branches": {"branch_a": {}, "branch_b": {}},
                "policy": "quorum",
                "quorum_count": 2,
            },
        )

        for b in ("branch_a", "branch_b"):
            graph.add_edge(b, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")

        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset({"common", "a_only", "b_only"})


class TestNestedCoalesceSchemaProgation:
    """Tests for schema propagation through nested coalesce nodes.

    When coalesce nodes chain (coalesce1 feeds into coalesce2), the downstream
    coalesce must see the computed guarantees of its upstream coalesce, not
    the raw branch schemas. This verifies transitive propagation works correctly.

    Example topology:
        source → branch_a ─┐
                           ├─→ coalesce_inner ─┐
        source → branch_b ─┘                   │
                                               ├─→ coalesce_outer → sink
        source → branch_c ───────────────────→─┘
    """

    def test_nested_coalesce_require_all_union_propagates(self) -> None:
        """Nested require_all coalesces propagate union guarantees transitively.

        Inner coalesce (require_all): union of branch_a | branch_b
        Outer coalesce (require_all): union of inner | branch_c

        The outer coalesce must see the inner's computed union, not recompute
        from the inner's raw predecessors.
        """
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Branch A: guarantees {common, a_only}
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only"]}},
        )

        # Branch B: guarantees {common, b_only}
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "b_only"]}},
        )

        # Branch C: guarantees {common, c_only}
        graph.add_node(
            "branch_c",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "c_only"]}},
        )

        # Inner coalesce (require_all): union of A | B = {common, a_only, b_only}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_inner",
            ["branch_a", "branch_b"],
            policy="require_all",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        # Outer coalesce (require_all): union of inner | C
        # Inner contributes {common, a_only, b_only}, C contributes {common, c_only}
        # Expected: {common, a_only, b_only, c_only}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_outer",
            ["coalesce_inner", "branch_c"],
            policy="require_all",
            extra_config={"branches": {"coalesce_inner": {}, "branch_c": {}}},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Wire up edges
        graph.add_edge("source", "branch_a", label="a")
        graph.add_edge("source", "branch_b", label="b")
        graph.add_edge("source", "branch_c", label="c")
        graph.add_edge("branch_a", "coalesce_inner", label="continue")
        graph.add_edge("branch_b", "coalesce_inner", label="continue")
        graph.add_edge("coalesce_inner", "coalesce_outer", label="continue")
        graph.add_edge("branch_c", "coalesce_outer", label="continue")
        graph.add_edge("coalesce_outer", "sink", label="continue")

        # Verify inner coalesce has correct guarantees
        inner_result = graph.get_effective_guaranteed_fields("coalesce_inner")
        assert inner_result == frozenset({"common", "a_only", "b_only"}), f"Inner coalesce should have union of A | B, got {inner_result}"

        # Verify outer coalesce sees inner's computed guarantees
        outer_result = graph.get_effective_guaranteed_fields("coalesce_outer")
        assert outer_result == frozenset({"common", "a_only", "b_only", "c_only"}), (
            f"Outer coalesce should have union of inner | C, got {outer_result}"
        )

    def test_nested_coalesce_best_effort_intersection_propagates(self) -> None:
        """Nested best_effort coalesces propagate intersection guarantees transitively.

        Inner coalesce (best_effort): intersection of branch_a ∩ branch_b
        Outer coalesce (best_effort): intersection of inner ∩ branch_c

        The outer coalesce must see the inner's computed intersection.
        """
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Branch A: guarantees {common, shared_ab, a_only}
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "shared_ab", "a_only"]}},
        )

        # Branch B: guarantees {common, shared_ab, b_only}
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "shared_ab", "b_only"]}},
        )

        # Branch C: guarantees {common, c_only}
        graph.add_node(
            "branch_c",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "c_only"]}},
        )

        # Inner coalesce (best_effort): intersection of A ∩ B = {common, shared_ab}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_inner",
            ["branch_a", "branch_b"],
            policy="best_effort",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        # Outer coalesce (best_effort): intersection of inner ∩ C
        # Inner contributes {common, shared_ab}, C contributes {common, c_only}
        # Expected: {common} (only field in both)
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_outer",
            ["coalesce_inner", "branch_c"],
            policy="best_effort",
            extra_config={"branches": {"coalesce_inner": {}, "branch_c": {}}},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Wire up edges
        graph.add_edge("source", "branch_a", label="a")
        graph.add_edge("source", "branch_b", label="b")
        graph.add_edge("source", "branch_c", label="c")
        graph.add_edge("branch_a", "coalesce_inner", label="continue")
        graph.add_edge("branch_b", "coalesce_inner", label="continue")
        graph.add_edge("coalesce_inner", "coalesce_outer", label="continue")
        graph.add_edge("branch_c", "coalesce_outer", label="continue")
        graph.add_edge("coalesce_outer", "sink", label="continue")

        # Verify inner coalesce has correct guarantees
        inner_result = graph.get_effective_guaranteed_fields("coalesce_inner")
        assert inner_result == frozenset({"common", "shared_ab"}), f"Inner coalesce should have intersection of A ∩ B, got {inner_result}"

        # Verify outer coalesce sees inner's computed guarantees
        outer_result = graph.get_effective_guaranteed_fields("coalesce_outer")
        assert outer_result == frozenset({"common"}), f"Outer coalesce should have intersection of inner ∩ C, got {outer_result}"

    def test_nested_coalesce_mixed_policies(self) -> None:
        """Mixed policies: inner require_all (union) feeds outer best_effort (intersection).

        This tests the edge case where policy changes between nesting levels.
        The outer must see the inner's union result, then intersect with branch_c.
        """
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Branch A: guarantees {common, a_only}
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only"]}},
        )

        # Branch B: guarantees {common, b_only}
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "b_only"]}},
        )

        # Branch C: guarantees {common, a_only, c_only}
        # Note: a_only is in both A and C, but not B
        graph.add_node(
            "branch_c",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["common", "a_only", "c_only"]}},
        )

        # Inner coalesce (require_all): union of A | B = {common, a_only, b_only}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_inner",
            ["branch_a", "branch_b"],
            policy="require_all",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        # Outer coalesce (best_effort): intersection of inner ∩ C
        # Inner contributes {common, a_only, b_only}, C contributes {common, a_only, c_only}
        # Expected: {common, a_only} (fields in both)
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_outer",
            ["coalesce_inner", "branch_c"],
            policy="best_effort",
            extra_config={"branches": {"coalesce_inner": {}, "branch_c": {}}},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Wire up edges
        graph.add_edge("source", "branch_a", label="a")
        graph.add_edge("source", "branch_b", label="b")
        graph.add_edge("source", "branch_c", label="c")
        graph.add_edge("branch_a", "coalesce_inner", label="continue")
        graph.add_edge("branch_b", "coalesce_inner", label="continue")
        graph.add_edge("coalesce_inner", "coalesce_outer", label="continue")
        graph.add_edge("branch_c", "coalesce_outer", label="continue")
        graph.add_edge("coalesce_outer", "sink", label="continue")

        # Verify inner coalesce has correct guarantees (union)
        inner_result = graph.get_effective_guaranteed_fields("coalesce_inner")
        assert inner_result == frozenset({"common", "a_only", "b_only"}), f"Inner coalesce should have union of A | B, got {inner_result}"

        # Verify outer coalesce: intersection of inner's union with C
        outer_result = graph.get_effective_guaranteed_fields("coalesce_outer")
        assert outer_result == frozenset({"common", "a_only"}), f"Outer coalesce should have intersection of inner | C, got {outer_result}"

    def test_three_level_nested_coalesce(self) -> None:
        """Three-level nesting: coalesce1 → coalesce2 → coalesce3.

        Verifies that schema propagation works through arbitrary depth.
        """
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Level 1 branches
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["all", "level1_shared", "a"]}},
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["all", "level1_shared", "b"]}},
        )

        # Level 2 branch
        graph.add_node(
            "branch_c",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["all", "level2_shared", "c"]}},
        )

        # Level 3 branch
        graph.add_node(
            "branch_d",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["all", "d"]}},
        )

        # Level 1 coalesce (best_effort): A ∩ B = {all, level1_shared}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_1",
            ["branch_a", "branch_b"],
            policy="best_effort",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        # Level 2 coalesce (best_effort): coalesce_1 ∩ C = {all}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_2",
            ["coalesce_1", "branch_c"],
            policy="best_effort",
            extra_config={"branches": {"coalesce_1": {}, "branch_c": {}}},
        )

        # Level 3 coalesce (best_effort): coalesce_2 ∩ D = {all}
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce_3",
            ["coalesce_2", "branch_d"],
            policy="best_effort",
            extra_config={"branches": {"coalesce_2": {}, "branch_d": {}}},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        # Wire up edges
        for branch in ("branch_a", "branch_b", "branch_c", "branch_d"):
            graph.add_edge("source", branch, label=branch)
        graph.add_edge("branch_a", "coalesce_1", label="continue")
        graph.add_edge("branch_b", "coalesce_1", label="continue")
        graph.add_edge("coalesce_1", "coalesce_2", label="continue")
        graph.add_edge("branch_c", "coalesce_2", label="continue")
        graph.add_edge("coalesce_2", "coalesce_3", label="continue")
        graph.add_edge("branch_d", "coalesce_3", label="continue")
        graph.add_edge("coalesce_3", "sink", label="continue")

        # Verify each level
        assert graph.get_effective_guaranteed_fields("coalesce_1") == frozenset({"all", "level1_shared"})
        assert graph.get_effective_guaranteed_fields("coalesce_2") == frozenset({"all"})
        assert graph.get_effective_guaranteed_fields("coalesce_3") == frozenset({"all"})


class TestPassThroughPropagation:
    """Tests for ADR-007 pass-through propagation through get_effective_guaranteed_fields.

    These tests pin the propagation-aware implementation of
    ``get_effective_guaranteed_fields``. A pass-through transform
    (``passes_through_input=True``) inherits the intersection of its
    predecessors' effective guarantees, unioned with its own declared fields.
    """

    @staticmethod
    def _build_simple_chain(
        *,
        source_config: dict[str, Any],
        transform_config: dict[str, Any],
        transform_passes_through: bool,
        sink_declared_required: frozenset[str] = frozenset(),
    ) -> ExecutionGraph:
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config=source_config,
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config=transform_config,
            passes_through_input=transform_passes_through,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=sink_declared_required,
        )
        graph.add_edge("source", "pt", label="continue")
        graph.add_edge("pt", "sink", label="continue")
        return graph

    def test_pass_through_transform_propagates_predecessor_guarantees(self) -> None:
        """Source(fixed{a,b}) → PT(declared={c}, passes_through_input=True) → sink requires {a,b,c}.

        Without propagation this would reject because PT's static contract
        contains only {c}. With propagation it must succeed — inherited {a,b}
        union own {c}.
        """
        graph = self._build_simple_chain(
            source_config={"schema": {"mode": "fixed", "fields": ["a: str", "b: str"], "guaranteed_fields": ["a", "b"]}},
            transform_config={
                "schema": {"mode": "observed", "guaranteed_fields": ["c"]},
            },
            transform_passes_through=True,
            sink_declared_required=frozenset({"a", "b", "c"}),
        )
        # Should not raise
        graph._validate_sink_required_fields()
        assert graph.get_effective_guaranteed_fields("pt") == frozenset({"a", "b", "c"})

    def test_non_pass_through_transform_still_rejects_missing_fields(self) -> None:
        """Same shape with passes_through_input=False must reject (regression guard)."""
        from elspeth.core.dag.models import GraphValidationError

        graph = self._build_simple_chain(
            source_config={"schema": {"mode": "fixed", "fields": ["a: str", "b: str"], "guaranteed_fields": ["a", "b"]}},
            transform_config={
                "schema": {"mode": "observed", "guaranteed_fields": ["c"]},
            },
            transform_passes_through=False,
            sink_declared_required=frozenset({"a", "b", "c"}),
        )
        with pytest.raises(GraphValidationError, match=r"does not guarantee"):
            graph._validate_sink_required_fields()

    def test_pass_through_intersection_across_multiple_predecessors(self) -> None:
        """Two predecessors guaranteeing different field sets — intersection.

        source → gate → t_a({a,b}) → coalesce(union,best_effort) with union-intersection
        → PT(passes_through_input=True) → sink requiring {a}.

        Only {a} is common across both branches, so intersection = {a}.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.schema import FieldDefinition

        branch_a_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("a", "str", required=True), FieldDefinition("b", "str", required=True)),
            guaranteed_fields=("a", "b"),
        )
        branch_b_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("a", "str", required=True),),
            guaranteed_fields=("a",),
        )
        # Coalesce pre-computes strategy-aware guarantees on its own schema.
        coalesce_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("a", "str", required=True),),
            guaranteed_fields=("a",),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node(
            "t_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            output_schema_config=branch_a_schema,
        )
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            output_schema_config=branch_b_schema,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "best_effort", "merge": "union"},
            output_schema_config=coalesce_schema,
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            passes_through_input=True,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"a"}),
        )
        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "pt", label="continue")
        graph.add_edge("pt", "sink", label="continue")

        # Coalesce leaf gives {a}; PT inherits {a} and has no own declaration.
        assert graph.get_effective_guaranteed_fields("pt") == frozenset({"a"})

    def test_pass_through_chain_diamond_topology_uses_memoization(self) -> None:
        """Diamond topology: source → pt_a, pt_b → coalesce → sink.

        Correctness: intersection of branches' effective guarantees.
        Memoization: each node visited at most once via the shared cache.
        """
        from unittest import mock

        from elspeth.contracts import RoutingMode
        from elspeth.contracts.schema import FieldDefinition

        leaf_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("a", "str", required=True), FieldDefinition("b", "str", required=True)),
            guaranteed_fields=("a", "b"),
        )
        coalesce_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("a", "str", required=True), FieldDefinition("b", "str", required=True)),
            guaranteed_fields=("a", "b"),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema_config=leaf_schema)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node(
            "pt_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_node(
            "pt_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "require_all", "merge": "union"},
            output_schema_config=coalesce_schema,
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv_sink")

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "pt_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "pt_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("pt_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("pt_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue")

        # Count _walk_effective_guaranteed_fields invocations.
        original = ExecutionGraph._walk_effective_guaranteed_fields
        calls: list[str] = []

        def counting_walk(self_inner: ExecutionGraph, node_id: str, cache: dict[str, frozenset[str]]) -> frozenset[str]:
            calls.append(node_id)
            return original(self_inner, node_id, cache)

        with mock.patch.object(ExecutionGraph, "_walk_effective_guaranteed_fields", new=counting_walk):
            result = graph.get_effective_guaranteed_fields("coalesce")

        # Coalesce pre-computed its own guarantees; recursion stops at the leaf.
        assert result == frozenset({"a", "b"})
        # Memoization: no node walked more than once per get_effective call.
        assert len(calls) == len(set(calls)), f"Duplicate walk calls: {calls}"

    def test_pass_through_with_abstaining_predecessor_skips_in_intersection(self) -> None:
        """Predecessor with guaranteed_fields=None abstains and is skipped."""
        from elspeth.contracts import RoutingMode

        # Source: abstains (guaranteed_fields=None via observed mode with nothing declared).
        # Pass-through consumer sees no participating predecessor → inherited = empty.
        graph = ExecutionGraph()
        # Abstaining source
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed"}},  # No guaranteed_fields → abstain
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_edge("source", "pt", label="continue", mode=RoutingMode.MOVE)

        # Source abstains → no predecessor participates → inherited = empty.
        # PT own = empty. Total empty.
        assert graph.get_effective_guaranteed_fields("pt") == frozenset()
        # But it doesn't crash — abstain-skip is the correct behavior.

    def test_pass_through_with_explicit_empty_predecessor_collapses_intersection(self) -> None:
        """Predecessor with guaranteed_fields=() participates; intersection collapses to empty."""
        empty_schema = SchemaConfig(
            mode="flexible",
            fields=(),
            guaranteed_fields=(),  # Explicit empty
        )
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            output_schema_config=empty_schema,
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_edge("source", "pt", label="continue")
        # Source declares explicit empty → participating with empty → intersection empty
        assert graph.get_effective_guaranteed_fields("pt") == frozenset()

    def test_pass_through_downstream_of_observed_aggregation_inherits_empty(self) -> None:
        """Observed-mode aggregation upstream yields empty inheritance."""
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "fixed", "fields": ["a: str", "b: str"], "guaranteed_fields": ["a", "b"]}},
        )
        graph.add_node(
            "agg",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={"input_schema": {"mode": "observed"}},
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_edge("source", "agg", label="continue")
        graph.add_edge("agg", "pt", label="continue")

        # Aggregation declares nothing → PT inherits empty → own empty → empty total.
        assert graph.get_effective_guaranteed_fields("pt") == frozenset()

    def test_pass_through_with_abstaining_self_propagates_predecessor_unchanged(self) -> None:
        """PT with output_schema_config=None propagates predecessor fields unchanged."""
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "fixed", "fields": ["x: str", "y: str"], "guaranteed_fields": ["x", "y"]}},
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},  # No own declaration
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_edge("source", "pt", label="continue")
        # Own empty, inherited {x, y}, total {x, y}.
        assert graph.get_effective_guaranteed_fields("pt") == frozenset({"x", "y"})

    def test_get_effective_guaranteed_fields_public_api_signature_unchanged(self) -> None:
        """Guards against future _cache=None parameter regression."""
        import inspect

        sig = inspect.signature(ExecutionGraph.get_effective_guaranteed_fields)
        params = list(sig.parameters.values())
        # (self, node_id) — two parameters total, both positional.
        assert len(params) == 2, f"Public API must take only (self, node_id); got {params!r}"
        assert params[1].name == "node_id"
        assert params[1].default is inspect.Parameter.empty

    def test_get_effective_guaranteed_fields_no_longer_delegates_to_get_guaranteed_fields(self) -> None:
        """Pin the semantic split: get_guaranteed_fields is raw, get_effective is propagation-aware."""
        graph = self._build_simple_chain(
            source_config={"schema": {"mode": "fixed", "fields": ["a: str", "b: str"], "guaranteed_fields": ["a", "b"]}},
            transform_config={
                "schema": {"mode": "observed", "guaranteed_fields": ["c"]},
            },
            transform_passes_through=True,
        )
        # get_guaranteed_fields: this node alone declares {c}.
        assert graph.get_guaranteed_fields("pt") == frozenset({"c"})
        # get_effective_guaranteed_fields: propagation-aware includes {a, b, c}.
        assert graph.get_effective_guaranteed_fields("pt") == frozenset({"a", "b", "c"})
        # Idempotent — two calls return identical results.
        assert graph.get_effective_guaranteed_fields("pt") == graph.get_effective_guaranteed_fields("pt")

    def test_node_info_passes_through_on_non_transform_raises(self) -> None:
        """NodeInfo guard: passes_through_input=True rejected on non-transform-class node types.

        Per ADR-007, the flag applies to nodes that execute transform-class
        plugins — TRANSFORM (plain transforms) and AGGREGATION (batch-aware
        transforms like BatchReplicate wired under `aggregations:` in YAML).
        All other node types are rejected at construction.
        """
        from elspeth.core.dag.models import GraphValidationError, NodeInfo

        for bad_type in (
            NodeType.SOURCE,
            NodeType.COALESCE,
            NodeType.SINK,
            NodeType.GATE,
        ):
            with pytest.raises(GraphValidationError, match="passes_through_input"):
                NodeInfo(
                    node_id="n",
                    node_type=bad_type,
                    plugin_name="p",
                    passes_through_input=True,
                )

        # TRANSFORM and AGGREGATION both accept the flag.
        NodeInfo(node_id="t", node_type=NodeType.TRANSFORM, plugin_name="p", passes_through_input=True)
        NodeInfo(node_id="a", node_type=NodeType.AGGREGATION, plugin_name="p", passes_through_input=True)

    def test_pass_through_with_no_predecessors_raises_framework_bug_error(self) -> None:
        """Source-position PT is impossible in a built DAG — raise FrameworkBugError."""
        from elspeth.contracts.errors import FrameworkBugError

        graph = ExecutionGraph()
        # Construct a pass-through transform with NO predecessors (manual graph, bypasses builder).
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            passes_through_input=True,
        )
        with pytest.raises(FrameworkBugError, match=r"Pass-through transform .* has no predecessors"):
            graph.get_effective_guaranteed_fields("pt")

    def test_cache_memory_bounded(self) -> None:
        """NFR: per-call cache memory <= 32 KiB per 100 nodes."""
        from tests.performance.benchmarks._deep_size import deep_sizeof

        graph = ExecutionGraph()
        # Build a 100-node pass-through chain: source → pt_0 → pt_1 → ... → pt_98.
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "fixed", "fields": ["a: str"], "guaranteed_fields": ["a"]}},
        )
        for i in range(99):
            graph.add_node(
                f"pt_{i}",
                node_type=NodeType.TRANSFORM,
                plugin_name="passthrough",
                config={"schema": {"mode": "observed"}},
                output_schema_config=None,
                passes_through_input=True,
            )
        # Wire
        graph.add_edge("source", "pt_0", label="continue")
        for i in range(98):
            graph.add_edge(f"pt_{i}", f"pt_{i + 1}", label="continue")

        # Measure cache size after one full walk.
        cache: dict[str, frozenset[str]] = {}
        graph._walk_effective_guaranteed_fields("pt_98", cache)
        size = deep_sizeof(cache)
        assert size < 32 * 1024, f"Cache exceeded 32 KiB budget: {size} bytes"

    def test_pass_through_schema_config_from_dict_participates(self) -> None:
        """PT downstream of a flexible-mode predecessor with explicit guaranteed fields inherits them."""
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={
                "schema": {
                    "mode": "flexible",
                    "fields": ["q: int", "r: str"],
                    "guaranteed_fields": ["q", "r"],
                },
            },
        )
        graph.add_node(
            "pt",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_edge("source", "pt", label="continue")
        assert graph.get_effective_guaranteed_fields("pt") == frozenset({"q", "r"})
