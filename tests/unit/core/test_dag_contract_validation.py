# tests/core/test_dag_contract_validation.py
"""Tests for DAG schema contract validation (guaranteed/required fields)."""

import pytest

from elspeth.contracts import NodeType
from elspeth.core.dag import ExecutionGraph


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

    def test_gate_inherits_from_upstream(self) -> None:
        """Gate node inherits guarantees from upstream."""
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
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source_1", "gate_1", label="continue")

        result = graph.get_effective_guaranteed_fields("gate_1")
        assert result == frozenset({"x", "y"})

    def test_coalesce_returns_intersection(self) -> None:
        """Coalesce node returns intersection of branch guarantees."""
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

        # Coalesce node
        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"schema": {"mode": "observed"}, "merge": "union", "branches": {}, "policy": "require_all"},
        )

        graph.add_edge("branch_a", "coalesce_1", label="path_a")
        graph.add_edge("branch_b", "coalesce_1", label="path_b")

        # Coalesce guarantees only the intersection
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
        """Gate node passes upstream guarantees to downstream consumer."""
        graph = ExecutionGraph()

        # Source with guarantees
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["field_a"]}},
        )

        # Gate (passthrough)
        graph.add_node(
            "gate_1",
            node_type=NodeType.GATE,
            plugin_name="config_gate",
            config={"schema": {"mode": "observed"}},
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
        """Fork gate to two branches with different transforms, coalesce guarantees intersection."""
        graph = ExecutionGraph()

        # Source guarantees base fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["id", "raw_data"]}},
        )

        # Fork gate (passes through source guarantees to both branches)
        graph.add_node(
            "fork_gate",
            node_type=NodeType.GATE,
            plugin_name="fork_gate",
            config={"schema": {"mode": "observed"}},
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

        # Coalesce merges branches - guarantees INTERSECTION (id, raw_data only)
        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"schema": {"mode": "observed"}, "merge": "union", "branches": {}, "policy": "require_all"},
        )

        # Sink requires only 'id' (which is in intersection)
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
        graph.add_edge("branch_a", "coalesce_1", label="merge")
        graph.add_edge("branch_b", "coalesce_1", label="merge")
        graph.add_edge("coalesce_1", "sink_1", label="continue")

        # Should pass - sink requires 'id' which is in intersection
        graph.validate_edge_compatibility()

    def test_coalesce_intersection_rejects_branch_specific_requirement(self) -> None:
        """Consumer after coalesce cannot require branch-specific fields."""
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

        # Coalesce
        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"schema": {"mode": "observed"}, "merge": "union", "branches": {}, "policy": "require_all"},
        )

        # Sink requires 'a_only' - NOT guaranteed by coalesce (only 'common' is)
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={
                "schema": {"mode": "observed", "required_fields": ["a_only"]},
            },
        )

        graph.add_edge("source_1", "branch_a", label="path_a")
        graph.add_edge("source_1", "branch_b", label="path_b")
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

        graph.add_node(
            "coalesce_1",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={"schema": {"mode": "observed"}, "merge": "union", "branches": {}, "policy": "require_all"},
        )

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

        # Should pass - 'id' is in intersection of all three branches
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
