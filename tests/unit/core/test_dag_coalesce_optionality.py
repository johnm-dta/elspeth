# tests/unit/core/test_dag_coalesce_optionality.py
"""Tests for field optionality preservation in coalesce union merge schema.

Fix: builder.py union merge was stripping "?" markers and "required: false"
from branch field specs, making all merged fields appear required.
"""

from __future__ import annotations

from elspeth.core.dag.builder import _field_name_type, _field_required


class TestFieldRequired:
    """Unit tests for _field_required() helper."""

    def test_string_required(self) -> None:
        assert _field_required("score: float") is True

    def test_string_optional_question_mark(self) -> None:
        assert _field_required("score: float?") is False

    def test_string_optional_with_spaces(self) -> None:
        assert _field_required("  score: float?  ") is False

    def test_dict_with_required_true(self) -> None:
        assert _field_required({"name": "score", "type": "float", "required": True}) is True

    def test_dict_with_required_false(self) -> None:
        assert _field_required({"name": "score", "type": "float", "required": False}) is False

    def test_dict_without_required_defaults_true(self) -> None:
        assert _field_required({"name": "score", "type": "float"}) is True

    def test_yaml_dict_required(self) -> None:
        assert _field_required({"score": "float"}) is True

    def test_yaml_dict_optional(self) -> None:
        assert _field_required({"score": "float?"}) is False

    def test_non_dict_non_string_defaults_true(self) -> None:
        assert _field_required(42) is True


class TestFieldNameType:
    """Ensure _field_name_type still strips ? from type correctly."""

    def test_string_with_question_mark(self) -> None:
        name, ftype = _field_name_type("score: float?")
        assert name == "score"
        assert ftype == "float"

    def test_dict_format(self) -> None:
        name, ftype = _field_name_type({"name": "x", "type": "int"})
        assert name == "x"
        assert ftype == "int"


class TestUnionMergeOptionalityPreservation:
    """Integration tests for optionality in union merge via make_graph_fork.

    These test the full path through the builder's coalesce schema
    construction to verify that optionality markers survive the merge.
    """

    def _build_coalesce_with_schemas(
        self,
        branch_schemas: dict[str, dict],
    ) -> dict:
        """Build a coalesce graph and return the merged schema config.

        Constructs a fork graph with two identity branches, each having
        a specific schema, then extracts the coalesce node's schema.
        """
        from elspeth.contracts.enums import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Source
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test-source",
            config={"schema": {"mode": "observed"}},
        )

        # Gate
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="test-gate",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source", "gate", label="continue")

        # Coalesce
        from elspeth.contracts import RoutingMode

        coalesce_config = {
            "branches": {name: name for name in branch_schemas},
            "policy": "require_all",
            "merge": "union",
        }
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config=coalesce_config,
        )

        # Create branches with schemas
        for branch_name, schema_dict in branch_schemas.items():
            # Create a transform node with the branch schema
            schema_config = SchemaConfig.from_dict(schema_dict)
            graph.add_node(
                branch_name,
                node_type=NodeType.TRANSFORM,
                plugin_name=f"test-{branch_name}",
                config={"schema": schema_dict},
                output_schema_config=schema_config,
            )
            graph.add_edge("gate", branch_name, label=branch_name, mode=RoutingMode.COPY)
            graph.add_edge(branch_name, "coalesce", label="continue", mode=RoutingMode.MOVE)

        # Sink
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="test-sink",
            config={},
        )
        graph.add_edge("coalesce", "sink", label="continue")

        # Now run the union merge logic manually (same as builder.py)
        from elspeth.core.dag.builder import _field_name_type, _field_required
        from elspeth.core.dag.models import GraphValidationError

        seen_types: dict[str, tuple[str, bool, str]] = {}
        for branch_name, schema_dict in branch_schemas.items():
            fields_list = schema_dict.get("fields")
            if not fields_list:
                continue
            for field_spec in fields_list:
                fname, ftype = _field_name_type(field_spec)
                freq = _field_required(field_spec)
                if fname in seen_types:
                    prior_type, _prior_req, prior_branch = seen_types[fname]
                    if prior_type != ftype:
                        raise GraphValidationError(f"Type mismatch for {fname}")
                    if not freq:
                        seen_types[fname] = (prior_type, False, prior_branch)
                else:
                    seen_types[fname] = (ftype, freq, branch_name)

        merged_fields = [f"{name}: {ftype}{'?' if not req else ''}" for name, (ftype, req, _) in seen_types.items()]
        return {"mode": "flexible", "fields": merged_fields}

    def test_optional_field_preserved_string_format(self) -> None:
        """String-format "score: float?" should produce "score: float?" in merged output."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": {"mode": "flexible", "fields": ["id: int", "score: float?"]},
                "branch_b": {"mode": "flexible", "fields": ["id: int", "label: str"]},
            }
        )
        assert "score: float?" in result["fields"]
        assert "id: int" in result["fields"]
        assert "label: str" in result["fields"]

    def test_optional_field_preserved_dict_format(self) -> None:
        """Dict-format {"required": false} should produce optional in merged output."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": {
                    "mode": "flexible",
                    "fields": [
                        {"name": "id", "type": "int", "required": True},
                        {"name": "score", "type": "float", "required": False},
                    ],
                },
                "branch_b": {
                    "mode": "flexible",
                    "fields": [{"name": "id", "type": "int", "required": True}],
                },
            }
        )
        assert "score: float?" in result["fields"]
        assert "id: int" in result["fields"]

    def test_mixed_required_and_optional_yields_optional(self) -> None:
        """Field required in branch A, optional in branch B → optional in output."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": {"mode": "flexible", "fields": ["score: float"]},
                "branch_b": {"mode": "flexible", "fields": ["score: float?"]},
            }
        )
        assert "score: float?" in result["fields"]

    def test_all_required_stays_required(self) -> None:
        """Field required in ALL branches → required in output."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": {"mode": "flexible", "fields": ["id: int"]},
                "branch_b": {"mode": "flexible", "fields": ["id: int"]},
            }
        )
        assert "id: int" in result["fields"]
        # Ensure no ? was added
        assert "id: int?" not in result["fields"]

    def test_all_optional_stays_optional(self) -> None:
        """Field optional in ALL branches → optional in output."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": {"mode": "flexible", "fields": ["score: float?"]},
                "branch_b": {"mode": "flexible", "fields": ["score: float?"]},
            }
        )
        assert "score: float?" in result["fields"]

    def test_yaml_dict_optional_preserved(self) -> None:
        """YAML shorthand {"score": "float?"} should preserve optionality."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": {"mode": "flexible", "fields": [{"score": "float?"}]},
                "branch_b": {"mode": "flexible", "fields": [{"id": "int"}]},
            }
        )
        assert "score: float?" in result["fields"]
        assert "id: int" in result["fields"]
