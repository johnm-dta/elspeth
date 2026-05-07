# tests/mcp/test_arg_validation.py
"""Tests for MCP tool argument validation at the Tier 3 boundary.

The ``_validate_tool_args`` function validates external MCP client arguments
before they reach analyzer methods. These tests verify:
  - Required fields raise ValueError when missing
  - Type mismatches raise TypeError
  - Defaults are applied for optional fields
  - JSON float-to-int coercion works for whole numbers
  - Boolean values are rejected for integer fields
"""

from __future__ import annotations

import pytest

from elspeth.mcp.server import _ToolDef, _validate_tool_args


class TestRequiredStringFields:
    """Required string fields must be present and be strings."""

    def test_missing_required_field_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="requires 'run_id'"):
            _validate_tool_args("get_run", {})

    def test_wrong_type_for_required_field_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match=r"must be string.*got int"):
            _validate_tool_args("get_run", {"run_id": 123})

    def test_valid_required_string(self) -> None:
        args = _validate_tool_args("get_run", {"run_id": "abc-123"})
        assert args["run_id"] == "abc-123"

    def test_multiple_required_strings(self) -> None:
        args = _validate_tool_args("explain_field", {"run_id": "r1", "field_name": "amount"})
        assert args["run_id"] == "r1"
        assert args["field_name"] == "amount"


class TestOptionalStringFields:
    """Optional string fields default to None and must be string when present."""

    def test_optional_string_defaults_to_none(self) -> None:
        args = _validate_tool_args("list_runs", {})
        assert args["status"] is None

    def test_optional_string_accepts_string(self) -> None:
        args = _validate_tool_args("list_runs", {"status": "running"})
        assert args["status"] == "running"

    def test_optional_string_accepts_null(self) -> None:
        args = _validate_tool_args("list_runs", {"status": None})
        assert args["status"] is None

    def test_optional_string_rejects_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="must be string or null"):
            _validate_tool_args("list_runs", {"status": 42})


class TestOptionalStringDefaults:
    """Optional strings with non-None defaults."""

    def test_default_applied_when_absent(self) -> None:
        args = _validate_tool_args("get_errors", {"run_id": "r1"})
        assert args["error_type"] == "all"

    def test_explicit_value_overrides_default(self) -> None:
        args = _validate_tool_args("get_errors", {"run_id": "r1", "error_type": "validation"})
        assert args["error_type"] == "validation"

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError, match="must be string"):
            _validate_tool_args("get_errors", {"run_id": "r1", "error_type": 123})


class TestOptionalIntFields:
    """Optional integer fields with defaults."""

    def test_default_applied_when_absent(self) -> None:
        args = _validate_tool_args("list_runs", {})
        assert args["limit"] == 50

    def test_explicit_int_accepted(self) -> None:
        args = _validate_tool_args("list_runs", {"limit": 25})
        assert args["limit"] == 25

    def test_whole_float_coerced_to_int(self) -> None:
        """JSON has no int/float distinction -- 100.0 should become 100."""
        args = _validate_tool_args("list_runs", {"limit": 100.0})
        assert args["limit"] == 100
        assert isinstance(args["limit"], int)

    def test_fractional_float_rejected(self) -> None:
        """Non-whole floats are not valid integers."""
        with pytest.raises(TypeError, match="must be integer"):
            _validate_tool_args("list_runs", {"limit": 50.5})

    def test_string_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be integer"):
            _validate_tool_args("list_runs", {"limit": "50"})

    def test_boolean_rejected(self) -> None:
        """bool is a subclass of int in Python -- must be explicitly rejected."""
        with pytest.raises(TypeError, match="must be integer"):
            _validate_tool_args("list_runs", {"limit": True})

    def test_multiple_int_defaults(self) -> None:
        args = _validate_tool_args("list_rows", {"run_id": "r1"})
        assert args["limit"] == 100
        assert args["offset"] == 0


class TestOptionalDictFields:
    """Optional dict fields default to None."""

    def test_default_to_none(self) -> None:
        args = _validate_tool_args("query", {"sql": "SELECT 1"})
        assert args["params"] is None

    def test_accepts_dict(self) -> None:
        args = _validate_tool_args("query", {"sql": "SELECT 1", "params": {"x": 1}})
        assert args["params"] == {"x": 1}

    def test_accepts_null(self) -> None:
        args = _validate_tool_args("query", {"sql": "SELECT 1", "params": None})
        assert args["params"] is None

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError, match="must be object or null"):
            _validate_tool_args("query", {"sql": "SELECT 1", "params": "not a dict"})


class TestUnknownTool:
    """Unknown tool names raise ValueError."""

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            _validate_tool_args("nonexistent_tool", {})


class TestExtraFieldsIgnored:
    """Extra fields in arguments are silently dropped (not passed through)."""

    def test_extra_fields_not_in_output(self) -> None:
        args = _validate_tool_args("get_run", {"run_id": "r1", "extra_field": "surprise"})
        assert "extra_field" not in args
        assert args["run_id"] == "r1"


class TestAllToolsHaveSpecs:
    """Every tool in the registry has a valid _ToolDef entry."""

    @pytest.mark.parametrize(
        "tool_name",
        [
            "list_runs",
            "get_run",
            "get_run_summary",
            "list_nodes",
            "list_rows",
            "list_tokens",
            "list_operations",
            "get_operation_calls",
            "explain_token",
            "get_errors",
            "get_node_states",
            "get_calls",
            "query",
            "get_dag_structure",
            "get_performance_report",
            "get_error_analysis",
            "get_llm_usage_report",
            "describe_schema",
            "get_outcome_analysis",
            "diagnose",
            "get_failure_context",
            "get_recent_activity",
            "get_run_contract",
            "explain_field",
            "list_contract_violations",
        ],
    )
    def test_tool_has_arg_spec(self, tool_name: str) -> None:
        """Every tool must have a _TOOLS registry entry."""
        from elspeth.mcp.server import _TOOLS

        assert tool_name in _TOOLS, f"Tool '{tool_name}' has no _TOOLS entry -- arguments will not be validated at the Tier 3 boundary"


class TestNoArgTools:
    """Tools with no arguments validate successfully with empty dict."""

    @pytest.mark.parametrize("tool_name", ["describe_schema", "diagnose"])
    def test_no_arg_tool_accepts_empty(self, tool_name: str) -> None:
        args = _validate_tool_args(tool_name, {})
        assert args == {}

    @pytest.mark.parametrize("tool_name", ["describe_schema", "diagnose"])
    def test_no_arg_tool_ignores_extras(self, tool_name: str) -> None:
        args = _validate_tool_args(tool_name, {"spurious": "value"})
        assert args == {}


class TestToolDefImmutability:
    """_ToolDef.schema_properties must be truly immutable."""

    def test_schema_properties_is_immutable(self) -> None:
        """schema_properties dict must resist mutation after construction."""
        from types import MappingProxyType

        from elspeth.mcp.server import _ArgSpec

        original = {"run_id": {"type": "string"}}
        tool = _ToolDef(
            description="test",
            args=_ArgSpec(required_str=("run_id",)),
            handler=lambda a, args: None,
            schema_properties=original,
        )

        assert isinstance(tool.schema_properties, MappingProxyType)
        with pytest.raises(TypeError):
            tool.schema_properties["injected"] = "evil"  # type: ignore[index]

        # Caller's original dict must be decoupled
        original["injected"] = "evil"  # type: ignore[assignment]
        assert "injected" not in tool.schema_properties


class TestSchemaAndValidatorConsistency:
    """JSON Schema must agree with _validate_tool_args on nullability.

    Bug fix for elspeth-0ff83f68dd: optional_str and optional_dict fields
    were published with non-null JSON Schema types, so MCP clients that
    serialize omitted optionals as null were rejected by the framework
    before ELSPETH's validator could handle them.
    """

    def test_optional_str_fields_are_nullable_in_schema(self) -> None:
        """Every optional_str field must have type ["string", "null"] in the JSON Schema."""
        from elspeth.mcp.server import _TOOLS

        for tool_name, tool_def in _TOOLS.items():
            for field_name in tool_def.args.optional_str:
                schema_prop = tool_def.schema_properties.get(field_name)
                assert schema_prop is not None, f"{tool_name}: optional_str '{field_name}' missing from schema_properties"
                schema_type = schema_prop["type"]
                assert schema_type == ["string", "null"], (
                    f"{tool_name}: optional_str '{field_name}' has schema type {schema_type!r}, "
                    f"expected ['string', 'null'] — validator accepts null but schema rejects it"
                )

    def test_optional_dict_fields_are_nullable_in_schema(self) -> None:
        """Every optional_dict field must have type ["object", "null"] in the JSON Schema."""
        from elspeth.mcp.server import _TOOLS

        for tool_name, tool_def in _TOOLS.items():
            for field_name in tool_def.args.optional_dict:
                schema_prop = tool_def.schema_properties.get(field_name)
                assert schema_prop is not None, f"{tool_name}: optional_dict '{field_name}' missing from schema_properties"
                schema_type = schema_prop["type"]
                assert schema_type == ["object", "null"], (
                    f"{tool_name}: optional_dict '{field_name}' has schema type {schema_type!r}, "
                    f"expected ['object', 'null'] — validator accepts null but schema rejects it"
                )

    def test_required_str_fields_are_not_nullable(self) -> None:
        """Required string fields must NOT be nullable — they must always be present."""
        from elspeth.mcp.server import _TOOLS

        for tool_name, tool_def in _TOOLS.items():
            for field_name in tool_def.args.required_str:
                schema_prop = tool_def.schema_properties.get(field_name)
                assert schema_prop is not None, f"{tool_name}: required_str '{field_name}' missing from schema_properties"
                schema_type = schema_prop["type"]
                assert schema_type == "string", (
                    f"{tool_name}: required_str '{field_name}' should be non-nullable 'string', got {schema_type!r}"
                )
