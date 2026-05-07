# tests/property/plugins/llm/test_multi_query_properties.py
"""Property-based tests for multi-query pure data transformations.

Tests cover QuerySpec (named variable mapping) used by the unified LLMTransform:

1. OutputFieldConfig.to_json_schema: type mapping correctness
2. QuerySpec.build_template_context: named variable mapping
3. OutputFieldConfig validation: enum requires values, others reject values
"""

from __future__ import annotations

from types import MappingProxyType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.plugins.transforms.llm.multi_query import (
    OutputFieldConfig,
    OutputFieldType,
    QuerySpec,
)

# =============================================================================
# Strategies
# =============================================================================

field_names = st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz_").filter(
    lambda s: s != "source_row"  # Reserved key in build_template_context
)

string_values = st.text(min_size=0, max_size=30)

# Non-enum field types
non_enum_types = st.sampled_from(
    [
        OutputFieldType.STRING,
        OutputFieldType.INTEGER,
        OutputFieldType.NUMBER,
        OutputFieldType.BOOLEAN,
    ]
)


# =============================================================================
# OutputFieldConfig.to_json_schema Properties
# =============================================================================


class TestOutputFieldSchemaProperties:
    """to_json_schema must produce valid JSON Schema fragments."""

    @given(field_type=non_enum_types)
    @settings(max_examples=20)
    def test_non_enum_schema_has_type_key(self, field_type: OutputFieldType) -> None:
        """Property: Non-enum types produce {"type": type_value}."""
        config = OutputFieldConfig(suffix="test", type=field_type)
        schema = config.to_json_schema()
        assert "type" in schema
        assert schema["type"] == field_type.value

    @given(values=st.lists(string_values, min_size=1, max_size=5, unique=True))
    @settings(max_examples=50)
    def test_enum_schema_has_enum_list(self, values: list[str]) -> None:
        """Property: Enum type produces {"type": "string", "enum": values}."""
        config = OutputFieldConfig(suffix="test", type=OutputFieldType.ENUM, values=values)
        schema = config.to_json_schema()
        assert schema["type"] == "string"
        assert "enum" in schema
        assert schema["enum"] == values

    def test_enum_without_values_rejected(self) -> None:
        """Property: Enum type without values raises ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="non-empty"):
            OutputFieldConfig(suffix="test", type=OutputFieldType.ENUM, values=[])

    def test_enum_with_none_values_rejected(self) -> None:
        """Property: Enum type with None values raises ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="non-empty"):
            OutputFieldConfig(suffix="test", type=OutputFieldType.ENUM, values=None)

    @given(field_type=non_enum_types, values=st.lists(string_values, min_size=1, max_size=3))
    @settings(max_examples=20)
    def test_non_enum_with_values_rejected(self, field_type: OutputFieldType, values: list[str]) -> None:
        """Property: Non-enum types with values raises ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="only valid for enum"):
            OutputFieldConfig(suffix="test", type=field_type, values=values)

    @given(field_type=non_enum_types)
    @settings(max_examples=20)
    def test_schema_has_no_extra_keys(self, field_type: OutputFieldType) -> None:
        """Property: Non-enum schemas only have 'type' key."""
        config = OutputFieldConfig(suffix="test", type=field_type)
        schema = config.to_json_schema()
        assert set(schema.keys()) == {"type"}


# =============================================================================
# QuerySpec.build_template_context Properties (Named Mapping)
# =============================================================================


class TestQuerySpecContextProperties:
    """build_template_context must create correct named variable mappings.

    QuerySpec uses named input_fields (dict mapping template variable
    name to row column name) instead of positional input_1, input_2 variables.
    """

    @given(
        n_fields=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_named_mapping(self, n_fields: int, data: st.DataObject) -> None:
        """Property: input_fields maps template_var -> row[column_name] correctly."""
        template_vars = data.draw(st.lists(field_names, min_size=n_fields, max_size=n_fields, unique=True))
        column_names = data.draw(st.lists(field_names, min_size=n_fields, max_size=n_fields, unique=True))
        values = data.draw(st.lists(string_values, min_size=n_fields, max_size=n_fields))

        # Build row from column_names -> values
        row = dict(zip(column_names, values, strict=False))

        # Build input_fields mapping: template_var -> column_name
        input_fields = dict(zip(template_vars, column_names, strict=False))

        spec = QuerySpec(
            name="test_query",
            input_fields=MappingProxyType(input_fields),
        )

        ctx = spec.build_template_context(row)

        # Each template variable should map to the correct row value
        for template_var, column_name in input_fields.items():
            assert template_var in ctx
            assert ctx[template_var] == row[column_name]

    @given(data=st.data())
    @settings(max_examples=50)
    def test_context_includes_source_row(self, data: st.DataObject) -> None:
        """Property: Context['source_row'] contains the full original row."""
        column_name = data.draw(field_names)
        value = data.draw(string_values)
        row = {column_name: value}

        spec = QuerySpec(
            name="test_query",
            input_fields=MappingProxyType({"var": column_name}),
        )

        ctx = spec.build_template_context(row)
        assert ctx["source_row"] == row

    def test_missing_column_raises_key_error(self) -> None:
        """Property: Missing row column raises KeyError."""
        spec = QuerySpec(
            name="test_query",
            input_fields=MappingProxyType({"template_var": "missing_column"}),
        )

        with pytest.raises(KeyError):
            spec.build_template_context({"other_field": "value"})

    @given(
        n_fields=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_context_has_exactly_expected_keys(self, n_fields: int, data: st.DataObject) -> None:
        """Property: Context has named variables and source_row only."""
        template_vars = data.draw(st.lists(field_names, min_size=n_fields, max_size=n_fields, unique=True))
        column_names = data.draw(st.lists(field_names, min_size=n_fields, max_size=n_fields, unique=True))
        row = dict.fromkeys(column_names, "v")

        input_fields = dict(zip(template_vars, column_names, strict=False))

        spec = QuerySpec(
            name="test_query",
            input_fields=MappingProxyType(input_fields),
        )

        ctx = spec.build_template_context(row)

        expected_keys = set(template_vars) | {"source_row"}
        assert set(ctx.keys()) == expected_keys

    def test_empty_name_rejected(self) -> None:
        """Property: Empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            QuerySpec(
                name="",
                input_fields=MappingProxyType({"var": "col"}),
            )

    def test_empty_input_fields_rejected(self) -> None:
        """Property: Empty input_fields raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            QuerySpec(
                name="test_query",
                input_fields=MappingProxyType({}),
            )
