"""Tests for JSONExplode deaggregation transform.

JSONExplode transforms one row containing an array field into multiple rows,
one for each element in the array. This is the inverse of aggregation.

THREE-TIER TRUST MODEL:
- JSONExplode TRUSTS that pipeline data types are correct
- Type violations (missing field, wrong type) indicate UPSTREAM BUGS and crash
- No TransformResult.error() for type violations - they are bugs to fix
"""

import pytest

from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestJSONExplodeHappyPath:
    """Happy path tests for JSONExplode transform."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_explodes_array_into_multiple_rows(self, ctx: PluginContext) -> None:
        """JSONExplode expands array field into multiple rows."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {
            "id": 1,
            "items": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        }

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 3

        # Each row should have the item and item_index
        assert result.rows[0] == {"id": 1, "item": {"name": "a"}, "item_index": 0}
        assert result.rows[1] == {"id": 1, "item": {"name": "b"}, "item_index": 1}
        assert result.rows[2] == {"id": 1, "item": {"name": "c"}, "item_index": 2}

    def test_creates_tokens_is_true(self) -> None:
        """JSONExplode has creates_tokens=True (deaggregation)."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        assert transform.creates_tokens is True

    def test_empty_array_returns_single_row(self, ctx: PluginContext) -> None:
        """Empty array returns single row with None item."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": []}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert not result.is_multi_row  # Single row result
        assert result.row is not None
        assert result.row == {"id": 1, "item": None, "item_index": None}

    def test_custom_output_field_name(self, ctx: PluginContext) -> None:
        """Custom output_field name is respected."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "tags",
                "output_field": "tag",
            }
        )

        row = {"id": 1, "tags": ["red", "green", "blue"]}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 3

        assert result.rows[0] == {"id": 1, "tag": "red", "item_index": 0}
        assert result.rows[1] == {"id": 1, "tag": "green", "item_index": 1}
        assert result.rows[2] == {"id": 1, "tag": "blue", "item_index": 2}

    def test_include_index_false(self, ctx: PluginContext) -> None:
        """Can disable item_index field."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
                "include_index": False,
            }
        )

        row = {"id": 1, "items": ["a", "b"]}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 2

        # No item_index field
        assert result.rows[0] == {"id": 1, "item": "a"}
        assert result.rows[1] == {"id": 1, "item": "b"}
        assert "item_index" not in result.rows[0]

    def test_preserves_all_non_array_fields(self, ctx: PluginContext) -> None:
        """All fields except array field are preserved in output."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {
            "id": 1,
            "name": "test",
            "metadata": {"source": "api"},
            "count": 42,
            "items": ["x"],
        }

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 1

        # All non-array fields preserved
        output = result.rows[0]
        assert output["id"] == 1
        assert output["name"] == "test"
        assert output["metadata"] == {"source": "api"}
        assert output["count"] == 42
        assert output["item"] == "x"
        assert output["item_index"] == 0

        # Array field is NOT preserved (replaced by output_field)
        assert "items" not in output


class TestJSONExplodeTypeViolations:
    """Tests for type violations - these should CRASH, not return errors.

    Per three-tier trust model:
    - Source validates that array field exists and is a list
    - Transform trusts source did its job
    - Type violations are UPSTREAM BUGS that should crash
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_missing_field_crashes(self, ctx: PluginContext) -> None:
        """Missing array field is upstream bug - should crash (KeyError)."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1}  # Missing 'items' field

        with pytest.raises(KeyError, match="items"):
            transform.process(row, ctx)

    def test_none_value_crashes(self, ctx: PluginContext) -> None:
        """None value for array field is upstream bug - should crash (TypeError)."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": None}

        with pytest.raises(TypeError):
            transform.process(row, ctx)

    def test_string_value_crashes_with_type_error(self, ctx: PluginContext) -> None:
        """String value is upstream bug - should crash with TypeError.

        Strings are iterable in Python, but JSONExplode requires lists.
        A string where a list was expected indicates a source validation bug
        or misconfigured pipeline. The transform crashes explicitly to surface
        this bug rather than producing garbage output (one row per character).
        """
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": "abc"}  # String, not array!

        # Should crash with clear error message
        with pytest.raises(TypeError, match=r"Field 'items' must be a list"):
            transform.process(row, ctx)

    def test_dict_value_crashes_with_type_error(self, ctx: PluginContext) -> None:
        """Dict value is upstream bug - should crash with TypeError.

        Dicts are iterable (over keys) in Python, but JSONExplode requires lists.
        A dict where a list was expected indicates an upstream validation bug.
        """
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": {"x": 1, "y": 2}}  # Dict, not list!

        with pytest.raises(TypeError, match=r"Field 'items' must be a list"):
            transform.process(row, ctx)

    def test_tuple_value_crashes_with_type_error(self, ctx: PluginContext) -> None:
        """Tuple value is upstream bug - should crash with TypeError.

        Tuples are iterable in Python, but JSONExplode requires lists.
        JSON has no tuple type - if data came from JSON it would be a list.
        A tuple indicates the data came from Python code that should be fixed.
        """
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": ("a", "b", "c")}  # Tuple, not list!

        with pytest.raises(TypeError, match=r"Field 'items' must be a list"):
            transform.process(row, ctx)

    def test_non_iterable_value_crashes(self, ctx: PluginContext) -> None:
        """Non-iterable value is upstream bug - should crash (TypeError)."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": 42}  # int is not iterable

        with pytest.raises(TypeError):
            transform.process(row, ctx)


class TestJSONExplodeConfiguration:
    """Tests for configuration validation."""

    def test_no_on_error_attribute(self) -> None:
        """JSONExplode has no on_error - _on_error should be None."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        # _on_error is None because JSONExplode uses DataPluginConfig,
        # not TransformDataConfig, and doesn't set _on_error
        assert transform._on_error is None

    def test_array_field_is_required(self) -> None:
        """array_field config is required - raises PluginConfigError."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        with pytest.raises(PluginConfigError, match="array_field"):
            JSONExplode({"schema": DYNAMIC_SCHEMA})

    def test_schema_is_required(self) -> None:
        """schema config is required - raises PluginConfigError."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        with pytest.raises(PluginConfigError, match="schema"):
            JSONExplode({"array_field": "items"})

    def test_implements_transform_protocol(self) -> None:
        """JSONExplode implements TransformProtocol."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        assert isinstance(transform, TransformProtocol)

    def test_has_name_attribute(self) -> None:
        """JSONExplode has name class attribute."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        assert JSONExplode.name == "json_explode"


class TestJSONExplodeOutputSchema:
    """Tests for output schema behavior of shape-changing transforms.

    Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch:
    JSONExplode changes row shape (removes array_field, adds output_field + item_index),
    so output_schema must be dynamic.
    """

    def test_output_schema_is_dynamic(self) -> None:
        """JSONExplode uses dynamic output_schema.

        JSONExplode removes array_field and adds output_field/item_index.
        The output shape depends on config, not input schema.
        Therefore output_schema must be dynamic.
        """
        from elspeth.plugins.transforms.json_explode import JSONExplode

        # Explicit schema with array field
        transform = JSONExplode(
            {
                "schema": {"mode": "strict", "fields": ["id: int", "items: any"]},
                "array_field": "items",
                "output_field": "item",
            }
        )

        # Output schema should be dynamic (no required fields, extra="allow")
        # Output has: id, item, item_index (NOT items)
        # Currently fails because output_schema = input_schema (has id, items)
        output_fields = transform.output_schema.model_fields

        assert len(output_fields) == 0, f"Expected dynamic schema with no required fields, got: {list(output_fields.keys())}"

        config = transform.output_schema.model_config
        assert config.get("extra") == "allow", "Output schema should allow extra fields (dynamic)"
