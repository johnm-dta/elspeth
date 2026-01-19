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

    def test_string_value_iterates_over_characters(self, ctx: PluginContext) -> None:
        """String value produces one row per character - likely a source bug.

        This test documents behavior, not crashes. Strings are iterable in Python,
        so they don't crash. However, iterating over a string character-by-character
        is almost certainly NOT what the user intended - this indicates a source
        validation bug that let a string through where an array was expected.

        The transform trusts the source, so it "works" - but produces garbage.
        """
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": "abc"}  # String, not array!

        result = transform.process(row, ctx)

        # This "works" but produces one row per character - almost certainly wrong
        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 3  # One row per character
        assert result.rows[0]["item"] == "a"
        assert result.rows[1]["item"] == "b"
        assert result.rows[2]["item"] == "c"

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
