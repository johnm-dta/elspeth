"""Tests for JSONExplode deaggregation transform.

JSONExplode transforms one row containing an array field into multiple rows,
one for each element in the array. This is the inverse of aggregation.

THREE-TIER TRUST MODEL:
- JSONExplode TRUSTS that pipeline data types are correct
- Type violations (missing field, wrong type) indicate UPSTREAM BUGS and crash
- No TransformResult.error() for type violations - they are bugs to fix
"""

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.plugins.config_base import PluginConfigError
from elspeth.testing import make_field, make_pipeline_row

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


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

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 3

        # Each row should have the item and item_index
        assert result.rows[0].to_dict() == {"id": 1, "item": {"name": "a"}, "item_index": 0}
        assert result.rows[1].to_dict() == {"id": 1, "item": {"name": "b"}, "item_index": 1}
        assert result.rows[2].to_dict() == {"id": 1, "item": {"name": "c"}, "item_index": 2}

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

    def test_empty_array_returns_error(self, ctx: PluginContext) -> None:
        """Empty array produces error (nothing to deaggregate)."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        row = {"id": 1, "items": []}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "invalid_input"
        assert result.reason["field"] == "items"
        assert not result.retryable

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

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 3

        assert result.rows[0].to_dict() == {"id": 1, "tag": "red", "item_index": 0}
        assert result.rows[1].to_dict() == {"id": 1, "tag": "green", "item_index": 1}
        assert result.rows[2].to_dict() == {"id": 1, "tag": "blue", "item_index": 2}

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

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 2

        # No item_index field
        assert result.rows[0].to_dict() == {"id": 1, "item": "a"}
        assert result.rows[1].to_dict() == {"id": 1, "item": "b"}
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

        result = transform.process(make_pipeline_row(row), ctx)

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

    def test_array_field_original_name_is_resolved(self, ctx: PluginContext) -> None:
        """Configured original array_field names resolve through PipelineRow contract."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "Line Items",
            }
        )
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                make_field("id", int, original_name="ID"),
                make_field("line_items", object, original_name="Line Items"),
            ),
            locked=True,
        )
        row = PipelineRow({"id": 1, "line_items": ["a", "b"]}, contract)

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert result.rows[0].to_dict() == {"id": 1, "item": "a", "item_index": 0}
        assert result.rows[1].to_dict() == {"id": 1, "item": "b", "item_index": 1}


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
            transform.process(make_pipeline_row(row), ctx)

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
            transform.process(make_pipeline_row(row), ctx)

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
            transform.process(make_pipeline_row(row), ctx)

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
            transform.process(make_pipeline_row(row), ctx)

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
            transform.process(make_pipeline_row(row), ctx)

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
            transform.process(make_pipeline_row(row), ctx)


class TestJSONExplodeConfiguration:
    """Tests for configuration validation."""

    def test_no_on_error_attribute(self) -> None:
        """JSONExplode has no on_error - on_error should be None."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
            }
        )

        # on_error is None because JSONExplode uses DataPluginConfig,
        # not TransformDataConfig, and doesn't set on_error
        assert transform.on_error is None

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

    def test_has_name_attribute(self) -> None:
        """JSONExplode has name class attribute."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        assert JSONExplode.name == "json_explode"

    def test_rejects_on_success_in_plugin_options(self) -> None:
        """on_success must be configured at TransformSettings, not plugin options."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        with pytest.raises(PluginConfigError, match="does not accept 'on_success'"):
            JSONExplode(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "array_field": "items",
                    "on_success": "output",
                }
            )


class TestJSONExplodeOutputSchema:
    """Tests for output schema behavior of shape-changing transforms.

    Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch:
    JSONExplode changes row shape (removes array_field, adds output_field + item_index),
    so output_schema must be dynamic.
    """

    def test_output_schema_is_observed(self) -> None:
        """JSONExplode uses dynamic output_schema.

        JSONExplode removes array_field and adds output_field/item_index.
        The output shape depends on config, not input schema.
        Therefore output_schema must be dynamic.
        """
        from elspeth.plugins.transforms.json_explode import JSONExplode

        # Explicit schema with array field
        transform = JSONExplode(
            {
                "schema": {"mode": "fixed", "fields": ["id: int", "items: any"]},
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


class TestJSONExplodeContractPropagation:
    """Tests for JSONExplode contract propagation."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_contract_contains_output_field_not_array_field(self, ctx: PluginContext) -> None:
        """Output contract contains output_field and item_index, not array_field."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
                "output_field": "item",
            }
        )

        row = make_pipeline_row({"id": 1, "items": ["a", "b"]})
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert isinstance(result.rows[0], PipelineRow)

        field_names = {f.normalized_name for f in result.rows[0].contract.fields}
        assert "item" in field_names
        assert "item_index" in field_names
        assert "items" not in field_names  # Array field removed
        assert "id" in field_names  # Other fields preserved

    def test_contract_empty_array_case(self, ctx: PluginContext) -> None:
        """Empty array returns error — no contract to propagate."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "items",
                "output_field": "item",
            }
        )

        row = make_pipeline_row({"id": 1, "items": []})
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "invalid_input"
        assert result.row is None

    def test_downstream_can_access_exploded_fields(self, ctx: PluginContext) -> None:
        """Downstream transforms can access exploded fields via contract."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": DYNAMIC_SCHEMA,
                "array_field": "tags",
                "output_field": "tag",
            }
        )

        row = make_pipeline_row({"id": 1, "tags": ["python", "elspeth"]})
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2
        assert isinstance(result.rows[0], PipelineRow)

        # result.rows[0] IS already a PipelineRow with contract
        output_row = result.rows[0]

        # Downstream access via contract should work
        assert output_row["tag"] == "python"
        assert output_row["item_index"] == 0
        assert output_row["id"] == 1

        # Original array field should not be accessible
        with pytest.raises(KeyError, match="not found in schema contract"):
            _ = output_row["tags"]

    def test_fixed_mode_object_elements_keep_output_field_in_contract(self, ctx: PluginContext) -> None:
        """FIXED mode keeps output_field contract when exploded elements are objects."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode(
            {
                "schema": {"mode": "fixed", "fields": ["id: int", "items: any"]},
                "array_field": "items",
                "output_field": "item",
            }
        )

        input_contract = SchemaContract(
            mode="FIXED",
            fields=(
                make_field("id", int, required=True, source="declared"),
                make_field("items", object, required=True, source="declared"),
            ),
            locked=True,
        )
        row = PipelineRow({"id": 1, "items": [{"sku": "A1"}]}, input_contract)

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.rows is not None
        output_row = result.rows[0]
        item_field = output_row.contract.get_field("item")
        assert item_field is not None
        assert item_field.python_type is object
        assert output_row["item"] == {"sku": "A1"}


class TestJSONExplodeCopyIsolation:
    """Tests that exploded rows have independent copies of nested data.

    Shallow copy shares nested dicts/lists between output rows.
    If a downstream transform mutates a nested value in one row,
    all sibling rows would be corrupted. Deep copy prevents this.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return PluginContext(run_id="test-run", config={})

    def test_nested_dict_mutation_does_not_corrupt_siblings(self, ctx: PluginContext) -> None:
        """Mutating a nested dict in one exploded row must not affect others."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode({"schema": DYNAMIC_SCHEMA, "array_field": "items"})

        row = make_pipeline_row(
            {
                "id": 1,
                "metadata": {"source": "api", "version": 2},
                "items": ["a", "b", "c"],
            }
        )

        result = transform.process(row, ctx)

        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 3

        # Mutate nested dict in first row
        row_0_dict = result.rows[0].to_dict()
        row_0_dict["metadata"]["corrupted"] = True

        # Sibling rows must NOT see the mutation
        row_1_dict = result.rows[1].to_dict()
        row_2_dict = result.rows[2].to_dict()
        assert "corrupted" not in row_1_dict["metadata"]
        assert "corrupted" not in row_2_dict["metadata"]

    def test_nested_list_mutation_does_not_corrupt_siblings(self, ctx: PluginContext) -> None:
        """Mutating a nested list in one exploded row must not affect others."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        transform = JSONExplode({"schema": DYNAMIC_SCHEMA, "array_field": "items"})

        row = make_pipeline_row(
            {
                "id": 1,
                "tags": ["original"],
                "items": ["x", "y"],
            }
        )

        result = transform.process(row, ctx)

        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 2

        # Mutate nested list in first row
        row_0_dict = result.rows[0].to_dict()
        row_0_dict["tags"].append("injected")

        # Second row must NOT see the mutation
        row_1_dict = result.rows[1].to_dict()
        assert row_1_dict["tags"] == ["original"]
