"""Tests for FieldMapper transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestFieldMapper:
    """Tests for FieldMapper transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """FieldMapper implements TransformProtocol."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"old": "new"},
            }
        )
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """FieldMapper has name and schemas."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        assert FieldMapper.name == "field_mapper"

    def test_rename_single_field(self, ctx: PluginContext) -> None:
        """Rename a single field."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"old_name": "new_name"},
            }
        )
        row = {"old_name": "value", "other": 123}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"new_name": "value", "other": 123}
        assert "old_name" not in result.row

    def test_rename_multiple_fields(self, ctx: PluginContext) -> None:
        """Rename multiple fields at once."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {
                    "first_name": "firstName",
                    "last_name": "lastName",
                },
            }
        )
        row = {"first_name": "Alice", "last_name": "Smith", "id": 1}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"firstName": "Alice", "lastName": "Smith", "id": 1}

    def test_select_fields_only(self, ctx: PluginContext) -> None:
        """Only include specified fields (drop others)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"id": "id", "name": "name"},
                "select_only": True,
            }
        )
        row = {"id": 1, "name": "alice", "secret": "password", "extra": "data"}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"id": 1, "name": "alice"}
        assert "secret" not in result.row
        assert "extra" not in result.row

    def test_missing_field_error(self, ctx: PluginContext) -> None:
        """Error when required field is missing and strict mode enabled."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"required_field": "output"},
                "strict": True,
            }
        )
        row = {"other_field": "value"}

        result = transform.process(row, ctx)

        assert result.status == "error"
        assert "required_field" in str(result.reason)

    def test_missing_field_skip_non_strict(self, ctx: PluginContext) -> None:
        """Skip missing fields when strict mode disabled."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"maybe_field": "output"},
                "strict": False,
            }
        )
        row = {"other_field": "value"}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"other_field": "value"}
        assert "output" not in result.row

    def test_default_is_non_strict(self, ctx: PluginContext) -> None:
        """Default behavior is non-strict (skip missing)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"missing": "output"},
            }
        )
        row = {"exists": "value"}

        result = transform.process(row, ctx)

        assert result.status == "success"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested fields with dot notation."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"meta.source": "origin"},
            }
        )
        row = {"id": 1, "meta": {"source": "api", "timestamp": 123}}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["origin"] == "api"
        assert "meta" in result.row  # Original nested structure preserved

    def test_empty_mapping_passthrough(self, ctx: PluginContext) -> None:
        """Empty mapping acts as passthrough."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {},
            }
        )
        row = {"a": 1, "b": 2}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row

    def test_requires_schema_config(self) -> None:
        """FieldMapper requires schema configuration."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="schema"):
            FieldMapper({"mapping": {"a": "b"}})

    def test_validate_input_rejects_wrong_type(self, ctx: PluginContext) -> None:
        """validate_input=True crashes on wrong types (upstream bug).

        Per three-tier trust model: transforms use allow_coercion=False,
        so string "42" is NOT coerced to int 42 - it raises ValidationError.
        """
        from pydantic import ValidationError

        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "strict", "fields": ["count: int"]},
                "mapping": {},
                "validate_input": True,
            }
        )

        with pytest.raises(ValidationError):
            transform.process({"count": "not_an_int"}, ctx)

    def test_validate_input_disabled_passes_wrong_type(
        self, ctx: PluginContext
    ) -> None:
        """validate_input=False (default) passes wrong types through.

        When validation is disabled, the transform doesn't check types.
        This is the default to avoid breaking existing pipelines.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "strict", "fields": ["count: int"]},
                "mapping": {},
                "validate_input": False,  # Explicit default
            }
        )

        # String passes through without validation
        result = transform.process({"count": "not_an_int"}, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == "not_an_int"

    def test_validate_input_skipped_for_dynamic_schema(
        self, ctx: PluginContext
    ) -> None:
        """validate_input=True with dynamic schema skips validation.

        Dynamic schemas accept anything, so validation is a no-op.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"fields": "dynamic"},
                "mapping": {},
                "validate_input": True,  # Would validate, but schema is dynamic
            }
        )

        # Any data passes with dynamic schema
        result = transform.process({"anything": "goes", "count": "string"}, ctx)
        assert result.status == "success"
