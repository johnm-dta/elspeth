"""Tests for FieldMapper transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol


class TestFieldMapper:
    """Tests for FieldMapper transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """FieldMapper implements TransformProtocol."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {"old": "new"}})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """FieldMapper has name and schemas."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        assert FieldMapper.name == "field_mapper"

    def test_rename_single_field(self, ctx: PluginContext) -> None:
        """Rename a single field."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {"old_name": "new_name"}})
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
                "mapping": {
                    "first_name": "firstName",
                    "last_name": "lastName",
                }
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

        transform = FieldMapper({"mapping": {"missing": "output"}})
        row = {"exists": "value"}

        result = transform.process(row, ctx)

        assert result.status == "success"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested fields with dot notation."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"meta.source": "origin"},
            }
        )
        row = {"id": 1, "meta": {"source": "api", "timestamp": 123}}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row["origin"] == "api"
        assert "meta" in result.row  # Original nested structure preserved

    def test_empty_mapping_passthrough(self, ctx: PluginContext) -> None:
        """Empty mapping acts as passthrough."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {}})
        row = {"a": 1, "b": 2}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
