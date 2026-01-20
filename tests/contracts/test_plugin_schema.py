"""Tests for PluginSchema in contracts module."""

import pytest


class TestPluginSchemaLocation:
    """Verify PluginSchema is importable from contracts."""

    def test_plugin_schema_importable_from_contracts(self) -> None:
        """PluginSchema should be importable from elspeth.contracts."""
        from elspeth.contracts import PluginSchema

        assert PluginSchema.model_config.get("extra") == "ignore"
        assert PluginSchema.model_config.get("frozen") is False

    def test_schema_validation_error_importable_from_contracts(self) -> None:
        """SchemaValidationError should be importable from contracts."""
        from elspeth.contracts import SchemaValidationError

        error = SchemaValidationError("field", "message", "value")
        assert error.field == "field"
        assert error.message == "message"

    def test_compatibility_result_importable_from_contracts(self) -> None:
        """CompatibilityResult should be importable from contracts."""
        from elspeth.contracts import CompatibilityResult

        result = CompatibilityResult(compatible=True)
        assert result.compatible is True

    def test_validate_row_importable_from_contracts(self) -> None:
        """validate_row should be importable from contracts."""
        from elspeth.contracts import PluginSchema, validate_row

        class TestSchema(PluginSchema):
            name: str

        errors = validate_row({"name": "test"}, TestSchema)
        assert errors == []

    def test_check_compatibility_importable_from_contracts(self) -> None:
        """check_compatibility should be importable from contracts."""
        from elspeth.contracts import PluginSchema, check_compatibility

        class SchemaA(PluginSchema):
            name: str

        class SchemaB(PluginSchema):
            name: str

        result = check_compatibility(SchemaA, SchemaB)
        assert result.compatible is True


class TestPluginSchemaNotInOldLocation:
    """Verify plugins/schemas.py has been deleted."""

    def test_old_import_path_removed(self) -> None:
        """Importing from plugins.schemas should fail - module deleted."""

        with pytest.raises(ModuleNotFoundError):
            from elspeth.plugins.schemas import (
                PluginSchema,  # type: ignore[import-not-found]  # noqa: F401
            )
