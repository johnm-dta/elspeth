# tests/plugins/test_schemas.py
"""Tests for plugin schema system."""

import pytest
from pydantic import ValidationError


class TestPluginSchema:
    """Base class for plugin schemas."""

    def test_schema_validates_fields(self) -> None:
        from pydantic import BaseModel

        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            temperature: float
            humidity: float

        # Valid data
        data = MySchema(temperature=20.5, humidity=65.0)
        assert data.temperature == 20.5

        # Invalid data
        with pytest.raises(ValidationError):
            MySchema(temperature="not a number", humidity=65.0)

    def test_schema_to_dict(self) -> None:
        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            value: int
            name: str

        data = MySchema(value=42, name="test")
        as_dict = data.to_row()
        assert as_dict == {"value": 42, "name": "test"}

    def test_schema_from_row(self) -> None:
        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            value: int
            name: str

        row = {"value": 42, "name": "test", "extra": "ignored"}
        data = MySchema.from_row(row)
        assert data.value == 42
        assert data.name == "test"

    def test_schema_extra_fields_ignored(self) -> None:
        from elspeth.plugins.schemas import PluginSchema

        class StrictSchema(PluginSchema):
            required_field: str

        # Extra fields should be ignored, not cause errors
        data = StrictSchema.from_row({"required_field": "value", "extra": "ignored"})
        assert data.required_field == "value"


class TestSchemaValidation:
    """Schema validation utilities."""

    def test_validate_row_against_schema(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, validate_row

        class MySchema(PluginSchema):
            x: int
            y: int

        # Valid
        errors = validate_row({"x": 1, "y": 2}, MySchema)
        assert errors == []

        # Invalid
        errors = validate_row({"x": "not int", "y": 2}, MySchema)
        assert len(errors) > 0

    def test_validate_missing_field(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, validate_row

        class MySchema(PluginSchema):
            required: str

        errors = validate_row({}, MySchema)
        assert len(errors) > 0
        assert "required" in str(errors[0])
