"""Tests for data contracts."""

import pytest
from pydantic import ValidationError


class TestPluginSchema:
    """Tests for PluginSchema base class."""

    def test_subclass_validates_input(self) -> None:
        """PluginSchema subclasses validate input."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str
            value: int

        schema = MySchema(name="test", value=42)
        assert schema.name == "test"

        with pytest.raises(ValidationError):
            MySchema(name="test", value="not_an_int")

    def test_coercion_with_strict_false(self) -> None:
        """PluginSchema coerces compatible types (strict=False)."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str
            value: int

        schema = MySchema(name="test", value="42")  # type: ignore[arg-type]
        assert schema.value == 42
        assert type(schema.value) is int

    def test_schema_is_mutable(self) -> None:
        """PluginSchema instances are mutable (Their Data trust boundary)."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        schema = MySchema(name="test")
        schema.name = "changed"
        assert schema.name == "changed"

    def test_schema_ignores_extra(self) -> None:
        """PluginSchema ignores unknown fields (Their Data trust boundary)."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        schema = MySchema(name="test", unknown_field="value")  # type: ignore[call-arg]
        assert schema.name == "test"
        field_names = set(MySchema.model_fields.keys())
        assert "unknown_field" not in field_names
