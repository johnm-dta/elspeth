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

        # Valid input
        schema = MySchema(name="test", value=42)
        assert schema.name == "test"

        # Invalid input raises
        with pytest.raises(ValidationError):
            MySchema(name="test", value="not_an_int")

    def test_schema_is_frozen(self) -> None:
        """PluginSchema instances are immutable."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        schema = MySchema(name="test")
        with pytest.raises(ValidationError):
            schema.name = "changed"  # type: ignore[misc]

    def test_schema_forbids_extra(self) -> None:
        """PluginSchema rejects unknown fields."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        with pytest.raises(ValidationError):
            MySchema(name="test", unknown_field="value")  # type: ignore[call-arg]
