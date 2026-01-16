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

    def test_schema_is_mutable(self) -> None:
        """PluginSchema instances are mutable (Their Data trust boundary)."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        schema = MySchema(name="test")
        # PluginSchema is NOT frozen - it validates "Their Data" which needs
        # permissive handling per the Data Manifesto
        schema.name = "changed"
        assert schema.name == "changed"

    def test_schema_ignores_extra(self) -> None:
        """PluginSchema ignores unknown fields (Their Data trust boundary)."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        # Extra fields are ignored, not rejected - this is correct for
        # "Their Data" which may have more fields than schema requires
        schema = MySchema(name="test", unknown_field="value")  # type: ignore[call-arg]
        assert schema.name == "test"
        # Extra field is silently ignored (not stored)
        assert not hasattr(schema, "unknown_field")
