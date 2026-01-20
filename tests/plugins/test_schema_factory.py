"""Tests for schema factory - creates Pydantic models from config.

Note: These tests create dynamic schemas at runtime and access their fields.
Mypy cannot know about dynamically-created attributes, so we use type: ignore
for attribute access on generated schema instances.
"""

# mypy: disable-error-code="attr-defined"

import pytest
from pydantic import ValidationError


class TestCreateSchemaFromConfig:
    """Tests for create_schema_from_config function."""

    def test_factory_exists(self) -> None:
        """Factory function can be imported."""
        from elspeth.plugins.schema_factory import create_schema_from_config

        assert create_schema_from_config is not None

    def test_dynamic_schema_accepts_anything(self) -> None:
        """Dynamic schema accepts arbitrary fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"fields": "dynamic"})
        Schema = create_schema_from_config(config, "TestSchema")

        # Should accept any fields
        instance = Schema(foo="bar", count=42, nested={"a": 1})
        assert instance.model_dump() == {"foo": "bar", "count": 42, "nested": {"a": 1}}

    def test_strict_schema_rejects_extra_fields(self) -> None:
        """Strict schema rejects extra fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["id: int", "name: str"],
            }
        )
        Schema = create_schema_from_config(config, "StrictSchema")

        # Should accept exact fields
        instance = Schema(id=1, name="Alice")
        assert instance.model_dump() == {"id": 1, "name": "Alice"}

        # Should reject extra fields
        with pytest.raises(ValidationError, match="extra"):
            Schema(id=1, name="Alice", extra_field="nope")

    def test_strict_schema_requires_all_fields(self) -> None:
        """Strict schema requires all non-optional fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["id: int", "name: str"],
            }
        )
        Schema = create_schema_from_config(config, "StrictSchema")

        # Should require all fields
        with pytest.raises(ValidationError, match="name"):
            Schema(id=1)

    def test_free_schema_allows_extra_fields(self) -> None:
        """Free schema allows extra fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "free",
                "fields": ["id: int", "name: str"],
            }
        )
        Schema = create_schema_from_config(config, "FreeSchema")

        # Should accept required + extra fields
        instance = Schema(id=1, name="Alice", extra="allowed")
        dumped = instance.model_dump()
        assert dumped["id"] == 1
        assert dumped["name"] == "Alice"
        assert dumped["extra"] == "allowed"

    def test_free_schema_requires_specified_fields(self) -> None:
        """Free schema still requires specified fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "free",
                "fields": ["id: int", "name: str"],
            }
        )
        Schema = create_schema_from_config(config, "FreeSchema")

        # Should require specified fields
        with pytest.raises(ValidationError, match="name"):
            Schema(id=1)

    def test_optional_field_can_be_missing(self) -> None:
        """Optional fields (?) can be missing or None."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["id: int", "score: float?"],
            }
        )
        Schema = create_schema_from_config(config, "OptionalSchema")

        # Should accept without optional field
        instance = Schema(id=1)
        assert instance.model_dump() == {"id": 1, "score": None}

        # Should accept with optional field
        instance2 = Schema(id=2, score=3.14)
        assert instance2.model_dump() == {"id": 2, "score": 3.14}

        # Should accept explicit None
        instance3 = Schema(id=3, score=None)
        assert instance3.model_dump() == {"id": 3, "score": None}

    def test_type_coercion_int_to_float(self) -> None:
        """Int values coerce to float fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "CoerceSchema")

        instance = Schema(value=42)  # int -> float
        assert instance.value == 42.0
        assert isinstance(instance.value, float)

    def test_type_coercion_string_to_int(self) -> None:
        """String numeric values coerce to int fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["count: int"],
            }
        )
        Schema = create_schema_from_config(config, "CoerceSchema")

        instance = Schema(count="42")  # str -> int
        assert instance.count == 42
        assert isinstance(instance.count, int)

    def test_type_coercion_string_to_float(self) -> None:
        """String numeric values coerce to float fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "CoerceSchema")

        instance = Schema(value="3.14")  # str -> float
        assert instance.value == 3.14

    def test_type_coercion_string_to_bool(self) -> None:
        """String boolean values coerce to bool fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["active: bool"],
            }
        )
        Schema = create_schema_from_config(config, "CoerceSchema")

        # Various truthy strings
        assert Schema(active="true").active is True
        assert Schema(active="True").active is True
        assert Schema(active="1").active is True
        assert Schema(active="yes").active is True

        # Various falsy strings
        assert Schema(active="false").active is False
        assert Schema(active="False").active is False
        assert Schema(active="0").active is False
        assert Schema(active="no").active is False

    def test_any_type_accepts_anything(self) -> None:
        """'any' type accepts any value without coercion."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["data: any"],
            }
        )
        Schema = create_schema_from_config(config, "AnySchema")

        # Accept various types
        assert Schema(data="string").data == "string"
        assert Schema(data=42).data == 42
        assert Schema(data=[1, 2, 3]).data == [1, 2, 3]
        assert Schema(data={"nested": "dict"}).data == {"nested": "dict"}

    def test_invalid_type_not_coercible(self) -> None:
        """Non-coercible values raise ValidationError."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["count: int"],
            }
        )
        Schema = create_schema_from_config(config, "CoerceSchema")

        with pytest.raises(ValidationError):
            Schema(count="not_a_number")


class TestCoercionControl:
    """Tests for coercion control - enforces three-tier trust model."""

    def test_coercion_enabled_by_default(self) -> None:
        """Default behavior allows coercion (for sources)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["count: int"],
            }
        )
        # Default: allow_coercion=True (source behavior)
        Schema = create_schema_from_config(config, "SourceSchema")

        instance = Schema(count="42")  # str -> int coercion
        assert instance.count == 42
        assert isinstance(instance.count, int)

    def test_coercion_disabled_rejects_string_to_int(self) -> None:
        """With coercion disabled, string '42' is rejected for int field."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["count: int"],
            }
        )
        # Transforms/sinks: allow_coercion=False
        Schema = create_schema_from_config(config, "TransformSchema", allow_coercion=False)

        # Should REJECT string, not coerce
        with pytest.raises(ValidationError, match="int"):
            Schema(count="42")

    def test_coercion_disabled_rejects_string_to_float(self) -> None:
        """With coercion disabled, string '3.14' is rejected for float field."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "TransformSchema", allow_coercion=False)

        with pytest.raises(ValidationError, match="float"):
            Schema(value="3.14")

    def test_coercion_disabled_still_accepts_correct_types(self) -> None:
        """With coercion disabled, correct types are still accepted."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["count: int", "value: float", "name: str"],
            }
        )
        Schema = create_schema_from_config(config, "TransformSchema", allow_coercion=False)

        # Correct types work fine
        instance = Schema(count=42, value=3.14, name="Alice")
        assert instance.count == 42
        assert instance.value == 3.14
        assert instance.name == "Alice"

    def test_coercion_disabled_allows_int_to_float(self) -> None:
        """Int -> float is allowed even without coercion (numeric widening)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "TransformSchema", allow_coercion=False)

        # int -> float is always safe (widening, not coercion)
        instance = Schema(value=42)
        assert instance.value == 42.0

    def test_dynamic_schema_with_coercion_disabled(self) -> None:
        """Dynamic schema with coercion disabled still accepts any types."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"fields": "dynamic"})
        Schema = create_schema_from_config(config, "DynamicSchema", allow_coercion=False)

        # Dynamic accepts anything - no type checking
        instance = Schema(foo="bar", count="42", value="3.14")
        assert instance.model_dump() == {"foo": "bar", "count": "42", "value": "3.14"}


class TestSchemaPluginSchemaCompliance:
    """Tests for PluginSchema compliance and conversion methods."""

    def test_schema_is_plugin_schema_subclass(self) -> None:
        """Generated schema is a PluginSchema subclass."""
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"fields": "dynamic"})
        Schema = create_schema_from_config(config, "TestSchema")

        assert issubclass(Schema, PluginSchema)

    def test_to_row_returns_all_fields(self) -> None:
        """to_row() returns all fields including extras in free mode."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "free",
                "fields": ["id: int"],
            }
        )
        Schema = create_schema_from_config(config, "FreeSchema")

        instance = Schema(id=1, extra="value")
        row = instance.to_row()
        assert row == {"id": 1, "extra": "value"}
