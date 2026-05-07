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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        assert create_schema_from_config is not None

    def test_dynamic_schema_accepts_anything(self) -> None:
        """Dynamic schema accepts arbitrary fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "TestSchema")

        # Should accept any fields
        instance = Schema(foo="bar", count=42, nested={"a": 1})
        assert instance.model_dump() == {"foo": "bar", "count": 42, "nested": {"a": 1}}

    def test_strict_schema_rejects_extra_fields(self) -> None:
        """Strict schema rejects extra fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "CoerceSchema")

        instance = Schema(value="3.14")  # str -> float
        assert instance.value == 3.14

    def test_type_coercion_string_to_bool(self) -> None:
        """String boolean values coerce to bool fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "TransformSchema", allow_coercion=False)

        with pytest.raises(ValidationError, match="float"):
            Schema(value="3.14")

    def test_coercion_disabled_still_accepts_correct_types(self) -> None:
        """With coercion disabled, correct types are still accepted."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
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
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "DynamicSchema", allow_coercion=False)

        # Dynamic accepts anything - no type checking
        instance = Schema(foo="bar", count="42", value="3.14")
        assert instance.model_dump() == {"foo": "bar", "count": "42", "value": "3.14"}


class TestNonFiniteFloatRejection:
    """Tests for NaN/Infinity rejection at source boundary.

    Per P2-2026-01-19-non-finite-floats-pass-source-validation:
    Non-finite floats (NaN, Infinity) must be rejected at the source boundary
    because they cannot be represented in canonical JSON (RFC 8785) and would
    crash later during hashing. This is a Tier 3 -> Tier 1 boundary enforcement.
    """

    def test_nan_string_rejected_in_float_field(self) -> None:
        """Source schema rejects 'nan' string for float fields.

        NaN cannot be represented in canonical JSON and would crash during
        hashing. Must be caught at source validation, not downstream.
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        # Source uses allow_coercion=True (default)
        Schema = create_schema_from_config(config, "SourceSchema")

        with pytest.raises(ValidationError):
            Schema(value="nan")

    def test_infinity_string_rejected_in_float_field(self) -> None:
        """Source schema rejects 'inf' string for float fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "SourceSchema")

        with pytest.raises(ValidationError):
            Schema(value="inf")

    def test_negative_infinity_rejected_in_float_field(self) -> None:
        """Source schema rejects '-inf' string for float fields."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "SourceSchema")

        with pytest.raises(ValidationError):
            Schema(value="-inf")

    def test_actual_nan_float_rejected(self) -> None:
        """Source schema rejects actual float('nan') value."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "SourceSchema")

        with pytest.raises(ValidationError):
            Schema(value=float("nan"))

    def test_actual_infinity_float_rejected(self) -> None:
        """Source schema rejects actual float('inf') value."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "SourceSchema")

        with pytest.raises(ValidationError):
            Schema(value=float("inf"))

    def test_optional_float_still_rejects_nan(self) -> None:
        """Optional float fields also reject NaN."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float?"],
            }
        )
        Schema = create_schema_from_config(config, "SourceSchema")

        # None is fine
        instance = Schema(value=None)
        assert instance.value is None

        # NaN is not
        with pytest.raises(ValidationError):
            Schema(value="nan")

    def test_finite_floats_still_accepted(self) -> None:
        """Normal finite floats are still accepted."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: float"],
            }
        )
        Schema = create_schema_from_config(config, "SourceSchema")

        # Finite values work
        assert Schema(value=3.14).value == 3.14
        assert Schema(value=-273.15).value == -273.15
        assert Schema(value=0.0).value == 0.0
        assert Schema(value="3.14").value == 3.14  # Coercion still works

    def test_observed_schema_rejects_nan(self) -> None:
        """Observed schemas reject NaN values before contract inference."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "ObservedSchema")

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": float("nan")})

    def test_observed_schema_rejects_nested_infinity(self) -> None:
        """Observed schemas reject nested Infinity values."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "ObservedSchema")

        with pytest.raises(ValidationError):
            Schema.model_validate({"outer": {"inner": [1.0, float("inf")]}})

    def test_explicit_any_field_rejects_nan_at_source_boundary(self) -> None:
        """Explicit schema with 'any' field rejects NaN at source boundary.

        Regression: Non-finite values bypassed source validation for explicit
        schemas because only observed schemas had the non-finite validator.
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "fixed", "fields": ["data: any"]})
        Schema = create_schema_from_config(config, "SourceSchema", allow_coercion=True)

        with pytest.raises(ValidationError, match=r"[Nn]on-finite"):
            Schema(data=float("nan"))

    def test_explicit_any_field_rejects_infinity_at_source_boundary(self) -> None:
        """Explicit schema with 'any' field rejects Infinity at source boundary."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "fixed", "fields": ["data: any"]})
        Schema = create_schema_from_config(config, "SourceSchema", allow_coercion=True)

        with pytest.raises(ValidationError, match=r"[Nn]on-finite"):
            Schema(data=float("inf"))

    def test_flexible_extras_reject_nan_at_source_boundary(self) -> None:
        """Flexible schema rejects NaN in extra fields at source boundary."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "flexible", "fields": ["id: int"]})
        Schema = create_schema_from_config(config, "SourceSchema", allow_coercion=True)

        with pytest.raises(ValidationError, match=r"[Nn]on-finite"):
            Schema(id=1, extra_score=float("nan"))

    def test_explicit_any_nested_nan_rejected_at_source_boundary(self) -> None:
        """Nested NaN in any-typed field is rejected at source boundary."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "fixed", "fields": ["data: any"]})
        Schema = create_schema_from_config(config, "SourceSchema", allow_coercion=True)

        with pytest.raises(ValidationError, match=r"[Nn]on-finite"):
            Schema(data={"nested": [1.0, float("inf")]})

    def test_observed_schema_rejects_numpy_longdouble_nan(self) -> None:
        """np.longdouble NaN must be caught — name heuristic missed non-'float' numpy types."""
        import numpy as np

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "ObservedSchema")

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": np.longdouble("nan")})

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": np.longdouble("inf")})

    def test_observed_schema_rejects_nan_inside_numpy_array(self) -> None:
        """NaN/Infinity inside NumPy arrays must be detected at source boundary.

        Bug fix for elspeth-214096f76b: _find_non_finite_value_path had no
        ndarray handling, so rows with NaN in arrays passed source validation
        and crashed later during audit hashing.
        """
        import numpy as np

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "ObservedSchema")

        with pytest.raises(ValidationError):
            Schema.model_validate({"values": np.array([1.0, float("nan"), 3.0])})

        with pytest.raises(ValidationError):
            Schema.model_validate({"values": np.array([1.0, 2.0, float("inf")])})

    def test_observed_schema_accepts_finite_numpy_array(self) -> None:
        """Finite NumPy arrays pass source validation."""
        import numpy as np

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "ObservedSchema")

        instance = Schema.model_validate({"values": np.array([1.0, 2.0, 3.0])})
        assert instance is not None

    def test_observed_schema_accepts_string_numpy_array(self) -> None:
        """String NumPy arrays (non-numeric) pass source validation without error."""
        import numpy as np

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "ObservedSchema")

        instance = Schema.model_validate({"labels": np.array(["a", "b", "c"])})
        assert instance is not None

    def test_find_non_finite_path_numpy_array_reports_correct_index(self) -> None:
        """_find_non_finite_value_path reports useful path for ndarray NaN."""
        import numpy as np

        from elspeth.plugins.infrastructure.schema_factory import _find_non_finite_value_path

        # 1-D array with NaN at index 2
        path = _find_non_finite_value_path(np.array([1.0, 2.0, float("nan")]), "$")
        assert path is not None
        assert "[2]" in path

    def test_find_non_finite_path_2d_numpy_array(self) -> None:
        """_find_non_finite_value_path handles multi-dimensional arrays."""
        import numpy as np

        from elspeth.plugins.infrastructure.schema_factory import _find_non_finite_value_path

        arr = np.array([[1.0, 2.0], [3.0, float("inf")]])
        path = _find_non_finite_value_path(arr, "$")
        assert path is not None
        assert "[1]" in path and "[1]" in path  # row 1, col 1

    def test_find_non_finite_path_longdouble_scalar_uses_native_check(self) -> None:
        """np.longdouble finite values must not be falsely reported as non-finite.

        Bug fix: the old code used math.isfinite(float(value)) which overflows.
        """
        import numpy as np

        from elspeth.plugins.infrastructure.schema_factory import _find_non_finite_value_path

        # Large but finite value in longdouble range
        large_val = np.longdouble(2.0) ** np.longdouble(1024)
        assert np.isfinite(large_val), "Test precondition"

        path = _find_non_finite_value_path(large_val, "$")
        assert path is None, "Finite np.longdouble must not be reported as non-finite"

    def test_explicit_schema_no_validator_at_transform_boundary(self) -> None:
        """Transform boundary (allow_coercion=False) does not add non-finite validator.

        Transforms should never see NaN — if they do, it's an upstream bug
        and should crash downstream (per trust model), not silently validate.
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "fixed", "fields": ["data: any"]})
        Schema = create_schema_from_config(config, "TransformSchema", allow_coercion=False)

        # NaN passes schema validation at transform boundary — crash happens downstream
        instance = Schema(data=float("nan"))
        assert instance.data != instance.data  # NaN != NaN


class TestSchemaPluginSchemaCompliance:
    """Tests for PluginSchema compliance and conversion methods."""

    def test_schema_is_plugin_schema_subclass(self) -> None:
        """Generated schema is a PluginSchema subclass."""
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict({"mode": "observed"})
        Schema = create_schema_from_config(config, "TestSchema")

        assert issubclass(Schema, PluginSchema)

    def test_to_row_returns_all_fields(self) -> None:
        """to_row() returns all fields including extras in free mode."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
                "fields": ["id: int"],
            }
        )
        Schema = create_schema_from_config(config, "FreeSchema")

        instance = Schema(id=1, extra="value")
        row = instance.to_row()
        assert row == {"id": 1, "extra": "value"}
