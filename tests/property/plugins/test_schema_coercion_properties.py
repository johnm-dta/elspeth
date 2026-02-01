# tests/property/plugins/test_schema_coercion_properties.py
"""Property-based tests for schema coercion idempotence.

These tests verify the fundamental property of ELSPETH's source coercion:
once data is coerced from external format to pipeline format, re-coercing
produces the same result. This is critical for audit integrity.

Key Properties:
- Idempotence: coerce(coerce(x)) == coerce(x)
- Stability: repeated coercion produces identical results
- Type preservation: coerced values maintain their Python type

Per CLAUDE.md Three-Tier Trust Model:
- Sources (Tier 3 → Tier 2): May coerce "42" → 42
- Transforms/Sinks: Expect already-coerced data (no re-coercion needed)
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.schema_factory import create_schema_from_config
from tests.property.conftest import json_primitives, row_data

# =============================================================================
# Strategies for generating coercible values
# =============================================================================

# String representations of integers (coercible to int)
str_integers = st.integers(min_value=-10000, max_value=10000).map(str)

# String representations of floats (coercible to float, excluding NaN/Inf)
str_floats = st.floats(
    min_value=-10000.0,
    max_value=10000.0,
    allow_nan=False,
    allow_infinity=False,
).map(lambda f: f"{f:.6f}")  # Format to avoid scientific notation edge cases

# String representations of booleans (Pydantic-coercible patterns)
str_bools = st.sampled_from(["true", "True", "false", "False", "1", "0", "yes", "no"])

# Native Python types (already correct type, should pass through unchanged)
native_ints = st.integers(min_value=-10000, max_value=10000)
native_floats = st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
native_bools = st.booleans()
native_strings = st.text(min_size=0, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",)))


# =============================================================================
# Coercion Idempotence Tests
# =============================================================================


class TestCoercionIdempotence:
    """Property tests verifying coercion is idempotent."""

    @given(str_value=str_integers)
    @settings(max_examples=100)
    def test_string_to_int_coercion_is_idempotent(self, str_value: str) -> None:
        """Property: Coercing str→int, then validating again, produces same int.

        This verifies that once "42" becomes 42, passing 42 through coercion
        again yields 42 (not some other transformation).
        """
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: int"],
            }
        )

        # Create schema with coercion enabled (like sources)
        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        # First coercion: str → int
        first_result = Schema.model_validate({"value": str_value})
        first_value = first_result.to_row()["value"]

        # Second coercion: int → int (should be unchanged)
        second_result = Schema.model_validate({"value": first_value})
        second_value = second_result.to_row()["value"]

        # IDEMPOTENCE: coerce(coerce(x)) == coerce(x)
        assert first_value == second_value, f"Coercion not idempotent: first={first_value!r}, second={second_value!r}"
        assert type(first_value) is type(second_value), f"Type changed: first={type(first_value)}, second={type(second_value)}"

    @given(str_value=str_floats)
    @settings(max_examples=100)
    def test_string_to_float_coercion_is_idempotent(self, str_value: str) -> None:
        """Property: Coercing str→float, then validating again, produces same float."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        # First coercion: str → float
        first_result = Schema.model_validate({"value": str_value})
        first_value = first_result.to_row()["value"]

        # Second coercion: float → float (should be unchanged)
        second_result = Schema.model_validate({"value": first_value})
        second_value = second_result.to_row()["value"]

        # IDEMPOTENCE
        assert first_value == second_value, f"Coercion not idempotent: first={first_value!r}, second={second_value!r}"

    @given(str_value=str_bools)
    @settings(max_examples=50)
    def test_string_to_bool_coercion_is_idempotent(self, str_value: str) -> None:
        """Property: Coercing str→bool, then validating again, produces same bool."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: bool"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        # First coercion: str → bool
        first_result = Schema.model_validate({"value": str_value})
        first_value = first_result.to_row()["value"]

        # Second coercion: bool → bool (should be unchanged)
        second_result = Schema.model_validate({"value": first_value})
        second_value = second_result.to_row()["value"]

        # IDEMPOTENCE
        assert first_value == second_value
        assert isinstance(first_value, bool)
        assert isinstance(second_value, bool)


class TestNativeTypePassthrough:
    """Property tests verifying native types pass through unchanged."""

    @given(value=native_ints)
    @settings(max_examples=100)
    def test_native_int_unchanged_by_coercion(self, value: int) -> None:
        """Property: Native int passes through coercion unchanged."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: int"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        result = Schema.model_validate({"value": value})
        result_value = result.to_row()["value"]

        assert result_value == value
        assert type(result_value) is int

    @given(value=native_floats)
    @settings(max_examples=100)
    def test_native_float_unchanged_by_coercion(self, value: float) -> None:
        """Property: Native float passes through coercion unchanged."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        result = Schema.model_validate({"value": value})
        result_value = result.to_row()["value"]

        assert result_value == value
        assert type(result_value) is float

    @given(value=native_bools)
    @settings(max_examples=20)
    def test_native_bool_unchanged_by_coercion(self, value: bool) -> None:
        """Property: Native bool passes through coercion unchanged."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: bool"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        result = Schema.model_validate({"value": value})
        result_value = result.to_row()["value"]

        assert result_value == value
        assert type(result_value) is bool

    @given(value=native_strings)
    @settings(max_examples=100)
    def test_native_string_unchanged_by_coercion(self, value: str) -> None:
        """Property: Native string passes through coercion unchanged."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: str"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        result = Schema.model_validate({"value": value})
        result_value = result.to_row()["value"]

        assert result_value == value
        assert type(result_value) is str


class TestIntToFloatWidening:
    """Property tests for int→float widening (always allowed)."""

    @given(value=native_ints)
    @settings(max_examples=100)
    def test_int_to_float_widening_is_idempotent(self, value: int) -> None:
        """Property: int→float widening produces stable float.

        Even with coercion disabled, int can widen to float (numeric promotion).
        This should be idempotent: once widened, re-validation is unchanged.
        """
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: float"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=False)

        # First pass: int → float
        first_result = Schema.model_validate({"value": value})
        first_value = first_result.to_row()["value"]

        # Second pass: float → float
        second_result = Schema.model_validate({"value": first_value})
        second_value = second_result.to_row()["value"]

        # IDEMPOTENCE
        assert first_value == second_value
        assert type(first_value) is float
        assert type(second_value) is float


class TestMultiFieldCoercionStability:
    """Property tests for multi-field schemas."""

    @given(
        int_str=str_integers,
        float_str=str_floats,
        bool_str=str_bools,
        string_val=native_strings,
    )
    @settings(max_examples=50)
    def test_multi_field_coercion_is_idempotent(
        self,
        int_str: str,
        float_str: str,
        bool_str: str,
        string_val: str,
    ) -> None:
        """Property: Multi-field schema coercion is idempotent for all fields."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": [
                    "int_field: int",
                    "float_field: float",
                    "bool_field: bool",
                    "str_field: str",
                ],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        # First coercion pass
        first_result = Schema.model_validate(
            {
                "int_field": int_str,
                "float_field": float_str,
                "bool_field": bool_str,
                "str_field": string_val,
            }
        )
        first_row = first_result.to_row()

        # Second coercion pass (using already-coerced values)
        second_result = Schema.model_validate(first_row)
        second_row = second_result.to_row()

        # IDEMPOTENCE for all fields
        assert first_row == second_row, f"Multi-field coercion not idempotent:\n  first:  {first_row}\n  second: {second_row}"


class TestCoercionDeterminism:
    """Property tests verifying coercion is deterministic."""

    @given(str_value=str_integers)
    @settings(max_examples=50)
    def test_same_input_same_output(self, str_value: str) -> None:
        """Property: Same input always produces same coerced output."""
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["value: int"],
            }
        )

        Schema = create_schema_from_config(schema_config, "TestSchema", allow_coercion=True)

        # Coerce the same value multiple times
        results = [Schema.model_validate({"value": str_value}).to_row()["value"] for _ in range(5)]

        # All results should be identical
        assert all(r == results[0] for r in results), f"Non-deterministic coercion: {results}"


class TestNoCoercionRejection:
    """Property tests verifying strict schemas reject coercion."""

    @given(str_value=str_integers)
    @settings(max_examples=50)
    def test_string_to_int_rejected_when_coercion_disabled(self, str_value: str) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: int"]})
        Schema = create_schema_from_config(schema_config, "NoCoerceInt", allow_coercion=False)

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": str_value})

    @given(str_value=str_floats)
    @settings(max_examples=50)
    def test_string_to_float_rejected_when_coercion_disabled(self, str_value: str) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float"]})
        Schema = create_schema_from_config(schema_config, "NoCoerceFloat", allow_coercion=False)

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": str_value})

    @given(str_value=str_bools)
    @settings(max_examples=50)
    def test_string_to_bool_rejected_when_coercion_disabled(self, str_value: str) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: bool"]})
        Schema = create_schema_from_config(schema_config, "NoCoerceBool", allow_coercion=False)

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": str_value})


class TestExtraFieldHandling:
    """Property tests for strict vs free extra field handling."""

    @given(extra_key=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))), extra_value=json_primitives)
    @settings(max_examples=50)
    def test_strict_schema_rejects_extra_fields(self, extra_key: str, extra_value: object) -> None:
        assume(extra_key != "value")

        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: int"]})
        Schema = create_schema_from_config(schema_config, "StrictSchema", allow_coercion=True)

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": 1, extra_key: extra_value})

    @given(extra_key=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))), extra_value=json_primitives)
    @settings(max_examples=50)
    def test_free_schema_allows_extra_fields(self, extra_key: str, extra_value: object) -> None:
        assume(extra_key != "value")

        schema_config = SchemaConfig.from_dict({"mode": "free", "fields": ["value: int"]})
        Schema = create_schema_from_config(schema_config, "FreeSchema", allow_coercion=True)

        result = Schema.model_validate({"value": 1, extra_key: extra_value})
        result_row = result.to_row()
        assert result_row["value"] == 1
        assert result_row[extra_key] == extra_value


class TestOptionalAndDynamicSchemas:
    """Property tests for optional and dynamic schema behavior."""

    @given(value=st.one_of(native_ints, st.none()))
    @settings(max_examples=50)
    def test_optional_field_accepts_none(self, value: int | None) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: int?"]})
        Schema = create_schema_from_config(schema_config, "OptionalSchema", allow_coercion=False)

        result = Schema.model_validate({"value": value})
        assert result.to_row()["value"] == value

    def test_optional_field_missing_defaults_to_none(self) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: int?"]})
        Schema = create_schema_from_config(schema_config, "OptionalMissingSchema", allow_coercion=False)

        result = Schema.model_validate({})
        assert result.to_row()["value"] is None

    def test_required_field_missing_rejected(self) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: int"]})
        Schema = create_schema_from_config(schema_config, "RequiredSchema", allow_coercion=False)

        with pytest.raises(ValidationError):
            Schema.model_validate({})

    @given(row=row_data)
    @settings(max_examples=50)
    def test_dynamic_schema_accepts_any_fields(self, row: dict[str, object]) -> None:
        schema_config = SchemaConfig.from_dict({"fields": "dynamic"})
        Schema = create_schema_from_config(schema_config, "DynamicSchema", allow_coercion=False)

        result = Schema.model_validate(row)
        assert result.to_row() == row


class TestFiniteFloatRejection:
    """Property tests for NaN/Infinity rejection at source boundary."""

    @given(value=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
    @settings(max_examples=50)
    def test_nan_and_infinity_rejected(self, value: float) -> None:
        schema_config = SchemaConfig.from_dict({"mode": "strict", "fields": ["value: float"]})
        Schema = create_schema_from_config(schema_config, "FiniteFloatSchema", allow_coercion=True)

        with pytest.raises(ValidationError):
            Schema.model_validate({"value": value})
