"""Tests for data contracts."""

from __future__ import annotations

from typing import Annotated, Optional

import pytest
from pydantic import Field, ValidationError

from elspeth.contracts.data import (
    PluginSchema,
    _check_field_constraints,
    _types_compatible,
    _unwrap_annotated,
    check_compatibility,
)


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

        schema = MySchema(name="test", value="42")
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

        schema = MySchema(name="test", unknown_field="value")
        assert schema.name == "test"
        field_names = set(MySchema.model_fields.keys())
        assert "unknown_field" not in field_names


class TestUnwrapAnnotated:
    """Tests for _unwrap_annotated helper."""

    def test_plain_type_unchanged(self) -> None:
        """Plain types pass through unchanged."""
        assert _unwrap_annotated(float) is float
        assert _unwrap_annotated(int) is int
        assert _unwrap_annotated(str) is str

    def test_annotated_unwraps_to_base_type(self) -> None:
        """Annotated[T, ...] unwraps to T."""
        assert _unwrap_annotated(Annotated[float, "metadata"]) is float
        assert _unwrap_annotated(Annotated[int, "constraint"]) is int

    def test_nested_annotated_fully_unwraps(self) -> None:
        """Nested Annotated types are fully unwrapped."""
        nested = Annotated[Annotated[float, "inner"], "outer"]
        assert _unwrap_annotated(nested) is float

    def test_optional_not_unwrapped(self) -> None:
        """Optional (Union) is not unwrapped by _unwrap_annotated."""
        from typing import Union, get_origin

        result = _unwrap_annotated(Optional[float])  # noqa: UP045
        # Optional[float] is Union[float, None] - should not be unwrapped
        assert get_origin(result) is Union


class TestTypesCompatibleAnnotated:
    """Tests for _types_compatible with Annotated type handling.

    Regression tests for P1 bug: check_compatibility incorrectly rejects
    float -> Optional[Annotated[float, ...]] because Annotated metadata
    was not unwrapped before comparison.
    """

    def test_float_compatible_with_annotated_float(self) -> None:
        """float is compatible with Annotated[float, ...]."""
        assert _types_compatible(float, Annotated[float, "constraint"]) is True

    def test_annotated_float_compatible_with_float(self) -> None:
        """Annotated[float, ...] is compatible with float."""
        assert _types_compatible(Annotated[float, "constraint"], float) is True

    def test_float_compatible_with_optional_annotated_float(self) -> None:
        """float is compatible with Optional[Annotated[float, ...]]."""
        assert _types_compatible(float, Optional[Annotated[float, "constraint"]]) is True  # noqa: UP045

    def test_int_compatible_with_annotated_float_nonstrict(self) -> None:
        """int -> Annotated[float, ...] is compatible when not strict."""
        assert _types_compatible(int, Annotated[float, "constraint"], consumer_strict=False) is True

    def test_int_incompatible_with_annotated_float_strict(self) -> None:
        """int -> Annotated[float, ...] is rejected when strict."""
        assert _types_compatible(int, Annotated[float, "constraint"], consumer_strict=True) is False

    def test_annotated_int_compatible_with_optional_annotated_float(self) -> None:
        """Annotated[int, ...] -> Optional[Annotated[float, ...]] compatible (int->float coercion)."""
        assert (
            _types_compatible(
                Annotated[int, "producer_meta"],
                Optional[Annotated[float, "consumer_meta"]],  # noqa: UP045
                consumer_strict=False,
            )
            is True
        )

    def test_annotated_str_incompatible_with_annotated_float(self) -> None:
        """Annotated[str, ...] is NOT compatible with Annotated[float, ...]."""
        assert (
            _types_compatible(
                Annotated[str, "meta"],
                Annotated[float, "meta"],
            )
            is False
        )


class TestCheckCompatibilityAnnotated:
    """Integration tests for check_compatibility with Annotated schemas.

    Tests the full path from PluginSchema through check_compatibility,
    ensuring config-generated schemas with FiniteFloat work correctly.
    """

    def test_float_producer_optional_annotated_float_consumer(self) -> None:
        """Producer float field compatible with consumer Optional[Annotated[float, ...]]."""

        class ProducerSchema(PluginSchema):
            score: float

        class ConsumerSchema(PluginSchema):
            score: Optional[Annotated[float, "finite"]] = None  # noqa: UP045

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        assert result.compatible is True
        assert result.type_mismatches == ()

    def test_annotated_producer_plain_consumer(self) -> None:
        """Producer with Annotated[float, ...] compatible with consumer plain float."""

        class ProducerSchema(PluginSchema):
            score: Annotated[float, "constrained"]

        class ConsumerSchema(PluginSchema):
            score: float

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        assert result.compatible is True


class TestCheckFieldConstraints:
    """Tests for _check_field_constraints with Pydantic FieldInfo constraints."""

    def test_no_constraints_compatible(self) -> None:
        """Fields with no constraints are compatible."""

        class Schema(PluginSchema):
            value: float

        field = Schema.model_fields["value"]
        assert _check_field_constraints("value", field, field) is None

    def test_both_finite_compatible(self) -> None:
        """Both fields with allow_inf_nan=False are compatible."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            value: FiniteFloat

        class ConsumerSchema(PluginSchema):
            value: FiniteFloat

        producer_field = ProducerSchema.model_fields["value"]
        consumer_field = ConsumerSchema.model_fields["value"]
        assert _check_field_constraints("value", producer_field, consumer_field) is None

    def test_producer_finite_consumer_plain_compatible(self) -> None:
        """Producer guarantees finite, consumer accepts anything -- compatible."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            value: FiniteFloat

        class ConsumerSchema(PluginSchema):
            value: float

        producer_field = ProducerSchema.model_fields["value"]
        consumer_field = ConsumerSchema.model_fields["value"]
        assert _check_field_constraints("value", producer_field, consumer_field) is None

    def test_producer_plain_consumer_finite_incompatible(self) -> None:
        """Producer emits unconstrained float, consumer requires finite -- incompatible."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            value: float

        class ConsumerSchema(PluginSchema):
            value: FiniteFloat

        producer_field = ProducerSchema.model_fields["value"]
        consumer_field = ConsumerSchema.model_fields["value"]
        reason = _check_field_constraints("value", producer_field, consumer_field)
        assert reason is not None
        assert "allow_inf_nan" in reason


class TestCheckCompatibilityConstraints:
    """Integration tests for check_compatibility with FieldInfo constraints.

    Regression tests for bug: check_compatibility treated constrained Annotated
    types (e.g., FiniteFloat) as plain base types, allowing incompatible schemas
    to pass validation when consumer required stricter constraints than producer.
    """

    def test_float_producer_finite_consumer_incompatible(self) -> None:
        """Producer float is incompatible with consumer FiniteFloat."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            score: float

        class ConsumerSchema(PluginSchema):
            score: FiniteFloat

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        assert result.compatible is False
        assert len(result.constraint_mismatches) == 1
        assert result.constraint_mismatches[0][0] == "score"
        assert "allow_inf_nan" in result.constraint_mismatches[0][1]
        assert result.type_mismatches == ()  # types match, only constraint differs

    def test_finite_producer_float_consumer_compatible(self) -> None:
        """Producer FiniteFloat is compatible with consumer float (stricter is fine)."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            score: FiniteFloat

        class ConsumerSchema(PluginSchema):
            score: float

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        assert result.compatible is True

    def test_finite_producer_finite_consumer_compatible(self) -> None:
        """Producer and consumer both FiniteFloat -- compatible."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            score: FiniteFloat

        class ConsumerSchema(PluginSchema):
            score: FiniteFloat

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        assert result.compatible is True

    def test_error_message_includes_constraint_info(self) -> None:
        """Incompatibility error message mentions constraint mismatch."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            score: float

        class ConsumerSchema(PluginSchema):
            score: FiniteFloat

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        assert result.error_message is not None
        assert "Constraint mismatches" in result.error_message
        assert "score" in result.error_message

    def test_int_producer_finite_consumer_compatible(self) -> None:
        """int -> FiniteFloat: compatible because int values are always finite."""
        FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

        class ProducerSchema(PluginSchema):
            score: int

        class ConsumerSchema(PluginSchema):
            score: FiniteFloat

        result = check_compatibility(ProducerSchema, ConsumerSchema)
        # int -> float coercion is allowed (consumer_strict=False by default),
        # and int values are always finite, so no constraint issue.
        assert result.compatible is True
