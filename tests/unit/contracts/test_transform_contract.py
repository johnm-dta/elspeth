"""Tests for transform contract creation and validation."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import ConfigDict

from elspeth.contracts.data import PluginSchema
from elspeth.contracts.errors import AuditIntegrityError, TypeMismatchViolation
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.transform_contract import (
    create_output_contract_from_schema,
    validate_output_against_contract,
)
from elspeth.testing import make_field


class TestValidateOutputAgainstContract:
    """Test output validation against contracts."""

    @pytest.fixture
    def output_contract(self) -> SchemaContract:
        """Fixed contract for output validation."""
        return SchemaContract(
            mode="FIXED",
            fields=(
                make_field("id", int, original_name="id", required=True, source="declared"),
                make_field("result", str, original_name="result", required=True, source="declared"),
            ),
            locked=True,
        )

    def test_valid_output_returns_empty(self, output_contract: SchemaContract) -> None:
        """Valid output returns no violations."""
        output = {"id": 1, "result": "success"}
        violations = validate_output_against_contract(output, output_contract)

        assert violations == []

    def test_type_mismatch_returns_violation(self, output_contract: SchemaContract) -> None:
        """Wrong type returns TypeMismatchViolation."""
        output = {"id": "not_an_int", "result": "success"}
        violations = validate_output_against_contract(output, output_contract)

        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].normalized_name == "id"

    def test_missing_field_returns_violation(self, output_contract: SchemaContract) -> None:
        """Missing required field returns violation."""
        output = {"id": 1}  # Missing "result"
        violations = validate_output_against_contract(output, output_contract)

        assert len(violations) == 1
        assert violations[0].normalized_name == "result"

    def test_extra_field_in_fixed_returns_violation(self, output_contract: SchemaContract) -> None:
        """Extra field in FIXED mode returns violation."""
        output = {"id": 1, "result": "ok", "extra": "field"}
        violations = validate_output_against_contract(output, output_contract)

        assert len(violations) == 1
        assert violations[0].normalized_name == "extra"


# --- Additional Tests for Edge Cases ---


class SchemaWithOptional(PluginSchema):
    """Schema with optional field."""

    id: int
    name: str | None = None


class TestOptionalFields:
    """Test optional field handling."""

    def test_optional_field_is_not_required(self) -> None:
        """Optional field with default is marked as not required."""
        contract = create_output_contract_from_schema(SchemaWithOptional)

        field_map = {f.normalized_name: f for f in contract.fields}
        assert field_map["id"].required is True
        assert field_map["name"].required is False

    def test_optional_extracts_inner_type(self) -> None:
        """Optional[str] extracts str as the type."""
        contract = create_output_contract_from_schema(SchemaWithOptional)

        field_map = {f.normalized_name: f for f in contract.fields}
        # Optional[str] should extract to str
        assert field_map["name"].python_type is str


class TestSchemaFactoryOptionalFloat:
    """Regression tests for optional float extraction from schema_factory output."""

    def test_optional_float_preserves_float_type_and_enforces_validation(self) -> None:
        """Optional float should map to float type (None still allowed)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        cfg = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["score: float?"],
            }
        )
        schema_class = create_schema_from_config(cfg, "OptionalFloatOutputSchema")
        contract = create_output_contract_from_schema(schema_class)

        field_map = {f.normalized_name: f for f in contract.fields}
        score = field_map["score"]

        assert score.required is False
        assert score.python_type is float
        assert validate_output_against_contract({"score": None}, contract) == []

        violations = validate_output_against_contract({"score": "not-a-float"}, contract)
        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].normalized_name == "score"


# --- Bug dbk9: Unsupported concrete types silently map to object ---


class TestUnsupportedTypeAnnotations:
    """Bug dbk9: Unsupported types should raise, not silently map to object."""

    def test_list_type_raises_type_error(self) -> None:
        class BadSchema(PluginSchema):
            tags: list[str]

        with pytest.raises(TypeError, match="Unsupported type"):
            create_output_contract_from_schema(BadSchema)

    def test_dict_type_raises_type_error(self) -> None:
        class BadSchema(PluginSchema):
            metadata: dict[str, str]

        with pytest.raises(TypeError, match="Unsupported type"):
            create_output_contract_from_schema(BadSchema)

    def test_any_type_maps_to_object(self) -> None:
        class AnySchema(PluginSchema):
            data: Any

        contract = create_output_contract_from_schema(AnySchema)
        field_map = {f.normalized_name: f for f in contract.fields}
        assert field_map["data"].python_type is object
        assert field_map["data"].nullable is False


# --- Bug kr5n: T | None nullable handling ---


class TestNullableFieldHandling:
    """Bug kr5n: T | None should be nullable, not falsely reject None."""

    def test_required_nullable_field_allows_none(self) -> None:
        """score: float | None (no default) should accept None values."""

        class NullableSchema(PluginSchema):
            model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
            score: float | None

        contract = create_output_contract_from_schema(NullableSchema)
        field_map = {f.normalized_name: f for f in contract.fields}
        assert field_map["score"].python_type is float
        assert field_map["score"].required is True
        assert field_map["score"].nullable is True

        # None should be valid
        violations = validate_output_against_contract({"score": None}, contract)
        assert violations == []

        # Float should be valid
        violations = validate_output_against_contract({"score": 3.14}, contract)
        assert violations == []

        # Wrong type should fail
        violations = validate_output_against_contract({"score": "bad"}, contract)
        assert len(violations) == 1

    def test_optional_with_default_not_nullable_flag(self) -> None:
        """name: str | None = None should be required=False, nullable=True."""

        class OptSchema(PluginSchema):
            name: str | None = None

        contract = create_output_contract_from_schema(OptSchema)
        field_map = {f.normalized_name: f for f in contract.fields}
        assert field_map["name"].required is False
        assert field_map["name"].nullable is True

    def test_multi_type_union_raises(self) -> None:
        """int | float should raise TypeError (not supported)."""

        class MultiUnionSchema(PluginSchema):
            value: int | float

        with pytest.raises(TypeError, match="Multi-type union"):
            create_output_contract_from_schema(MultiUnionSchema)

    def test_missing_required_nullable_field_fails(self) -> None:
        """Required nullable field must be present (even if value can be None)."""

        class NullableRequired(PluginSchema):
            model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
            score: float | None

        contract = create_output_contract_from_schema(NullableRequired)
        # Missing field entirely - should fail
        violations = validate_output_against_contract({}, contract)
        assert len(violations) == 1
        assert violations[0].normalized_name == "score"


# --- Nullable checkpoint round-trip ---


class TestFieldContractNullableCheckpoint:
    """Test nullable field survives checkpoint round-trip."""

    def test_nullable_preserved_in_checkpoint(self) -> None:
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("score", float, required=True, source="declared", nullable=True),),
            locked=True,
        )
        data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)
        assert restored.fields[0].nullable is True

    def test_missing_nullable_crashes(self) -> None:
        """Checkpoint without 'nullable' key must crash — it's our data, not legacy compat."""
        data = {
            "mode": "FIXED",
            "locked": True,
            "version_hash": "placeholder",
            "fields": [
                {
                    "normalized_name": "x",
                    "original_name": "x",
                    "python_type": "int",
                    "required": True,
                    "source": "declared",
                }
            ],
        }
        with pytest.raises(AuditIntegrityError, match="Corrupt SchemaContract"):
            SchemaContract.from_checkpoint(data)
