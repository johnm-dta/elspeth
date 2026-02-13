"""Tests for transform contract creation and validation."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ConfigDict

from elspeth.contracts.data import PluginSchema
from elspeth.contracts.errors import TypeMismatchViolation
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.transform_contract import (
    create_output_contract_from_schema,
    validate_output_against_contract,
)
from elspeth.testing import make_field


class OutputSchema(PluginSchema):
    """Test output schema."""

    id: int
    result: str
    score: float


class TestCreateOutputContract:
    """Test creating output contracts from PluginSchema."""

    def test_creates_fixed_contract_from_schema(self) -> None:
        """PluginSchema creates FLEXIBLE contract by default (extra='ignore')."""
        contract = create_output_contract_from_schema(OutputSchema)

        # Default behavior: extra='ignore' -> FLEXIBLE mode
        # This allows extra fields to pass through while enforcing declared field types
        assert contract.mode == "FLEXIBLE"
        assert contract.locked is True
        assert len(contract.fields) == 3

    def test_field_types_from_annotations(self) -> None:
        """Field types extracted from schema annotations."""
        contract = create_output_contract_from_schema(OutputSchema)

        type_map = {f.normalized_name: f.python_type for f in contract.fields}
        assert type_map["id"] is int
        assert type_map["result"] is str
        assert type_map["score"] is float

    def test_fields_are_declared(self) -> None:
        """All fields marked as declared (from schema)."""
        contract = create_output_contract_from_schema(OutputSchema)

        for field in contract.fields:
            assert field.source == "declared"

    def test_original_equals_normalized(self) -> None:
        """Without resolution, original_name equals normalized_name."""
        contract = create_output_contract_from_schema(OutputSchema)

        for field in contract.fields:
            assert field.original_name == field.normalized_name


class DynamicSchema(PluginSchema):
    """Dynamic schema that accepts any fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class TestDynamicOutputContract:
    """Test dynamic schemas create FLEXIBLE contracts."""

    def test_extra_allow_creates_flexible(self) -> None:
        """Schema with extra='allow' creates FLEXIBLE contract."""
        contract = create_output_contract_from_schema(DynamicSchema)

        assert contract.mode == "FLEXIBLE"


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


class SchemaWithBool(PluginSchema):
    """Schema with bool field."""

    active: bool


class TestBoolField:
    """Test bool type handling."""

    def test_bool_type_preserved(self) -> None:
        """Bool type is correctly preserved."""
        contract = create_output_contract_from_schema(SchemaWithBool)

        field_map = {f.normalized_name: f for f in contract.fields}
        assert field_map["active"].python_type is bool


class TestSchemaFactoryOptionalFloat:
    """Regression tests for optional float extraction from schema_factory output."""

    def test_optional_float_preserves_float_type_and_enforces_validation(self) -> None:
        """Optional float should map to float type (None still allowed)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.schema_factory import create_schema_from_config

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
