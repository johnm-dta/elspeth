"""Tests for contract propagation through transforms."""

from __future__ import annotations

import pytest

from elspeth.contracts.contract_propagation import (
    merge_contract_with_output,
    propagate_contract,
)
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.testing import make_field


class TestPropagateContract:
    """Test contract propagation through transform output."""

    @pytest.fixture
    def input_contract(self) -> SchemaContract:
        """Input contract with source fields."""
        field_id = make_field("id", int, original_name="ID", required=True, source="declared")
        field_name = make_field("name", str, original_name="Name", required=True, source="declared")
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id, field_name),
            locked=True,
        )

    def test_passthrough_preserves_contract(self, input_contract: SchemaContract) -> None:
        """Passthrough transform preserves input contract."""
        output_row = {"id": 1, "name": "Alice"}  # Same data

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=False,
        )

        assert output_contract.mode == input_contract.mode
        assert len(output_contract.fields) == len(input_contract.fields)

    def test_transform_adds_field(self, input_contract: SchemaContract) -> None:
        """Transform adding field creates new contract with added field."""
        output_row = {"id": 1, "name": "Alice", "score": 95.5}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        assert len(output_contract.fields) == 3
        score_field = next(f for f in output_contract.fields if f.normalized_name == "score")
        assert score_field.python_type is float
        assert score_field.source == "inferred"

    def test_preserves_original_names(self, input_contract: SchemaContract) -> None:
        """Original names preserved through propagation."""
        output_row = {"id": 1, "name": "Alice"}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=False,
        )

        id_field = next(f for f in output_contract.fields if f.normalized_name == "id")
        assert id_field.original_name == "ID"

    def test_transform_adds_multiple_fields(self, input_contract: SchemaContract) -> None:
        """Transform adding multiple fields infers all types."""
        output_row = {"id": 1, "name": "Alice", "score": 95.5, "active": True}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        assert len(output_contract.fields) == 4

        score_field = next(f for f in output_contract.fields if f.normalized_name == "score")
        assert score_field.python_type is float

        active_field = next(f for f in output_contract.fields if f.normalized_name == "active")
        assert active_field.python_type is bool

    def test_inferred_fields_not_required(self, input_contract: SchemaContract) -> None:
        """Inferred fields are marked as not required."""
        output_row = {"id": 1, "name": "Alice", "score": 95.5}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        score_field = next(f for f in output_contract.fields if f.normalized_name == "score")
        assert score_field.required is False

    def test_new_fields_have_normalized_original_name(self, input_contract: SchemaContract) -> None:
        """New fields created by transforms have original_name = normalized_name."""
        output_row = {"id": 1, "name": "Alice", "new_field": "value"}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        new_field = next(f for f in output_contract.fields if f.normalized_name == "new_field")
        assert new_field.original_name == "new_field"

    def test_contract_locked_preserved(self, input_contract: SchemaContract) -> None:
        """Output contract preserves locked state."""
        output_row = {"id": 1, "name": "Alice", "score": 95.5}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        assert output_contract.locked is True

    def test_no_new_fields_returns_same_contract(self, input_contract: SchemaContract) -> None:
        """When no new fields added with transform_adds_fields=True, returns input contract."""
        output_row = {"id": 1, "name": "Alice"}  # Same fields as input

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        # Should return input contract unchanged when no new fields
        assert output_contract is input_contract


class TestMergeContractWithOutput:
    """Test merging output schema with propagated contract."""

    def test_output_schema_adds_guaranteed_fields(self) -> None:
        """Output schema fields become guaranteed in merged contract."""
        field_id = make_field("id", int, original_name="id", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

        field_id_out = make_field("id", int, original_name="id", required=True, source="declared")
        field_result = make_field("result", str, original_name="result", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FIXED",
            fields=(field_id_out, field_result),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        # Output schema guarantees 'result' field
        result_field = next(f for f in merged.fields if f.normalized_name == "result")
        assert result_field is not None
        assert result_field.required is True

    def test_preserves_original_names_from_input(self) -> None:
        """Original names from input contract preserved in merge."""
        field_amount = make_field("amount_usd", int, original_name="'Amount USD'", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_amount,),
            locked=True,
        )

        field_amount_out = make_field("amount_usd", int, original_name="amount_usd", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FIXED",
            fields=(field_amount_out,),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        # Original name from input should be preserved
        amount_field = next(f for f in merged.fields if f.normalized_name == "amount_usd")
        assert amount_field.original_name == "'Amount USD'"

    def test_mode_most_restrictive_wins(self) -> None:
        """Merged contract uses most restrictive mode."""
        field_id = make_field("id", int, original_name="id", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

        field_id_out = make_field("id", int, original_name="id", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FIXED",
            fields=(field_id_out,),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        # FIXED is more restrictive than FLEXIBLE
        assert merged.mode == "FIXED"

    def test_merged_contract_locked(self) -> None:
        """Merged contract is always locked."""
        field_id = make_field("id", int, original_name="id", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

        field_id_out = make_field("id", int, original_name="id", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FIXED",
            fields=(field_id_out,),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        assert merged.locked is True

    def test_new_fields_use_output_original_name(self) -> None:
        """Fields only in output schema use their own original_name."""
        field_id = make_field("id", int, original_name="ID", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

        field_id_out = make_field("id", int, original_name="id", required=True, source="declared")
        field_result = make_field("result", str, original_name="Result", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FIXED",
            fields=(field_id_out, field_result),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        # New field 'result' not in input uses output's original_name
        result_field = next(f for f in merged.fields if f.normalized_name == "result")
        assert result_field.original_name == "Result"

    def test_observed_mode_is_least_restrictive(self) -> None:
        """OBSERVED is least restrictive, FIXED wins over it."""
        field_id = make_field("id", int, original_name="id", required=False, source="inferred")
        input_contract = SchemaContract(
            mode="OBSERVED",
            fields=(field_id,),
            locked=True,
        )

        field_id_out = make_field("id", int, original_name="id", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id_out,),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        # FLEXIBLE is more restrictive than OBSERVED
        assert merged.mode == "FLEXIBLE"

    def test_preserves_source_from_output(self) -> None:
        """Source information comes from output schema."""
        field_id = make_field("id", int, original_name="ID", required=True, source="inferred")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

        field_id_out = make_field("id", int, original_name="id", required=True, source="declared")
        output_schema_contract = SchemaContract(
            mode="FIXED",
            fields=(field_id_out,),
            locked=True,
        )

        merged = merge_contract_with_output(
            input_contract=input_contract,
            output_schema_contract=output_schema_contract,
        )

        id_field = next(f for f in merged.fields if f.normalized_name == "id")
        assert id_field.source == "declared"


class TestPropagateContractEdgeCases:
    """Edge case tests for contract propagation."""

    def test_field_rename_loses_original_name_metadata(self) -> None:
        """When a field is renamed, original_name metadata is lost.

        This documents the current behavior: narrow_contract_to_output()
        treats renamed fields as NEW fields, so they get original_name = normalized_name.

        Example: FieldMapper renames customer_id -> customer
        - Input contract has: normalized="customer_id", original="Customer ID"
        - Output row has: {"customer": "Alice", "id": 1}
        - Output contract has: normalized="customer", original="customer" (metadata lost)
        """
        from elspeth.contracts.contract_propagation import narrow_contract_to_output

        field_customer = make_field("customer_id", str, original_name="Customer ID", required=True, source="declared")
        field_id = make_field("id", int, original_name="id", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_customer, field_id),
            locked=True,
        )

        # Transform renamed customer_id -> customer
        output_row = {"customer": "Alice", "id": 1}

        output_contract = narrow_contract_to_output(
            input_contract=input_contract,
            output_row=output_row,
        )

        # customer_id field is gone, customer field is new
        field_names = {f.normalized_name for f in output_contract.fields}
        assert "customer_id" not in field_names
        assert "customer" in field_names

        # New field loses original name metadata (becomes normalized name)
        customer_field = next(f for f in output_contract.fields if f.normalized_name == "customer")
        assert customer_field.original_name == "customer"  # Lost "Customer ID" metadata

    def test_field_rename_preserves_metadata_when_mapping_uses_original_name(self) -> None:
        """Rename metadata is preserved when renamed_fields uses original source names."""
        from elspeth.contracts.contract_propagation import narrow_contract_to_output

        field_amount = make_field("amount_usd", float, original_name="Amount USD", required=True, source="declared")
        field_id = make_field("id", int, original_name="ID", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_amount, field_id),
            locked=True,
        )

        output_row = {"price": 12.5, "id": 1}
        output_contract = narrow_contract_to_output(
            input_contract=input_contract,
            output_row=output_row,
            renamed_fields={"Amount USD": "price"},
        )

        price_field = output_contract.get_field("price")
        assert price_field is not None
        assert price_field.original_name == "Amount USD"
        assert price_field.python_type is float
        assert price_field.required is True
        assert price_field.source == "declared"

    def test_type_conflict_between_contract_and_actual_data(self) -> None:
        """Input contract declares int, but output has string - documents mismatch behavior.

        propagate_contract() doesn't validate that existing fields match their
        declared types. It only infers types for NEW fields.

        This test documents that type mismatches are NOT caught at propagation time.
        """
        field_amount = make_field("amount", int, original_name="amount", required=True, source="declared")
        field_id = make_field("id", int, original_name="id", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_amount, field_id),
            locked=True,
        )

        # Transform outputs amount as string, violating contract
        output_row = {"amount": "99.5", "id": 1}  # amount is string, not int!

        # propagate_contract with transform_adds_fields=False just returns input contract
        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=False,
        )

        # Contract still says int, but data is string - mismatch not detected
        amount_field = next(f for f in output_contract.fields if f.normalized_name == "amount")
        assert amount_field.python_type is int  # Contract unchanged
        assert isinstance(output_row["amount"], str)  # Actual data is string

        # NOTE: This mismatch would be caught later during validation or sink processing

    def test_none_value_included_in_contract(self) -> None:
        """None values in output rows are included in contract with type(None)."""
        field_id = make_field("id", int, original_name="id", required=True, source="declared")
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

        # Transform adds field with None value
        output_row = {"id": 1, "optional_field": None}

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        # None field should be in contract with type(None)
        field_names = {f.normalized_name for f in output_contract.fields}
        assert "optional_field" in field_names

        optional_field = next(f for f in output_contract.fields if f.normalized_name == "optional_field")
        assert optional_field.python_type is type(None)
        assert optional_field.required is False  # Inferred fields are never required


class TestPropagateContractNonPrimitiveTypes:
    """Test contract propagation with non-primitive types (dict, list)."""

    @pytest.fixture
    def input_contract(self) -> SchemaContract:
        """Input contract with source fields."""
        field_id = make_field("id", int, original_name="ID", required=True, source="declared")
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(field_id,),
            locked=True,
        )

    def test_dict_field_does_not_crash_propagation(self, input_contract: SchemaContract) -> None:
        """Dict fields (like LLM _usage) should not crash propagation.

        P1 bug: LLM transforms add _usage dict fields which caused TypeError
        when propagate_contract tried to normalize the type.
        """
        output_row = {
            "id": 1,
            "response_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        # Should not raise TypeError
        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        # Dict field should be preserved as object in the inferred contract
        field_names = {f.normalized_name for f in output_contract.fields}
        assert "id" in field_names
        assert "response_usage" in field_names
        usage_field = output_contract.get_field("response_usage")
        assert usage_field.python_type is object
        assert usage_field.source == "inferred"
        assert usage_field.required is False

    def test_list_field_does_not_crash_propagation(self, input_contract: SchemaContract) -> None:
        """List fields should not crash propagation."""
        output_row = {
            "id": 1,
            "tags": ["red", "green", "blue"],
        }

        # Should not raise TypeError
        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        field_names = {f.normalized_name for f in output_contract.fields}
        assert "id" in field_names
        assert "tags" in field_names
        tags_field = output_contract.get_field("tags")
        assert tags_field.python_type is object
        assert tags_field.source == "inferred"
        assert tags_field.required is False

    def test_mixed_primitive_and_nonprimitive_fields(self, input_contract: SchemaContract) -> None:
        """Primitive and non-primitive fields are all represented in contract."""
        output_row = {
            "id": 1,
            "score": 95.5,
            "metadata": {"key": "value"},
            "active": True,
        }

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        field_names = {f.normalized_name for f in output_contract.fields}
        assert "id" in field_names
        assert "score" in field_names
        assert "active" in field_names
        assert "metadata" in field_names
        metadata_field = output_contract.get_field("metadata")
        assert metadata_field.python_type is object

    def test_unsupported_non_dict_list_type_is_still_skipped(self, input_contract: SchemaContract) -> None:
        """Unsupported non-dict/list values preserve existing skip behavior."""

        class _CustomUnsupported:
            pass

        output_row = {
            "id": 1,
            "custom": _CustomUnsupported(),
        }

        output_contract = propagate_contract(
            input_contract=input_contract,
            output_row=output_row,
            transform_adds_fields=True,
        )

        field_names = {f.normalized_name for f in output_contract.fields}
        assert "id" in field_names
        assert "custom" not in field_names

    def test_non_finite_float_still_raises_value_error(self, input_contract: SchemaContract) -> None:
        """Non-finite floats remain invalid for contract inference."""
        output_row = {"id": 1, "bad": float("nan")}

        with pytest.raises(ValueError, match="non-finite float"):
            propagate_contract(
                input_contract=input_contract,
                output_row=output_row,
                transform_adds_fields=True,
            )
