"""Tests for ContractBuilder - handles first-row inference and locking."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from elspeth.contracts.errors import TypeMismatchViolation
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.testing import make_field


class TestContractBuilderInference:
    """Test type inference from first row."""

    def test_infer_types_from_first_row(self) -> None:
        """OBSERVED mode infers types from first row."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"id": 1, "name": "Alice", "score": 3.14}
        field_resolution = {"id": "id", "name": "name", "score": "score"}

        updated = builder.process_first_row(first_row, field_resolution)

        assert updated.locked is True
        assert len(updated.fields) == 3

        types = {f.normalized_name: f.python_type for f in updated.fields}
        assert types["id"] is int
        assert types["name"] is str
        assert types["score"] is float

    def test_infer_preserves_original_names(self) -> None:
        """Inferred fields get correct original names from resolution."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"amount_usd": 100}
        field_resolution = {"'Amount USD'": "amount_usd"}

        updated = builder.process_first_row(first_row, field_resolution)

        field = updated.fields[0]
        assert field.normalized_name == "amount_usd"
        assert field.original_name == "'Amount USD'"

    def test_flexible_adds_extra_fields(self) -> None:
        """FLEXIBLE mode adds extra fields to declared ones."""
        from elspeth.contracts.contract_builder import ContractBuilder

        declared = make_field("id", int, original_name="id", required=True, source="declared")
        contract = SchemaContract(mode="FLEXIBLE", fields=(declared,), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"id": 1, "extra": "value"}
        field_resolution = {"id": "id", "extra": "extra"}

        updated = builder.process_first_row(first_row, field_resolution)

        assert len(updated.fields) == 2
        extra_field = next(f for f in updated.fields if f.normalized_name == "extra")
        assert extra_field.python_type is str
        assert extra_field.source == "inferred"
        assert extra_field.required is False  # Inferred fields are optional

    def test_fixed_ignores_first_row(self) -> None:
        """FIXED mode doesn't infer - already locked."""
        from elspeth.contracts.contract_builder import ContractBuilder

        declared = make_field("id", int, original_name="id", required=True, source="declared")
        contract = SchemaContract(mode="FIXED", fields=(declared,), locked=True)
        builder = ContractBuilder(contract)

        first_row = {"id": 1}
        field_resolution = {"id": "id"}

        # Should return same contract (already locked)
        updated = builder.process_first_row(first_row, field_resolution)
        assert updated is contract

    def test_infer_numpy_types(self) -> None:
        """numpy types normalize to Python primitives."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {
            "np_int": np.int64(42),
            "np_float": np.float64(3.14),
            "np_bool": np.bool_(True),
        }
        field_resolution = {k: k for k in first_row}

        updated = builder.process_first_row(first_row, field_resolution)

        types = {f.normalized_name: f.python_type for f in updated.fields}
        assert types["np_int"] is int
        assert types["np_float"] is float
        assert types["np_bool"] is bool

    def test_infer_pandas_timestamp(self) -> None:
        """pandas.Timestamp normalizes to datetime."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"ts": pd.Timestamp("2024-01-01")}
        field_resolution = {"ts": "ts"}

        updated = builder.process_first_row(first_row, field_resolution)

        ts_field = updated.fields[0]
        assert ts_field.python_type is datetime

    def test_infer_none_type(self) -> None:
        """None values infer as type(None)."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"nullable": None}
        field_resolution = {"nullable": "nullable"}

        updated = builder.process_first_row(first_row, field_resolution)

        field = updated.fields[0]
        assert field.python_type is type(None)

    def test_infer_pandas_na_type(self) -> None:
        """pd.NA values infer as type(None) (missing sentinel)."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"nullable": pd.NA}
        field_resolution = {"nullable": "nullable"}

        updated = builder.process_first_row(first_row, field_resolution)

        field = updated.fields[0]
        assert field.python_type is type(None)


class TestContractBuilderValidation:
    """Test validation after locking."""

    def test_validate_subsequent_row(self) -> None:
        """Subsequent rows validate against locked contract."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"id": 1, "name": "Alice"}
        field_resolution = {"id": "id", "name": "name"}
        locked_contract = builder.process_first_row(first_row, field_resolution)

        # Valid row
        violations = locked_contract.validate({"id": 2, "name": "Bob"})
        assert violations == []

    def test_type_mismatch_after_lock(self) -> None:
        """Type mismatch on subsequent row returns violation."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"amount": 100}  # int
        field_resolution = {"amount": "amount"}
        locked_contract = builder.process_first_row(first_row, field_resolution)

        # Wrong type
        violations = locked_contract.validate({"amount": "not_an_int"})

        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].expected_type is int
        assert violations[0].actual_type is str

    def test_flexible_inferred_extra_type_mismatch_after_lock(self) -> None:
        """FLEXIBLE inferred extras are type-validated after first-row lock."""
        from elspeth.contracts.contract_builder import ContractBuilder

        declared = make_field("id", int, original_name="id", required=True, source="declared")
        contract = SchemaContract(mode="FLEXIBLE", fields=(declared,), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"id": 1, "extra": "alpha"}
        field_resolution = {"id": "id", "extra": "extra"}
        locked_contract = builder.process_first_row(first_row, field_resolution)

        violations = locked_contract.validate({"id": 2, "extra": 42})

        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].normalized_name == "extra"
        assert violations[0].expected_type is str
        assert violations[0].actual_type is int

    def test_any_type_field_accepts_different_types(self) -> None:
        """Fields declared as 'any' (python_type=object) accept any type."""
        # Pre-declare a field with 'any' type (object)
        any_field = make_field("data", object, original_name="Data", required=True, source="declared")
        contract = SchemaContract(mode="FLEXIBLE", fields=(any_field,), locked=True)

        # Should accept int
        violations = contract.validate({"data": 42})
        assert violations == []

        # Should accept str
        violations = contract.validate({"data": "hello"})
        assert violations == []

        # Should accept list
        violations = contract.validate({"data": [1, 2, 3]})
        assert violations == []

        # Should accept None
        violations = contract.validate({"data": None})
        assert violations == []


class TestContractBuilderProperty:
    """Test ContractBuilder property access."""

    def test_contract_property_returns_current_state(self) -> None:
        """contract property returns the current contract state."""
        from elspeth.contracts.contract_builder import ContractBuilder

        initial = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(initial)

        # Before processing
        assert builder.contract is initial
        assert builder.contract.locked is False

        # After processing
        first_row = {"x": 1}
        field_resolution = {"x": "x"}
        locked = builder.process_first_row(first_row, field_resolution)

        assert builder.contract is locked
        assert builder.contract.locked is True

    def test_multiple_process_first_row_calls_on_locked(self) -> None:
        """Calling process_first_row on already locked contract returns same contract."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        # First call locks
        first_row = {"x": 1}
        field_resolution = {"x": "x"}
        locked = builder.process_first_row(first_row, field_resolution)
        assert locked.locked is True

        # Second call returns same (already locked)
        second_row = {"x": 2, "y": 3}
        second_resolution = {"x": "x", "y": "y"}
        same = builder.process_first_row(second_row, second_resolution)
        assert same is locked  # Same object, no changes


class TestContractBuilderDeclaredFieldPreservation:
    """Test that declared fields are preserved during inference."""

    def test_declared_field_type_takes_precedence(self) -> None:
        """Declared field type takes precedence over inferred type."""
        from elspeth.contracts.contract_builder import ContractBuilder

        # Declare 'amount' as float
        declared = make_field("amount", float, original_name="Amount", required=True, source="declared")
        contract = SchemaContract(mode="FLEXIBLE", fields=(declared,), locked=False)
        builder = ContractBuilder(contract)

        # First row has int value (100), but declared type is float
        first_row = {"amount": 100, "extra": "text"}
        field_resolution = {"Amount": "amount", "extra": "extra"}

        updated = builder.process_first_row(first_row, field_resolution)

        # Check declared field preserved
        amount_field = next(f for f in updated.fields if f.normalized_name == "amount")
        assert amount_field.python_type is float  # Declared type preserved
        assert amount_field.source == "declared"
        assert amount_field.required is True

        # Check inferred field added
        extra_field = next(f for f in updated.fields if f.normalized_name == "extra")
        assert extra_field.python_type is str
        assert extra_field.source == "inferred"

    def test_multiple_declared_fields_preserved(self) -> None:
        """Multiple declared fields are all preserved."""
        from elspeth.contracts.contract_builder import ContractBuilder

        declared1 = make_field("id", int, original_name="ID", required=True, source="declared")
        declared2 = make_field("name", str, original_name="Name", required=True, source="declared")
        contract = SchemaContract(mode="FLEXIBLE", fields=(declared1, declared2), locked=False)
        builder = ContractBuilder(contract)

        first_row = {"id": 1, "name": "Alice", "extra": 3.14}
        field_resolution = {"ID": "id", "Name": "name", "extra": "extra"}

        updated = builder.process_first_row(first_row, field_resolution)

        assert len(updated.fields) == 3

        id_field = next(f for f in updated.fields if f.normalized_name == "id")
        assert id_field.python_type is int
        assert id_field.source == "declared"

        name_field = next(f for f in updated.fields if f.normalized_name == "name")
        assert name_field.python_type is str
        assert name_field.source == "declared"


class TestContractBuilderEdgeCases:
    """Edge case tests for ContractBuilder."""

    def test_empty_row(self) -> None:
        """Empty row still locks the contract."""
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        updated = builder.process_first_row({}, {})

        assert updated.locked is True
        assert len(updated.fields) == 0

    def test_field_in_row_not_in_resolution_crashes(self) -> None:
        """Field in row but not in resolution raises KeyError (Tier 1 integrity).

        Per CLAUDE.md: Sources are system code. If a field is in the row but
        not in field_resolution, that's a bug in the source plugin.
        Silent fallback corrupts the audit trail with wrong original_name.
        """
        from elspeth.contracts.contract_builder import ContractBuilder

        contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        builder = ContractBuilder(contract)

        # Field 'orphan' is in row but not in resolution - this is a source bug
        first_row = {"known": 1, "orphan": 2}
        field_resolution = {"Known": "known"}  # 'orphan' not in resolution

        with pytest.raises(KeyError, match="orphan"):
            builder.process_first_row(first_row, field_resolution)
