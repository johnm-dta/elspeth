"""Tests for SchemaContract - core structure and name resolution.

Task 4 of Phase 1: Core Contracts Implementation.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.testing import make_field

# --- Fixtures ---


@pytest.fixture
def sample_fields() -> tuple[FieldContract, ...]:
    """Sample fields for testing schema contracts."""
    return (
        make_field("amount_usd", float, original_name="'Amount USD'", required=True, source="declared"),
        make_field("customer_id", str, original_name="customer_id", required=True, source="declared"),
        make_field("is_active", bool, original_name="'Is Active?'", required=False, source="inferred"),
    )


# --- FieldContract Tests ---


class TestFieldContract:
    """Tests for FieldContract validation."""

    def test_valid_primitive_types_accepted(self) -> None:
        """FieldContract accepts all valid primitive types."""
        from datetime import datetime

        # All these should succeed
        make_field("a", int, original_name="A", required=True, source="declared")
        make_field("b", str, original_name="B", required=True, source="declared")
        make_field("c", float, original_name="C", required=True, source="declared")
        make_field("d", bool, original_name="D", required=True, source="declared")
        make_field("e", datetime, original_name="E", required=True, source="declared")
        make_field("f", type(None), original_name="F", required=True, source="declared")
        make_field("g", object, original_name="G", required=True, source="declared")  # 'any' type

    def test_invalid_type_raises_typeerror(self) -> None:
        """FieldContract rejects types not supported by checkpoint serialization."""
        with pytest.raises(TypeError, match=r"Invalid python_type.*list"):
            FieldContract("bad", "Bad", list, True, "declared")

    def test_decimal_type_raises_typeerror(self) -> None:
        """FieldContract rejects Decimal - not serializable in checkpoint."""
        from decimal import Decimal

        with pytest.raises(TypeError, match=r"Invalid python_type.*Decimal"):
            FieldContract("bad", "Bad", Decimal, True, "declared")

    def test_custom_class_raises_typeerror(self) -> None:
        """FieldContract rejects custom classes - not serializable."""

        class CustomClass:
            pass

        with pytest.raises(TypeError, match=r"Invalid python_type.*CustomClass"):
            FieldContract("bad", "Bad", CustomClass, True, "declared")


# --- Creation Tests ---


class TestSchemaContractCreation:
    """Tests for SchemaContract creation."""

    def test_create_fixed_mode_contract(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """Create FIXED mode contract with fields tuple and locked=True."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
            locked=True,
        )

        assert contract.mode == "FIXED"
        assert contract.fields == sample_fields
        assert contract.locked is True

    def test_create_flexible_mode_contract(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """Create FLEXIBLE mode contract with locked=False."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=sample_fields,
            locked=False,
        )

        assert contract.mode == "FLEXIBLE"
        assert contract.fields == sample_fields
        assert contract.locked is False

    def test_create_observed_mode_contract(self) -> None:
        """Create OBSERVED mode contract with no declared fields."""
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(),
            locked=False,
        )

        assert contract.mode == "OBSERVED"
        assert contract.fields == ()
        assert contract.locked is False

    def test_duplicate_normalized_name_raises_valueerror(self) -> None:
        """Duplicate normalized_name in fields raises ValueError.

        Defense-in-depth: even if with_field() prevents dynamic duplicates,
        direct construction must also reject duplicates.
        """
        fc1 = make_field("amount", int, original_name="'Amount'", required=True, source="declared")
        fc2 = make_field("amount", float, original_name="'AMOUNT'", required=False, source="declared")

        with pytest.raises(ValueError, match="Duplicate normalized_name"):
            SchemaContract(mode="FIXED", fields=(fc1, fc2), locked=True)

    def test_frozen_immutable_raises_attribute_error(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """Frozen dataclass - assigning raises AttributeError."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
            locked=False,
        )

        with pytest.raises(AttributeError):
            contract.mode = "FLEXIBLE"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            contract.locked = True  # type: ignore[misc]

        with pytest.raises(AttributeError):
            contract.fields = ()  # type: ignore[misc]

    def test_locked_defaults_to_false(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """locked defaults to False when not specified."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        assert contract.locked is False


# --- Name Resolution Tests (O(1) dual-name access) ---


class TestSchemaContractNameResolution:
    """Tests for SchemaContract name resolution."""

    def test_resolve_name_already_normalized(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """resolve_name("amount_usd") returns "amount_usd" (already normalized)."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        result = contract.resolve_name("amount_usd")
        assert result == "amount_usd"

    def test_find_name_already_normalized(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """find_name("amount_usd") returns normalized name."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        result = contract.find_name("amount_usd")
        assert result == "amount_usd"

    def test_resolve_name_original_to_normalized(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """resolve_name("'Amount USD'") returns "amount_usd" (original -> normalized)."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        result = contract.resolve_name("'Amount USD'")
        assert result == "amount_usd"

    def test_find_name_original_to_normalized(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """find_name("'Amount USD'") returns normalized name."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        result = contract.find_name("'Amount USD'")
        assert result == "amount_usd"

    def test_resolve_name_nonexistent_raises_keyerror(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """resolve_name("nonexistent") raises KeyError."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        with pytest.raises(KeyError, match="'nonexistent' not found in schema contract"):
            contract.resolve_name("nonexistent")

    def test_find_name_nonexistent_returns_none(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """find_name("nonexistent") returns None."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        assert contract.find_name("nonexistent") is None

    def test_indices_populated(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """_by_normalized and _by_original indices are populated."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        # _by_normalized maps normalized_name -> FieldContract
        assert "amount_usd" in contract._by_normalized
        assert "customer_id" in contract._by_normalized
        assert "is_active" in contract._by_normalized
        assert len(contract._by_normalized) == 3

        # _by_original maps original_name -> normalized_name
        assert contract._by_original["'Amount USD'"] == "amount_usd"
        assert contract._by_original["customer_id"] == "customer_id"
        assert contract._by_original["'Is Active?'"] == "is_active"
        assert len(contract._by_original) == 3

    def test_get_field_returns_field_contract(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """get_field("amount_usd") returns the FieldContract."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        field = contract.get_field("amount_usd")
        assert field is not None
        assert field.normalized_name == "amount_usd"
        assert field.original_name == "'Amount USD'"
        assert field.python_type is float

    def test_get_field_nonexistent_raises(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """get_field("nonexistent") raises KeyError."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        with pytest.raises(KeyError, match="'nonexistent' not found in schema contract"):
            contract.get_field("nonexistent")

    def test_find_field_nonexistent_returns_none(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """find_field("nonexistent") returns None."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        result = contract.find_field("nonexistent")
        assert result is None

    def test_resolve_all_fields_both_directions(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """All fields resolve correctly in both directions."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        # Test all fields can be resolved by normalized name
        for fc in sample_fields:
            assert contract.resolve_name(fc.normalized_name) == fc.normalized_name

        # Test all fields can be resolved by original name
        for fc in sample_fields:
            assert contract.resolve_name(fc.original_name) == fc.normalized_name


# --- Edge Cases ---


class TestSchemaContractEdgeCases:
    """Edge case tests for SchemaContract."""

    def test_empty_fields_tuple(self) -> None:
        """Schema with no fields works correctly."""
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(),
        )

        assert len(contract._by_normalized) == 0
        assert len(contract._by_original) == 0
        assert contract.find_field("anything") is None

        with pytest.raises(KeyError):
            contract.get_field("anything")

        with pytest.raises(KeyError):
            contract.resolve_name("anything")

    def test_field_where_original_equals_normalized(self) -> None:
        """Field where original_name == normalized_name is indexed correctly."""
        field = make_field("simple_field", str, original_name="simple_field", required=True, source="declared")
        contract = SchemaContract(
            mode="FIXED",
            fields=(field,),
        )

        # Both lookups should work
        assert contract.resolve_name("simple_field") == "simple_field"
        assert contract.get_field("simple_field") == field

        # Indices should both contain it
        assert "simple_field" in contract._by_normalized
        assert "simple_field" in contract._by_original

    def test_indices_are_read_only(self, sample_fields: tuple[FieldContract, ...]) -> None:
        """Internal indices are immutable after construction."""
        contract = SchemaContract(
            mode="FIXED",
            fields=sample_fields,
        )

        with pytest.raises(TypeError):
            contract._by_normalized["new_field"] = sample_fields[0]  # type: ignore[index]

        with pytest.raises(TypeError):
            contract._by_original["new_original"] = "new_field"  # type: ignore[index]


# --- Mutation Tests ---


class TestSchemaContractMutation:
    """Test immutable 'mutation' methods that return new instances."""

    def test_with_locked_returns_new_instance(self) -> None:
        """with_locked() returns new contract with locked=True."""
        original = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        locked = original.with_locked()

        assert locked.locked is True
        assert original.locked is False  # Original unchanged
        assert locked is not original

    def test_with_locked_preserves_fields(self) -> None:
        """with_locked() preserves all fields."""
        fc = make_field("x", int, original_name="X", required=True, source="declared")
        original = SchemaContract(mode="FLEXIBLE", fields=(fc,), locked=False)
        locked = original.with_locked()

        assert locked.fields == original.fields
        assert locked.mode == original.mode

    def test_with_field_adds_inferred_field(self) -> None:
        """with_field() adds new inferred field to contract."""
        original = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        updated = original.with_field("amount", "'Amount'", 100)

        assert len(updated.fields) == 1
        assert updated.fields[0].normalized_name == "amount"
        assert updated.fields[0].original_name == "'Amount'"
        assert updated.fields[0].python_type is int
        assert updated.fields[0].source == "inferred"
        assert updated.fields[0].required is False

    def test_with_field_normalizes_numpy_type(self) -> None:
        """with_field() normalizes numpy.int64 to int."""
        import numpy as np

        original = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        updated = original.with_field("count", "Count", np.int64(42))

        assert updated.fields[0].python_type is int  # Not numpy.int64

    def test_with_field_normalizes_pandas_timestamp(self) -> None:
        """with_field() normalizes pandas.Timestamp to datetime."""
        from datetime import datetime

        import pandas as pd

        original = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        updated = original.with_field("ts", "Timestamp", pd.Timestamp("2024-01-01"))

        assert updated.fields[0].python_type is datetime

    def test_with_field_rejects_nan(self) -> None:
        """with_field() rejects NaN values."""
        original = SchemaContract(mode="OBSERVED", fields=(), locked=False)

        with pytest.raises(ValueError, match="non-finite"):
            original.with_field("bad", "Bad", float("nan"))

    def test_with_field_locked_rejects_existing(self) -> None:
        """with_field() raises if field already exists (locked contract)."""
        fc = make_field("amount", int, original_name="'Amount'", required=True, source="declared")
        locked = SchemaContract(mode="FLEXIBLE", fields=(fc,), locked=True)

        with pytest.raises(TypeError, match="already exists"):
            locked.with_field("amount", "'Amount'", 200)

    def test_with_field_unlocked_allows_new_field(self) -> None:
        """with_field() allows adding NEW fields when not locked."""
        fc = make_field("a", int, original_name="A", required=True, source="declared")
        unlocked = SchemaContract(mode="FLEXIBLE", fields=(fc,), locked=False)

        # Can add new field
        updated = unlocked.with_field("b", "B", "hello")
        assert len(updated.fields) == 2

    def test_with_field_unlocked_rejects_duplicate(self) -> None:
        """with_field() rejects duplicate field even when unlocked.

        Per CLAUDE.md: Adding duplicate is a bug in caller code.
        Prevents broken O(1) lookup invariant from duplicate fields.
        """
        fc = make_field("amount", int, original_name="'Amount'", required=True, source="declared")
        unlocked = SchemaContract(mode="FLEXIBLE", fields=(fc,), locked=False)

        with pytest.raises(TypeError, match="already exists"):
            unlocked.with_field("amount", "'Amount'", 200)

    def test_with_field_updates_indices(self) -> None:
        """with_field() updates name resolution indices."""
        original = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        updated = original.with_field("amount", "'Amount USD'", 100)

        # Both names should resolve
        assert updated.resolve_name("amount") == "amount"
        assert updated.resolve_name("'Amount USD'") == "amount"


# --- Validation Tests ---


class TestSchemaContractValidation:
    """Test contract validation."""

    def test_validate_valid_row_returns_empty(self) -> None:
        """Valid row returns empty violations list."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="ID", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"id": 42})
        assert violations == []

    def test_validate_missing_required_field(self) -> None:
        """Missing required field returns MissingFieldViolation."""
        from elspeth.contracts.errors import MissingFieldViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="ID", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({})

        assert len(violations) == 1
        assert isinstance(violations[0], MissingFieldViolation)
        assert violations[0].normalized_name == "id"

    def test_validate_optional_field_can_be_missing(self) -> None:
        """Optional field (required=False) can be missing."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("note", str, original_name="Note", required=False, source="declared"),),
            locked=True,
        )
        violations = contract.validate({})
        assert violations == []

    def test_validate_type_mismatch(self) -> None:
        """Wrong type returns TypeMismatchViolation."""
        from elspeth.contracts.errors import TypeMismatchViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="ID", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"id": "not_an_int"})

        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].expected_type is int
        assert violations[0].actual_type is str

    def test_validate_numpy_type_matches_primitive(self) -> None:
        """numpy.int64 value matches int contract type."""
        import numpy as np

        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("count", int, original_name="Count", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"count": np.int64(42)})
        assert violations == []  # np.int64 normalizes to int

    def test_validate_none_matches_nonetype(self) -> None:
        """None value matches type(None) contract."""
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("x", type(None), original_name="X", required=False, source="inferred"),),
            locked=True,
        )
        violations = contract.validate({"x": None})
        assert violations == []

    def test_validate_optional_field_allows_none_value(self) -> None:
        """Optional field (required=False) allows None value even if typed as str.

        P2 fix: This matches Pydantic semantics where Optional[T] = T | None.
        An optional field declared as 'str?' should accept None values.
        """
        contract = SchemaContract(
            mode="FIXED",
            fields=(
                make_field("note", str, original_name="Note", required=False, source="declared"),  # optional str
            ),
            locked=True,
        )
        # None value for optional str field should be valid
        violations = contract.validate({"note": None})
        assert violations == []

    def test_validate_required_field_rejects_none_value(self) -> None:
        """Required field with typed value rejects None (type mismatch).

        A required field declared as 'str' (not 'str?') should reject None
        because None != str.
        """
        from elspeth.contracts.errors import TypeMismatchViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(
                make_field("name", str, original_name="Name", required=True, source="declared"),  # required str
            ),
            locked=True,
        )
        # None value for required str field should be type mismatch
        violations = contract.validate({"name": None})

        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].expected_type is str
        assert violations[0].actual_type is type(None)

    def test_validate_required_datetime_rejects_numpy_nat(self) -> None:
        """Required datetime field rejects np.datetime64('NaT') as missing."""
        from datetime import datetime

        import numpy as np

        from elspeth.contracts.errors import TypeMismatchViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("event_time", datetime, original_name="Event Time", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"event_time": np.datetime64("NaT")})

        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].expected_type is datetime
        assert violations[0].actual_type is type(None)

    def test_validate_fixed_rejects_extras(self) -> None:
        """FIXED mode rejects extra fields."""
        from elspeth.contracts.errors import ExtraFieldViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="ID", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"id": 1, "extra": "field"})

        assert len(violations) == 1
        assert isinstance(violations[0], ExtraFieldViolation)
        assert violations[0].normalized_name == "extra"

    def test_validate_flexible_allows_extras(self) -> None:
        """FLEXIBLE mode allows extra fields (returns no violation)."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("id", int, original_name="ID", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"id": 1, "extra": "field"})
        assert violations == []

    def test_validate_observed_allows_extras(self) -> None:
        """OBSERVED mode allows extra fields."""
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(),
            locked=True,
        )
        violations = contract.validate({"anything": "goes"})
        assert violations == []

    def test_validate_multiple_violations(self) -> None:
        """Multiple problems return multiple violations."""
        from elspeth.contracts.errors import MissingFieldViolation, TypeMismatchViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(
                make_field("a", int, original_name="A", required=True, source="declared"),
                make_field("b", str, original_name="B", required=True, source="declared"),
            ),
            locked=True,
        )
        violations = contract.validate({"a": "wrong_type"})  # Missing b, wrong type a

        assert len(violations) == 2
        types = {type(v) for v in violations}
        assert MissingFieldViolation in types
        assert TypeMismatchViolation in types


# --- Checkpoint Serialization Tests ---


class TestSchemaContractCheckpoint:
    """Test checkpoint serialization/deserialization."""

    @pytest.fixture
    def sample_contract(self) -> SchemaContract:
        """Sample contract for testing."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("amount", int, original_name="'Amount USD'", required=True, source="declared"),
                make_field("note", str, original_name="Note", required=False, source="inferred"),
            ),
            locked=True,
        )

    def test_version_hash_deterministic(self, sample_contract: SchemaContract) -> None:
        """version_hash() returns same value for same contract."""
        hash1 = sample_contract.version_hash()
        hash2 = sample_contract.version_hash()
        assert hash1 == hash2
        assert len(hash1) == 32  # Truncated SHA-256 (128 bits)

    def test_version_hash_changes_on_field_change(self) -> None:
        """Different fields produce different hashes."""
        c1 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", int, original_name="A", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FIXED",
            fields=(make_field("b", int, original_name="B", required=True, source="declared"),),
            locked=True,
        )
        assert c1.version_hash() != c2.version_hash()

    def test_version_hash_changes_on_type_change(self) -> None:
        """Different type produces different hash."""
        c1 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", int, original_name="A", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", str, original_name="A", required=True, source="declared"),),
            locked=True,
        )
        assert c1.version_hash() != c2.version_hash()

    def test_to_checkpoint_format(self, sample_contract: SchemaContract) -> None:
        """to_checkpoint_format() returns serializable dict."""
        data = sample_contract.to_checkpoint_format()

        assert data["mode"] == "FLEXIBLE"
        assert data["locked"] is True
        assert "version_hash" in data
        assert len(data["fields"]) == 2
        assert data["fields"][0]["normalized_name"] == "amount"
        assert data["fields"][0]["python_type"] == "int"

    def test_from_checkpoint_round_trip(self, sample_contract: SchemaContract) -> None:
        """Contract survives checkpoint round-trip."""
        data = sample_contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)

        assert restored.mode == sample_contract.mode
        assert restored.locked == sample_contract.locked
        assert len(restored.fields) == len(sample_contract.fields)
        assert restored.fields[0].normalized_name == "amount"
        assert restored.fields[0].python_type is int

    def test_from_checkpoint_validates_hash(self, sample_contract: SchemaContract) -> None:
        """from_checkpoint() validates hash integrity."""
        data = sample_contract.to_checkpoint_format()
        data["version_hash"] = "corrupted_hash"

        with pytest.raises(ValueError, match="integrity"):
            SchemaContract.from_checkpoint(data)

    def test_from_checkpoint_missing_hash_crashes(self, sample_contract: SchemaContract) -> None:
        """from_checkpoint() crashes on missing version_hash (Tier 1 integrity).

        Per CLAUDE.md Tier 1: "Bad data in the audit trail = crash immediately."
        to_checkpoint_format() ALWAYS writes version_hash, so if it's missing
        that's corruption - not an older format to silently accept.
        """
        data = sample_contract.to_checkpoint_format()
        del data["version_hash"]  # Simulate corruption

        with pytest.raises(KeyError):
            SchemaContract.from_checkpoint(data)

    def test_from_checkpoint_unknown_type_crashes(self) -> None:
        """from_checkpoint() crashes on unknown type (Tier 1 integrity)."""
        data = {
            "mode": "FIXED",
            "locked": True,
            "fields": [
                {
                    "normalized_name": "x",
                    "original_name": "X",
                    "python_type": "UnknownType",
                    "required": True,
                    "source": "declared",
                }
            ],
        }
        with pytest.raises(KeyError):
            SchemaContract.from_checkpoint(data)

    def test_from_checkpoint_nonetype_round_trip(self) -> None:
        """type(None) survives checkpoint round-trip."""
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("x", type(None), original_name="X", required=False, source="inferred"),),
            locked=True,
        )
        data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)

        assert restored.fields[0].python_type is type(None)

    def test_from_checkpoint_detects_locked_tampering(self) -> None:
        """from_checkpoint() detects tampering with 'locked' flag.

        Per CLAUDE.md Tier 1: integrity hash must cover ALL serialized state.
        Flipping locked=False could allow type inference on resume.
        """
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("id", int, original_name="id", required=True, source="declared"),),
            locked=True,
        )
        data = contract.to_checkpoint_format()

        # Tamper with locked flag
        data["locked"] = False

        with pytest.raises(ValueError, match="integrity"):
            SchemaContract.from_checkpoint(data)

    def test_from_checkpoint_detects_source_tampering(self) -> None:
        """from_checkpoint() detects tampering with field 'source'.

        Per CLAUDE.md Tier 1: integrity hash must cover ALL serialized state.
        Changing source could falsify audit trail (declared vs inferred).
        """
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("id", int, original_name="id", required=True, source="declared"),),
            locked=True,
        )
        data = contract.to_checkpoint_format()

        # Tamper with source field
        data["fields"][0]["source"] = "inferred"

        with pytest.raises(ValueError, match="integrity"):
            SchemaContract.from_checkpoint(data)

    def test_version_hash_changes_on_locked_change(self) -> None:
        """Different 'locked' values produce different hashes."""
        c1 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", int, original_name="A", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", int, original_name="A", required=True, source="declared"),),
            locked=False,
        )
        assert c1.version_hash() != c2.version_hash()

    def test_version_hash_changes_on_source_change(self) -> None:
        """Different 'source' values produce different hashes."""
        c1 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", int, original_name="A", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FIXED",
            fields=(make_field("a", int, original_name="A", required=True, source="inferred"),),
            locked=True,
        )
        assert c1.version_hash() != c2.version_hash()


# --- Merge Tests for Coalesce ---


class TestSchemaContractMerge:
    """Test contract merging for fork/join coalesce."""

    def test_merge_same_field_same_type(self) -> None:
        """Same field, same type merges successfully."""
        from elspeth.contracts.errors import ContractMergeError  # noqa: F401 - imported for test setup

        c1 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        merged = c1.merge(c2)

        assert len(merged.fields) == 1
        assert merged.fields[0].python_type is int

    def test_merge_different_types_raises(self) -> None:
        """Different types raise ContractMergeError."""
        from elspeth.contracts.errors import ContractMergeError

        c1 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", str, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        with pytest.raises(ContractMergeError, match="conflicting types"):
            c1.merge(c2)

    def test_merge_field_only_in_one_path_becomes_optional(self) -> None:
        """Field in only one path becomes optional (required=False)."""
        c1 = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("x", int, original_name="X", required=True, source="declared"),
                make_field("y", str, original_name="Y", required=True, source="declared"),
            ),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        merged = c1.merge(c2)

        # y is only in c1, so becomes optional in merged
        y_field = next(f for f in merged.fields if f.normalized_name == "y")
        assert y_field.required is False

    def test_merge_mode_precedence(self) -> None:
        """Most restrictive mode wins: FIXED > FLEXIBLE > OBSERVED."""
        fixed = SchemaContract(mode="FIXED", fields=(), locked=True)
        flexible = SchemaContract(mode="FLEXIBLE", fields=(), locked=True)
        observed = SchemaContract(mode="OBSERVED", fields=(), locked=True)

        assert fixed.merge(observed).mode == "FIXED"
        assert observed.merge(fixed).mode == "FIXED"
        assert flexible.merge(observed).mode == "FLEXIBLE"

    def test_merge_locked_if_either_locked(self) -> None:
        """Merged contract is locked if either input is locked."""
        locked = SchemaContract(mode="OBSERVED", fields=(), locked=True)
        unlocked = SchemaContract(mode="OBSERVED", fields=(), locked=False)

        assert locked.merge(unlocked).locked is True
        assert unlocked.merge(locked).locked is True

    def test_merge_required_if_either_required(self) -> None:
        """Field is required if required in either path."""
        c1 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=False, source="inferred"),),
            locked=True,
        )
        merged = c1.merge(c2)

        assert merged.fields[0].required is True

    def test_merge_source_declared_wins(self) -> None:
        """Field source is 'declared' if either is declared."""
        c1 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", int, original_name="X", required=True, source="inferred"),),
            locked=True,
        )
        merged = c1.merge(c2)

        assert merged.fields[0].source == "declared"

    def test_merge_empty_contracts(self) -> None:
        """Merging empty contracts works."""
        c1 = SchemaContract(mode="OBSERVED", fields=(), locked=True)
        c2 = SchemaContract(mode="OBSERVED", fields=(), locked=True)
        merged = c1.merge(c2)

        assert len(merged.fields) == 0

    def test_merge_orders_fields_by_normalized_name(self) -> None:
        """Merge produces deterministic field ordering by normalized name."""
        c1 = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("zeta", int, original_name="Zeta", required=True, source="declared"),
                make_field("bravo", str, original_name="Bravo", required=True, source="declared"),
            ),
            locked=True,
        )
        c2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("alpha", float, original_name="Alpha", required=True, source="declared"),
                make_field("yankee", bool, original_name="Yankee", required=True, source="declared"),
            ),
            locked=True,
        )

        merged = c1.merge(c2)

        assert [field.normalized_name for field in merged.fields] == [
            "alpha",
            "bravo",
            "yankee",
            "zeta",
        ]


# --- Any Type Tests ---


class TestSchemaContractAnyType:
    """Test 'any' type handling (python_type=object)."""

    def test_validate_any_type_accepts_int(self) -> None:
        """Field with python_type=object accepts int values."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"data": 42})
        assert violations == []

    def test_validate_any_type_accepts_string(self) -> None:
        """Field with python_type=object accepts str values."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"data": "hello"})
        assert violations == []

    def test_validate_any_type_accepts_list(self) -> None:
        """Field with python_type=object accepts list values."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"data": [1, 2, 3]})
        assert violations == []

    def test_validate_any_type_accepts_dict(self) -> None:
        """Field with python_type=object accepts dict values."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"data": {"nested": "value"}})
        assert violations == []

    def test_validate_any_type_accepts_none(self) -> None:
        """Field with python_type=object accepts None values."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({"data": None})
        assert violations == []

    def test_validate_any_type_still_requires_presence(self) -> None:
        """Required 'any' field still must be present."""
        from elspeth.contracts.errors import MissingFieldViolation

        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        violations = contract.validate({})  # Missing required field

        assert len(violations) == 1
        assert isinstance(violations[0], MissingFieldViolation)

    def test_validate_any_type_optional_can_be_missing(self) -> None:
        """Optional 'any' field can be missing."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=False, source="declared"),),
            locked=True,
        )
        violations = contract.validate({})
        assert violations == []

    def test_checkpoint_object_type_round_trip(self) -> None:
        """python_type=object survives checkpoint round-trip."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("data", object, original_name="Data", required=True, source="declared"),),
            locked=True,
        )
        data = contract.to_checkpoint_format()

        # Verify serialization
        assert data["fields"][0]["python_type"] == "object"

        # Verify restoration
        restored = SchemaContract.from_checkpoint(data)
        assert restored.fields[0].python_type is object

    def test_any_type_mixed_with_typed_fields(self) -> None:
        """Contract with both 'any' and typed fields validates correctly."""
        contract = SchemaContract(
            mode="FIXED",
            fields=(
                make_field("id", int, original_name="ID", required=True, source="declared"),
                make_field("data", object, original_name="Data", required=True, source="declared"),
                make_field("name", str, original_name="Name", required=True, source="declared"),
            ),
            locked=True,
        )

        # Valid: id=int, data=anything, name=str
        violations = contract.validate(
            {
                "id": 42,
                "data": {"complex": ["nested", 123]},
                "name": "test",
            }
        )
        assert violations == []

        # Invalid: id should be int
        from elspeth.contracts.errors import TypeMismatchViolation

        violations = contract.validate(
            {
                "id": "not_int",
                "data": "anything",
                "name": "test",
            }
        )
        assert len(violations) == 1
        assert isinstance(violations[0], TypeMismatchViolation)
        assert violations[0].normalized_name == "id"
