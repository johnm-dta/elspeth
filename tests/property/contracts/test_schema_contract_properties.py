# tests/property/contracts/test_schema_contract_properties.py
"""Property-based tests for SchemaContract merge, validation, and checkpoint invariants.

SchemaContract is the central type contract in the pipeline. At coalesce (join)
points, contracts from parallel paths must merge. The merge operation has:
- Mode precedence: FIXED > FLEXIBLE > OBSERVED (most restrictive wins)
- Field union: fields present in both must have matching types
- Fields from only one side become non-required
- Locked status: True if either input is locked

Properties tested:
- Merge field commutativity: A.merge(B).fields ≡ B.merge(A).fields (same set)
- Merge mode determinism: same inputs → same mode (most restrictive wins)
- Merge type conflict detection: mismatched types always raise
- Checkpoint round-trip: to_checkpoint_format → from_checkpoint preserves all state
- version_hash determinism: same contract → same hash
- version_hash sensitivity: different contracts → different hashes
- Field uniqueness invariant: duplicate normalized_name always raises
- PipelineRow immutability: __setitem__ always raises TypeError
- PipelineRow dual-name access consistency
- Validation: detects missing required, type mismatch, extra fields
"""

from __future__ import annotations

from typing import Literal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts.errors import ContractMergeError
from elspeth.contracts.schema_contract import (
    FieldContract,
    PipelineRow,
    SchemaContract,
)
from elspeth.contracts.type_normalization import ALLOWED_CONTRACT_TYPES

# Type aliases for Literal parameters
FieldSource = Literal["declared", "inferred"]
SchemaMode = Literal["FIXED", "FLEXIBLE", "OBSERVED"]

# =============================================================================
# Strategies
# =============================================================================

# Contract modes
modes: st.SearchStrategy[SchemaMode] = st.sampled_from(["FIXED", "FLEXIBLE", "OBSERVED"])

# Allowed Python types for FieldContract
allowed_types = st.sampled_from(list(ALLOWED_CONTRACT_TYPES))

# Field names (valid Python identifiers)
normalized_names = st.text(
    min_size=1,
    max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyz_",
).filter(lambda s: not s.startswith("_"))  # Avoid private-looking names

# Original names (can have spaces, special chars)
original_names = st.text(min_size=1, max_size=30)

# Source values
sources: st.SearchStrategy[FieldSource] = st.sampled_from(["declared", "inferred"])


@st.composite
def field_contracts(draw: st.DrawFn) -> FieldContract:
    """Generate a valid FieldContract."""
    norm = draw(normalized_names)
    orig = draw(original_names)
    ptype = draw(allowed_types)
    req = draw(st.booleans())
    src = draw(sources)
    return FieldContract(
        normalized_name=norm,
        original_name=orig,
        python_type=ptype,
        required=req,
        source=src,
    )


@st.composite
def schema_contracts(draw: st.DrawFn, max_fields: int = 5) -> SchemaContract:
    """Generate a valid SchemaContract with unique normalized and original names."""
    mode = draw(modes)
    locked = draw(st.booleans())
    n_fields = draw(st.integers(min_value=0, max_value=max_fields))

    # Generate unique normalized AND original names
    names = draw(
        st.lists(
            normalized_names,
            min_size=n_fields,
            max_size=n_fields,
            unique=True,
        )
    )
    origs = draw(
        st.lists(
            original_names,
            min_size=n_fields,
            max_size=n_fields,
            unique=True,
        )
    )

    fields = []
    for name, orig in zip(names, origs, strict=True):
        ptype = draw(allowed_types)
        req = draw(st.booleans())
        src = draw(sources)
        fields.append(
            FieldContract(
                normalized_name=name,
                original_name=orig,
                python_type=ptype,
                required=req,
                source=src,
            )
        )

    return SchemaContract(mode=mode, fields=tuple(fields), locked=locked)


@st.composite
def mergeable_contract_pairs(draw: st.DrawFn) -> tuple[SchemaContract, SchemaContract]:
    """Generate two SchemaContracts that can be merged (no type conflicts)."""
    # Generate shared fields (same type in both)
    n_shared = draw(st.integers(min_value=0, max_value=3))
    n_only_a = draw(st.integers(min_value=0, max_value=3))
    n_only_b = draw(st.integers(min_value=0, max_value=3))

    all_names = draw(
        st.lists(
            normalized_names,
            min_size=n_shared + n_only_a + n_only_b,
            max_size=n_shared + n_only_a + n_only_b,
            unique=True,
        )
    )

    shared_names = all_names[:n_shared]
    only_a_names = all_names[n_shared : n_shared + n_only_a]
    only_b_names = all_names[n_shared + n_only_a :]

    fields_a = []
    fields_b = []

    # Shared fields: same type in both contracts
    for name in shared_names:
        ptype = draw(allowed_types)
        fields_a.append(
            FieldContract(
                normalized_name=name,
                original_name=name.replace("_", " ").title(),
                python_type=ptype,
                required=draw(st.booleans()),
                source=draw(sources),
                nullable=draw(st.booleans()),  # D7 fix: test nullable propagation
            )
        )
        fields_b.append(
            FieldContract(
                normalized_name=name,
                original_name=name.replace("_", " ").title(),
                python_type=ptype,  # SAME type
                required=draw(st.booleans()),
                source=draw(sources),
                nullable=draw(st.booleans()),  # D7 fix: test nullable propagation
            )
        )

    # Fields only in A
    for name in only_a_names:
        fields_a.append(
            FieldContract(
                normalized_name=name,
                original_name=name.replace("_", " ").title(),
                python_type=draw(allowed_types),
                required=draw(st.booleans()),
                source=draw(sources),
                nullable=draw(st.booleans()),  # D7 fix: test nullable propagation
            )
        )

    # Fields only in B
    for name in only_b_names:
        fields_b.append(
            FieldContract(
                normalized_name=name,
                original_name=name.replace("_", " ").title(),
                python_type=draw(allowed_types),
                required=draw(st.booleans()),
                source=draw(sources),
                nullable=draw(st.booleans()),  # D7 fix: test nullable propagation
            )
        )

    mode_a = draw(modes)
    mode_b = draw(modes)

    contract_a = SchemaContract(mode=mode_a, fields=tuple(fields_a), locked=draw(st.booleans()))
    contract_b = SchemaContract(mode=mode_b, fields=tuple(fields_b), locked=draw(st.booleans()))

    return contract_a, contract_b


# =============================================================================
# Merge Commutativity Properties
# =============================================================================


class TestMergeCommutativityProperties:
    """A.merge(B) and B.merge(A) must produce equivalent field sets."""

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=200)
    def test_merge_field_set_is_commutative(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: A.merge(B) has the same normalized_names as B.merge(A)."""
        a, b = pair
        ab = a.merge(b)
        ba = b.merge(a)

        ab_names = {fc.normalized_name for fc in ab.fields}
        ba_names = {fc.normalized_name for fc in ba.fields}
        assert ab_names == ba_names

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=200)
    def test_merge_types_are_commutative(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Merged field types match regardless of merge order."""
        a, b = pair
        ab = a.merge(b)
        ba = b.merge(a)

        ab_types = {fc.normalized_name: fc.python_type for fc in ab.fields}
        ba_types = {fc.normalized_name: fc.python_type for fc in ba.fields}
        assert ab_types == ba_types

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=200)
    def test_merge_mode_is_commutative(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Merged mode is the same regardless of merge order."""
        a, b = pair
        assert a.merge(b).mode == b.merge(a).mode

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=100)
    def test_merge_locked_is_commutative(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Merged locked status is the same regardless of order."""
        a, b = pair
        assert a.merge(b).locked == b.merge(a).locked

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=200)
    def test_merge_nullable_is_commutative(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Merged field nullable is the same regardless of merge order.

        D7 fix: The nullable OR propagation must be commutative. For any shared
        field, A.merge(B)[field].nullable == B.merge(A)[field].nullable.
        """
        a, b = pair
        ab = a.merge(b)
        ba = b.merge(a)

        ab_nullable = {fc.normalized_name: fc.nullable for fc in ab.fields}
        ba_nullable = {fc.normalized_name: fc.nullable for fc in ba.fields}
        assert ab_nullable == ba_nullable


# =============================================================================
# Merge Nullable OR Semantics Properties
# =============================================================================


class TestMergeNullableOrSemantics:
    """Property tests for nullable OR propagation in shared fields (D7 fix).

    When two contracts share a field, the merged field's nullable flag uses
    OR semantics: nullable if EITHER input is nullable. This reflects the
    runtime behavior where any nullable branch can contribute None via
    collision resolution.
    """

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=200)
    def test_shared_field_nullable_uses_or_semantics(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Shared field nullable = a.nullable OR b.nullable."""
        a, b = pair
        merged = a.merge(b)

        a_by_name = {fc.normalized_name: fc for fc in a.fields}
        b_by_name = {fc.normalized_name: fc for fc in b.fields}

        for fc in merged.fields:
            in_a = fc.normalized_name in a_by_name
            in_b = fc.normalized_name in b_by_name

            if in_a and in_b:
                # Shared field: OR semantics
                expected_nullable = a_by_name[fc.normalized_name].nullable or b_by_name[fc.normalized_name].nullable
                assert fc.nullable == expected_nullable, (
                    f"Shared field '{fc.normalized_name}' nullable should be "
                    f"{a_by_name[fc.normalized_name].nullable} OR {b_by_name[fc.normalized_name].nullable} = {expected_nullable}, "
                    f"got {fc.nullable}"
                )

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=200)
    def test_branch_exclusive_field_forced_nullable(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Branch-exclusive fields are forced nullable=True.

        A field present in only one branch becomes nullable in the merged
        contract because the source branch may not arrive (e.g., timeout
        under best_effort policy). The merged row would have None for that field.
        """
        a, b = pair
        merged = a.merge(b)

        a_names = {fc.normalized_name for fc in a.fields}
        b_names = {fc.normalized_name for fc in b.fields}

        for fc in merged.fields:
            in_a = fc.normalized_name in a_names
            in_b = fc.normalized_name in b_names

            if in_a != in_b:  # Branch-exclusive (XOR)
                assert fc.nullable is True, f"Branch-exclusive field '{fc.normalized_name}' must be forced nullable=True"


# =============================================================================
# Merge Mode Precedence Properties
# =============================================================================


class TestMergeModePrecedenceProperties:
    """Mode precedence: FIXED > FLEXIBLE > OBSERVED."""

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=100)
    def test_merged_mode_is_most_restrictive(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Merged mode is the most restrictive of the two inputs."""
        a, b = pair
        mode_order = {"FIXED": 0, "FLEXIBLE": 1, "OBSERVED": 2}

        merged = a.merge(b)
        expected_rank = min(mode_order[a.mode], mode_order[b.mode])
        actual_rank = mode_order[merged.mode]
        assert actual_rank == expected_rank

    def test_fixed_always_wins(self) -> None:
        """Property: Merging with FIXED always produces FIXED."""
        fixed = SchemaContract(mode="FIXED", fields=())
        observed = SchemaContract(mode="OBSERVED", fields=())
        assert fixed.merge(observed).mode == "FIXED"
        assert observed.merge(fixed).mode == "FIXED"


# =============================================================================
# Merge Type Conflict Detection
# =============================================================================


class TestMergeTypeConflictProperties:
    """Type mismatches in shared fields must always raise ContractMergeError."""

    @given(
        name=normalized_names,
        type_a=allowed_types,
        type_b=allowed_types,
    )
    @settings(max_examples=100)
    def test_type_conflict_raises(self, name: str, type_a: type, type_b: type) -> None:
        """Property: Shared field with different types raises ContractMergeError."""
        assume(type_a != type_b)

        a = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=type_a, required=True, source="declared"),),
        )
        b = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=type_b, required=True, source="declared"),),
        )

        with pytest.raises(ContractMergeError):
            a.merge(b)

    @given(
        name=normalized_names,
        ptype=allowed_types,
    )
    @settings(max_examples=100)
    def test_same_type_merges_cleanly(self, name: str, ptype: type) -> None:
        """Property: Shared field with same type merges without error."""
        a = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=ptype, required=True, source="declared"),),
        )
        b = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=ptype, required=False, source="inferred"),),
        )

        merged = a.merge(b)
        field = merged.get_field(name)
        assert field is not None
        assert field.python_type == ptype


# =============================================================================
# Merge Field Union Properties
# =============================================================================


class TestMergeFieldUnionProperties:
    """Merged contract must contain the union of fields from both inputs."""

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=100)
    def test_merged_fields_are_union(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Merged field set is the union of both contracts' fields."""
        a, b = pair
        merged = a.merge(b)

        a_names = {fc.normalized_name for fc in a.fields}
        b_names = {fc.normalized_name for fc in b.fields}
        merged_names = {fc.normalized_name for fc in merged.fields}

        assert merged_names == a_names | b_names

    @given(pair=mergeable_contract_pairs())
    @settings(max_examples=100)
    def test_fields_only_in_one_side_are_not_required(self, pair: tuple[SchemaContract, SchemaContract]) -> None:
        """Property: Fields from only one side are marked non-required after merge."""
        a, b = pair
        merged = a.merge(b)

        a_names = {fc.normalized_name for fc in a.fields}
        b_names = {fc.normalized_name for fc in b.fields}
        shared = a_names & b_names
        only_one_side = (a_names | b_names) - shared

        for fc in merged.fields:
            if fc.normalized_name in only_one_side:
                assert not fc.required, f"Field '{fc.normalized_name}' from only one side should be non-required"


# =============================================================================
# Checkpoint Round-Trip Properties
# =============================================================================


class TestCheckpointRoundTripProperties:
    """to_checkpoint_format → from_checkpoint must preserve all state."""

    @given(contract=schema_contracts())
    @settings(max_examples=200)
    def test_checkpoint_roundtrip_preserves_mode(self, contract: SchemaContract) -> None:
        """Property: Mode survives checkpoint round-trip."""
        data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)
        assert restored.mode == contract.mode

    @given(contract=schema_contracts())
    @settings(max_examples=200)
    def test_checkpoint_roundtrip_preserves_locked(self, contract: SchemaContract) -> None:
        """Property: Locked status survives checkpoint round-trip."""
        data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)
        assert restored.locked == contract.locked

    @given(contract=schema_contracts())
    @settings(max_examples=200)
    def test_checkpoint_roundtrip_preserves_fields(self, contract: SchemaContract) -> None:
        """Property: All fields survive checkpoint round-trip with correct types."""
        data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)

        assert len(restored.fields) == len(contract.fields)
        original_map = {fc.normalized_name: fc for fc in contract.fields}
        for fc in restored.fields:
            orig = original_map[fc.normalized_name]
            assert fc.original_name == orig.original_name
            assert fc.python_type == orig.python_type
            assert fc.required == orig.required
            assert fc.source == orig.source

    @given(contract=schema_contracts())
    @settings(max_examples=100)
    def test_checkpoint_roundtrip_preserves_hash(self, contract: SchemaContract) -> None:
        """Property: version_hash is the same before and after round-trip."""
        data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(data)
        assert restored.version_hash() == contract.version_hash()


# =============================================================================
# Version Hash Properties
# =============================================================================


class TestVersionHashProperties:
    """version_hash must be deterministic and sensitive to changes."""

    @given(contract=schema_contracts())
    @settings(max_examples=200)
    def test_hash_is_deterministic(self, contract: SchemaContract) -> None:
        """Property: Same contract always produces the same hash."""
        h1 = contract.version_hash()
        h2 = contract.version_hash()
        assert h1 == h2

    @given(contract=schema_contracts())
    @settings(max_examples=100)
    def test_hash_is_32_char_hex(self, contract: SchemaContract) -> None:
        """Property: version_hash is always a 32-character hex string."""
        h = contract.version_hash()
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    @given(contract=schema_contracts(max_fields=3))
    @settings(max_examples=100)
    def test_hash_changes_with_locked(self, contract: SchemaContract) -> None:
        """Property: Changing locked status changes the hash."""
        locked = SchemaContract(mode=contract.mode, fields=contract.fields, locked=True)
        unlocked = SchemaContract(mode=contract.mode, fields=contract.fields, locked=False)
        assert locked.version_hash() != unlocked.version_hash()


# =============================================================================
# Field Uniqueness Invariant
# =============================================================================


class TestFieldUniquenessProperties:
    """Duplicate normalized_name in fields must always raise."""

    @given(name=normalized_names, mode=modes)
    @settings(max_examples=50)
    def test_duplicate_normalized_name_raises(self, name: str, mode: SchemaMode) -> None:
        """Property: Duplicate normalized_name in fields raises ValueError."""
        field_a = FieldContract(normalized_name=name, original_name="A", python_type=int, required=True, source="declared")
        field_b = FieldContract(normalized_name=name, original_name="B", python_type=str, required=False, source="inferred")

        with pytest.raises(ValueError, match="Duplicate"):
            SchemaContract(mode=mode, fields=(field_a, field_b))


# =============================================================================
# PipelineRow Immutability Properties
# =============================================================================


class TestPipelineRowImmutabilityProperties:
    """PipelineRow must be immutable for audit integrity."""

    def test_setitem_raises_typeerror(self) -> None:
        """Property: __setitem__ always raises TypeError."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name="x", original_name="x", python_type=int, required=True, source="declared"),),
        )
        row = PipelineRow({"x": 42}, contract)

        with pytest.raises(TypeError, match="immutable"):
            row["x"] = 99


# =============================================================================
# PipelineRow Access Consistency Properties
# =============================================================================


class TestPipelineRowAccessProperties:
    """__contains__ and __getitem__ must be consistent."""

    @given(
        field_name=normalized_names,
        value=st.integers(),
    )
    @settings(max_examples=100)
    def test_contains_implies_getitem_succeeds(self, field_name: str, value: int) -> None:
        """Property: If 'key in row' is True, then row[key] does not raise."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name=field_name,
                    original_name=field_name.replace("_", " "),
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
        )
        row = PipelineRow({field_name: value}, contract)

        if field_name in row:
            result = row[field_name]
            assert result == value

    def test_original_name_access(self) -> None:
        """Property: Fields are accessible by original name."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="customer_id",
                    original_name="Customer ID",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
        )
        row = PipelineRow({"customer_id": "abc123"}, contract)

        assert row["Customer ID"] == "abc123"
        assert row["customer_id"] == "abc123"
        assert "Customer ID" in row
        assert "customer_id" in row

    def test_to_dict_returns_copy(self) -> None:
        """Property: to_dict returns a mutable copy of data."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name="x", original_name="x", python_type=int, required=True, source="declared"),),
        )
        row = PipelineRow({"x": 42}, contract)
        d = row.to_dict()
        d["x"] = 999  # Mutating copy

        assert row["x"] == 42  # Original unchanged


# =============================================================================
# Validation Properties
# =============================================================================


class TestValidationProperties:
    """validate() must detect all contract violations."""

    @given(name=normalized_names)
    @settings(max_examples=50)
    def test_missing_required_field_detected(self, name: str) -> None:
        """Property: Missing required field always produces MissingFieldViolation."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=int, required=True, source="declared"),),
        )
        violations = contract.validate({})
        assert len(violations) == 1
        assert violations[0].normalized_name == name

    @given(name=normalized_names, value=st.text(min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_type_mismatch_detected(self, name: str, value: str) -> None:
        """Property: Wrong type always produces TypeMismatchViolation."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=int, required=True, source="declared"),),
        )
        violations = contract.validate({name: value})
        assert any(v.normalized_name == name for v in violations)

    @given(
        declared=normalized_names,
        extra=normalized_names,
    )
    @settings(max_examples=50)
    def test_extra_field_detected_in_fixed_mode(self, declared: str, extra: str) -> None:
        """Property: Extra fields in FIXED mode produce ExtraFieldViolation."""
        assume(declared != extra)

        contract = SchemaContract(
            mode="FIXED",
            fields=(FieldContract(normalized_name=declared, original_name=declared, python_type=int, required=False, source="declared"),),
        )
        violations = contract.validate({declared: 42, extra: "surprise"})
        extra_violations = [v for v in violations if v.normalized_name == extra]
        assert len(extra_violations) == 1

    @given(name=normalized_names, value=st.integers())
    @settings(max_examples=50)
    def test_valid_row_has_no_violations(self, name: str, value: int) -> None:
        """Property: Correct data produces zero violations."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=int, required=True, source="declared"),),
        )
        violations = contract.validate({name: value})
        assert len(violations) == 0

    @given(name=normalized_names)
    @settings(max_examples=50)
    def test_object_type_accepts_anything(self, name: str) -> None:
        """Property: Fields with python_type=object skip type validation."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract(normalized_name=name, original_name=name, python_type=object, required=True, source="declared"),),
        )
        # Any value should pass type validation
        for value in [42, "hello", 3.14, True, None, [1, 2, 3], {"nested": True}]:
            violations = contract.validate({name: value})
            type_violations = [v for v in violations if hasattr(v, "expected_type")]
            assert len(type_violations) == 0
