# tests/property/contracts/test_enum_coercion.py
"""Property-based tests for enum coercion in audit data.

Per ELSPETH's Three-Tier Trust Model:
- Tier 1 (Audit Database) has FULL TRUST
- Bad data in audit = crash immediately
- No coercion, no defaults, no silent recovery

These tests verify that:
1. Valid enum values (both enum members and strings) work correctly
2. Invalid enum strings MUST raise ValueError (not silently coerce)
3. Enum roundtrips (enum → string → enum) are identity

This is critical for audit integrity. If an auditor asks "why did row 42
get routed here?" and we give a confident wrong answer because we coerced
garbage into a valid-looking value, we've committed fraud.
"""

from __future__ import annotations

import string
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingKind,
    RoutingMode,
    RowOutcome,
    RunStatus,
    TriggerType,
)

# =============================================================================
# Helper: Get all (str, Enum) types that are stored in the database
# =============================================================================

# These enums use (str, Enum) and are stored in the database.
# Invalid string values MUST crash, not silently coerce.
DATABASE_STORED_ENUMS: list[type] = [
    RunStatus,
    NodeStateStatus,
    ExportStatus,
    BatchStatus,
    TriggerType,
    NodeType,
    Determinism,
    RoutingKind,
    RoutingMode,
    CallType,
    CallStatus,
]

# RowOutcome is plain Enum (derived at query time, not stored)
# It has different behavior - we test it separately
DERIVED_ENUMS: list[type] = [
    RowOutcome,
]


# =============================================================================
# Strategies for generating enum values
# =============================================================================


def valid_enum_values(enum_type: type) -> st.SearchStrategy[Any]:
    """Strategy that generates valid members of an enum type."""
    return st.sampled_from(list(enum_type))


def valid_enum_strings(enum_type: type) -> st.SearchStrategy[str]:
    """Strategy that generates valid string values for an enum type."""
    return st.sampled_from([e.value for e in enum_type])


def invalid_enum_strings(enum_type: type) -> st.SearchStrategy[str]:
    """Strategy that generates INVALID string values for an enum type.

    These should all raise ValueError when passed to the enum constructor.
    """
    valid_values = {e.value for e in enum_type}

    # Generate random strings that are NOT in the valid set
    return st.text(
        alphabet=string.ascii_lowercase + string.digits + "_",
        min_size=1,
        max_size=30,
    ).filter(lambda s: s not in valid_values)


# =============================================================================
# Property Tests for Database-Stored Enums
# =============================================================================


class TestRunStatusCoercion:
    """Property tests for RunStatus enum coercion."""

    @given(status=valid_enum_values(RunStatus))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, status: RunStatus) -> None:
        """Property: RunStatus member → string → RunStatus is identity."""
        string_value = status.value
        recovered = RunStatus(string_value)
        assert recovered == status
        assert recovered is status  # Same object (enum singleton)

    @given(string_value=valid_enum_strings(RunStatus))
    @settings(max_examples=50)
    def test_valid_string_coerces_to_enum(self, string_value: str) -> None:
        """Property: Valid string values coerce to correct enum member."""
        result = RunStatus(string_value)
        assert isinstance(result, RunStatus)
        assert result.value == string_value

    @given(invalid=invalid_enum_strings(RunStatus))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError (Tier 1 integrity)."""
        with pytest.raises(ValueError):
            RunStatus(invalid)


class TestNodeStateStatusCoercion:
    """Property tests for NodeStateStatus enum coercion."""

    @given(status=valid_enum_values(NodeStateStatus))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, status: NodeStateStatus) -> None:
        """Property: NodeStateStatus member → string → NodeStateStatus is identity."""
        string_value = status.value
        recovered = NodeStateStatus(string_value)
        assert recovered == status

    @given(invalid=invalid_enum_strings(NodeStateStatus))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            NodeStateStatus(invalid)


class TestBatchStatusCoercion:
    """Property tests for BatchStatus enum coercion."""

    @given(status=valid_enum_values(BatchStatus))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, status: BatchStatus) -> None:
        """Property: BatchStatus member → string → BatchStatus is identity."""
        string_value = status.value
        recovered = BatchStatus(string_value)
        assert recovered == status

    @given(invalid=invalid_enum_strings(BatchStatus))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            BatchStatus(invalid)


class TestDeterminismCoercion:
    """Property tests for Determinism enum coercion.

    Determinism is CRITICAL - every plugin MUST declare one.
    Undeclared or invalid determinism = crash.
    """

    @given(det=valid_enum_values(Determinism))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, det: Determinism) -> None:
        """Property: Determinism member → string → Determinism is identity."""
        string_value = det.value
        recovered = Determinism(string_value)
        assert recovered == det

    @given(invalid=invalid_enum_strings(Determinism))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid Determinism strings MUST raise ValueError.

        Per CLAUDE.md: "There is no 'unknown' - undeclared determinism
        crashes at registration time."
        """
        with pytest.raises(ValueError):
            Determinism(invalid)


class TestNodeTypeCoercion:
    """Property tests for NodeType enum coercion."""

    @given(node_type=valid_enum_values(NodeType))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, node_type: NodeType) -> None:
        """Property: NodeType member → string → NodeType is identity."""
        string_value = node_type.value
        recovered = NodeType(string_value)
        assert recovered == node_type

    @given(invalid=invalid_enum_strings(NodeType))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            NodeType(invalid)


class TestRoutingKindCoercion:
    """Property tests for RoutingKind enum coercion."""

    @given(kind=valid_enum_values(RoutingKind))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, kind: RoutingKind) -> None:
        """Property: RoutingKind member → string → RoutingKind is identity."""
        string_value = kind.value
        recovered = RoutingKind(string_value)
        assert recovered == kind

    @given(invalid=invalid_enum_strings(RoutingKind))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            RoutingKind(invalid)


class TestRoutingModeCoercion:
    """Property tests for RoutingMode enum coercion."""

    @given(mode=valid_enum_values(RoutingMode))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, mode: RoutingMode) -> None:
        """Property: RoutingMode member → string → RoutingMode is identity."""
        string_value = mode.value
        recovered = RoutingMode(string_value)
        assert recovered == mode

    @given(invalid=invalid_enum_strings(RoutingMode))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            RoutingMode(invalid)


class TestTriggerTypeCoercion:
    """Property tests for TriggerType enum coercion."""

    @given(trigger=valid_enum_values(TriggerType))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, trigger: TriggerType) -> None:
        """Property: TriggerType member → string → TriggerType is identity."""
        string_value = trigger.value
        recovered = TriggerType(string_value)
        assert recovered == trigger

    @given(invalid=invalid_enum_strings(TriggerType))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            TriggerType(invalid)


class TestCallTypeCoercion:
    """Property tests for CallType enum coercion (Phase 6)."""

    @given(call_type=valid_enum_values(CallType))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, call_type: CallType) -> None:
        """Property: CallType member → string → CallType is identity."""
        string_value = call_type.value
        recovered = CallType(string_value)
        assert recovered == call_type

    @given(invalid=invalid_enum_strings(CallType))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            CallType(invalid)


class TestCallStatusCoercion:
    """Property tests for CallStatus enum coercion (Phase 6)."""

    @given(status=valid_enum_values(CallStatus))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, status: CallStatus) -> None:
        """Property: CallStatus member → string → CallStatus is identity."""
        string_value = status.value
        recovered = CallStatus(string_value)
        assert recovered == status

    @given(invalid=invalid_enum_strings(CallStatus))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            CallStatus(invalid)


class TestExportStatusCoercion:
    """Property tests for ExportStatus enum coercion."""

    @given(status=valid_enum_values(ExportStatus))
    @settings(max_examples=50)
    def test_valid_enum_member_roundtrips(self, status: ExportStatus) -> None:
        """Property: ExportStatus member → string → ExportStatus is identity."""
        string_value = status.value
        recovered = ExportStatus(string_value)
        assert recovered == status

    @given(invalid=invalid_enum_strings(ExportStatus))
    @settings(max_examples=100)
    def test_invalid_string_raises_valueerror(self, invalid: str) -> None:
        """Property: Invalid strings MUST raise ValueError."""
        with pytest.raises(ValueError):
            ExportStatus(invalid)


# =============================================================================
# Property Tests for Derived Enums (not stored in DB)
# =============================================================================


class TestRowOutcomeCoercion:
    """Property tests for RowOutcome enum.

    RowOutcome is a plain Enum (not str, Enum) because it's DERIVED
    at query time from node_states, routing_events, and batch_members,
    NOT stored directly in the database.

    Different behavior: we access .value explicitly when needed.
    """

    @given(outcome=valid_enum_values(RowOutcome))
    @settings(max_examples=50)
    def test_valid_enum_member_has_value(self, outcome: RowOutcome) -> None:
        """Property: RowOutcome members have string values."""
        assert isinstance(outcome.value, str)
        assert len(outcome.value) > 0

    @given(outcome=valid_enum_values(RowOutcome))
    @settings(max_examples=50)
    def test_is_terminal_property_consistent(self, outcome: RowOutcome) -> None:
        """Property: is_terminal property matches expected behavior."""
        # BUFFERED is the only non-terminal outcome
        if outcome == RowOutcome.BUFFERED:
            assert outcome.is_terminal is False
        else:
            assert outcome.is_terminal is True


# =============================================================================
# Parametrized Tests Across All Enums
# =============================================================================


class TestAllDatabaseEnumsConsistent:
    """Parametrized tests that verify consistent behavior across all enums."""

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_all_members_have_string_values(self, enum_type: type) -> None:
        """All enum members have non-empty string values."""
        for member in enum_type:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_all_values_are_lowercase(self, enum_type: type) -> None:
        """All enum values are lowercase (database convention)."""
        for member in enum_type:
            assert (
                member.value == member.value.lower()
            ), f"{enum_type.__name__}.{member.name} has non-lowercase value: {member.value}"

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_no_duplicate_values(self, enum_type: type) -> None:
        """All enum values are unique within the type."""
        values = [m.value for m in enum_type]
        assert len(values) == len(
            set(values)
        ), f"{enum_type.__name__} has duplicate values"

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_string_conversion_roundtrips(self, enum_type: type) -> None:
        """All members roundtrip through string conversion."""
        for member in enum_type:
            string_value = member.value
            recovered = enum_type(string_value)
            assert recovered is member

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_invalid_empty_string_rejected(self, enum_type: type) -> None:
        """Empty string is always rejected."""
        with pytest.raises(ValueError):
            enum_type("")

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_invalid_uppercase_variant_rejected(self, enum_type: type) -> None:
        """Uppercase variants of valid values are rejected (case-sensitive)."""
        # Get first member
        first_member = next(iter(enum_type))
        uppercase_value = first_member.value.upper()

        # Only test if uppercase is different
        if uppercase_value != first_member.value:
            with pytest.raises(ValueError):
                enum_type(uppercase_value)

    @pytest.mark.parametrize("enum_type", DATABASE_STORED_ENUMS)
    def test_invalid_whitespace_variant_rejected(self, enum_type: type) -> None:
        """Whitespace variants of valid values are rejected."""
        first_member = next(iter(enum_type))
        whitespace_value = f" {first_member.value} "

        with pytest.raises(ValueError):
            enum_type(whitespace_value)
