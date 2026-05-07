"""Tests for FieldContract dataclass.

FieldContract represents a single field in a schema contract with:
- normalized_name: Python identifier for dict access (e.g., "amount_usd")
- original_name: Display name from source (e.g., "'Amount USD'")
- python_type: Python primitive type (int, str, float, bool, etc.)
- required: Whether field must be present
- source: "declared" (from config) or "inferred" (from first row)

The dataclass is frozen (immutable) and uses __slots__ for memory efficiency.
"""

from elspeth.testing import make_field


class TestFieldContractEquality:
    """Tests for equality and hashing."""

    def test_equality_same_values(self) -> None:
        """Two FieldContracts with same values are equal."""
        field1 = make_field("amount", float, original_name="Amount", required=True, source="declared")
        field2 = make_field("amount", float, original_name="Amount", required=True, source="declared")

        assert field1 == field2

    def test_equality_different_normalized_name(self) -> None:
        """Different normalized_name means not equal."""
        field1 = make_field("amount", float, original_name="Amount", required=True, source="declared")
        field2 = make_field("price", float, original_name="Amount", required=True, source="declared")

        assert field1 != field2

    def test_equality_different_type(self) -> None:
        """Different python_type means not equal."""
        field1 = make_field("value", int, required=True, source="declared")
        field2 = make_field("value", float, required=True, source="declared")

        assert field1 != field2

    def test_equality_different_source(self) -> None:
        """Different source means not equal."""
        field1 = make_field("field", str, required=True, source="declared")
        field2 = make_field("field", str, required=True, source="inferred")

        assert field1 != field2


class TestFieldContractHashable:
    """Tests for hashability (can use in sets/dict keys)."""

    def test_hashable_can_use_in_set(self) -> None:
        """FieldContract is hashable - can be used in sets."""
        field1 = make_field("amount", float, original_name="Amount", required=True, source="declared")
        field2 = make_field("amount", float, original_name="Amount", required=True, source="declared")
        field3 = make_field("count", int, original_name="Count", required=True, source="declared")

        # Can add to set
        field_set = {field1, field2, field3}

        # field1 and field2 are equal, so set should have 2 items
        assert len(field_set) == 2
        assert field1 in field_set
        assert field3 in field_set

    def test_hashable_can_use_as_dict_key(self) -> None:
        """FieldContract is hashable - can be used as dict key."""
        field = make_field("status", str, original_name="Status", required=True, source="declared")

        # Can use as dict key
        field_dict = {field: "some_value"}

        assert field_dict[field] == "some_value"

    def test_equal_fields_have_same_hash(self) -> None:
        """Equal FieldContracts have the same hash (hash consistency)."""
        field1 = make_field("x", int, original_name="X", required=True, source="declared")
        field2 = make_field("x", int, original_name="X", required=True, source="declared")

        assert field1 == field2
        assert hash(field1) == hash(field2)


class TestFieldContractSourceValidation:
    """Tests for source field validation.

    Source must be Literal["declared", "inferred"] - this is enforced
    by the type system, but we document the expected values here.
    """

    def test_source_declared_accepted(self) -> None:
        """source='declared' is valid."""
        field = make_field("field", str, required=True, source="declared")
        assert field.source == "declared"

    def test_source_inferred_accepted(self) -> None:
        """source='inferred' is valid."""
        field = make_field("field", str, required=False, source="inferred")
        assert field.source == "inferred"
