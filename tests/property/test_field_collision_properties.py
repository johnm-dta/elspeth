# tests/property/test_field_collision_properties.py
"""Property-based tests for field collision detection.

Tests the structural guarantees of detect_field_collisions():
- Soundness: None ↔ no intersection between existing and new fields
- Sorted output: collision lists are always alphabetically sorted
- Declared field coverage: transforms with declared_output_fields are fully checkable
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.field_collision import detect_field_collisions

# --- Field name strategy ---
# Realistic field names: lowercase alphanumeric with underscores (e.g., "llm_response_usage")
field_name_st = st.from_regex(r"[a-z][a-z0-9_]{0,30}", fullmatch=True)


class TestDetectFieldCollisionsSoundness:
    """detect_field_collisions is sound: None ↔ disjoint sets."""

    @given(
        existing=st.frozensets(field_name_st, min_size=0, max_size=20),
        new_fields=st.lists(field_name_st, min_size=0, max_size=20),
    )
    def test_none_iff_no_intersection(
        self,
        existing: frozenset[str],
        new_fields: list[str],
    ) -> None:
        """detect_field_collisions returns None if and only if fields are disjoint.

        This is the fundamental safety property: if there ARE overlapping fields,
        the function MUST detect them (no false negatives). If there are NOT
        overlapping fields, it MUST return None (no false positives).
        """
        result = detect_field_collisions(set(existing), new_fields)

        actual_intersection = existing & set(new_fields)

        if actual_intersection:
            # Overlapping fields → must detect (no false negatives)
            assert result is not None
            assert set(result) == actual_intersection
        else:
            # Disjoint → must return None (no false positives)
            assert result is None


class TestDetectFieldCollisionsSortOrder:
    """detect_field_collisions results are always sorted."""

    @given(
        existing=st.frozensets(field_name_st, min_size=1, max_size=20),
        new_fields=st.lists(field_name_st, min_size=1, max_size=20),
    )
    def test_collision_list_is_sorted(
        self,
        existing: frozenset[str],
        new_fields: list[str],
    ) -> None:
        """When collisions exist, the returned list is alphabetically sorted."""
        result = detect_field_collisions(set(existing), new_fields)

        if result is not None:
            assert result == sorted(result)


class TestDeclaredOutputFieldsCoverage:
    """For any transform with declared_output_fields, detect_field_collisions catches all overlaps."""

    @given(
        input_keys=st.frozensets(field_name_st, min_size=1, max_size=30),
        declared_fields=st.frozensets(field_name_st, min_size=1, max_size=15),
    )
    def test_declared_fields_fully_detectable(
        self,
        input_keys: frozenset[str],
        declared_fields: frozenset[str],
    ) -> None:
        """Every declared output field that overlaps with input is detected.

        This verifies the executor-level enforcement property: if a transform
        declares its output fields, and any of those fields already exist in
        the input row, detect_field_collisions will catch ALL of them (not
        just the first one found).
        """
        result = detect_field_collisions(set(input_keys), declared_fields)

        expected_collisions = input_keys & declared_fields

        if expected_collisions:
            assert result is not None
            assert set(result) == expected_collisions
        else:
            assert result is None
