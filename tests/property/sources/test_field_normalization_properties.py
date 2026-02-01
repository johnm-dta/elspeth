# tests/property/sources/test_field_normalization_properties.py
"""Property-based tests for field name normalization.

These tests verify invariants at ELSPETH's Tier 3 trust boundary:
external data normalization must be deterministic, idempotent, and
produce valid Python identifiers.

Properties tested:
1. Idempotence: normalize(normalize(x)) == normalize(x)
2. Valid output: result.isidentifier() == True
3. Collision detection symmetry: order-independent
4. Header count preservation: len(input) == len(output)
"""

from __future__ import annotations

import keyword
from typing import TYPE_CHECKING

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.plugins.sources.field_normalization import (
    check_normalization_collisions,
    normalize_field_name,
    resolve_field_names,
)
from tests.property.conftest import messy_headers, normalizable_headers

if TYPE_CHECKING:
    pass


class TestNormalizeFieldNameProperties:
    """Property tests for normalize_field_name()."""

    @given(raw=messy_headers)
    @settings(max_examples=500)
    def test_normalize_field_name_is_idempotent(self, raw: str) -> None:
        """Property: normalize(normalize(x)) == normalize(x).

        Once a field is normalized, normalizing again produces identical output.
        This ensures transforms can safely re-normalize without drift.
        """
        try:
            normalized = normalize_field_name(raw)
        except ValueError:
            # Skip inputs that normalize to empty string
            assume(False)
            return

        # Second normalization should be identical
        double_normalized = normalize_field_name(normalized)
        assert normalized == double_normalized, f"Idempotence violated: '{raw}' -> '{normalized}' -> '{double_normalized}'"

    @given(raw=messy_headers)
    @settings(max_examples=500)
    def test_normalize_produces_valid_identifier(self, raw: str) -> None:
        """Property: Output is always a valid Python identifier.

        Per ELSPETH's design, normalized field names must be usable as
        Python identifiers for downstream processing.
        """
        try:
            normalized = normalize_field_name(raw)
        except ValueError:
            # Empty normalization is explicitly handled by the function
            return

        assert normalized.isidentifier(), f"'{raw}' -> '{normalized}' is not a valid identifier"

    @given(raw=messy_headers)
    @settings(max_examples=300)
    def test_normalize_never_produces_keywords(self, raw: str) -> None:
        """Property: Output is never a bare Python keyword.

        The normalization algorithm appends underscore to keywords,
        so 'class' -> 'class_', ensuring safe attribute access.
        """
        try:
            normalized = normalize_field_name(raw)
        except ValueError:
            return

        # If it looks like a keyword, it must have trailing underscore
        if normalized.rstrip("_") in keyword.kwlist:
            assert normalized.endswith("_"), f"'{raw}' -> '{normalized}' is a Python keyword without trailing underscore"

    @given(raw=st.sampled_from(list(keyword.kwlist)))
    @settings(max_examples=len(keyword.kwlist))
    def test_normalize_handles_all_keywords(self, raw: str) -> None:
        """Property: All Python keywords are handled correctly.

        The algorithm lowercases BEFORE checking keywords, so:
        - 'False' -> 'false' (not a keyword after lowercase)
        - 'class' -> 'class' -> 'class_' (still a keyword)

        If the lowercased form is still a keyword, it gets underscore suffix.
        """
        normalized = normalize_field_name(raw)
        lowercased = raw.lower()

        if keyword.iskeyword(lowercased):
            # Lowercased form is still a keyword, should have underscore
            assert normalized == f"{lowercased}_", (
                f"Keyword '{raw}' (lowercased: '{lowercased}') should normalize to '{lowercased}_', got '{normalized}'"
            )
        else:
            # Lowercased form is NOT a keyword (e.g., False -> false)
            assert normalized == lowercased, (
                f"'{raw}' lowercases to non-keyword '{lowercased}', should normalize to '{lowercased}', got '{normalized}'"
            )


class TestCollisionDetectionProperties:
    """Property tests for check_normalization_collisions()."""

    @given(headers=st.lists(normalizable_headers, min_size=2, max_size=10, unique=True), data=st.data())
    @settings(max_examples=200)
    def test_collision_detection_is_order_independent(self, headers: list[str], data: st.DataObject) -> None:
        """Property: Collision detection doesn't depend on header order.

        Whether we check [A, B, C] or [C, A, B], if there's a collision
        it should be detected in both cases.
        """
        try:
            normalized_map = {h: normalize_field_name(h) for h in headers}
        except ValueError:
            assume(False)
            return

        # Check if original order has collision
        has_collision_original = False
        try:
            check_normalization_collisions(headers, [normalized_map[h] for h in headers])
        except ValueError:
            has_collision_original = True

        # Permute deterministically via Hypothesis data to avoid flakiness
        permuted = list(data.draw(st.permutations(headers)))
        shuffled_raw = permuted
        shuffled_norm = [normalized_map[h] for h in shuffled_raw]

        has_collision_shuffled = False
        try:
            check_normalization_collisions(list(shuffled_raw), list(shuffled_norm))
        except ValueError:
            has_collision_shuffled = True

        assert has_collision_original == has_collision_shuffled, "Collision detection gave different result after shuffle"

    @given(headers=st.lists(normalizable_headers, min_size=1, max_size=10, unique=True))
    @settings(max_examples=200)
    def test_unique_headers_never_collide(self, headers: list[str]) -> None:
        """Property: Unique headers that normalize to unique values don't raise.

        If we generate unique raw headers and they happen to normalize uniquely,
        no collision should be detected.
        """
        try:
            normalized = [normalize_field_name(h) for h in headers]
        except ValueError:
            assume(False)
            return

        # If normalized values are unique, should not raise
        if len(normalized) == len(set(normalized)):
            # This should NOT raise
            check_normalization_collisions(headers, normalized)


class TestResolveFieldNamesProperties:
    """Property tests for resolve_field_names()."""

    @given(headers=st.lists(normalizable_headers, min_size=1, max_size=10, unique=True))
    @settings(max_examples=200)
    def test_resolve_preserves_header_count(self, headers: list[str]) -> None:
        """Property: Resolution never changes number of headers.

        Input count must equal output count - no headers lost or duplicated.
        """
        try:
            result = resolve_field_names(
                raw_headers=headers,
                normalize_fields=True,
                field_mapping=None,
                columns=None,
            )
        except ValueError:
            # Collisions or empty normalizations are valid rejections
            assume(False)
            return

        assert len(result.final_headers) == len(headers), f"Header count changed: {len(headers)} -> {len(result.final_headers)}"
        assert len(result.resolution_mapping) == len(headers), "Resolution mapping has wrong size"

    @given(headers=st.lists(normalizable_headers, min_size=1, max_size=10, unique=True))
    @settings(max_examples=200)
    def test_resolve_mapping_covers_all_inputs(self, headers: list[str]) -> None:
        """Property: Resolution mapping has entry for every input header.

        The audit trail needs to trace every original header to its final name.
        """
        try:
            result = resolve_field_names(
                raw_headers=headers,
                normalize_fields=True,
                field_mapping=None,
                columns=None,
            )
        except ValueError:
            assume(False)
            return

        for original in headers:
            assert original in result.resolution_mapping, f"Original header '{original}' missing from resolution_mapping"

    @given(columns=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_resolve_columns_mode_passthrough(self, columns: list[str]) -> None:
        """Property: Explicit columns pass through unchanged.

        When columns are explicitly specified (headerless mode),
        they should appear in output without normalization.
        """
        result = resolve_field_names(
            raw_headers=None,
            normalize_fields=False,  # Doesn't apply in columns mode
            field_mapping=None,
            columns=columns,
        )

        assert result.final_headers == columns, "Columns mode should pass through unchanged"
        assert result.normalization_version is None, "Columns mode should not set normalization_version"

    @given(headers=st.lists(normalizable_headers, min_size=1, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_resolve_without_normalization_passthrough(self, headers: list[str]) -> None:
        """Property: With normalize_fields=False, headers pass through unchanged."""
        result = resolve_field_names(
            raw_headers=headers,
            normalize_fields=False,
            field_mapping=None,
            columns=None,
        )

        assert result.final_headers == headers, "Without normalization, headers should pass through"
        assert result.normalization_version is None, "Without normalization, version should be None"

    @given(headers=st.lists(normalizable_headers, min_size=2, max_size=10, unique=True), data=st.data())
    @settings(max_examples=100)
    def test_resolve_applies_field_mapping(self, headers: list[str], data: st.DataObject) -> None:
        """Property: field_mapping overrides only specified headers."""
        # Choose a subset of headers to map
        keys_to_map = data.draw(st.lists(st.sampled_from(headers), min_size=1, max_size=len(headers), unique=True))
        field_mapping = {key: f"mapped_{i}" for i, key in enumerate(keys_to_map)}

        result = resolve_field_names(
            raw_headers=headers,
            normalize_fields=False,
            field_mapping=field_mapping,
            columns=None,
        )

        for original in headers:
            expected = field_mapping.get(original, original)
            assert result.resolution_mapping[original] == expected
        assert result.final_headers == [field_mapping.get(h, h) for h in headers]

    @given(headers=st.lists(normalizable_headers, min_size=1, max_size=10, unique=True))
    @settings(max_examples=50)
    def test_resolve_rejects_missing_mapping_keys(self, headers: list[str]) -> None:
        """Property: field_mapping keys must exist in effective headers."""
        # normalizable_headers never include underscores, so this is guaranteed missing
        missing_key = "missing_key"
        field_mapping = {missing_key: "mapped_missing"}

        try:
            resolve_field_names(
                raw_headers=headers,
                normalize_fields=False,
                field_mapping=field_mapping,
                columns=None,
            )
        except ValueError:
            return
        raise AssertionError("Expected ValueError for missing field_mapping keys")

    @given(headers=st.lists(normalizable_headers, min_size=2, max_size=10, unique=True), data=st.data())
    @settings(max_examples=50)
    def test_resolve_rejects_mapping_collisions(self, headers: list[str], data: st.DataObject) -> None:
        """Property: field_mapping cannot collapse multiple headers to same name."""
        key_a = data.draw(st.sampled_from(headers))
        key_b = data.draw(st.sampled_from(headers))
        assume(key_a != key_b)
        field_mapping = {key_a: "collision", key_b: "collision"}

        try:
            resolve_field_names(
                raw_headers=headers,
                normalize_fields=False,
                field_mapping=field_mapping,
                columns=None,
            )
        except ValueError:
            return
        raise AssertionError("Expected ValueError for field_mapping collision")
