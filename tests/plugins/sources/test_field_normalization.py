"""Tests for field normalization algorithm."""

import concurrent.futures
import keyword

import pytest
from hypothesis import given
from hypothesis import strategies as st


class TestNormalizeFieldName:
    """Unit tests for normalize_field_name function."""

    def test_basic_normalization_spaces_to_underscore(self) -> None:
        """Spaces are replaced with underscores."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("User ID") == "user_id"

    def test_basic_normalization_lowercase(self) -> None:
        """Mixed case is lowercased."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("CaSE Study1 !!!! xx!") == "case_study1_xx"

    def test_special_chars_replaced(self) -> None:
        """Special characters become underscores, collapsed."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("data.field") == "data_field"
        assert normalize_field_name("amount$$$") == "amount"

    def test_leading_digit_prefixed(self) -> None:
        """Leading digits get underscore prefix."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("123_field") == "_123_field"

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace stripped."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("  Amount  ") == "amount"

    def test_empty_result_raises_error(self) -> None:
        """Headers that normalize to empty raise ValueError."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        with pytest.raises(ValueError, match="normalizes to empty"):
            normalize_field_name("!!!")

    def test_algorithm_version_available(self) -> None:
        """Algorithm version is accessible for audit trail."""
        from elspeth.plugins.sources.field_normalization import (
            NORMALIZATION_ALGORITHM_VERSION,
            get_normalization_version,
        )

        assert NORMALIZATION_ALGORITHM_VERSION == "1.0.0"
        assert get_normalization_version() == "1.0.0"

    def test_unicode_bom_stripped(self) -> None:
        """BOM character at start is stripped."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("\ufeffid") == "id"

    def test_zero_width_chars_stripped(self) -> None:
        """Zero-width characters are stripped."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("id\u200b") == "id"

    def test_emoji_stripped(self) -> None:
        """Emoji characters are stripped."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("Status 🔥") == "status"

    def test_python_keyword_gets_suffix(self) -> None:
        """Python keywords get underscore suffix."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("class") == "class_"
        assert normalize_field_name("for") == "for_"
        assert normalize_field_name("import") == "import_"

    def test_header_normalizing_to_keyword_gets_suffix(self) -> None:
        """Headers that normalize to keywords also get suffix."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("CLASS") == "class_"
        assert normalize_field_name("For ") == "for_"

    def test_accented_chars_preserved(self) -> None:
        """Accented characters are valid identifiers (PEP 3131)."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        assert normalize_field_name("café") == "café"
        assert normalize_field_name("naïve") == "naïve"

    def test_unicode_nfc_normalization_consistent(self) -> None:
        """Unicode characters in different forms normalize to same result."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        # "é" can be: precomposed U+00E9, or decomposed U+0065 U+0301
        precomposed = "café"
        decomposed = "caf\u0065\u0301"

        assert precomposed != decomposed  # Different byte representations
        assert normalize_field_name(precomposed) == normalize_field_name(decomposed)


class TestNormalizationProperties:
    """Property-based tests for normalization invariants."""

    @given(raw=st.text(min_size=1, max_size=100))
    def test_property_normalized_result_is_identifier(self, raw: str) -> None:
        """Property: All normalized results are valid Python identifiers."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        try:
            result = normalize_field_name(raw)
            # If it didn't raise, result must be valid identifier
            assert result.isidentifier(), f"'{result}' is not a valid identifier"
            # And not a keyword (keywords get suffix)
            assert not keyword.iskeyword(result), f"'{result}' is a keyword without suffix"
        except ValueError as e:
            # Accept expected error types:
            # - "normalizes to empty" for inputs that become empty after normalization
            # - "not a valid identifier" for defense-in-depth rejection (e.g., Unicode
            #   chars like '¼' that pass regex but aren't valid identifiers)
            error_msg = str(e)
            valid_errors = "normalizes to empty" in error_msg or "not a valid identifier" in error_msg
            assert valid_errors, f"Unexpected error: {e}"

    @given(raw=st.text(min_size=1, max_size=100))
    def test_property_normalization_is_idempotent(self, raw: str) -> None:
        """Property: Normalizing twice gives same result as normalizing once."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        try:
            once = normalize_field_name(raw)
            twice = normalize_field_name(once)
            assert once == twice, f"Not idempotent: '{once}' != '{twice}'"
        except ValueError:
            pass  # Empty result expected for some inputs


class TestNormalizationThreadSafety:
    """Thread safety tests - module-level regex patterns are immutable but verify."""

    def test_concurrent_normalization_no_interference(self) -> None:
        """Multiple threads normalizing fields doesn't cause interference."""
        from elspeth.plugins.sources.field_normalization import normalize_field_name

        headers = ["User ID", "Amount $", "CaSE Study1", "data.field"]
        expected = ["user_id", "amount", "case_study1", "data_field"]

        def normalize_batch(batch: list[str]) -> list[str]:
            return [normalize_field_name(h) for h in batch]

        # Run 100 iterations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(normalize_batch, headers) for _ in range(100)]
            results = [f.result() for f in futures]

        # All results should be identical
        for result in results:
            assert result == expected


class TestCollisionDetection:
    """Tests for collision detection functions."""

    def test_no_collision_passes(self) -> None:
        """No collision when all normalized names are unique."""
        from elspeth.plugins.sources.field_normalization import check_normalization_collisions

        raw = ["User ID", "Amount", "Date"]
        normalized = ["user_id", "amount", "date"]
        # Should not raise
        check_normalization_collisions(raw, normalized)

    def test_two_way_collision_raises(self) -> None:
        """Two headers normalizing to same value raises error."""
        from elspeth.plugins.sources.field_normalization import check_normalization_collisions

        raw = ["Case Study 1", "case-study-1"]
        normalized = ["case_study_1", "case_study_1"]

        with pytest.raises(ValueError, match="collision") as exc_info:
            check_normalization_collisions(raw, normalized)

        # Error should mention both original headers
        assert "Case Study 1" in str(exc_info.value)
        assert "case-study-1" in str(exc_info.value)

    def test_three_way_collision_lists_all(self) -> None:
        """Three+ headers colliding lists all of them."""
        from elspeth.plugins.sources.field_normalization import check_normalization_collisions

        raw = ["A B", "a-b", "A  B"]
        normalized = ["a_b", "a_b", "a_b"]

        with pytest.raises(ValueError, match="collision") as exc_info:
            check_normalization_collisions(raw, normalized)

        error = str(exc_info.value)
        assert "A B" in error
        assert "a-b" in error
        assert "A  B" in error


class TestMappingCollisionDetection:
    """Tests for field_mapping collision detection."""

    def test_no_collision_passes(self) -> None:
        """No collision when mapping targets are unique."""
        from elspeth.plugins.sources.field_normalization import check_mapping_collisions

        mapping = {"user_id": "uid", "amount": "amt"}
        headers = ["user_id", "amount", "date"]
        final = ["uid", "amt", "date"]
        # Should not raise
        check_mapping_collisions(headers, final, mapping)

    def test_mapping_collision_raises(self) -> None:
        """Mapping two fields to same target raises error."""
        from elspeth.plugins.sources.field_normalization import check_mapping_collisions

        mapping = {"a": "x", "b": "x"}
        headers = ["a", "b", "c"]
        final = ["x", "x", "c"]

        with pytest.raises(ValueError, match="collision") as exc_info:
            check_mapping_collisions(headers, final, mapping)

        error = str(exc_info.value)
        assert "'a'" in error
        assert "'b'" in error
        assert "'x'" in error

    def test_mapping_to_existing_field_raises(self) -> None:
        """Mapping to name that already exists as unmapped field raises error."""
        from elspeth.plugins.sources.field_normalization import check_mapping_collisions

        # "uid" exists naturally, and we try to map "user_id" to "uid"
        mapping = {"user_id": "uid"}
        headers = ["user_id", "uid", "date"]
        final = ["uid", "uid", "date"]

        with pytest.raises(ValueError, match="collision"):
            check_mapping_collisions(headers, final, mapping)


class TestResolveFieldNames:
    """Tests for the complete field resolution flow."""

    def test_normalize_only(self) -> None:
        """Resolution with normalize_fields=True, no mapping."""
        from elspeth.plugins.sources.field_normalization import resolve_field_names

        raw_headers = ["User ID", "Amount $"]
        result = resolve_field_names(
            raw_headers=raw_headers,
            normalize_fields=True,
            field_mapping=None,
            columns=None,
        )

        assert result.final_headers == ["user_id", "amount"]
        assert result.resolution_mapping == {
            "User ID": "user_id",
            "Amount $": "amount",
        }
        assert result.normalization_version == "1.0.0"

    def test_normalize_with_mapping(self) -> None:
        """Resolution with normalize + mapping override."""
        from elspeth.plugins.sources.field_normalization import resolve_field_names

        raw_headers = ["User ID", "Amount $"]
        result = resolve_field_names(
            raw_headers=raw_headers,
            normalize_fields=True,
            field_mapping={"user_id": "uid"},
            columns=None,
        )

        assert result.final_headers == ["uid", "amount"]
        assert result.resolution_mapping == {
            "User ID": "uid",
            "Amount $": "amount",
        }

    def test_columns_mode(self) -> None:
        """Resolution with explicit columns (headerless mode)."""
        from elspeth.plugins.sources.field_normalization import resolve_field_names

        result = resolve_field_names(
            raw_headers=None,
            normalize_fields=False,
            field_mapping=None,
            columns=["id", "name", "amount"],
        )

        assert result.final_headers == ["id", "name", "amount"]
        assert result.resolution_mapping == {
            "id": "id",
            "name": "name",
            "amount": "amount",
        }
        assert result.normalization_version is None  # No normalization

    def test_columns_with_mapping(self) -> None:
        """Resolution with columns + mapping override."""
        from elspeth.plugins.sources.field_normalization import resolve_field_names

        result = resolve_field_names(
            raw_headers=None,
            normalize_fields=False,
            field_mapping={"id": "customer_id"},
            columns=["id", "name"],
        )

        assert result.final_headers == ["customer_id", "name"]
        assert result.resolution_mapping == {
            "id": "customer_id",
            "name": "name",
        }

    def test_no_normalization_passthrough(self) -> None:
        """Without normalize_fields, headers pass through unchanged."""
        from elspeth.plugins.sources.field_normalization import resolve_field_names

        raw_headers = ["User ID", "Amount $"]
        result = resolve_field_names(
            raw_headers=raw_headers,
            normalize_fields=False,
            field_mapping=None,
            columns=None,
        )

        assert result.final_headers == ["User ID", "Amount $"]
        assert result.resolution_mapping == {
            "User ID": "User ID",
            "Amount $": "Amount $",
        }

    def test_mapping_key_not_found_raises(self) -> None:
        """Mapping key not in headers raises helpful error."""
        from elspeth.plugins.sources.field_normalization import resolve_field_names

        with pytest.raises(ValueError, match="not found") as exc_info:
            resolve_field_names(
                raw_headers=["user_id", "amount"],
                normalize_fields=True,
                field_mapping={"nonexistent": "x"},
                columns=None,
            )

        error = str(exc_info.value)
        assert "nonexistent" in error
        assert "user_id" in error  # Shows available headers
