# tests/property/core/test_identifiers_properties.py
"""Property-based tests for identifier validation.

These tests verify the field name validation at ELSPETH's trust boundary:

Acceptance Properties:
- Valid Python identifiers are accepted
- Empty lists are accepted (no duplicates possible)

Rejection Properties:
- Invalid identifiers are rejected
- Python keywords are rejected
- Duplicates are rejected

Error Message Properties:
- Error messages include context and index
- First violation is reported

These invariants enforce the source boundary - only valid field names
can enter the pipeline and appear in the audit trail.
"""

from __future__ import annotations

import keyword
import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.identifiers import validate_field_names

# =============================================================================
# Strategies for generating field names
# =============================================================================

# Valid Python identifiers (letter/underscore start, then letters/digits/underscores)
valid_identifiers = st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,20}", fullmatch=True).filter(lambda s: not keyword.iskeyword(s))

# Invalid identifiers - strings that fail str.isidentifier()
invalid_identifiers = st.one_of(
    # Starts with digit
    st.from_regex(r"[0-9][a-zA-Z0-9_]{0,10}", fullmatch=True),
    # Contains invalid characters
    st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]*[-./!@#$%^&*()+=\[\]{}|\\:;<>,?~` ][a-zA-Z0-9_]*", fullmatch=True),
    # Empty string
    st.just(""),
    # Contains spaces
    st.text(min_size=2, max_size=10, alphabet=string.ascii_letters + " ").filter(lambda s: " " in s),
)

# All Python keywords
python_keywords = st.sampled_from(keyword.kwlist)

# Contexts for error messages
contexts = st.sampled_from(["columns", "field_mapping values", "output_fields", "schema"])


# =============================================================================
# Acceptance Property Tests
# =============================================================================


class TestIdentifierAcceptanceProperties:
    """Property tests for valid identifier acceptance."""

    @given(names=st.lists(valid_identifiers, min_size=0, max_size=20, unique=True))
    @settings(max_examples=200)
    def test_valid_unique_identifiers_accepted(self, names: list[str]) -> None:
        """Property: Lists of valid unique identifiers pass validation.

        This is the core acceptance property - valid field names should
        always be accepted without raising any exception.
        """
        # Should not raise
        validate_field_names(names, context="test_fields")

    @given(context=contexts)
    @settings(max_examples=20)
    def test_empty_list_accepted(self, context: str) -> None:
        """Property: Empty list is always accepted (no duplicates possible).

        Sources may have no fields (e.g., single-column data). This should
        be valid input.
        """
        validate_field_names([], context=context)

    @given(name=valid_identifiers, context=contexts)
    @settings(max_examples=100)
    def test_single_valid_identifier_accepted(self, name: str, context: str) -> None:
        """Property: Single valid identifier is always accepted."""
        validate_field_names([name], context=context)

    @given(names=st.lists(valid_identifiers, min_size=2, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_validation_is_deterministic(self, names: list[str]) -> None:
        """Property: Same input produces same result (no exception) every time."""
        # Multiple calls should all succeed
        for _ in range(3):
            validate_field_names(names, context="test")


# =============================================================================
# Invalid Identifier Rejection Property Tests
# =============================================================================


class TestInvalidIdentifierRejectionProperties:
    """Property tests for invalid identifier rejection."""

    @given(invalid=invalid_identifiers, context=contexts)
    @settings(max_examples=100)
    def test_invalid_identifier_rejected(self, invalid: str, context: str) -> None:
        """Property: Invalid identifiers are rejected with ValueError."""
        # Skip empty string edge case for regex-generated invalids
        assume(invalid != "")

        with pytest.raises(ValueError, match="not a valid Python identifier"):
            validate_field_names([invalid], context=context)

    @given(context=contexts)
    @settings(max_examples=20)
    def test_empty_string_rejected(self, context: str) -> None:
        """Property: Empty string is not a valid identifier."""
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            validate_field_names([""], context=context)

    @given(
        valid_before=st.lists(valid_identifiers, min_size=0, max_size=5, unique=True),
        invalid=invalid_identifiers,
        valid_after=st.lists(valid_identifiers, min_size=0, max_size=5, unique=True),
        context=contexts,
    )
    @settings(max_examples=100)
    def test_invalid_identifier_in_middle_rejected(
        self,
        valid_before: list[str],
        invalid: str,
        valid_after: list[str],
        context: str,
    ) -> None:
        """Property: Invalid identifier anywhere in list causes rejection."""
        assume(invalid != "")
        # Ensure no duplicates across the combined list
        all_names = [*valid_before, invalid, *valid_after]
        assume(len(all_names) == len(set(all_names)))

        with pytest.raises(ValueError, match="not a valid Python identifier"):
            validate_field_names(all_names, context=context)


# =============================================================================
# Keyword Rejection Property Tests
# =============================================================================


class TestKeywordRejectionProperties:
    """Property tests for Python keyword rejection."""

    @given(kw=python_keywords, context=contexts)
    @settings(max_examples=50)
    def test_all_keywords_rejected(self, kw: str, context: str) -> None:
        """Property: All Python keywords are rejected.

        Keywords like 'class', 'def', 'import' would cause syntax errors
        if used as field names in certain contexts (e.g., namedtuples).
        """
        with pytest.raises(ValueError, match="is a Python keyword"):
            validate_field_names([kw], context=context)

    @given(
        valid_before=st.lists(valid_identifiers, min_size=0, max_size=3, unique=True),
        kw=python_keywords,
        valid_after=st.lists(valid_identifiers, min_size=0, max_size=3, unique=True),
        context=contexts,
    )
    @settings(max_examples=100)
    def test_keyword_in_middle_rejected(
        self,
        valid_before: list[str],
        kw: str,
        valid_after: list[str],
        context: str,
    ) -> None:
        """Property: Keyword anywhere in list causes rejection."""
        # Ensure no duplicates
        all_names = [*valid_before, kw, *valid_after]
        assume(len(all_names) == len(set(all_names)))

        with pytest.raises(ValueError, match="is a Python keyword"):
            validate_field_names(all_names, context=context)

    def test_common_keywords_rejected(self) -> None:
        """Property: Common problematic keywords are all rejected.

        Explicit test for keywords most likely to appear in data schemas.
        """
        common_keywords = [
            "class",
            "def",
            "return",
            "import",
            "from",
            "if",
            "else",
            "for",
            "in",
            "is",
            "not",
            "and",
            "or",
            "True",
            "False",
            "None",
        ]

        for kw in common_keywords:
            with pytest.raises(ValueError, match="is a Python keyword"):
                validate_field_names([kw], context="test")


# =============================================================================
# Duplicate Rejection Property Tests
# =============================================================================


class TestDuplicateRejectionProperties:
    """Property tests for duplicate name rejection."""

    @given(name=valid_identifiers, context=contexts)
    @settings(max_examples=100)
    def test_duplicate_rejected(self, name: str, context: str) -> None:
        """Property: Duplicate names are rejected."""
        with pytest.raises(ValueError, match="Duplicate field name"):
            validate_field_names([name, name], context=context)

    @given(
        name=valid_identifiers,
        count=st.integers(min_value=2, max_value=5),
        context=contexts,
    )
    @settings(max_examples=50)
    def test_multiple_duplicates_rejected(self, name: str, count: int, context: str) -> None:
        """Property: Multiple copies of same name are rejected."""
        with pytest.raises(ValueError, match="Duplicate field name"):
            validate_field_names([name] * count, context=context)

    @given(
        before=st.lists(valid_identifiers, min_size=1, max_size=5, unique=True),
        dup=valid_identifiers,
        between=st.lists(valid_identifiers, min_size=0, max_size=3, unique=True),
        context=contexts,
    )
    @settings(max_examples=100)
    def test_non_adjacent_duplicates_rejected(
        self,
        before: list[str],
        dup: str,
        between: list[str],
        context: str,
    ) -> None:
        """Property: Non-adjacent duplicates are detected and rejected."""
        # Ensure dup isn't already in other lists
        assume(dup not in before)
        assume(dup not in between)
        # Ensure between doesn't have duplicates with before
        assume(len(set(before + between)) == len(before) + len(between))

        names = [*before, dup, *between, dup]

        with pytest.raises(ValueError, match="Duplicate field name"):
            validate_field_names(names, context=context)


# =============================================================================
# Error Message Property Tests
# =============================================================================


class TestErrorMessageProperties:
    """Property tests for error message quality."""

    @given(invalid=invalid_identifiers, context=contexts)
    @settings(max_examples=50)
    def test_error_includes_context(self, invalid: str, context: str) -> None:
        """Property: Error message includes the context string."""
        assume(invalid != "")

        with pytest.raises(ValueError) as exc_info:
            validate_field_names([invalid], context=context)

        assert context in str(exc_info.value)

    @given(invalid=invalid_identifiers)
    @settings(max_examples=50)
    def test_error_includes_index(self, invalid: str) -> None:
        """Property: Error message includes the index of the invalid field."""
        assume(invalid != "")

        with pytest.raises(ValueError) as exc_info:
            validate_field_names(["valid_field", invalid], context="test")

        # Should mention index 1 (the invalid one)
        assert "[1]" in str(exc_info.value)

    @given(name=valid_identifiers, context=contexts)
    @settings(max_examples=50)
    def test_duplicate_error_includes_name(self, name: str, context: str) -> None:
        """Property: Duplicate error message includes the duplicated name."""
        with pytest.raises(ValueError) as exc_info:
            validate_field_names([name, name], context=context)

        assert name in str(exc_info.value)

    @given(kw=python_keywords, context=contexts)
    @settings(max_examples=50)
    def test_keyword_error_includes_keyword(self, kw: str, context: str) -> None:
        """Property: Keyword error message includes the keyword."""
        with pytest.raises(ValueError) as exc_info:
            validate_field_names([kw], context=context)

        assert kw in str(exc_info.value)


# =============================================================================
# Order Independence Property Tests
# =============================================================================


class TestValidationOrderProperties:
    """Property tests for validation order behavior."""

    @given(names=st.lists(valid_identifiers, min_size=2, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_order_of_valid_names_doesnt_matter(self, names: list[str]) -> None:
        """Property: Valid names pass regardless of order."""
        # Original order
        validate_field_names(names, context="test")

        # Reversed order
        validate_field_names(list(reversed(names)), context="test")

        # Rotated order
        rotated = names[1:] + names[:1] if len(names) > 1 else names
        validate_field_names(rotated, context="test")

    @given(
        valid1=valid_identifiers,
        valid2=valid_identifiers,
        invalid=invalid_identifiers,
    )
    @settings(max_examples=50)
    def test_first_invalid_is_reported(self, valid1: str, valid2: str, invalid: str) -> None:
        """Property: First invalid field's index is in error message."""
        assume(invalid != "")
        assume(len({valid1, valid2, invalid}) == 3)  # All unique

        # Invalid at index 1
        with pytest.raises(ValueError) as exc_info:
            validate_field_names([valid1, invalid, valid2], context="test")

        assert "[1]" in str(exc_info.value)
