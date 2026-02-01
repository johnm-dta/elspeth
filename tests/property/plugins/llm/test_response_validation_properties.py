# tests/property/plugins/llm/test_response_validation_properties.py
"""Property-based tests for LLM response validation.

Per ELSPETH's Three-Tier Trust Model:
- LLM responses are Tier 3 (external data) - zero trust
- Validation must happen IMMEDIATELY at the boundary
- Invalid responses must be caught, not silently coerced

These tests exercise the PRODUCTION validation code in:
    src/elspeth/plugins/llm/validation.py

They verify that validation correctly handles:
- Non-JSON responses
- Wrong JSON types (array when object expected)
- Truncated/partial JSON
- Valid responses (positive cases)
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.plugins.llm.validation import (
    ValidationError,
    ValidationSuccess,
    validate_json_object_response,
)
from tests.property.conftest import json_primitives

# =============================================================================
# Strategies for generating LLM-like responses
# =============================================================================

# Valid JSON object responses
valid_json_objects = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
    values=json_primitives,
    min_size=0,  # Empty object {} is valid
    max_size=10,
)

# Non-JSON strings - use explicit patterns instead of filter() for efficiency
# (Reviewer fix: avoid .filter(not _is_valid_json) which rejects >99% of inputs)
# NOTE: "NaN" is valid JSON (parses to float) so it goes in wrong_type_json, not here
non_json_strings = st.one_of(
    st.just(""),
    st.just("{"),
    st.just('{"incomplete": '),
    st.just("[1, 2, 3"),
    st.just("{invalid}"),
    st.just("{{double braces}}"),
    st.sampled_from(
        [
            "This is not JSON",
            "Error: API rate limit exceeded",
            "<html>Error</html>",
            "None",
            "undefined",
            "{'single': 'quotes'}",  # Python dict, not JSON
        ]
    ),
)

# JSON that parses but is wrong type (array, not object)
wrong_type_json = st.one_of(
    st.lists(json_primitives, min_size=0, max_size=5).map(json.dumps),
    st.just("null"),
    st.just("true"),
    st.just("false"),
    st.integers().map(str),
    st.text(max_size=50).map(lambda s: json.dumps(s)),
)


# =============================================================================
# Property Tests: JSON Parse Boundary (Testing PRODUCTION code)
# =============================================================================


class TestLLMResponseParsingProperties:
    """Property tests for LLM response JSON parsing."""

    @given(response=valid_json_objects)
    @settings(max_examples=100)
    def test_valid_json_object_succeeds(self, response: dict[str, Any]) -> None:
        """Property: Valid JSON objects are accepted."""
        content = json.dumps(response)
        result = validate_json_object_response(content)

        assert isinstance(result, ValidationSuccess)
        assert result.data == response

    @given(response=non_json_strings)
    @settings(max_examples=100)
    def test_non_json_rejected(self, response: str) -> None:
        """Property: Non-JSON strings are rejected with clear error."""
        result = validate_json_object_response(response)

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json"

    @given(response=wrong_type_json)
    @settings(max_examples=100)
    def test_wrong_json_type_rejected(self, response: str) -> None:
        """Property: JSON that isn't an object is rejected.

        LLM transforms expect {"field": value} responses, not arrays,
        primitives, or null.
        """
        result = validate_json_object_response(response)

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json_type"
        assert result.expected == "object"


class TestLLMResponseEdgeCases:
    """Property tests for LLM response edge cases."""

    def test_empty_object_accepted(self) -> None:
        """Property: Empty object {} is valid (may have optional fields)."""
        result = validate_json_object_response("{}")

        assert isinstance(result, ValidationSuccess)
        assert result.data == {}

    def test_deeply_nested_accepted(self) -> None:
        """Property: Deeply nested objects are accepted."""
        deep = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        result = validate_json_object_response(json.dumps(deep))

        assert isinstance(result, ValidationSuccess)
        assert result.data == deep

    @given(whitespace=st.sampled_from([" ", "\n", "\t", "\r\n"]))
    @settings(max_examples=20)
    def test_whitespace_padded_json_accepted(self, whitespace: str) -> None:
        """Property: JSON with leading/trailing whitespace is accepted."""
        content = f'{whitespace}{{"key": "value"}}{whitespace}'
        result = validate_json_object_response(content)

        assert isinstance(result, ValidationSuccess)

    def test_null_json_rejected(self) -> None:
        """Property: JSON null is rejected (not an object)."""
        result = validate_json_object_response("null")

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json_type"

    def test_array_json_rejected(self) -> None:
        """Property: JSON array is rejected (not an object)."""
        result = validate_json_object_response("[1, 2, 3]")

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json_type"
        assert result.actual == "list"


class TestLLMResponseDeterminism:
    """Property tests for validation determinism."""

    @given(content=st.text(max_size=200))
    @settings(max_examples=100)
    def test_validation_is_deterministic(self, content: str) -> None:
        """Property: Same input always produces same validation result."""
        result1 = validate_json_object_response(content)
        result2 = validate_json_object_response(content)

        # Both should be same type
        assert type(result1) is type(result2)

        if isinstance(result1, ValidationSuccess):
            assert isinstance(result2, ValidationSuccess)
            assert result1.data == result2.data
        else:
            assert isinstance(result2, ValidationError)
            assert result1.reason == result2.reason
