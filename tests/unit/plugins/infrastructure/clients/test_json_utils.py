"""Tests for shared strict JSON parsing utilities.

Covers all branches per the spec:
- parse_json_strict: valid JSON → parsed, malformed → error tuple,
  NaN in nested object → rejected, Infinity in array → rejected
- contains_non_finite: float NaN, float Infinity, nested dict, nested list, clean data
"""

from __future__ import annotations

from elspeth.plugins.infrastructure.clients.json_utils import (
    contains_non_finite,
    parse_json_strict,
)


class TestContainsNonFinite:
    """Branch coverage for contains_non_finite()."""

    def test_nan_float(self) -> None:
        assert contains_non_finite(float("nan")) is True

    def test_positive_infinity(self) -> None:
        assert contains_non_finite(float("inf")) is True

    def test_negative_infinity(self) -> None:
        assert contains_non_finite(float("-inf")) is True

    def test_normal_float(self) -> None:
        assert contains_non_finite(42.5) is False

    def test_zero_float(self) -> None:
        assert contains_non_finite(0.0) is False

    def test_nan_in_dict_values(self) -> None:
        assert contains_non_finite({"a": 1, "b": float("nan")}) is True

    def test_infinity_in_dict_values(self) -> None:
        assert contains_non_finite({"x": float("inf")}) is True

    def test_clean_dict(self) -> None:
        assert contains_non_finite({"a": 1, "b": "hello", "c": 3.14}) is False

    def test_nan_in_list(self) -> None:
        assert contains_non_finite([1, 2, float("nan"), 4]) is True

    def test_infinity_in_list(self) -> None:
        assert contains_non_finite([float("inf")]) is True

    def test_clean_list(self) -> None:
        assert contains_non_finite([1, 2, 3, "hello"]) is False

    def test_nan_in_nested_dict(self) -> None:
        """NaN deeply nested in dict hierarchy."""
        assert contains_non_finite({"a": {"b": {"c": float("nan")}}}) is True

    def test_infinity_in_nested_list(self) -> None:
        """Infinity deeply nested in list hierarchy."""
        assert contains_non_finite([[[float("inf")]]]) is True

    def test_nan_in_dict_inside_list(self) -> None:
        """NaN in a dict within a list."""
        assert contains_non_finite([{"key": float("nan")}]) is True

    def test_clean_nested_structure(self) -> None:
        assert contains_non_finite({"a": [1, {"b": [2, 3]}], "c": "text"}) is False

    def test_non_numeric_types(self) -> None:
        """Strings, ints, bools, None return False."""
        assert contains_non_finite("hello") is False
        assert contains_non_finite(42) is False
        assert contains_non_finite(True) is False
        assert contains_non_finite(None) is False

    def test_empty_containers(self) -> None:
        assert contains_non_finite({}) is False
        assert contains_non_finite([]) is False


class TestParseJsonStrict:
    """Branch coverage for parse_json_strict()."""

    def test_valid_json_object(self) -> None:
        """Valid JSON object parses successfully."""
        parsed, error = parse_json_strict('{"key": "value", "num": 42}')
        assert error is None
        assert parsed == {"key": "value", "num": 42}

    def test_valid_json_array(self) -> None:
        """Valid JSON array parses successfully."""
        parsed, error = parse_json_strict("[1, 2, 3]")
        assert error is None
        assert parsed == [1, 2, 3]

    def test_valid_json_with_floats(self) -> None:
        """Normal floats pass validation."""
        parsed, error = parse_json_strict('{"pi": 3.14, "e": 2.718}')
        assert error is None
        assert parsed == {"pi": 3.14, "e": 2.718}

    def test_malformed_json_returns_error(self) -> None:
        """Malformed JSON returns (None, error_message)."""
        parsed, error = parse_json_strict("{not valid json}")
        assert parsed is None
        assert error is not None
        assert len(error) > 0

    def test_empty_string_returns_error(self) -> None:
        """Empty string is malformed JSON."""
        parsed, error = parse_json_strict("")
        assert parsed is None
        assert error is not None

    def test_nan_in_nested_object_rejected(self) -> None:
        """NaN in nested object is rejected."""
        # Python json.loads accepts NaN by default
        import json

        text = json.dumps({"outer": {"inner": float("nan")}}, allow_nan=True)
        parsed, error = parse_json_strict(text)
        assert parsed is None
        assert error is not None
        assert "non-finite" in error.lower() or "NaN" in error

    def test_infinity_in_array_rejected(self) -> None:
        """Infinity in array is rejected."""
        import json

        text = json.dumps([1.0, float("inf"), 3.0], allow_nan=True)
        parsed, error = parse_json_strict(text)
        assert parsed is None
        assert error is not None
        assert "non-finite" in error.lower() or "Infinity" in error

    def test_negative_infinity_rejected(self) -> None:
        """Negative Infinity is also rejected."""
        import json

        text = json.dumps({"val": float("-inf")}, allow_nan=True)
        parsed, error = parse_json_strict(text)
        assert parsed is None
        assert error is not None

    def test_valid_json_primitive_string(self) -> None:
        """JSON primitive (string) parses successfully."""
        parsed, error = parse_json_strict('"hello"')
        assert error is None
        assert parsed == "hello"

    def test_valid_json_primitive_number(self) -> None:
        """JSON primitive (number) parses successfully."""
        parsed, error = parse_json_strict("42")
        assert error is None
        assert parsed == 42

    def test_valid_json_null(self) -> None:
        """JSON null parses successfully."""
        parsed, error = parse_json_strict("null")
        assert error is None
        assert parsed is None

    def test_duplicate_keys_rejected(self) -> None:
        """Duplicate keys in JSON object are rejected."""
        parsed, error = parse_json_strict('{"a": 1, "a": 2}')
        assert parsed is None
        assert error is not None
        assert "Duplicate" in error

    def test_nested_duplicate_keys_rejected(self) -> None:
        """Duplicate keys in nested JSON object are rejected."""
        parsed, error = parse_json_strict('{"outer": {"a": 1, "a": 2}}')
        assert parsed is None
        assert error is not None
        assert "Duplicate" in error

    def test_unique_keys_still_accepted(self) -> None:
        """Normal JSON with unique keys still parses fine (regression check)."""
        parsed, error = parse_json_strict('{"a": 1, "b": 2, "c": 3}')
        assert error is None
        assert parsed == {"a": 1, "b": 2, "c": 3}
