"""Tests for safety_utils.validate_fields_not_empty().

get_fields_to_scan() is covered by property tests in
tests/property/plugins/transforms/azure/test_azure_safety_properties.py.
These tests cover the Pydantic validator helper.
"""

import pytest

from elspeth.plugins.transforms.safety_utils import validate_fields_not_empty


class TestValidateFieldsNotEmpty:
    """Pydantic validator: security transforms must scan at least one field."""

    def test_accepts_non_empty_string(self) -> None:
        assert validate_fields_not_empty("content") == "content"

    def test_accepts_all_keyword(self) -> None:
        assert validate_fields_not_empty("all") == "all"

    def test_accepts_non_empty_list(self) -> None:
        result = validate_fields_not_empty(["title", "body"])
        assert result == ["title", "body"]

    def test_accepts_single_element_list(self) -> None:
        assert validate_fields_not_empty(["content"]) == ["content"]

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="fields cannot be empty"):
            validate_fields_not_empty("")

    def test_rejects_whitespace_only_string(self) -> None:
        with pytest.raises(ValueError, match="fields cannot be empty"):
            validate_fields_not_empty("   ")

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="fields list cannot be empty"):
            validate_fields_not_empty([])

    def test_rejects_list_with_empty_string_element(self) -> None:
        with pytest.raises(ValueError, match=r"fields\[0\] cannot be empty"):
            validate_fields_not_empty([""])

    def test_rejects_list_with_whitespace_element(self) -> None:
        with pytest.raises(ValueError, match=r"fields\[1\] cannot be empty"):
            validate_fields_not_empty(["valid", "  "])

    def test_rejects_list_with_empty_element_after_valid(self) -> None:
        with pytest.raises(ValueError, match=r"fields\[2\] cannot be empty"):
            validate_fields_not_empty(["a", "b", ""])
