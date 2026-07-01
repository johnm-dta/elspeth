"""Tests for placeholder sentinel detection helpers."""

import pytest

from elspeth.contracts.wire_visible_identity import (
    is_operator_required_placeholder_value,
    is_placeholder_value,
    is_wire_visible_placeholder,
    reject_operator_required_placeholder_value,
    reject_placeholder_value,
)


@pytest.mark.parametrize(
    "value",
    [
        "<OPERATOR_REQUIRED>",
        "operator required",
        "operator_required",
        "TODO",
        "unknown",
        "unset",
    ],
)
def test_placeholder_detection_covers_composer_sentinels(value: str) -> None:
    assert is_placeholder_value(value) is True
    assert is_wire_visible_placeholder(value) is True


def test_placeholder_rejection_names_field() -> None:
    with pytest.raises(ValueError, match=r"resource_name.*placeholder"):
        reject_placeholder_value("operator required", field_name="resource_name")


@pytest.mark.parametrize("value", ["<OPERATOR_REQUIRED>", "operator required", "operator_required"])
def test_operator_required_placeholder_detection_covers_composer_sentinel(value: str) -> None:
    assert is_operator_required_placeholder_value(value) is True
    with pytest.raises(ValueError, match=r"resource_name.*placeholder"):
        reject_operator_required_placeholder_value(value, field_name="resource_name")


@pytest.mark.parametrize("value", ["todo", "unknown", "unset", "required", "<literal>"])
def test_operator_required_placeholder_detection_accepts_resource_identifiers(value: str) -> None:
    assert is_operator_required_placeholder_value(value) is False
    assert reject_operator_required_placeholder_value(value, field_name="resource_name") == value


def test_placeholder_detection_accepts_real_values() -> None:
    assert is_placeholder_value("contacts") is False
    assert is_wire_visible_placeholder("ops@example.org") is False
    assert reject_placeholder_value("contacts", field_name="entity") == "contacts"
