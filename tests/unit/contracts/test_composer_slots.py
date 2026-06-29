"""Tests for Layer-0 composer slot contracts."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.composer_slots import SlotSpec, SlotType


@pytest.mark.parametrize(
    ("slot_type", "raw_default", "expected_default", "expected_type"),
    [
        ("float", 2, 2.0, float),
        ("int", "7", 7, int),
        ("str_list", ["a", "b"], ("a", "b"), tuple),
    ],
)
def test_optional_default_is_stored_in_canonical_form(
    slot_type: SlotType,
    raw_default: Any,
    expected_default: Any,
    expected_type: type[object],
) -> None:
    spec = SlotSpec(slot_type=slot_type, required=False, default=raw_default)

    assert spec.default == expected_default
    assert type(spec.default) is expected_type
