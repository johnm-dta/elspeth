from __future__ import annotations

from typing import Any, cast

import pytest

from elspeth.web.catalog.knob_schema import (
    KnobSchema,
    KnobSchemaLoweringError,
    VisibilityPredicate,
    validate_knob_schema,
)


def _ks_with_predicate(predicate: dict[str, Any]) -> KnobSchema:
    return {
        "fields": [
            {
                "name": "x",
                "label": "x",
                "kind": "enum",
                "enum": ["a", "b"],
                "required": True,
                "nullable": False,
            },
            {
                "name": "y",
                "label": "y",
                "kind": "text",
                "required": False,
                "nullable": False,
                "visible_when": cast(VisibilityPredicate, predicate),
            },
        ]
    }


def test_well_formed_predicate_validates() -> None:
    ks = _ks_with_predicate({"field": "x", "equals": "a"})
    validate_knob_schema(ks, plugin_kind="t", plugin_name="p")


def test_extra_keys_rejected() -> None:
    ks = _ks_with_predicate({"field": "x", "equals": "a", "operator": "and"})
    with pytest.raises(KnobSchemaLoweringError) as exc:
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")
    assert "keys" in exc.value.constraint.lower()


def test_missing_keys_rejected() -> None:
    ks = _ks_with_predicate({"field": "x"})
    with pytest.raises(KnobSchemaLoweringError):
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")


def test_forward_reference_rejected() -> None:
    ks: KnobSchema = {
        "fields": [
            {
                "name": "y",
                "label": "y",
                "kind": "text",
                "required": False,
                "nullable": False,
                "visible_when": {"field": "x", "equals": "a"},
            },
            {
                "name": "x",
                "label": "x",
                "kind": "enum",
                "enum": ["a"],
                "required": True,
                "nullable": False,
            },
        ]
    }
    with pytest.raises(KnobSchemaLoweringError) as exc:
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")
    assert "forward" in exc.value.constraint.lower()


def test_unknown_field_reference_rejected() -> None:
    ks = _ks_with_predicate({"field": "nonexistent", "equals": "a"})
    with pytest.raises(KnobSchemaLoweringError):
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")


def test_nested_visibility_rejected() -> None:
    ks: KnobSchema = {
        "fields": [
            {
                "name": "x",
                "label": "x",
                "kind": "enum",
                "enum": ["a"],
                "required": True,
                "nullable": False,
            },
            {
                "name": "y",
                "label": "y",
                "kind": "text",
                "required": False,
                "nullable": False,
                "visible_when": {"field": "x", "equals": "a"},
            },
            {
                "name": "z",
                "label": "z",
                "kind": "text",
                "required": False,
                "nullable": False,
                "visible_when": {"field": "y", "equals": "anything"},
            },
        ]
    }
    with pytest.raises(KnobSchemaLoweringError) as exc:
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")
    assert "nest" in exc.value.constraint.lower()
