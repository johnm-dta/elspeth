from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, create_model

from elspeth.web.catalog.knob_schema import (
    FieldKind,
    lower_model_to_knob_schema,
    validate_knob_schema,
)

_FIELD_KINDS = frozenset(FieldKind.__args__)


@dataclass(frozen=True)
class _FieldCase:
    annotation: Any
    expected_kind: str
    expected_nullable: bool = False


_FIELD_CASES = (
    _FieldCase(str, "text"),
    _FieldCase(int, "number-int"),
    _FieldCase(float, "number-float"),
    _FieldCase(bool, "checkbox"),
    _FieldCase(Literal["alpha", "beta"], "enum"),
    _FieldCase(str | None, "text", expected_nullable=True),
    _FieldCase(int | None, "number-int", expected_nullable=True),
    _FieldCase(list[str], "string-list"),
    _FieldCase(list[int], "json-array"),
    _FieldCase(dict[str, int], "json-object"),
    _FieldCase(int | str, "json-value"),
)


def _field_name() -> st.SearchStrategy[str]:
    return st.from_regex(r"field_[a-z]{1,8}", fullmatch=True)


def _field_case() -> st.SearchStrategy[_FieldCase]:
    return st.sampled_from(_FIELD_CASES)


def _model_case() -> st.SearchStrategy[list[tuple[str, _FieldCase]]]:
    return st.lists(
        st.tuples(_field_name(), _field_case()),
        min_size=1,
        max_size=6,
        unique_by=lambda item: item[0],
    )


def _model_from_cases(cases: list[tuple[str, _FieldCase]]) -> type[BaseModel]:
    fields = {name: (case.annotation, ...) for name, case in cases}
    return create_model("GeneratedKnobModel", **fields)


@given(cases=_model_case())
@settings(max_examples=60)
def test_lower_model_to_knob_schema_returns_valid_schema_for_generated_models(
    cases: list[tuple[str, _FieldCase]],
) -> None:
    model_cls = _model_from_cases(cases)

    schema = lower_model_to_knob_schema(
        model_cls,
        plugin_kind="property",
        plugin_name="generated",
    )

    validate_knob_schema(schema, plugin_kind="property", plugin_name="generated")
    assert len(schema["fields"]) == len(cases)

    fields_by_name = {field["name"]: field for field in schema["fields"]}
    for field_name, case in cases:
        field = fields_by_name[field_name]
        assert field["kind"] in _FIELD_KINDS
        assert field["kind"] == case.expected_kind
        assert field["nullable"] is case.expected_nullable


def test_rich_shapes_lower_to_fallback_knobs_without_raising() -> None:
    class NestedModel(BaseModel):
        value: str

    class RichModel(BaseModel):
        nested: NestedModel
        array: list[int]
        union_value: int | str
        mapping: dict[str, int]

    schema = lower_model_to_knob_schema(
        RichModel,
        plugin_kind="property",
        plugin_name="rich",
    )

    validate_knob_schema(schema, plugin_kind="property", plugin_name="rich")
    fields_by_name = {field["name"]: field for field in schema["fields"]}
    assert fields_by_name["nested"]["kind"] == "json-object"
    assert fields_by_name["array"]["kind"] == "json-array"
    assert fields_by_name["union_value"]["kind"] == "json-value"
    assert fields_by_name["mapping"]["kind"] == "json-object"
