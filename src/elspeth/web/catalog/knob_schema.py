"""Composer one-knob wire shape.

Lowering happens at catalog load time inside ``CatalogServiceImpl.__init__``;
this module exposes the result types and the lowering entry points. See
docs/superpowers/specs/2026-05-14-composer-one-knob-design.md.

Trust tier: L3 web layer. ``KnobSchema`` instances are Tier 1 because we write
them from plugin models we control. Prefilled values from
``SourceInspectionFacts`` remain Tier 3.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from inspect import isclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict, Union, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from elspeth.contracts.composer_slots import SlotSpec

FieldKind = Literal[
    "text",
    "number-int",
    "number-float",
    "checkbox",
    "enum",
    "string-list",
    "blob-ref",
    "json-object",
    "json-array",
    "json-value",
]
FieldTier = Literal["essential", "common", "advanced"]


class VisibilityPredicate(TypedDict):
    """Conditional-visibility predicate for a ``KnobField``.

    ``field`` must reference an earlier-declared ``KnobField`` in the same
    ``KnobSchema``. ``equals`` is an exact value match against current form
    state. No other keys are permitted; invalid predicates raise
    ``KnobSchemaLoweringError`` at catalog load.
    """

    field: str
    equals: Any


class KnobField(TypedDict):
    name: str
    label: str
    description: NotRequired[str]
    kind: FieldKind
    tier: NotRequired[FieldTier]
    required: bool
    default: NotRequired[object]
    nullable: bool
    enum: NotRequired[list[str]]
    item_kind: NotRequired[Literal["text", "number-int", "number-float"]]
    visible_when: NotRequired[VisibilityPredicate]


class KnobSchema(TypedDict):
    fields: list[KnobField]


class RecipeContext(TypedDict):
    recipe_name: str
    description: str
    alternatives: list[str]


class _PluginOptionsPayload(TypedDict):
    mode: Literal["plugin_options"]
    plugin: str
    knobs: KnobSchema
    prefilled: dict[str, object]


class _RecipeDecisionPayload(TypedDict):
    mode: Literal["recipe_decision"]
    knobs: KnobSchema
    prefilled: dict[str, object]
    recipe_context: RecipeContext


SchemaFormPayload = _PluginOptionsPayload | _RecipeDecisionPayload


class KnobSchemaLoweringError(Exception):
    """Raised at catalog load for malformed schemas or one-knob violations.

    Valid-but-rich fields lower to ``json-object``, ``json-array``, or
    ``json-value`` fallback knobs. True invariant violations halt startup.
    """

    def __init__(
        self,
        *,
        plugin_kind: str,
        plugin_name: str,
        field_path: str,
        constraint: str,
        remediation: str,
    ) -> None:
        message = f"Plugin {plugin_kind}/{plugin_name} field {field_path!r}: {constraint}. Remediation: {remediation}"
        super().__init__(message)
        self.plugin_kind = plugin_kind
        self.plugin_name = plugin_name
        self.field_path = field_path
        self.constraint = constraint
        self.remediation = remediation


_TYPE_TO_KIND: dict[type, Literal["text", "number-int", "number-float", "checkbox"]] = {
    str: "text",
    int: "number-int",
    float: "number-float",
    bool: "checkbox",
}


def _unwrap_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, nullable)`` for ``T | None`` and ``Optional[T]``."""
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            return _unwrap_annotated(non_none[0]), True
    return annotation, False


def _kind_for_scalar(
    inner: Any,
) -> tuple[Literal["text", "number-int", "number-float", "checkbox", "enum", "json-value"], list[str] | None]:
    """Map a Python scalar or Literal type to a field kind and enum values."""
    if get_origin(inner) is Literal:
        values = [str(value) for value in get_args(inner)]
        return "enum", values
    if inner in _TYPE_TO_KIND:
        return _TYPE_TO_KIND[inner], None
    return "json-value", None


def _base_field(
    *,
    name: str,
    info: FieldInfo,
    kind: FieldKind,
    nullable: bool,
) -> KnobField:
    field: KnobField = {
        "name": name,
        "label": info.title or name,
        "kind": kind,
        "required": info.is_required(),
        "nullable": nullable,
    }
    if info.description:
        field["description"] = info.description
    _attach_default(field, info)
    _attach_tier(field, info)
    return field


def _lower_field(
    name: str,
    info: FieldInfo,
    *,
    plugin_kind: str,
    plugin_name: str,
    composer_tier_default: str,
) -> KnobField:
    del plugin_kind, plugin_name, composer_tier_default
    inner, nullable = _unwrap_optional(info.annotation)
    origin = get_origin(inner)

    if origin is list:
        list_args = get_args(inner)
        if len(list_args) == 1 and _unwrap_annotated(list_args[0]) is str:
            field = _base_field(name=name, info=info, kind="string-list", nullable=nullable)
            field["item_kind"] = "text"
            return field
        return _base_field(name=name, info=info, kind="json-array", nullable=nullable)

    is_model_cls = isclass(inner) and issubclass(inner, BaseModel)
    if origin in (dict, Mapping) or is_model_cls:
        return _base_field(name=name, info=info, kind="json-object", nullable=nullable)

    kind, enum_values = _kind_for_scalar(inner)
    field = _base_field(name=name, info=info, kind=kind, nullable=nullable)
    if enum_values is not None:
        field["enum"] = enum_values
    return field


def _attach_default(field: KnobField, info: FieldInfo) -> None:
    if info.is_required() or info.default is PydanticUndefined:
        return
    field["default"] = info.default


def _attach_tier(field: KnobField, info: FieldInfo) -> None:
    extra = info.json_schema_extra
    if type(extra) is not dict:
        return
    if "composer_tier" not in extra:
        return
    tier = extra["composer_tier"]
    if tier in ("essential", "common", "advanced"):
        field["tier"] = cast(FieldTier, tier)


def lower_model_to_knob_schema(
    model_cls: type[BaseModel],
    *,
    plugin_kind: str,
    plugin_name: str,
    composer_tier_default: str = "common",
) -> KnobSchema:
    """Lower a single-model Pydantic config class to ``KnobSchema``."""
    fields: list[KnobField] = []
    for name, info in model_cls.model_fields.items():
        fields.append(
            _lower_field(
                name,
                info,
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                composer_tier_default=composer_tier_default,
            )
        )
    return {"fields": fields}


_SLOT_TYPE_TO_KIND: dict[str, FieldKind] = {
    "blob_id": "blob-ref",
    "str": "text",
    "int": "number-int",
    "float": "number-float",
    "str_list": "string-list",
}


def lower_slot_specs_to_knob_schema(slots: Mapping[str, SlotSpec]) -> KnobSchema:
    """Lower recipe slot specs to the one-knob schema."""
    fields: list[KnobField] = []
    for name, spec in slots.items():
        kind = _SLOT_TYPE_TO_KIND[spec.slot_type]
        field: KnobField = {
            "name": name,
            "label": name,
            "kind": kind,
            "required": spec.required,
            "nullable": not spec.required,
        }
        if spec.description:
            field["description"] = spec.description
        if kind == "string-list":
            field["item_kind"] = "text"
        if spec.default is not None:
            field["default"] = spec.default
        fields.append(field)
    return {"fields": fields}


def lower_discriminated_to_knob_schema(
    plugin_cls: type,
    *,
    plugin_kind: str,
    plugin_name: str,
    composer_tier_default: str = "common",
) -> KnobSchema:
    """Lower a discriminated-union plugin to a flat visible_when schema."""
    try:
        discriminated_variants = cast(Any, plugin_cls).discriminated_variants
    except AttributeError as exc:
        raise KnobSchemaLoweringError(
            plugin_kind=plugin_kind,
            plugin_name=plugin_name,
            field_path="<class>",
            constraint=("plugin lacks discriminated_variants() classmethod required by DiscriminatedPlugin protocol"),
            remediation=("Implement discriminated_variants() returning (discriminator_field_name, {literal_value: variant_cls})."),
        ) from exc
    if not callable(discriminated_variants):
        raise KnobSchemaLoweringError(
            plugin_kind=plugin_kind,
            plugin_name=plugin_name,
            field_path="<class>",
            constraint=("plugin lacks discriminated_variants() classmethod required by DiscriminatedPlugin protocol"),
            remediation=("Implement discriminated_variants() returning (discriminator_field_name, {literal_value: variant_cls})."),
        )
    discriminator, variants = discriminated_variants()

    fields: list[KnobField] = [
        {
            "name": discriminator,
            "label": discriminator,
            "kind": "enum",
            "enum": list(variants.keys()),
            "required": True,
            "nullable": False,
        }
    ]
    for variant_value, variant_cls in variants.items():
        for fname, info in variant_cls.model_fields.items():
            if fname == discriminator:
                continue
            inner_field = _lower_field(
                fname,
                info,
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                composer_tier_default=composer_tier_default,
            )
            inner_field["visible_when"] = {"field": discriminator, "equals": variant_value}
            fields.append(inner_field)
    return {"fields": fields}
