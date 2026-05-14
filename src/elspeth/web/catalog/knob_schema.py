"""Composer one-knob wire shape.

Lowering happens at catalog load time inside ``CatalogServiceImpl.__init__``;
this module exposes the result types and the lowering entry points. See
docs/superpowers/specs/2026-05-14-composer-one-knob-design.md.

Trust tier: L3 web layer. ``KnobSchema`` instances are Tier 1 because we write
them from plugin models we control. Prefilled values from
``SourceInspectionFacts`` remain Tier 3.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


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
    kind: Literal[
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
    tier: NotRequired[Literal["essential", "common", "advanced"]]
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
