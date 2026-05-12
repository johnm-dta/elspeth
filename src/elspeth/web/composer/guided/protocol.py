"""Guided-mode protocol: turn types, payloads, responses, legal-turn matrix.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, TypedDict

from elspeth.web.composer.recipes import SlotType


class TurnType(StrEnum):
    """The closed taxonomy of turn types the protocol allows."""

    INSPECT_AND_CONFIRM = "inspect_and_confirm"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT_WITH_CUSTOM = "multi_select_with_custom"
    SCHEMA_FORM = "schema_form"
    PROPOSE_CHAIN = "propose_chain"
    RECIPE_OFFER = "recipe_offer"


class _Option(TypedDict):
    id: str
    label: str
    hint: str | None


class _Observed(TypedDict):
    columns: Sequence[str]
    samples: Sequence[Mapping[str, Any]]
    warnings: Sequence[str]


class InspectAndConfirmPayload(TypedDict):
    observed: _Observed


class SingleSelectPayload(TypedDict):
    question: str
    options: Sequence[_Option]
    allow_custom: bool


class MultiSelectWithCustomPayload(TypedDict):
    question: str
    options: Sequence[_Option]
    default_chosen: Sequence[str]
    escape_label: str | None


class SchemaFormPayload(TypedDict):
    plugin: str
    schema_block: Mapping[str, Any]
    prefilled: Mapping[str, Any]


class _ProposedStep(TypedDict):
    plugin: str
    options: Mapping[str, Any]
    rationale: str


class ProposeChainPayload(TypedDict):
    steps: Sequence[_ProposedStep]
    why: str
    blockers: Sequence[str]


class _RecipeSlotInput(TypedDict):
    """Wire shape for one unsatisfied required slot in a recipe_offer turn.

    ``slot_type`` reuses :data:`elspeth.web.composer.recipes.SlotType` directly
    so the wire schema cannot drift from the source-of-truth Literal: adding a
    new member to ``SlotType`` immediately fails type-checking here, in the
    emitter, and (mirrored manually) in ``guided.ts``. The frontend renders an
    editable input keyed by ``name`` and submits the typed value back as part
    of ``edited_values.slots``.

    ``required`` is intentionally absent from this wire shape: every entry in
    ``RecipeOfferPayload.unsatisfied_slots`` is guaranteed required by the
    :class:`~elspeth.web.composer.guided.recipe_match.RecipeMatch` invariant
    (commit 83b17ca6, ``__post_init__`` Invariant 2). Sending a field that is
    always ``True`` is dead information on the wire — the frontend can treat
    all entries as required without reading a flag.
    """

    name: str
    slot_type: SlotType
    description: str


class RecipeOfferPayload(TypedDict):
    recipe_name: str
    slots: Mapping[str, Any]
    alternatives: Sequence[str]
    unsatisfied_slots: Sequence[_RecipeSlotInput]


class ControlSignal(StrEnum):
    """Out-of-band signals carried in a TurnResponse instead of (or alongside) data."""

    EXIT_TO_FREEFORM = "exit_to_freeform"
    REQUEST_ADVISOR = "request_advisor"
    REJECT = "reject"


class TurnResponse(TypedDict):
    """The user's typed response to a turn."""

    chosen: Sequence[str] | None
    edited_values: Mapping[str, Any] | None
    custom_inputs: Sequence[str] | None
    accepted_step_index: int | None
    edit_step_index: int | None
    control_signal: str | None  # ControlSignal value, or None


class Turn(TypedDict):
    """A turn emitted to the user (server-emitted or LLM-emitted)."""

    type: str  # TurnType value
    step_index: int
    payload: Mapping[str, Any]


class GuidedStep(StrEnum):
    """Wizard step pointer."""

    STEP_1_SOURCE = "step_1_source"
    STEP_2_SINK = "step_2_sink"
    STEP_2_5_RECIPE_MATCH = "step_2_5_recipe_match"
    STEP_3_TRANSFORMS = "step_3_transforms"


_LEGAL_TURN_MATRIX: Mapping[GuidedStep, frozenset[TurnType]] = {
    GuidedStep.STEP_1_SOURCE: frozenset(
        {
            TurnType.INSPECT_AND_CONFIRM,
            TurnType.SINGLE_SELECT,
            TurnType.SCHEMA_FORM,
        }
    ),
    GuidedStep.STEP_2_SINK: frozenset(
        {
            TurnType.SINGLE_SELECT,
            TurnType.MULTI_SELECT_WITH_CUSTOM,
            TurnType.SCHEMA_FORM,
        }
    ),
    GuidedStep.STEP_2_5_RECIPE_MATCH: frozenset({TurnType.RECIPE_OFFER}),
    GuidedStep.STEP_3_TRANSFORMS: frozenset(
        {
            TurnType.PROPOSE_CHAIN,
            TurnType.SINGLE_SELECT,
        }
    ),
}


def legal_turn_types_for(step: GuidedStep) -> frozenset[TurnType]:
    """Return the frozen set of TurnType values legal at the given step."""
    return _LEGAL_TURN_MATRIX[step]


_REQUIRED_KEYS: Mapping[TurnType, frozenset[str]] = {
    TurnType.INSPECT_AND_CONFIRM: frozenset({"observed"}),
    TurnType.SINGLE_SELECT: frozenset({"question", "options", "allow_custom"}),
    TurnType.MULTI_SELECT_WITH_CUSTOM: frozenset(
        {
            "question",
            "options",
            "default_chosen",
            "escape_label",
        }
    ),
    TurnType.SCHEMA_FORM: frozenset({"plugin", "schema_block", "prefilled"}),
    TurnType.PROPOSE_CHAIN: frozenset({"steps", "why", "blockers"}),
    TurnType.RECIPE_OFFER: frozenset({"recipe_name", "slots", "alternatives", "unsatisfied_slots"}),
}

# Nested shape spec for recursive payload validation.
#
# Each entry maps a TurnType to a list of (field_path_component, field_kind,
# required_keys) triples that describe nested structures.
#
# ``field_kind`` is one of:
#   "mapping"  — the field value must be a Mapping; validate its required_keys.
#   "sequence_of_mappings" — the field value must be a Sequence; each element
#                            must be a Mapping with required_keys.
#
# Path-rooted error messages use dot notation: ``"payload.observed.columns"``.
#
# Only fields whose nested shapes are meaningful to validate are listed. Scalar
# and pass-through fields (e.g., ``allow_custom: bool``, ``why: str``) are
# covered by the top-level required-key check and need no further descend.
_NestedSpec = tuple[str, str, frozenset[str]]
_NESTED_SHAPES: Mapping[TurnType, tuple[_NestedSpec, ...]] = {
    TurnType.INSPECT_AND_CONFIRM: (
        # "observed" must be a Mapping with these keys
        ("observed", "mapping", frozenset({"columns", "samples", "warnings"})),
    ),
    TurnType.RECIPE_OFFER: (
        # "unsatisfied_slots" must be a Sequence; each element is a Mapping
        # with these keys.  "required" is intentionally absent — the
        # RecipeMatch invariant guarantees every entry is required.
        ("unsatisfied_slots", "sequence_of_mappings", frozenset({"name", "slot_type", "description"})),
    ),
}


def validate_payload(turn_type: TurnType, payload: Mapping[str, Any]) -> str | None:
    """Validate that *payload* satisfies the schema for *turn_type*.

    Validates top-level required keys and recursively walks nested TypedDicts
    listed in ``_NESTED_SHAPES``.  Error messages are path-rooted using dot
    notation so the caller can locate the offending field:
    ``"payload.observed.columns missing"`` rather than ``"columns missing"``.

    Returns None on success, or a human-readable error string on failure.
    Raises ValueError if turn_type is not a known TurnType.
    """
    if not isinstance(turn_type, TurnType):
        raise ValueError(f"unknown turn type: {turn_type!r}")
    required = _REQUIRED_KEYS[turn_type]
    missing = required - payload.keys()
    if missing:
        return f"payload for {turn_type.value} missing required keys: {sorted(missing)}"

    # Recursive nested-shape validation.
    for field_name, field_kind, nested_required in _NESTED_SHAPES.get(turn_type, ()):
        field_value = payload[field_name]
        prefix = f"payload.{field_name}"
        if field_kind == "mapping":
            if not isinstance(field_value, Mapping):
                return f"{prefix} must be a mapping (got {type(field_value).__name__})"
            nested_missing = nested_required - field_value.keys()
            if nested_missing:
                return f"{prefix} missing required keys: {sorted(nested_missing)}"
        elif field_kind == "sequence_of_mappings":
            if not isinstance(field_value, Sequence) or isinstance(field_value, str):
                return f"{prefix} must be a sequence (got {type(field_value).__name__})"
            for idx, item in enumerate(field_value):
                if not isinstance(item, Mapping):
                    return f"{prefix}[{idx}] must be a mapping (got {type(item).__name__})"
                item_missing = nested_required - item.keys()
                if item_missing:
                    return f"{prefix}[{idx}] missing required keys: {sorted(item_missing)}"

    return None
