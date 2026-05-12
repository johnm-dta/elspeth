"""Guided-mode protocol: turn types, payloads, responses, legal-turn matrix.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, TypedDict


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

    ``slot_type`` mirrors :data:`elspeth.web.composer.recipes.SlotType`.
    The frontend renders an editable input keyed by ``name`` and submits the
    typed value back as part of ``edited_values.slots``.
    """

    name: str
    slot_type: str  # one of recipes.SlotType: "blob_id"/"str"/"float"/"int"/"str_list"
    description: str
    required: bool


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


def validate_payload(turn_type: TurnType, payload: Mapping[str, Any]) -> str | None:
    """Validate that *payload* satisfies the schema for *turn_type*.

    Returns None on success, or a human-readable error string on failure.
    Raises ValueError if turn_type is not a known TurnType.
    """
    if not isinstance(turn_type, TurnType):
        raise ValueError(f"unknown turn type: {turn_type!r}")
    required = _REQUIRED_KEYS[turn_type]
    missing = required - payload.keys()
    if missing:
        return f"payload for {turn_type.value} missing required keys: {sorted(missing)}"
    return None
