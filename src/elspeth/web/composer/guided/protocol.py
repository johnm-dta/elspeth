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


class RecipeOfferPayload(TypedDict):
    recipe_name: str
    slots: Mapping[str, Any]
    alternatives: Sequence[str]
