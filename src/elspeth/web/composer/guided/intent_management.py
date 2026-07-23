"""Pure stable-id management of pending guided deferred intents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_freeze, deep_thaw
from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.guided.deferred_intents import (
    DeferredIntentAccepted,
    DeferredIntentCancelAction,
    DeferredIntentEditAction,
    DeferredIntentManagementAction,
    DeferredIntentValidation,
    create_deferred_stage_intent,
    validate_deferred_intent_action,
)
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.stage_subjects import OptionValueConstraint, StageName
from elspeth.web.composer.guided.state_machine import DeferredStageIntent, GuidedSession


@dataclass(frozen=True, slots=True)
class DeferredIntentManagementUnknown:
    """The requested stable ID did not name one current pending intent."""


@dataclass(frozen=True, slots=True)
class DeferredIntentManagementBindingMismatch:
    """The model paired a real intent ID with the wrong server selection token."""


@dataclass(frozen=True, slots=True)
class DeferredIntentManagementAmbiguous:
    """The private request did not explicitly authorize the exact mutation."""


@dataclass(frozen=True, slots=True)
class DeferredIntentManagementOption:
    """Provider-safe, server-authored destructive-selection binding."""

    intent_id: str
    selection_token: str
    receiving_stage: StageName
    target_stage: StageName
    catalog_kind: str | None
    catalog_name: str | None
    redacted_summary: str
    structural_constraints: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_constraints", deep_freeze(self.structural_constraints))

    def to_provider_dict(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "selection_token": self.selection_token,
            "receiving_stage": self.receiving_stage,
            "target_stage": self.target_stage,
            "catalog_kind": self.catalog_kind,
            "catalog_name": self.catalog_name,
            "redacted_summary": self.redacted_summary,
            "structural_constraints": deep_thaw(self.structural_constraints),
        }


def _provider_safe_constraint(constraint: object) -> dict[str, object]:
    if not hasattr(constraint, "to_dict"):
        raise AuditIntegrityError("deferred intent contains a constraint without a closed projection")
    projected = cast(dict[str, Any], constraint.to_dict())
    if type(constraint) is OptionValueConstraint:
        value = projected.pop("value")
        projected["value_hash"] = stable_hash({"schema": "guided.deferred-option-value.v1", "value": value})
    return projected


def _structural_projection(intent: DeferredStageIntent) -> tuple[Mapping[str, object], ...]:
    return cast(tuple[Mapping[str, object], ...], deep_freeze(tuple(_provider_safe_constraint(item) for item in intent.constraints)))


def _structural_identity(intent: DeferredStageIntent) -> str:
    return stable_hash(
        {
            "schema": "guided.deferred-management-structure.v1",
            "receiving_stage": intent.receiving_stage,
            "target_stage": intent.target_stage,
            "catalog_kind": intent.catalog_kind,
            "catalog_name": intent.catalog_name,
            "constraints": deep_thaw(_structural_projection(intent)),
        }
    )


def deferred_intent_management_structural_identity(intent: DeferredStageIntent) -> str:
    """Return the provider-safe identity used to detect destructive ambiguity."""

    return _structural_identity(intent)


def deferred_intent_management_option(intent: DeferredStageIntent) -> DeferredIntentManagementOption:
    structural_identity = _structural_identity(intent)
    return DeferredIntentManagementOption(
        intent_id=intent.intent_id,
        selection_token=stable_hash(
            {
                "schema": "guided.deferred-management-selection.v1",
                "intent_id": intent.intent_id,
                "structural_identity": structural_identity,
            }
        ),
        receiving_stage=intent.receiving_stage,
        target_stage=intent.target_stage,
        catalog_kind=intent.catalog_kind,
        catalog_name=intent.catalog_name,
        redacted_summary=intent.redacted_summary,
        structural_constraints=_structural_projection(intent),
    )


def deferred_intent_management_user_authority_matches(
    action: DeferredIntentManagementAction,
    *,
    deferred_intents: Sequence[DeferredStageIntent],
    originating_message_content: str,
) -> bool:
    """Require an exact action-specific command for every destructive mutation.

    The provider action and its server-bound selection token identify a
    proposed mutation; neither grants user authority. Cancellation requires
    ``Cancel exact intent <UUID>.`` as the whole private request. Editing
    requires ``Edit exact intent <UUID>: <new instruction>`` so the request
    explicitly supplies both the operation and replacement directive.
    """

    matching_current_ids = tuple(intent.intent_id for intent in deferred_intents if intent.intent_id == action.intent_id)
    if matching_current_ids != (action.intent_id,):
        return False
    escaped_intent_id = re.escape(action.intent_id)
    if type(action) is DeferredIntentCancelAction:
        pattern = rf"\s*cancel\s+exact\s+intent\s+{escaped_intent_id}\.?\s*"
    elif type(action) is DeferredIntentEditAction:
        pattern = rf"\s*edit\s+exact\s+intent\s+{escaped_intent_id}\s*:\s*\S(?:[\s\S]*\S)?\s*"
    else:  # pragma: no cover - the closed action union owns this guard
        return False
    return re.fullmatch(pattern, originating_message_content, flags=re.IGNORECASE) is not None


@dataclass(frozen=True, slots=True)
class DeferredIntentManagementApplied:
    """Exact ordered intent mutation prepared without changing GuidedSession."""

    prior_intent: DeferredStageIntent
    effective_intent: DeferredStageIntent
    deferred_intents: tuple[DeferredStageIntent, ...]
    assistant_message: str


type DeferredIntentManagementResolution = (
    DeferredIntentManagementApplied
    | DeferredIntentManagementUnknown
    | DeferredIntentManagementBindingMismatch
    | DeferredIntentManagementAmbiguous
    | DeferredIntentValidation
)


def resolve_deferred_intent_management(
    action: DeferredIntentManagementAction,
    *,
    guided: GuidedSession,
    catalog: PolicyCatalogView,
    originating_message_id: str,
    originating_message_content: str,
) -> DeferredIntentManagementResolution:
    """Resolve one exact cancel/edit while preserving stable identity and order."""

    matches = [(index, intent) for index, intent in enumerate(guided.deferred_intents) if intent.intent_id == action.intent_id]
    if len(matches) != 1:
        return DeferredIntentManagementUnknown()
    intent_index, existing = matches[0]
    expected_option = deferred_intent_management_option(existing)
    if action.selection_token != expected_option.selection_token:
        return DeferredIntentManagementBindingMismatch()
    if not deferred_intent_management_user_authority_matches(
        action,
        deferred_intents=guided.deferred_intents,
        originating_message_content=originating_message_content,
    ):
        return DeferredIntentManagementAmbiguous()
    if type(action) is DeferredIntentCancelAction:
        return DeferredIntentManagementApplied(
            prior_intent=existing,
            effective_intent=existing,
            deferred_intents=(*guided.deferred_intents[:intent_index], *guided.deferred_intents[intent_index + 1 :]),
            assistant_message=f"I cancelled that saved {existing.target_stage.replace('_', ' ')} instruction.",
        )
    if type(action) is DeferredIntentEditAction:
        disposition = validate_deferred_intent_action(
            action.replacement,
            receiving_stage=existing.receiving_stage,
            catalog=catalog,
            guided=guided,
        )
        if type(disposition) is not DeferredIntentAccepted:
            return disposition
        replacement = create_deferred_stage_intent(
            disposition.action,
            receiving_stage=existing.receiving_stage,
            intent_id=existing.intent_id,
            originating_message_id=originating_message_id,
            originating_message_content=originating_message_content,
        )
        return DeferredIntentManagementApplied(
            prior_intent=existing,
            effective_intent=replacement,
            deferred_intents=(
                *guided.deferred_intents[:intent_index],
                replacement,
                *guided.deferred_intents[intent_index + 1 :],
            ),
            assistant_message=f"I revised that saved {replacement.target_stage.replace('_', ' ')} instruction.",
        )
    raise TypeError("action must be an exact deferred intent management action")


def schema8_deferred_management_rewind_step(*, current_step: GuidedStep, target_stage: StageName) -> GuidedStep | None:
    """Return schema-8's sole topology replan entry for a passed intent stage.

    Schema 8 has no empty Step-3/replan turn. Finishing Step-2 output review is
    the only legal transition that invokes the shared planner and stages a
    proposal. Passed output/topology changes therefore rewind to that explicit
    boundary. Schema 9 replaces this topology path rather than retaining it.
    """

    stage_index = {"source": 0, "output": 1, "topology": 2, "wire_review": 3}
    current_index = {
        GuidedStep.STEP_1_SOURCE: 0,
        GuidedStep.STEP_2_SINK: 1,
        GuidedStep.STEP_3_TRANSFORMS: 2,
        GuidedStep.STEP_4_WIRE: 3,
    }[current_step]
    target_index = stage_index[target_stage]
    if target_stage == "source" and current_index > target_index:
        raise AuditIntegrityError("schema-8 deferred intent authority cannot target the already-passed source stage")
    if target_stage == "wire_review" and current_index > target_index:  # pragma: no cover - closed final stage
        raise AuditIntegrityError("schema-8 deferred intent authority cannot pass the final wire-review stage")
    if current_step is GuidedStep.STEP_3_TRANSFORMS and target_stage == "topology":
        return GuidedStep.STEP_2_SINK
    if current_index > target_index:
        if target_stage not in {"output", "topology"}:
            raise AuditIntegrityError("schema-8 passed deferred intent target is outside the closed rewind vocabulary")
        return GuidedStep.STEP_2_SINK
    return None


__all__ = [
    "DeferredIntentManagementAmbiguous",
    "DeferredIntentManagementApplied",
    "DeferredIntentManagementBindingMismatch",
    "DeferredIntentManagementOption",
    "DeferredIntentManagementUnknown",
    "deferred_intent_management_option",
    "deferred_intent_management_structural_identity",
    "deferred_intent_management_user_authority_matches",
    "resolve_deferred_intent_management",
    "schema8_deferred_management_rewind_step",
]
