"""Schema-8 guided Chat deferred-intent application and rewind authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast
from uuid import UUID

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.payload_store import PayloadStore
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.guided.deferred_intents import (
    DeferredIntentAccepted,
    DeferredIntentAction,
    DeferredIntentCancelAction,
    DeferredIntentClarification,
    DeferredIntentEditAction,
    DeferredIntentManagementAction,
    DeferredIntentRejected,
    DeferredIntentUnsupported,
    create_deferred_stage_intent,
    validate_deferred_intent_action,
)
from elspeth.web.composer.guided.intent_management import (
    DeferredIntentManagementAmbiguous,
    DeferredIntentManagementApplied,
    DeferredIntentManagementBindingMismatch,
    DeferredIntentManagementUnknown,
    resolve_deferred_intent_management,
    schema8_deferred_management_rewind_step,
)
from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
from elspeth.web.composer.guided.protocol import GuidedStep, Turn, TurnType
from elspeth.web.composer.guided.stage_subjects import StageName
from elspeth.web.composer.guided.state_machine import DeferredStageIntent, GuidedProposalRef, GuidedSession, TurnRecord
from elspeth.web.composer.state import CompositionState
from elspeth.web.sessions._guided_step_chat import StepChatResult
from elspeth.web.sessions.guided_payloads import prepare_guided_json_payload
from elspeth.web.sessions.protocol import (
    GuidedOriginatingUserMessageDraft,
    GuidedPendingProposalInvalidation,
    PreparedGuidedJsonPayload,
)


class Schema8GuidedRouteAuthority(Protocol):
    """Narrow deterministic turn-building surface used by intent rewind."""

    def _build_get_guided_turn(
        self,
        state: CompositionState,
        guided: GuidedSession,
        *,
        catalog: PolicyCatalogView,
    ) -> Turn | None: ...

    def _finalize_guided_turn(self, turn: Mapping[str, Any], *, shield_available: bool) -> Turn: ...

    def _prepare_server_turn_occurrence(
        self,
        guided: GuidedSession,
        *,
        current_step: GuidedStep,
        turn: Turn,
        payload_store: PayloadStore,
    ) -> tuple[GuidedSession, TurnRecord, TurnType, PreparedGuidedJsonPayload]: ...


@dataclass(frozen=True, slots=True)
class ManagementRewind:
    state: CompositionState
    response_payload: PreparedGuidedJsonPayload
    next_turn: Turn
    next_payload: PreparedGuidedJsonPayload
    invalidated_proposal: GuidedPendingProposalInvalidation | None


@dataclass(frozen=True, slots=True)
class ManagementRewindAuthority:
    """Frozen schema-8 checkpoint authority used to prepare one rewind."""

    guided_route: Schema8GuidedRouteAuthority
    current_state: CompositionState
    current_guided: GuidedSession
    prospective: GuidedSession
    catalog: PolicyCatalogView
    shield_available: bool
    payload_store: PayloadStore


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredRequestUnchanged:
    guided: GuidedSession
    chat: StepChatResult


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredRequestRetained:
    guided: GuidedSession
    chat: StepChatResult
    retained_intent_id: UUID

    def __post_init__(self) -> None:
        if type(self.retained_intent_id) is not UUID:
            raise TypeError("DeferredRequestRetained.retained_intent_id must be an exact UUID")
        if not any(intent.intent_id == str(self.retained_intent_id) for intent in self.guided.deferred_intents):
            raise AuditIntegrityError("retained deferred request lost its exact stable intent")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredRequestCancelled:
    guided: GuidedSession
    chat: StepChatResult
    action: DeferredIntentCancelAction
    effective_intent: DeferredStageIntent
    deferred_intents: tuple[DeferredStageIntent, ...]
    invalidated_active_proposal: GuidedProposalRef | None

    def __post_init__(self) -> None:
        if type(self.action) is not DeferredIntentCancelAction:
            raise TypeError("DeferredRequestCancelled.action must be exact")
        if type(self.effective_intent) is not DeferredStageIntent or type(self.deferred_intents) is not tuple:
            raise TypeError("DeferredRequestCancelled intent fields must be exact")
        if self.effective_intent.intent_id != self.action.intent_id:
            raise AuditIntegrityError("cancelled request action and effective intent identities differ")
        if any(intent.intent_id == self.action.intent_id for intent in self.deferred_intents):
            raise AuditIntegrityError("cancelled request retained the cancelled stable intent")
        _validate_managed_request_checkpoint(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredRequestEdited:
    guided: GuidedSession
    chat: StepChatResult
    action: DeferredIntentEditAction
    effective_intent: DeferredStageIntent
    deferred_intents: tuple[DeferredStageIntent, ...]
    invalidated_active_proposal: GuidedProposalRef | None

    def __post_init__(self) -> None:
        if type(self.action) is not DeferredIntentEditAction:
            raise TypeError("DeferredRequestEdited.action must be exact")
        if type(self.effective_intent) is not DeferredStageIntent or type(self.deferred_intents) is not tuple:
            raise TypeError("DeferredRequestEdited intent fields must be exact")
        matches = tuple(intent for intent in self.deferred_intents if intent.intent_id == self.action.intent_id)
        if self.effective_intent.intent_id != self.action.intent_id or matches != (self.effective_intent,):
            raise AuditIntegrityError("edited request lost its one exact effective stable intent")
        _validate_managed_request_checkpoint(self)


type DeferredRequestApplication = DeferredRequestUnchanged | DeferredRequestRetained | DeferredRequestCancelled | DeferredRequestEdited
type DeferredRequestManaged = DeferredRequestCancelled | DeferredRequestEdited


def _validate_managed_request_checkpoint(request: DeferredRequestManaged) -> None:
    if request.invalidated_active_proposal is None:
        if request.guided.deferred_intents != request.deferred_intents:
            raise AuditIntegrityError("managed request checkpoint does not contain its exact deferred-intent mutation")
        return
    if request.guided.active_proposal != request.invalidated_active_proposal:
        raise AuditIntegrityError("managed request invalidation does not bind the active proposal")


@dataclass(frozen=True, slots=True)
class DeferredRequestAuthority:
    """Stable authority needed to retain or manage one deferred request."""

    guided: GuidedSession
    catalog: PolicyCatalogView
    originating_message: GuidedOriginatingUserMessageDraft
    new_intent_id: UUID


def _guided_stage_name(step: GuidedStep) -> StageName:
    if step is GuidedStep.STEP_1_SOURCE:
        return "source"
    if step is GuidedStep.STEP_2_SINK:
        return "output"
    if step is GuidedStep.STEP_3_TRANSFORMS:
        return "topology"
    if step is GuidedStep.STEP_4_WIRE:
        return "wire_review"
    raise AuditIntegrityError("Guided Chat step is outside the closed stage vocabulary")


def _deferred_disposition_chat(
    disposition: DeferredIntentAccepted | DeferredIntentClarification | DeferredIntentUnsupported | DeferredIntentRejected,
    *,
    latency_ms: int,
) -> StepChatResult:
    if type(disposition) is DeferredIntentAccepted:
        message = f"I saved that instruction for the {disposition.action.target_stage.replace('_', ' ')} stage."
    elif type(disposition) is DeferredIntentClarification:
        kinds = ", ".join(disposition.plugin_kinds)
        message = f"I found {disposition.plugin_name!r} in more than one plugin category ({kinds}). Which category did you mean?"
    elif type(disposition) is DeferredIntentUnsupported:
        if disposition.reason.value == "plugin_not_enabled":
            message = f"The {disposition.plugin_kind} plugin {disposition.plugin_name!r} is not enabled by the current policy."
        elif disposition.reason.value == "plugin_not_installed":
            message = f"The {disposition.plugin_kind} plugin {disposition.plugin_name!r} is not installed."
        else:
            message = f"The {disposition.plugin_kind} plugin {disposition.plugin_name!r} is currently unavailable."
    else:
        message = "I couldn't safely retain that as a future-stage instruction. Please clarify the target stage and structural requirement."
    return StepChatResult(
        assistant_message=message,
        status=ComposerChatTurnStatus.SUCCESS,
        latency_ms=latency_ms,
        error_class=None,
    )


def _apply_deferred_management(
    action: DeferredIntentManagementAction,
    *,
    guided: GuidedSession,
    catalog: PolicyCatalogView,
    originating_message: GuidedOriginatingUserMessageDraft,
    chat: StepChatResult,
) -> DeferredRequestApplication:
    management = resolve_deferred_intent_management(
        action,
        guided=guided,
        catalog=catalog,
        originating_message_id=str(originating_message.message_id),
        originating_message_content=originating_message.content,
    )
    if type(management) in {
        DeferredIntentManagementUnknown,
        DeferredIntentManagementBindingMismatch,
        DeferredIntentManagementAmbiguous,
    }:
        if type(management) is DeferredIntentManagementUnknown:
            assistant_message = "I couldn't find one current saved instruction with that stable identity, so I didn't change anything."
            error_class = "DeferredIntentUnknown"
        elif type(management) is DeferredIntentManagementBindingMismatch:
            assistant_message = "That saved-instruction selection did not match its server binding, so I didn't change anything."
            error_class = "DeferredIntentBindingMismatch"
        else:
            assistant_message = (
                "More than one saved instruction has that structure. Name the exact intent UUID so I can change the right one."
            )
            error_class = "DeferredIntentAmbiguous"
        unavailable = StepChatResult(
            assistant_message=assistant_message,
            status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
            latency_ms=chat.latency_ms,
            error_class=error_class,
        )
        return DeferredRequestUnchanged(guided=guided, chat=unavailable)
    if type(management) is DeferredIntentManagementApplied:
        invalidated = guided.active_proposal
        prospective = guided if invalidated is not None else replace(guided, deferred_intents=management.deferred_intents)
        applied_chat = StepChatResult(
            assistant_message=management.assistant_message,
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=chat.latency_ms,
            error_class=None,
        )
        if type(action) is DeferredIntentCancelAction:
            return DeferredRequestCancelled(
                guided=prospective,
                chat=applied_chat,
                action=action,
                effective_intent=management.effective_intent,
                deferred_intents=management.deferred_intents,
                invalidated_active_proposal=invalidated,
            )
        if type(action) is not DeferredIntentEditAction:  # pragma: no cover - closed action union
            raise TypeError("deferred management action is outside the closed union")
        return DeferredRequestEdited(
            guided=prospective,
            chat=applied_chat,
            action=action,
            effective_intent=management.effective_intent,
            deferred_intents=management.deferred_intents,
            invalidated_active_proposal=invalidated,
        )
    rejected_chat = _deferred_disposition_chat(
        cast(
            DeferredIntentAccepted | DeferredIntentClarification | DeferredIntentUnsupported | DeferredIntentRejected,
            management,
        ),
        latency_ms=chat.latency_ms,
    )
    return DeferredRequestUnchanged(guided=guided, chat=rejected_chat)


def apply_deferred_request(
    deferred_action: DeferredIntentAction | None,
    management_action: DeferredIntentManagementAction | None,
    *,
    authority: DeferredRequestAuthority,
    chat: StepChatResult,
) -> DeferredRequestApplication:
    if deferred_action is not None:
        disposition = validate_deferred_intent_action(
            deferred_action,
            receiving_stage=_guided_stage_name(authority.guided.step),
            catalog=authority.catalog,
            guided=authority.guided,
        )
        resolved_chat = _deferred_disposition_chat(disposition, latency_ms=chat.latency_ms)
        if type(disposition) is not DeferredIntentAccepted:
            return DeferredRequestUnchanged(guided=authority.guided, chat=resolved_chat)
        retained = create_deferred_stage_intent(
            disposition.action,
            receiving_stage=_guided_stage_name(authority.guided.step),
            intent_id=str(authority.new_intent_id),
            originating_message_id=str(authority.originating_message.message_id),
            originating_message_content=authority.originating_message.content,
        )
        prospective = replace(authority.guided, deferred_intents=(*authority.guided.deferred_intents, retained))
        return DeferredRequestRetained(
            guided=prospective,
            chat=resolved_chat,
            retained_intent_id=authority.new_intent_id,
        )
    if management_action is not None:
        return _apply_deferred_management(
            management_action,
            guided=authority.guided,
            catalog=authority.catalog,
            originating_message=authority.originating_message,
            chat=chat,
        )
    return DeferredRequestUnchanged(guided=authority.guided, chat=chat)


def deferred_request_retained_intent_id(application: DeferredRequestApplication) -> UUID | None:
    if type(application) is DeferredRequestRetained:
        return application.retained_intent_id
    return None


def deferred_request_management(application: DeferredRequestApplication) -> DeferredRequestManaged | None:
    if type(application) is DeferredRequestCancelled:
        return application
    if type(application) is DeferredRequestEdited:
        return application
    return None


def _prepare_schema8_management_rewind(
    *,
    authority: ManagementRewindAuthority,
    managed_intent: DeferredStageIntent,
    managed_deferred_intents: tuple[DeferredStageIntent, ...],
    action: DeferredIntentManagementAction,
    invalidated_active_proposal: GuidedProposalRef | None,
) -> ManagementRewind:
    """Build the exact schema-8 rewind candidate without invoking the planner."""

    rewind_step = GuidedStep.STEP_2_SINK
    if not authority.prospective.history or authority.prospective.history[-1].response_hash is not None:
        raise AuditIntegrityError("schema-8 deferred management has no exact current turn to invalidate")
    response_payload = prepare_guided_json_payload(
        authority.payload_store,
        purpose="turn_response",
        payload={
            "action": "manage_deferred_intent",
            "management": "cancel" if type(action) is DeferredIntentCancelAction else "edit",
            "intent_id": managed_intent.intent_id,
            "target_stage": managed_intent.target_stage,
        },
    )
    answered = replace(
        authority.prospective.history[-1],
        response_hash=response_payload.payload_id,
        summary="Saved instruction changed; downstream guided review invalidated.",
    )
    invalidated_proposal = None
    if invalidated_active_proposal is not None:
        invalidated_proposal = GuidedPendingProposalInvalidation(
            proposal_id=invalidated_active_proposal.proposal_id,
            draft_hash=invalidated_active_proposal.draft_hash,
            reviewed_facts=guided_private_reviewed_facts(authority.current_guided),
        )
    rewound_guided = replace(
        authority.prospective,
        step=rewind_step,
        history=(*authority.prospective.history[:-1], answered),
        deferred_intents=managed_deferred_intents,
        active_proposal=None,
        active_edit_target=None,
    )
    rewound_state = replace(authority.current_state, guided_session=rewound_guided)
    review_turn = authority.guided_route._build_get_guided_turn(rewound_state, rewound_guided, catalog=authority.catalog)
    if review_turn is None:
        raise AuditIntegrityError("schema-8 deferred management rewind did not produce output review")
    review_turn = authority.guided_route._finalize_guided_turn(
        review_turn,
        shield_available=authority.shield_available,
    )
    rewound_guided, _record, _turn_type, next_payload = authority.guided_route._prepare_server_turn_occurrence(
        rewound_guided,
        current_step=rewind_step,
        turn=review_turn,
        payload_store=authority.payload_store,
    )
    return ManagementRewind(
        state=replace(authority.current_state, guided_session=rewound_guided),
        response_payload=response_payload,
        next_turn=review_turn,
        next_payload=next_payload,
        invalidated_proposal=invalidated_proposal,
    )


def maybe_prepare_schema8_management_rewind(
    *,
    authority: ManagementRewindAuthority,
    management: DeferredRequestManaged | None,
) -> ManagementRewind | None:
    if management is None:
        return None
    rewind_step = schema8_deferred_management_rewind_step(
        current_step=authority.prospective.step,
        target_stage=management.effective_intent.target_stage,
    )
    if rewind_step is None and management.invalidated_active_proposal is None:
        return None
    return _prepare_schema8_management_rewind(
        authority=authority,
        managed_intent=management.effective_intent,
        managed_deferred_intents=management.deferred_intents,
        action=management.action,
        invalidated_active_proposal=management.invalidated_active_proposal,
    )


__all__ = [
    "DeferredRequestAuthority",
    "DeferredRequestCancelled",
    "DeferredRequestEdited",
    "DeferredRequestRetained",
    "DeferredRequestUnchanged",
    "ManagementRewindAuthority",
    "apply_deferred_request",
    "deferred_request_management",
    "deferred_request_retained_intent_id",
    "maybe_prepare_schema8_management_rewind",
]
