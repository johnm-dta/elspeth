"""Pure projection of immutable guided settlement state onto strict responses."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.payload_store import IntegrityError as PayloadIntegrityError
from elspeth.contracts.payload_store import PayloadNotFoundError, PayloadStore
from elspeth.web.composer.guided.profile import EMPTY_PROFILE
from elspeth.web.composer.guided.protocol import ChatRole, GuidedStep, validate_current_turn
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.redaction import redact_guided_snapshot_storage_paths, redact_source_storage_path
from elspeth.web.sessions.protocol import (
    CompositionProposalRecord,
    CompositionStateData,
    CompositionStateRecord,
    GuidedJsonPayloadPurpose,
    GuidedResponseDescriptor,
    PreparedGuidedJsonPayload,
)
from elspeth.web.sessions.schemas import (
    ChatTurnResponse,
    CompositionProposalResponse,
    CompositionStateResponse,
    GetGuidedResponse,
    GuidedChatResponse,
    GuidedRespondResponse,
    GuidedSessionResponse,
    PipelineProposalMetadataResponse,
    PluginPolicyFindingResponse,
    TerminalStateResponse,
    TurnPayloadResponse,
    TurnRecordResponse,
    WorkflowProfileResponse,
)

GUIDED_REPLAY_META_KEY = "guided_operation_replay"
_GUIDED_INVALID_STATUS = ("guided_composition_invalid",)


def project_composition_proposal(record: CompositionProposalRecord) -> CompositionProposalResponse:
    """Project one immutable proposal record to its ordinary strict wire body."""

    metadata = record.pipeline_metadata
    return CompositionProposalResponse(
        id=str(record.id),
        session_id=str(record.session_id),
        tool_call_id=record.tool_call_id,
        tool_name=record.tool_name,
        status=record.status,
        summary=record.summary,
        rationale=record.rationale,
        affects=list(record.affects),
        arguments_redacted_json=deep_thaw(record.arguments_redacted_json),
        base_state_id=str(record.base_state_id) if record.base_state_id is not None else None,
        committed_state_id=str(record.committed_state_id) if record.committed_state_id is not None else None,
        audit_event_id=str(record.audit_event_id) if record.audit_event_id is not None else None,
        pipeline_metadata=(
            PipelineProposalMetadataResponse(
                surface=metadata.surface,
                draft_hash=metadata.draft_hash,
                base=deep_thaw(metadata.base),
                reviewed_anchor_hash=metadata.reviewed_anchor_hash,
                repair_count=metadata.repair_count,
                skill_hash=metadata.skill_hash,
                audit_payload_hash=metadata.audit_payload_hash,
                custody_result=metadata.custody_result,
            )
            if metadata is not None
            else None
        ),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def guided_validation_errors(*, is_valid: bool) -> tuple[str, ...] | None:
    """Return the closed persisted validity status for a guided state."""

    if type(is_valid) is not bool:
        raise TypeError("is_valid must be an exact bool")
    return None if is_valid else _GUIDED_INVALID_STATUS


def validation_errors_for_composer_surface(
    *,
    composer_meta: Mapping[str, Any] | None,
    is_valid: bool,
    validation_errors: Sequence[str] | None,
) -> Sequence[str] | None:
    """Close validator text only for states that carry guided custody."""

    if composer_meta is not None and ("guided_session" in composer_meta or GUIDED_REPLAY_META_KEY in composer_meta):
        return guided_validation_errors(is_valid=is_valid)
    return validation_errors


_GUIDED_STEP_INDEX = {
    GuidedStep.STEP_1_SOURCE: 0,
    GuidedStep.STEP_2_SINK: 1,
    GuidedStep.STEP_3_TRANSFORMS: 2,
    GuidedStep.STEP_4_WIRE: 3,
}


def guided_turn_token(guided: GuidedSession) -> str:
    """Bind the current unanswered turn to its persisted history occurrence."""

    if type(guided) is not GuidedSession:
        raise TypeError("guided must be an exact GuidedSession")
    if not guided.history:
        raise AuditIntegrityError("Guided turn token requires a persisted turn record")
    history_index = len(guided.history) - 1
    record = guided.history[history_index]
    if (
        record.step is not guided.step
        or record.response_hash is not None
        or any(previous.response_hash is None for previous in guided.history[:-1])
    ):
        raise AuditIntegrityError("Guided turn token requires the final current unanswered turn")
    return stable_hash(
        {
            "schema": "guided.turn-token.v1",
            "history_index": history_index,
            "step": record.step.value,
            "turn_type": record.turn_type.value,
            "payload_hash": record.payload_hash,
        }
    )


def load_guided_json_payload(
    payload_store: PayloadStore,
    *,
    payload_id: str,
    purpose: GuidedJsonPayloadPurpose,
) -> PreparedGuidedJsonPayload:
    """Load and fully revalidate one canonical guided JSON payload for replay."""

    if not isinstance(payload_store, PayloadStore):
        raise TypeError("payload_store must implement PayloadStore")
    if type(payload_id) is not str or len(payload_id) != 64 or any(char not in "0123456789abcdef" for char in payload_id):
        raise AuditIntegrityError("Guided replay payload id is malformed")
    if purpose not in {"turn", "turn_response"}:
        raise AuditIntegrityError("Guided replay payload purpose is outside the closed vocabulary")
    try:
        raw = payload_store.retrieve(payload_id)
    except (PayloadNotFoundError, PayloadIntegrityError) as exc:
        raise AuditIntegrityError("Guided replay payload is unavailable or corrupt") from exc
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), payload_id):
        raise AuditIntegrityError("Guided replay payload bytes do not match the content id")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        envelope = json.loads(decoded)
        canonical = canonical_json(envelope).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise AuditIntegrityError("Guided replay payload is not valid canonical JSON") from exc
    if type(envelope) is not dict or set(envelope) != {"schema", "purpose", "payload"}:
        raise AuditIntegrityError("Guided replay payload envelope is malformed")
    if envelope["schema"] != "guided.json-payload.v1" or envelope["purpose"] != purpose:
        raise AuditIntegrityError("Guided replay payload schema or purpose does not match its reference")
    payload = envelope["payload"]
    if type(payload) is not dict or not hmac.compare_digest(raw, canonical):
        raise AuditIntegrityError("Guided replay payload is not a canonical JSON object")
    return PreparedGuidedJsonPayload(payload_id=payload_id, purpose=purpose, payload=payload)


def with_guided_response_descriptor(
    state: CompositionStateData,
    descriptor: GuidedResponseDescriptor,
) -> CompositionStateData:
    """Replace inherited replay metadata with the current operation descriptor."""

    if type(state) is not CompositionStateData:
        raise TypeError("state must be an exact CompositionStateData")
    if type(descriptor) is not GuidedResponseDescriptor:
        raise TypeError("descriptor must be an exact GuidedResponseDescriptor")
    composer_meta = deep_thaw(state.composer_meta) if state.composer_meta is not None else {}
    composer_meta[GUIDED_REPLAY_META_KEY] = descriptor.to_dict()
    # Free-form validator messages can echo paths, credentials, and provider
    # diagnostics. Persist only the closed guided status while retaining the
    # truthful relationship between ``is_valid`` and ``validation_errors``.
    validation_errors = guided_validation_errors(is_valid=state.is_valid)
    return replace(state, composer_meta=composer_meta, validation_errors=validation_errors)


def parse_guided_response_descriptor(record: CompositionStateRecord) -> GuidedResponseDescriptor:
    """Parse the closed replay descriptor from an immutable state row."""

    if record.composer_meta is None:
        raise AuditIntegrityError("Guided result state has no composer metadata")
    composer_meta = deep_thaw(record.composer_meta)
    raw = composer_meta.get(GUIDED_REPLAY_META_KEY)
    if not isinstance(raw, Mapping):
        raise AuditIntegrityError("Guided result state has no valid replay descriptor")
    return GuidedResponseDescriptor.from_dict(raw)


def _guided_session(record: CompositionStateRecord) -> GuidedSession:
    if record.composer_meta is None:
        raise AuditIntegrityError("Guided result state has no composer metadata")
    raw = deep_thaw(record.composer_meta).get("guided_session")
    if type(raw) is not dict:
        raise AuditIntegrityError("Guided result state has no guided checkpoint")
    return GuidedSession.from_dict(raw)


def _terminal_response(guided: GuidedSession) -> TerminalStateResponse | None:
    terminal = guided.terminal
    if terminal is None:
        return None
    return TerminalStateResponse(
        kind=terminal.kind.value,
        reason=terminal.reason.value if terminal.reason is not None else None,
        pipeline_yaml=terminal.pipeline_yaml,
    )


def _profile_response(guided: GuidedSession) -> WorkflowProfileResponse | None:
    if guided.profile == EMPTY_PROFILE:
        return None
    return WorkflowProfileResponse(
        coaching=guided.profile.coaching,
        bookends=guided.profile.bookends,
        advisor_checkpoints=guided.profile.advisor_checkpoints,
    )


def _guided_session_response(guided: GuidedSession) -> GuidedSessionResponse:
    return GuidedSessionResponse(
        step=guided.step.value,
        history=[
            TurnRecordResponse(
                step=record.step.value,
                turn_type=record.turn_type.value,
                payload_hash=record.payload_hash,
                response_hash=record.response_hash,
                summary=record.summary,
                emitter=record.emitter,
            )
            for record in guided.history
        ],
        terminal=_terminal_response(guided),
        chat_history=[
            ChatTurnResponse(
                role=turn.role.value,
                content=turn.content,
                seq=turn.seq,
                step=turn.step.value,
                ts_iso=turn.ts_iso,
                assistant_message_kind=turn.assistant_message_kind,
                synthetic_failure_reason=turn.synthetic_failure_reason,
            )
            for turn in guided.chat_history
        ],
        chat_turn_seq=guided.chat_turn_seq,
        profile=_profile_response(guided),
    )


def _composition_state_response(
    state: CompositionStateRecord,
    descriptor: GuidedResponseDescriptor,
) -> CompositionStateResponse:
    sources = deep_thaw(state.sources)
    if sources is not None:
        sources = redact_source_storage_path({"sources": sources})["sources"]
    composer_meta = deep_thaw(state.composer_meta) if state.composer_meta is not None else None
    sources, composer_meta = redact_guided_snapshot_storage_paths(sources, composer_meta)
    expected_errors = guided_validation_errors(is_valid=state.is_valid)
    if state.validation_errors != expected_errors:
        raise AuditIntegrityError("Guided result state has an invalid closed validation status")
    return CompositionStateResponse(
        id=str(state.id),
        session_id=str(state.session_id),
        version=state.version,
        sources=sources,
        nodes=deep_thaw(state.nodes),
        edges=deep_thaw(state.edges),
        outputs=deep_thaw(state.outputs),
        metadata=deep_thaw(state.metadata_),
        is_valid=state.is_valid,
        validation_errors=list(expected_errors) if expected_errors is not None else None,
        validation_warnings=None,
        validation_suggestions=None,
        derived_from_state_id=str(state.derived_from_state_id) if state.derived_from_state_id is not None else None,
        created_at=state.created_at,
        composer_meta=composer_meta,
        plugin_policy_findings=[
            PluginPolicyFindingResponse(
                component_id=finding.component_id,
                plugin_id=str(finding.plugin_id),
                reason_code=finding.reason_code.value,
                snapshot_fingerprint=finding.snapshot_fingerprint,
            )
            for finding in descriptor.policy_findings
        ],
    )


def _turn_response(
    guided: GuidedSession,
    descriptor: GuidedResponseDescriptor,
    payloads: Sequence[PreparedGuidedJsonPayload],
) -> TurnPayloadResponse | None:
    turn = descriptor.next_turn
    if turn is None:
        return None
    matches = [payload for payload in payloads if payload.payload_id == turn.payload_id]
    if len(matches) != 1 or matches[0].purpose != "turn":
        raise AuditIntegrityError("Guided replay turn payload is absent, duplicated, or purpose-mismatched")
    if not guided.history:
        raise AuditIntegrityError("Guided replay turn has no matching persisted turn record")
    turn_record = guided.history[-1]
    if (
        turn_record.turn_type is not turn.turn_type
        or _GUIDED_STEP_INDEX[turn_record.step] != turn.step_index
        or turn_record.payload_hash != turn.payload_id
    ):
        raise AuditIntegrityError("Guided replay turn does not match the persisted turn record")
    if turn_record.step is not guided.step or turn_record.response_hash is not None:
        raise AuditIntegrityError("Guided replay turn record is not the final current unanswered turn")
    projected_turn = {
        "type": turn.turn_type.value,
        "step_index": turn.step_index,
        "payload": matches[0].payload,
    }
    try:
        validate_current_turn(turn_record.step, projected_turn)
    except ValueError as exc:
        raise AuditIntegrityError(f"Guided replay current-schema turn is invalid: {exc}") from exc
    return TurnPayloadResponse(
        type=turn.turn_type.value,
        step_index=turn.step_index,
        turn_token=guided_turn_token(guided),
        payload=deep_thaw(matches[0].payload),
    )


def project_guided_response(
    state: CompositionStateRecord,
    *,
    payloads: Sequence[PreparedGuidedJsonPayload],
) -> GuidedRespondResponse | GuidedChatResponse | GetGuidedResponse:
    """Construct the exact strict response committed by a guided operation."""

    descriptor = parse_guided_response_descriptor(state)
    guided = _guided_session(state)
    next_turn = _turn_response(guided, descriptor, payloads)
    terminal = _terminal_response(guided)
    if terminal is not None and next_turn is not None:
        raise AuditIntegrityError("Terminal guided replay cannot also carry a next turn")
    guided_response = _guided_session_response(guided)
    state_response = _composition_state_response(state, descriptor)
    if descriptor.kind == "guided_respond":
        return GuidedRespondResponse(
            guided_session=guided_response,
            next_turn=next_turn,
            terminal=terminal,
            composition_state=state_response,
        )

    if descriptor.kind == "guided_reenter":
        return GetGuidedResponse(
            guided_session=guided_response,
            next_turn=next_turn,
            terminal=terminal,
            composition_state=state_response,
        )

    if descriptor.assistant_turn_seq is None:
        raise AuditIntegrityError("Guided chat replay descriptor has no assistant sequence")
    matches = [turn for turn in guided.chat_history if turn.seq == descriptor.assistant_turn_seq]
    if len(matches) != 1 or matches[0].role is not ChatRole.ASSISTANT or matches[0] is not guided.chat_history[-1]:
        raise AuditIntegrityError("Guided chat replay assistant sequence does not identify the final assistant turn")
    assistant_turn = matches[0]
    if assistant_turn.assistant_message_kind not in {"assistant", "synthetic_failure"}:
        raise AuditIntegrityError("Guided chat replay assistant turn has no exact message kind")
    return GuidedChatResponse(
        assistant_message=assistant_turn.content,
        assistant_message_kind=assistant_turn.assistant_message_kind,
        guided_session=guided_response,
        next_turn=next_turn,
        terminal=terminal,
        composition_state=state_response,
    )


def guided_response_projection_hash(response: BaseModel) -> str:
    """Hash one strictly revalidated guided response projection."""

    config = type(response).model_config
    if config.get("strict") is not True or config.get("extra") != "forbid":
        raise AuditIntegrityError("Guided operation replay requires a strict, extra-forbid response DTO")
    strict_response = type(response).model_validate(response.model_dump(mode="python"), strict=True)
    return stable_hash(strict_response.model_dump(mode="json"))


def response_json(response: BaseModel) -> Mapping[str, Any]:
    """Return the exact JSON representation used by the replay hash."""

    return cast(Mapping[str, Any], response.model_dump(mode="json"))


__all__ = [
    "GUIDED_REPLAY_META_KEY",
    "guided_response_projection_hash",
    "guided_turn_token",
    "guided_validation_errors",
    "load_guided_json_payload",
    "parse_guided_response_descriptor",
    "project_composition_proposal",
    "project_guided_response",
    "response_json",
    "validation_errors_for_composer_surface",
    "with_guided_response_descriptor",
]
