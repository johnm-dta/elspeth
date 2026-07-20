"""Prepare bounded guided audit chat rows before opening a SQL transaction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_llm_audit import ComposerChatTurn, ComposerLLMCall, ComposerLLMCallStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.web.composer.audit import chat_turn_audit_envelope, llm_call_audit_envelope
from elspeth.web.composer.audit_storage import redacted_tool_invocation_content_and_envelope
from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import TerminalReason
from elspeth.web.composer.redaction import MANIFEST
from elspeth.web.sessions.protocol import (
    GUIDED_FAILURE_AUDIT_LINEAGE_KEY,
    GuidedFailureAuditLineage,
    PreparedGuidedAuditRow,
    PreparedGuidedJsonPayload,
)

_GUIDED_SYNTHETIC_TOOLS = frozenset(
    {
        "guided_turn_emitted",
        "guided_turn_answered",
        "guided_step_advanced",
        "guided_dropped_to_freeform",
        "guided_intent_cancelled",
    }
)
_ADVANCE_REASONS = frozenset({"user_advanced", "auto_advanced"})


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_canonical_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _valid_guided_synthetic_payload(tool_name: str, payload: object) -> bool:
    """Validate the exact safe-by-construction schemas of synthetic events."""

    if type(payload) is not dict:
        return False
    value: dict[str, Any] = payload
    steps = {member.value for member in GuidedStep}
    turns = {member.value for member in TurnType}
    if tool_name == "guided_turn_emitted":
        return (
            set(value) == {"step_index", "turn_type", "payload_hash", "payload_payload_id", "emitter"}
            and value["step_index"] in steps
            and value["turn_type"] in turns
            and _is_sha256(value["payload_hash"])
            and _is_sha256(value["payload_payload_id"])
            and value["emitter"] in {"server", "llm"}
        )
    if tool_name == "guided_turn_answered":
        required = {"step_index", "turn_type", "response_hash", "response_payload_id"}
        return (
            (set(value) == required or set(value) == required | {"control_signal"})
            and value["step_index"] in steps
            and value["turn_type"] in turns
            and _is_sha256(value["response_hash"])
            and _is_sha256(value["response_payload_id"])
            and ("control_signal" not in value or value["control_signal"] in {member.value for member in ControlSignal})
        )
    if tool_name == "guided_step_advanced":
        return (
            set(value) == {"prev_step", "next_step", "reason"}
            and value["prev_step"] in steps
            and value["next_step"] in steps
            and value["reason"] in _ADVANCE_REASONS
        )
    if tool_name == "guided_dropped_to_freeform":
        return (
            set(value) == {"prev_step", "drop_reason"}
            and value["prev_step"] in steps
            and value["drop_reason"] == TerminalReason.USER_PRESSED_EXIT.value
        )
    return (
        tool_name == "guided_intent_cancelled"
        and set(value) == {"intent_id", "receiving_stage", "target_stage"}
        and type(value["intent_id"]) is str
        and _is_canonical_uuid(value["intent_id"])
        and value["receiving_stage"] in {"source", "output", "topology", "wire_review"}
        and value["target_stage"] in {"source", "output", "topology", "wire_review"}
    )


def _omitted_success_invocation(invocation: ComposerToolInvocation) -> tuple[str, dict[str, Any]]:
    omitted = {"_redaction_status": "unmanifested_success_payload_omitted"}
    projection = deep_thaw(redacted_tool_invocation_content_and_envelope(invocation)[1]["invocation"])
    projection["arguments_canonical"] = canonical_json(omitted)
    projection["arguments_hash"] = stable_hash(omitted)
    projection["result_canonical"] = None
    projection["result_hash"] = None
    projection["error_message"] = None
    return (
        json.dumps({"_kind": "guided_tool_audit", "status": invocation.status.value, "tool_name": invocation.tool_name}),
        {"_kind": "audit", "invocation": projection},
    )


def is_authentic_guided_synthetic_invocation(invocation: ComposerToolInvocation) -> bool:
    """Return whether an invocation is an exact server synthetic event."""

    if type(invocation) is not ComposerToolInvocation or invocation.status is not ComposerToolStatus.SUCCESS:
        return False
    try:
        arguments = json.loads(invocation.arguments_canonical)
    except (TypeError, ValueError):
        return False
    return (
        invocation.tool_name in _GUIDED_SYNTHETIC_TOOLS
        and stable_hash(arguments) == invocation.arguments_hash
        and invocation.result_canonical is None
        and invocation.result_hash is None
        and invocation.error_class is None
        and invocation.error_message is None
        and invocation.version_after == invocation.version_before
        and _valid_guided_synthetic_payload(invocation.tool_name, arguments)
    )


def prepare_guided_audit_rows(
    *,
    invocations: tuple[ComposerToolInvocation, ...],
    llm_calls: tuple[ComposerLLMCall, ...],
    chat_turns: tuple[ComposerChatTurn, ...],
) -> tuple[PreparedGuidedAuditRow, ...]:
    """Apply existing redaction/public projections to all three audit channels."""

    if type(invocations) is not tuple or any(type(item) is not ComposerToolInvocation for item in invocations):
        raise TypeError("invocations must be an exact tuple[ComposerToolInvocation, ...]")
    if type(llm_calls) is not tuple or any(type(item) is not ComposerLLMCall for item in llm_calls):
        raise TypeError("llm_calls must be an exact tuple[ComposerLLMCall, ...]")
    if type(chat_turns) is not tuple or any(type(item) is not ComposerChatTurn for item in chat_turns):
        raise TypeError("chat_turns must be an exact tuple[ComposerChatTurn, ...]")

    rows: list[PreparedGuidedAuditRow] = []
    for invocation in invocations:
        content, envelope = redacted_tool_invocation_content_and_envelope(invocation)
        if invocation.status is not ComposerToolStatus.SUCCESS:
            omitted_arguments = canonical_json({"_redaction_status": "guided_failure_payload_omitted"})
            invocation_projection = deep_thaw(envelope["invocation"])
            invocation_projection["arguments_canonical"] = omitted_arguments
            invocation_projection["arguments_hash"] = stable_hash({"_redaction_status": "guided_failure_payload_omitted"})
            invocation_projection["result_canonical"] = None
            invocation_projection["result_hash"] = None
            invocation_projection["error_message"] = None
            envelope = {"_kind": "audit", "invocation": invocation_projection}
            content = json.dumps(
                {
                    "_kind": "guided_tool_failure_audit",
                    "status": invocation.status.value,
                    "error_class": invocation.error_class,
                }
            )
        elif invocation.tool_name not in MANIFEST:
            if not is_authentic_guided_synthetic_invocation(invocation):
                content, envelope = _omitted_success_invocation(invocation)
        rows.append(PreparedGuidedAuditRow(kind="tool", content=content, envelope=envelope))
    for call in llm_calls:
        content = json.dumps(
            {
                "_kind": "llm_call_audit",
                "status": call.status.value,
                "model_requested": call.model_requested,
                "model_returned": call.model_returned,
                "total_tokens": call.total_tokens,
                "reasoning_tokens": call.reasoning_tokens,
                "provider_cost": call.provider_cost,
            }
        )
        envelope = llm_call_audit_envelope(call)
        if call.status is not ComposerLLMCallStatus.SUCCESS:
            public_call = deep_thaw(envelope["call"])
            public_call["error_message"] = None
            public_call["reasoning_content"] = None
            public_call["reasoning_details"] = None
            public_call["thinking_blocks"] = None
            envelope = {"_kind": "llm_call_audit", "call": public_call}
        rows.append(
            PreparedGuidedAuditRow(
                kind="llm",
                content=content,
                envelope=envelope,
            )
        )
    for turn in chat_turns:
        content = json.dumps(
            {
                "_kind": "chat_turn_audit",
                "status": turn.status.value,
                "step": turn.step,
                "initiator": turn.initiator.value,
                "chat_turn_seq": turn.chat_turn_seq,
                "model": turn.model,
                "latency_ms": turn.latency_ms,
                "error_class": turn.error_class,
            }
        )
        rows.append(
            PreparedGuidedAuditRow(
                kind="chat",
                content=content,
                envelope=chat_turn_audit_envelope(turn),
            )
        )
    return tuple(rows)


def bind_guided_failure_audit_rows(
    rows: tuple[PreparedGuidedAuditRow, ...],
    *,
    lineage: GuidedFailureAuditLineage,
) -> tuple[PreparedGuidedAuditRow, ...]:
    """Bind every sanitized failure-evidence row to one terminal event tuple."""

    if type(rows) is not tuple or any(type(row) is not PreparedGuidedAuditRow for row in rows):
        raise TypeError("rows must be an exact tuple[PreparedGuidedAuditRow, ...]")
    if type(lineage) is not GuidedFailureAuditLineage:
        raise TypeError("lineage must be an exact GuidedFailureAuditLineage")
    bound: list[PreparedGuidedAuditRow] = []
    lineage_envelope = lineage.envelope()
    for row in rows:
        try:
            content = json.loads(row.content)
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("guided failure audit content is not a JSON object") from exc
        envelope = deep_thaw(row.envelope)
        if type(content) is not dict or type(envelope) is not dict:
            raise AuditIntegrityError("guided failure audit row must contain exact JSON objects")
        if GUIDED_FAILURE_AUDIT_LINEAGE_KEY in content or GUIDED_FAILURE_AUDIT_LINEAGE_KEY in envelope:
            raise AuditIntegrityError("guided failure audit evidence may not supply its own lineage")
        content[GUIDED_FAILURE_AUDIT_LINEAGE_KEY] = lineage_envelope
        envelope[GUIDED_FAILURE_AUDIT_LINEAGE_KEY] = lineage_envelope
        bound.append(
            PreparedGuidedAuditRow(
                kind=row.kind,
                content=json.dumps(content),
                envelope=envelope,
            )
        )
    return tuple(bound)


def validate_guided_audit_payload_references(
    rows: tuple[PreparedGuidedAuditRow, ...],
    payloads: tuple[PreparedGuidedJsonPayload, ...],
) -> None:
    """Require every synthetic payload reference to bind the verified set."""

    by_id = {payload.payload_id: payload for payload in payloads}
    for row in rows:
        if row.kind != "tool":
            continue
        invocation = row.envelope.get("invocation")
        if not isinstance(invocation, Mapping):
            raise AuditIntegrityError("guided tool audit invocation envelope is malformed")
        tool_name = invocation.get("tool_name")
        if tool_name not in {"guided_turn_emitted", "guided_turn_answered"}:
            continue
        raw_arguments = invocation.get("arguments_canonical")
        if type(raw_arguments) is not str:
            raise AuditIntegrityError("guided synthetic audit arguments are malformed")
        arguments = json.loads(raw_arguments)
        if tool_name == "guided_turn_emitted":
            payload_id = arguments["payload_payload_id"]
            expected_purpose = "turn"
            hash_field = "payload_hash"
        else:
            payload_id = arguments["response_payload_id"]
            expected_purpose = "turn_response"
            hash_field = "response_hash"
        payload = by_id.get(payload_id)
        if payload is None or payload.purpose != expected_purpose or arguments[hash_field] != payload_id:
            raise AuditIntegrityError("guided synthetic audit payload reference is absent or purpose-mismatched")


__all__ = [
    "bind_guided_failure_audit_rows",
    "prepare_guided_audit_rows",
    "validate_guided_audit_payload_references",
]
