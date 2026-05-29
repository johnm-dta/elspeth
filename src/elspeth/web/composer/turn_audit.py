"""Turn-audit persistence phase (P4) for the composer compose loop.

Extracted verbatim from ComposerServiceImpl._persist_turn_audit (service.py)
to reduce the god-class surface. The logic is UNCHANGED; the enclosing
self reference is made explicit via the ``service`` parameter.

Behaviour-preservation contract: the redaction + persist sub-steps are
identical to the pre-extraction method. Pinned by the tests that exercise
_persist_turn_audit call sites (compose-loop integration tests).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts.errors import AuditIntegrityError, FailedTurnMetadata
from elspeth.web.composer._compose_loop_carriers import _PersistOutcome
from elspeth.web.composer.protocol import ComposerPluginCrashError
from elspeth.web.composer.service import _INVALID_TOOL_ARGUMENTS_REDACTION_STATUS
from elspeth.web.sessions._persist_payload import RedactedToolRow, _ToolOutcome

if TYPE_CHECKING:
    from elspeth.web.composer.service import ComposerServiceImpl


async def persist_turn_audit(
    service: ComposerServiceImpl,
    *,
    tool_outcomes: tuple[_ToolOutcome, ...],
    decoded_args_by_call_id: Mapping[str, Mapping[str, Any]],
    assistant_message: Any,
    raw_assistant_content: str | None,
    assistant_tool_calls: tuple[Any, ...],
    plugin_crash: ComposerPluginCrashError | None,
    session_id: str | None,
    current_state_id: str | None,
    persisted_tool_call_turn: bool,
    persisted_assistant_message_id: str | None,
) -> _PersistOutcome:
    """Phase P4 of the compose loop — redact then persist the turn audit.

    Two sub-steps that share an invariant (a mid-step raise must leave
    the DB in its pre-step shape):

    1. **Redaction (pure / async).** Walks ``tool_outcomes`` via the
       redaction manifest and builds the immutable
       ``redacted_assistant_tool_calls`` / ``redacted_tool_rows`` shapes
       the persister expects.
    2. **Persistence.** Calls
       ``sessions_service.persist_compose_turn_async`` exactly once
       when ``session_id`` is set. The AuditIntegrityError catch
       stamps ``failed_turn`` with ``tool_responses_persisted=0`` and
       re-raises so the route handler sees the partial-write story.
       Unwind-audit invariants are checked after the persist returns;
       failures raise additional AuditIntegrityError(s).

    Plugin-crash propagation (the post-persist re-raise of a
    ``ComposerPluginCrashError`` captured in P3) is intentionally
    *not* in this helper: the driver decides whether to raise based
    on ``dispatch.plugin_crash is not None`` so the carrier never
    carries a "post-crash" disposition.
    """
    from pydantic import ValidationError as PydanticValidationError

    from elspeth.contracts.freeze import deep_thaw
    from elspeth.web.composer.redaction import MANIFEST, redact_tool_call_arguments

    phase3_self = cast(Any, service)
    redaction_telemetry = phase3_self._redaction_telemetry
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...] = ()
    for tool_outcome in tool_outcomes:
        tc = tool_outcome.call
        decoded_args: dict[str, Any]
        if tc.id in decoded_args_by_call_id:
            # deep_thaw restores plain dict/list types from any
            # MappingProxyType / tuple introduced by the carrier's
            # freeze_fields contract. redact_tool_call_arguments has a
            # ``dict[str, Any]`` signature and json.dumps cannot serialise
            # MappingProxyType, so the thaw here is load-bearing once
            # _DispatchOutcome carries frozen args.
            decoded_args = deep_thaw(decoded_args_by_call_id[tc.id])
        elif tool_outcome.error_class is not None:
            decoded_args = {
                "_redaction_status": _INVALID_TOOL_ARGUMENTS_REDACTION_STATUS,
                "error_class": tool_outcome.error_class,
            }
        else:
            decoded_args = {"_raw_arguments": tc.function.arguments}
        if tc.function.name in MANIFEST:
            try:
                persisted_arguments = redact_tool_call_arguments(
                    tc.function.name,
                    decoded_args,
                    telemetry=redaction_telemetry,
                )
            except PydanticValidationError:
                if tool_outcome.error_class is None:
                    raise
                persisted_arguments = {
                    "_redaction_status": _INVALID_TOOL_ARGUMENTS_REDACTION_STATUS,
                    "error_class": tool_outcome.error_class,
                }
        else:
            # Unknown tool names are Tier-3 LLM hallucinations handled
            # by execute_tool as a semantic failure ToolResult. The
            # manifest is intentionally closed, so do not call the
            # walker for names it cannot know about.
            persisted_arguments = decoded_args
        redacted_assistant_tool_calls = (
            *redacted_assistant_tool_calls,
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": json.dumps(persisted_arguments),
                },
            },
        )
    redacted_tool_rows = tuple(
        RedactedToolRow(
            tool_call_id=tool_outcome.call.id,
            content=phase3_self._serialize_response_via_walker(tool_outcome, telemetry=redaction_telemetry),
            composition_state_payload=(
                phase3_self._state_payload_for_compose_turn_for_test(tool_outcome.response)
                if tool_outcome.post_version > tool_outcome.pre_version
                else None
            ),
        )
        for tool_outcome in tool_outcomes
    )
    service._phase3_last_redacted_assistant_tool_calls = redacted_assistant_tool_calls
    service._phase3_last_redacted_tool_rows = redacted_tool_rows
    failed_turn: FailedTurnMetadata | None = None
    if session_id is not None:
        sessions_service = service._require_sessions_service()
        try:
            audit_outcome = await sessions_service.persist_compose_turn_async(
                session_id=session_id,
                assistant_content=assistant_message.content or "",
                raw_content=raw_assistant_content,
                redacted_assistant_tool_calls=redacted_assistant_tool_calls,
                redacted_tool_rows=redacted_tool_rows,
                parent_composition_state_id=current_state_id,
                expected_current_state_id=current_state_id,
                writer_principal="compose_loop",
                plugin_crash_pending=plugin_crash is not None,
            )
        except AuditIntegrityError as exc:
            exc.failed_turn = FailedTurnMetadata(
                assistant_message_id=None,
                tool_calls_attempted=len(assistant_tool_calls),
                tool_responses_persisted=0,
            )
            raise
        service._phase3_last_audit_outcome = audit_outcome
        current_state_id = audit_outcome.current_state_id
        failed_turn = FailedTurnMetadata(
            assistant_message_id=audit_outcome.assistant_id,
            tool_calls_attempted=len(assistant_tool_calls),
        )
        if audit_outcome.assistant_id is None and plugin_crash is None:
            raise AuditIntegrityError(
                "persist_compose_turn_async returned unwind_audit_failed without an in-flight plugin crash",
                failed_turn=failed_turn,
            )
        if audit_outcome.assistant_id is None and not audit_outcome.unwind_audit_failed:
            raise AuditIntegrityError(
                "persist_compose_turn_async returned no assistant id without the unwind-audit-failed disposition",
                failed_turn=failed_turn,
            )
        persisted_assistant_message_id = audit_outcome.assistant_id
        persisted_tool_call_turn = True
    return _PersistOutcome(
        current_state_id=current_state_id,
        persisted_assistant_message_id=persisted_assistant_message_id,
        persisted_tool_call_turn=persisted_tool_call_turn,
        failed_turn=failed_turn,
    )
