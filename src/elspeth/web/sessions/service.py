"""SessionService implementation -- CRUD, state versioning, active run enforcement.

Uses SQLAlchemy Core with a synchronous engine. Database calls run in a
thread pool executor to avoid blocking the async event loop.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from uuid import UUID

import structlog
from opentelemetry import metrics
from sqlalchemy import ColumnElement, Connection, Engine, delete, desc, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from elspeth.contracts.auth import AuthProviderType
from elspeth.contracts.blobs_inline import ResolvedBlobContent
from elspeth.contracts.composer_audit import ComposerToolStatus, PipelineDispatchAuditPayload
from elspeth.contracts.composer_interpretation import (
    INTERPRETATION_HASH_DOMAIN_V2,
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.composer.pipeline_commit import PipelineDispatchAuditBinding
from elspeth.web.composer.pipeline_planner import PipelinePlanResult
from elspeth.web.composer.pipeline_proposal import (
    AbsentBase,
    PipelineProposal,
    PlannerSurface,
    PresentBase,
    composition_content_hash,
    reviewed_anchor_hash,
)
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

# Phase 8 cohort-emit helper (Sub-task 7e — B3 cohort b1). The opt-out
# audit row is committed inside ``record_session_interpretation_opt_out``
# below, and the helper must fire only on the INSERT path (not the F-29
# idempotent re-fire). The helper module lives under ``web/composer``
# per project plan; the sessions→composer import direction follows the
# precedent set by ``_auto_title.py``, ``_guided_step_chat.py``, and
# ``converters.py``. The W5 try/except wrap inside the helper preserves
# the audit-primacy rule (a broken OTel exporter must not 500 a POST
# whose audit row already wrote).
from elspeth.web.composer.telemetry_phase8 import record_interpretation_opt_out
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PENDING_INTERPRETATION_AUTHORING_TEXT,
    PROMPT_TEMPLATE_PARTS_KEY,
    SOURCE_AUTHORING_KEY,
    SOURCE_COMPONENT_ID,
    model_choice_artifact_hash,
    pipeline_decision_artifact_hash,
    prompt_structure_hash_from_options,
    validate_pipeline_decision_node_semantics,
)
from elspeth.web.sessions._persist_payload import AuditOutcome, RedactedToolRow, StatePayload
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.locking import (
    acquire_session_advisory_xact_lock,
    process_session_lock,
    sqlite_session_mutex,
    sqlite_transaction_session_lock,
)
from elspeth.web.sessions.models import (
    audit_access_log_table,
    blob_inline_resolutions_table,
    chat_messages_table,
    composer_completion_events_table,
    composition_proposals_table,
    composition_states_table,
    interpretation_events_table,
    proposal_events_table,
    run_events_table,
    runs_table,
    sessions_table,
    skill_markdown_history_table,
)
from elspeth.web.sessions.proposal_blob_refs import validate_proposal_blob_references
from elspeth.web.sessions.protocol import (
    AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST,
    AUDIT_GRADE_VIEW_WRITER_PRINCIPAL,
    LEGAL_RUN_TRANSITIONS,
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SESSION_RUN_EVENT_TYPE_VALUES,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    AuditAccessLogRecord,
    AuditAccessLogWriteError,
    AuthoritativeCompositionProposal,
    AuthoritativePipelineProposal,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageWriterPrincipal,
    ComposerDensityDefault,
    ComposerSessionPreferencesRecord,
    ComposerSessionPreferencesTransition,
    ComposerTrustMode,
    CompositionProposalRecord,
    CompositionStateData,
    CompositionStateProvenance,
    CompositionStateRecord,
    IllegalRunTransitionError,
    PipelineDispatchRecovery,
    PipelineProposalPublicMetadata,
    PipelineProposalRejectionReason,
    PipelineProposalSettlementResult,
    ProposalEventRecord,
    ProposalLifecycleStatus,
    RunAlreadyActiveError,
    RunEventRecord,
    RunRecord,
    SessionNotFoundError,
    SessionRecord,
    SessionRunEventType,
    SessionRunStatus,
    StaleComposeStateError,
    ToolCallIDMismatchError,
)
from elspeth.web.sessions.telemetry import _SessionsTelemetry
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE, _validate_accepted_value_content

if TYPE_CHECKING:
    from elspeth.web.catalog.protocol import CatalogService
    from elspeth.web.composer.state import CompositionState, ValidationSummary
    from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
    from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry

# Process-wide SQLite session-write lock registry.
#
# These three globals back ``_session_write_lock`` and
# ``_assert_session_write_lock_held`` on ``SessionServiceImpl``. They
# are MODULE-LEVEL ON PURPOSE -- not instance-level -- because the
# correctness contract is process-wide, not service-instance-wide.
#
# Why process-wide is required (do not refactor to instance state):
#   * ``run_sync_in_worker`` dispatches DB writes to a thread pool, so
#     two coroutines holding two different ``SessionServiceImpl``
#     instances against the same SQLite file MUST serialise on the
#     same lock or they race on the ``SELECT MAX(...) + 1``
#     allocator. Instance-local locks would let two services for the
#     same DB skip past each other's allocator reads.
#   * Multiple ``SessionServiceImpl`` instances against the same
#     engine URL are legal (the web app constructs them per request
#     scope in some configurations). The (database_url, session_id)
#     key in ``_SQLITE_SESSION_LOCKS`` is what makes process-wide
#     locking honour multi-instance scope without serialising
#     UNRELATED databases inside the same process.
#   * ``ContextVar`` is the standard Python primitive for per-task /
#     per-thread scoped state. ``_assert_session_write_lock_held``
#     reads it to verify the calling helper is inside a
#     ``_session_write_lock`` block without threading the lock state
#     through every function signature.
#
# Module-level mutable state is normally a smell; here it is the
# correct shape because the resource being guarded (a SQLite file on
# disk) is itself process-scoped. Refactoring to instance state would
# break the multi-instance correctness invariant the comment above
# describes.
_SESSION_WRITE_LOCK_HELD: ContextVar[frozenset[tuple[int, str]]] = ContextVar(
    "_SESSION_WRITE_LOCK_HELD",
    default=frozenset(),
)
_PROPOSAL_COMPOSER_PROVENANCE_FIELDS = (
    "composer_model_identifier",
    "composer_model_version",
    "composer_provider",
    "composer_skill_hash",
    "tool_arguments_hash",
)
_PIPELINE_CREATED_SCHEMA = "pipeline_proposal_created.v1"
_PIPELINE_CREATED_FIELDS = frozenset(
    {
        "schema",
        "tool_call_id",
        "tool_name",
        "status",
        "surface",
        "draft_hash",
        "base",
        "reviewed_anchor_hash",
        "repair_count",
        "skill_hash",
        "covered_deferred_intent_ids",
        "supersedes_draft_hash",
        "supersedes_proposal_id",
        "custody_result",
        "private_arguments_hash",
        "provenance_hash",
        "audit_payload_hash",
    }
)
_LEGACY_PROPOSAL_CREATED_FIELDS = frozenset({"tool_call_id", "tool_name", "status"})
_PIPELINE_METER = metrics.get_meter(__name__)
_PIPELINE_PLANNER_COUNTER = _PIPELINE_METER.create_counter("composer.pipeline_proposal.created_total")
_PIPELINE_CUSTODY_COUNTER = _PIPELINE_METER.create_counter("composer.pipeline_proposal.custody_total")
_PIPELINE_SETTLEMENT_COUNTER = _PIPELINE_METER.create_counter("composer.pipeline_proposal.settled_total")


class _PipelineCreatedEventPayload(TypedDict):
    schema: str
    tool_call_id: str
    tool_name: str
    status: str
    surface: str
    draft_hash: str
    base: dict[str, str]
    reviewed_anchor_hash: str
    repair_count: int
    skill_hash: str
    covered_deferred_intent_ids: list[str]
    supersedes_draft_hash: str | None
    supersedes_proposal_id: str | None
    custody_result: str
    private_arguments_hash: str
    provenance_hash: str
    audit_payload_hash: str


class _PipelineAcceptedEventPayload(TypedDict):
    schema: str
    tool_call_id: str
    tool_name: str
    status: str
    outcome: str
    draft_hash: str
    committed_state_id: str
    committed_state_content_hash: str
    final_composer_metadata_hash: str
    dispatch: PipelineDispatchAuditPayload


class _PipelineRejectedEventPayload(TypedDict):
    schema: str
    tool_call_id: str
    tool_name: str
    status: str
    outcome: str
    reason_code: PipelineProposalRejectionReason
    draft_hash: str
    dispatch: PipelineDispatchAuditPayload | None


def _pipeline_private_arguments_hash(arguments: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "schema": "composer.pipeline-proposal-private-arguments.v1",
            "arguments": deep_thaw(arguments),
        }
    )


def _pipeline_audit_payload_hash(
    *,
    summary: str,
    rationale: str,
    affects: Sequence[str],
    arguments_redacted_json: Mapping[str, Any],
) -> str:
    return stable_hash(
        {
            "schema": "composer.pipeline-proposal-audit-payload.v1",
            "summary": summary,
            "rationale": rationale,
            "affects": list(affects),
            "arguments_redacted_json": deep_thaw(arguments_redacted_json),
        }
    )


def _pipeline_provenance_hash(
    *,
    user_message_id: UUID | None,
    composer_model_identifier: str,
    composer_model_version: str,
    composer_provider: str,
    composer_skill_hash: str,
    tool_arguments_hash: str,
) -> str:
    return stable_hash(
        {
            "schema": "composer.pipeline-proposal-provenance.v1",
            "user_message_id": str(user_message_id) if user_message_id is not None else None,
            "composer_model_identifier": composer_model_identifier,
            "composer_model_version": composer_model_version,
            "composer_provider": composer_provider,
            "composer_skill_hash": composer_skill_hash,
            "tool_arguments_hash": tool_arguments_hash,
        }
    )


def _proposal_base_payload(base: AbsentBase | PresentBase) -> dict[str, str]:
    if type(base) is AbsentBase:
        return {"kind": "absent"}
    if type(base) is PresentBase:
        return {
            "kind": "present",
            "state_id": str(base.state_id),
            "composition_content_hash": base.composition_content_hash,
        }
    raise AuditIntegrityError("pipeline proposal base must be explicitly absent or present")


def _pipeline_created_payload(
    *,
    plan: PipelinePlanResult,
    user_message_id: UUID | None,
    composer_model_identifier: str,
    composer_model_version: str,
    composer_provider: str,
    summary: str,
    rationale: str,
    affects: Sequence[str],
    arguments_redacted_json: Mapping[str, Any],
    supersedes_proposal_id: UUID | None,
) -> _PipelineCreatedEventPayload:
    proposal = plan.proposal
    tool_arguments_hash = stable_hash(proposal.pipeline)
    payload: _PipelineCreatedEventPayload = {
        "schema": _PIPELINE_CREATED_SCHEMA,
        "tool_call_id": plan.tool_call_id,
        "tool_name": "set_pipeline",
        "status": "pending",
        "surface": proposal.surface.value,
        "draft_hash": proposal.draft_hash,
        "base": _proposal_base_payload(proposal.base),
        "reviewed_anchor_hash": proposal.reviewed_anchor_hash,
        "repair_count": proposal.repair_count,
        "skill_hash": proposal.skill_hash,
        "covered_deferred_intent_ids": list(proposal.covered_deferred_intent_ids),
        "supersedes_draft_hash": proposal.supersedes_draft_hash,
        "supersedes_proposal_id": str(supersedes_proposal_id) if supersedes_proposal_id is not None else None,
        "custody_result": plan.custody_result,
        "private_arguments_hash": _pipeline_private_arguments_hash(proposal.pipeline),
        "audit_payload_hash": _pipeline_audit_payload_hash(
            summary=summary,
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=arguments_redacted_json,
        ),
        "provenance_hash": _pipeline_provenance_hash(
            user_message_id=user_message_id,
            composer_model_identifier=composer_model_identifier,
            composer_model_version=composer_model_version,
            composer_provider=composer_provider,
            composer_skill_hash=proposal.skill_hash,
            tool_arguments_hash=tool_arguments_hash,
        ),
    }
    return payload


def _composition_state_data_content_hash(state: CompositionStateData) -> str:
    return stable_hash(
        {
            "sources": state.sources,
            "nodes": state.nodes,
            "edges": state.edges,
            "outputs": state.outputs,
            "metadata": state.metadata_,
        }
    )


def _final_composer_metadata_hash(metadata: Mapping[str, Any] | None) -> str:
    return stable_hash(
        {
            "schema": "composer.pipeline-final-metadata.v1",
            "metadata": deep_thaw(metadata),
        }
    )


def _pipeline_accepted_payload(
    *,
    authority: AuthoritativePipelineProposal,
    state_id: str,
    state_content_hash: str,
    final_composer_metadata: Mapping[str, Any] | None,
    dispatch: PipelineDispatchAuditBinding,
) -> _PipelineAcceptedEventPayload:
    return {
        "schema": "pipeline_proposal_accepted.v1",
        "tool_call_id": authority.row.tool_call_id,
        "tool_name": "set_pipeline",
        "status": "committed",
        "outcome": "accepted",
        "draft_hash": authority.proposal.draft_hash,
        "committed_state_id": state_id,
        "committed_state_content_hash": state_content_hash,
        "final_composer_metadata_hash": _final_composer_metadata_hash(final_composer_metadata),
        "dispatch": dispatch.to_dict(),
    }


_PIPELINE_REJECTION_REASONS = frozenset(
    {
        "operator_rejected",
        "candidate_executor_mismatch",
        "validation_failed",
        "policy_changed",
        "base_conflict",
        "request_cancelled",
        "superseded",
    }
)


def _validated_pipeline_rejection_reason(value: object) -> PipelineProposalRejectionReason:
    if type(value) is not str or value not in _PIPELINE_REJECTION_REASONS:
        raise AuditIntegrityError("pipeline proposal rejection reason is outside the closed vocabulary")
    return cast(PipelineProposalRejectionReason, value)


_PIPELINE_ACCEPTED_FIELDS = frozenset(
    {
        "schema",
        "tool_call_id",
        "tool_name",
        "status",
        "outcome",
        "draft_hash",
        "committed_state_id",
        "committed_state_content_hash",
        "final_composer_metadata_hash",
        "dispatch",
    }
)

_PIPELINE_REJECTED_FIELDS = frozenset(
    {
        "schema",
        "tool_call_id",
        "tool_name",
        "status",
        "outcome",
        "reason_code",
        "draft_hash",
        "dispatch",
    }
)


def _pipeline_rejected_payload(
    *,
    authority: AuthoritativePipelineProposal,
    reason: PipelineProposalRejectionReason,
    dispatch: PipelineDispatchAuditBinding | None,
) -> _PipelineRejectedEventPayload:
    reason = _validated_pipeline_rejection_reason(reason)
    outcome = "rejected" if reason == "operator_rejected" else "superseded" if reason == "superseded" else "failed"
    return {
        "schema": "pipeline_proposal_rejected.v1",
        "tool_call_id": authority.row.tool_call_id,
        "tool_name": "set_pipeline",
        "status": "rejected",
        "outcome": outcome,
        "reason_code": reason,
        "draft_hash": authority.proposal.draft_hash,
        "dispatch": dispatch.to_dict() if dispatch is not None else None,
    }


def _persisted_pipeline_dispatch_content_hashes(
    conn: Connection,
    *,
    session_id: str,
    dispatch: PipelineDispatchAuditBinding,
) -> tuple[str, ...]:
    """Return content hashes from exact durable redacted dispatch envelopes."""
    rows = conn.execute(select(chat_messages_table.c.tool_calls).where(chat_messages_table.c.session_id == session_id)).fetchall()
    matches: list[str] = []
    for row in rows:
        tool_calls = row.tool_calls
        if type(tool_calls) is not list:
            continue
        for envelope in tool_calls:
            try:
                recovery = _pipeline_dispatch_recovery_from_envelope(
                    envelope,
                    expected_tool_call_id=dispatch.tool_call_id,
                )
            except AuditIntegrityError as exc:
                raise AuditIntegrityError("pipeline dispatch audit evidence is malformed") from exc
            if recovery is None:
                continue
            if recovery.binding != dispatch:
                raise AuditIntegrityError("pipeline successful dispatch evidence does not match the terminal binding")
            matches.append(recovery.executor_content_hash)
    return tuple(matches)


def _pipeline_dispatch_recovery_from_envelope(
    envelope: object,
    *,
    expected_tool_call_id: str,
) -> PipelineDispatchRecovery | None:
    """Restore one successful same-call dispatch; ignore unrelated and failed attempts."""
    if type(envelope) is not dict:
        return None
    invocation = envelope.get("invocation")
    if type(invocation) is not dict or invocation.get("tool_call_id") != expected_tool_call_id:
        return None
    if envelope.get("_kind") != "audit":
        raise AuditIntegrityError("pipeline dispatch envelope kind is malformed")
    if invocation.get("tool_name") != "set_pipeline":
        raise AuditIntegrityError("pipeline dispatch call id is bound to a different tool")
    raw_status = invocation.get("status")
    if type(raw_status) is not str:
        raise AuditIntegrityError("pipeline dispatch status is malformed")
    try:
        status = ComposerToolStatus(raw_status)
    except (TypeError, ValueError) as exc:
        raise AuditIntegrityError("pipeline dispatch status is malformed") from exc
    if status is not ComposerToolStatus.SUCCESS:
        return None
    binding = PipelineDispatchAuditBinding.from_persisted_envelope(envelope)
    result_canonical = invocation.get("result_canonical")
    assert type(result_canonical) is str
    try:
        result_payload = json.loads(result_canonical)
    except json.JSONDecodeError as exc:
        raise AuditIntegrityError("pipeline dispatch result canonical is malformed") from exc
    if type(result_payload) is not dict:
        raise AuditIntegrityError("pipeline dispatch result payload is malformed")
    if result_payload.get("pipeline_content_hash_schema") != "composer.pipeline-dispatch-result.v1":
        raise AuditIntegrityError("pipeline dispatch result content schema is malformed")
    content_hash = result_payload.get("pipeline_content_hash")
    if type(content_hash) is not str or len(content_hash) != 64 or any(character not in "0123456789abcdef" for character in content_hash):
        raise AuditIntegrityError("pipeline dispatch result content hash is malformed")
    return PipelineDispatchRecovery(binding=binding, executor_content_hash=content_hash)


def _verify_committed_pipeline_authority(
    conn: Connection,
    *,
    service: SessionServiceImpl,
    authority: AuthoritativePipelineProposal,
) -> None:
    """Verify the complete already-committed outcome used by HTTP retries."""
    sid = str(authority.row.session_id)
    pid = str(authority.row.id)
    terminal_rows = conn.execute(
        select(proposal_events_table)
        .where(proposal_events_table.c.session_id == sid)
        .where(proposal_events_table.c.proposal_id == pid)
        .where(proposal_events_table.c.event_type.in_(("proposal.accepted", "proposal.rejected")))
    ).fetchall()
    if len(terminal_rows) != 1 or terminal_rows[0].event_type != "proposal.accepted":
        raise AuditIntegrityError("committed pipeline proposal must have one accepted terminal event")
    terminal = _proposal_event_record_from_row(terminal_rows[0])
    payload = deep_thaw(terminal.payload)
    if type(payload) is not dict or set(payload) != _PIPELINE_ACCEPTED_FIELDS:
        raise AuditIntegrityError("committed pipeline proposal terminal payload is malformed")
    row = authority.row
    if row.audit_event_id != terminal.id:
        raise AuditIntegrityError("committed pipeline proposal terminal event pointer is malformed")
    if row.committed_state_id is None:
        raise AuditIntegrityError("committed pipeline proposal is missing committed state")
    dispatch_payload = payload["dispatch"]
    if type(dispatch_payload) is not dict or set(dispatch_payload) != {
        "tool_call_id",
        "tool_name",
        "status",
        "arguments_hash",
        "result_hash",
    }:
        raise AuditIntegrityError("committed pipeline proposal dispatch binding is malformed")
    try:
        dispatch = PipelineDispatchAuditBinding(
            tool_call_id=dispatch_payload["tool_call_id"],
            tool_name=dispatch_payload["tool_name"],
            status=ComposerToolStatus(dispatch_payload["status"]),
            arguments_hash=dispatch_payload["arguments_hash"],
            result_hash=dispatch_payload["result_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise AuditIntegrityError("committed pipeline proposal dispatch binding is malformed") from exc
    state_row = conn.execute(
        select(composition_states_table)
        .where(composition_states_table.c.session_id == sid)
        .where(composition_states_table.c.id == str(row.committed_state_id))
    ).one_or_none()
    if state_row is None:
        raise AuditIntegrityError("committed pipeline proposal state is missing")
    state_record = service._row_to_state_record(state_row)
    state_hash = composition_content_hash(state_from_record(state_record))
    expected = _pipeline_accepted_payload(
        authority=authority,
        state_id=str(state_record.id),
        state_content_hash=state_hash,
        final_composer_metadata=state_record.composer_meta,
        dispatch=dispatch,
    )
    if payload != expected:
        raise AuditIntegrityError("committed pipeline proposal exact retry binding mismatch")
    if _persisted_pipeline_dispatch_content_hashes(conn, session_id=sid, dispatch=dispatch) != (state_hash,):
        raise AuditIntegrityError("committed pipeline proposal dispatch audit is missing or duplicated")


def _verify_rejected_pipeline_authority(
    conn: Connection,
    *,
    authority: AuthoritativePipelineProposal,
) -> None:
    """Verify the closed terminal authority for a canonical rejection."""
    sid = str(authority.row.session_id)
    pid = str(authority.row.id)
    terminal_rows = conn.execute(
        select(proposal_events_table)
        .where(proposal_events_table.c.session_id == sid)
        .where(proposal_events_table.c.proposal_id == pid)
        .where(proposal_events_table.c.event_type.in_(("proposal.accepted", "proposal.rejected")))
    ).fetchall()
    if len(terminal_rows) != 1 or terminal_rows[0].event_type != "proposal.rejected":
        raise AuditIntegrityError("rejected pipeline proposal must have one rejected terminal event")
    terminal = _proposal_event_record_from_row(terminal_rows[0])
    row = authority.row
    if row.audit_event_id != terminal.id or row.committed_state_id is not None:
        raise AuditIntegrityError("rejected pipeline proposal row terminal binding is malformed")
    payload = deep_thaw(terminal.payload)
    if type(payload) is not dict or set(payload) != _PIPELINE_REJECTED_FIELDS:
        raise AuditIntegrityError("rejected pipeline proposal terminal payload is malformed")
    reason = _validated_pipeline_rejection_reason(payload["reason_code"])
    dispatch_payload = payload["dispatch"]
    if dispatch_payload is None:
        dispatch = None
    elif type(dispatch_payload) is dict and set(dispatch_payload) == {
        "tool_call_id",
        "tool_name",
        "status",
        "arguments_hash",
        "result_hash",
    }:
        try:
            dispatch = PipelineDispatchAuditBinding(
                tool_call_id=dispatch_payload["tool_call_id"],
                tool_name=dispatch_payload["tool_name"],
                status=ComposerToolStatus(dispatch_payload["status"]),
                arguments_hash=dispatch_payload["arguments_hash"],
                result_hash=dispatch_payload["result_hash"],
            )
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("rejected pipeline proposal dispatch binding is malformed") from exc
    else:
        raise AuditIntegrityError("rejected pipeline proposal dispatch binding is malformed")
    if reason == "candidate_executor_mismatch" and dispatch is None:
        raise AuditIntegrityError("rejected pipeline proposal mismatch outcome is missing dispatch evidence")
    expected = _pipeline_rejected_payload(authority=authority, reason=reason, dispatch=dispatch)
    if payload != expected:
        raise AuditIntegrityError("rejected pipeline proposal exact terminal binding mismatch")
    if dispatch is not None and len(_persisted_pipeline_dispatch_content_hashes(conn, session_id=sid, dispatch=dispatch)) != 1:
        raise AuditIntegrityError("rejected pipeline proposal dispatch audit is missing or duplicated")


def _verify_pipeline_lifecycle_authority(
    conn: Connection,
    *,
    service: SessionServiceImpl,
    authority: AuthoritativePipelineProposal,
) -> None:
    """Verify the complete canonical lifecycle before exposing or mutating it."""
    if authority.row.status == "committed":
        _verify_committed_pipeline_authority(conn, service=service, authority=authority)
        return
    if authority.row.status == "rejected":
        _verify_rejected_pipeline_authority(conn, authority=authority)
        return
    if authority.row.status != "pending":
        raise AuditIntegrityError("pipeline proposal lifecycle status is malformed")
    sid = str(authority.row.session_id)
    pid = str(authority.row.id)
    terminal_count = conn.execute(
        select(func.count())
        .select_from(proposal_events_table)
        .where(proposal_events_table.c.session_id == sid)
        .where(proposal_events_table.c.proposal_id == pid)
        .where(proposal_events_table.c.event_type.in_(("proposal.accepted", "proposal.rejected")))
    ).scalar_one()
    if terminal_count != 0:
        raise AuditIntegrityError("pending pipeline proposal must not have a terminal event")
    if authority.row.audit_event_id != authority.creation_event_id or authority.row.committed_state_id is not None:
        raise AuditIntegrityError("pending pipeline proposal row authority binding is malformed")


def _normalize_optional_provenance_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _normalize_proposal_composer_provenance(
    *,
    composer_model_identifier: str | None,
    composer_model_version: str | None,
    composer_provider: str | None,
    composer_skill_hash: str | None,
    tool_arguments_hash: str | None,
) -> dict[str, str | None]:
    raw = {
        "composer_model_identifier": composer_model_identifier,
        "composer_model_version": composer_model_version,
        "composer_provider": composer_provider,
        "composer_skill_hash": composer_skill_hash,
        "tool_arguments_hash": tool_arguments_hash,
    }
    normalized = {name: _normalize_optional_provenance_text(value) for name, value in raw.items()}
    if any(value is not None for value in raw.values()):
        missing = tuple(name for name in _PROPOSAL_COMPOSER_PROVENANCE_FIELDS if normalized[name] is None)
        if missing:
            raise AuditIntegrityError(
                "composer provenance for composition proposals requires all fields to be non-blank "
                f"when any are supplied; missing: {', '.join(missing)}"
            )
    return normalized


def _assert_state_in_session(
    conn: Connection,
    *,
    state_id: str,
    expected_session_id: str,
    caller: str,
) -> None:
    """Offensive guard: composition state must belong to the expected session.

    Catches cross-session reference bugs at the service boundary, before
    they hit the DB-level composite FK. Produces a diagnostic naming the
    caller, the state, and the session mismatch — something a generic
    ``IntegrityError`` cannot.

    Raises ``RuntimeError`` because a cross-session reference is a bug
    in caller code, not invalid user input. The audit trail records the
    attempted violation through the standard exception path.

    Contrast with ``set_active_state``, which raises ``ValueError`` for
    an equivalent-looking cross-session check on purpose: that method
    receives the state_id from the HTTP body and must map an unknown /
    non-owned state to 404 rather than 500. The exception type is
    load-bearing and encodes whether the caller (RuntimeError) or the
    user (ValueError) is wrong.
    """
    state_session_id = conn.execute(select(composition_states_table.c.session_id).where(composition_states_table.c.id == state_id)).scalar()
    if state_session_id is None:
        raise RuntimeError(f"{caller}: composition_state_id={state_id!r} does not exist (expected in session={expected_session_id!r})")
    if state_session_id != expected_session_id:
        raise RuntimeError(
            f"{caller}: composition_state_id={state_id!r} belongs to session "
            f"{state_session_id!r}, not {expected_session_id!r} — cross-session "
            f"reference is a contract violation"
        )


def _assert_parent_assistant_message(
    conn: Connection,
    *,
    parent_assistant_id: str,
    session_id: str,
    caller: str,
) -> None:
    """Offensive guard for tool rows.

    The composite FK on ``(parent_assistant_id, session_id)`` proves
    same-session existence at the DB layer, but SQL CHECK constraints
    cannot portably inspect the referenced row's ``role`` column.
    Service writers must therefore reject tool rows whose parent id
    exists in the same session but does not belong to an assistant
    message — otherwise a tool row could legally point at a user or
    system message and the audit trail would record a false parent
    relationship.

    Raises ``RuntimeError`` because a wrong-role parent reference is a
    bug in caller code, not invalid user input. The exception type
    mirrors ``_assert_state_in_session`` above and is load-bearing:
    it routes to a 500 in the route layer, not a 4xx for the user.
    """
    role = conn.execute(
        select(chat_messages_table.c.role).where(
            chat_messages_table.c.id == parent_assistant_id,
            chat_messages_table.c.session_id == session_id,
        )
    ).scalar_one_or_none()
    if role != "assistant":
        raise RuntimeError(
            f"{caller}: parent_assistant_id={parent_assistant_id!r} must reference "
            f"an assistant message in session={session_id!r}; got role={role!r}"
        )


def _assert_assistant_row_has_audit_content(
    *,
    content: str,
    raw_content: str | None,
    tool_calls: Any,
    caller: str,
) -> None:
    """Reject assistant rows that carry no auditable model output."""
    if content.strip():
        return
    if raw_content is not None and raw_content.strip():
        return
    if tool_calls:
        return
    raise AuditIntegrityError(f"{caller}: refusing to persist empty assistant audit row with no raw_content and no tool_calls")


def _validate_tool_call_id_set_equality(
    *,
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
    redacted_tool_rows: tuple[RedactedToolRow, ...],
) -> None:
    """Raise ``ToolCallIDMismatchError`` if the assistant's
    ``tool_calls`` IDs and the tool rows' ``tool_call_id`` values are
    not the same unique set.

    Four failure axes — any of them raises:

    - ``missing``: assistant ID with no tool row
    - ``extra``: tool row with no assistant ID
    - ``duplicates_in_assistant``: ID twice in tool_calls
    - ``duplicates_in_rows``: ID twice in tool rows

    All four are reported simultaneously so the diagnostic shows the
    full picture in one shot. The empty-empty case is valid.

    Pure function of caller arguments; called BEFORE
    ``_engine.begin()`` (pre-lock, pre-transaction) so a contract
    violation cannot leave a half-written audit trail behind.
    """
    assistant_ids: list[str] = [
        # ``id`` key is contractually present (OpenAI/LiteLLM tool-call
        # shape requires it). If it's missing, that's an upstream
        # framework bug, not data we should defend against.
        tc["id"]
        for tc in redacted_assistant_tool_calls
    ]
    row_ids: list[str] = [row.tool_call_id for row in redacted_tool_rows]

    assistant_set = set(assistant_ids)
    row_set = set(row_ids)
    missing = frozenset(assistant_set - row_set)
    extra = frozenset(row_set - assistant_set)
    duplicates_in_assistant = frozenset(i for i in assistant_set if assistant_ids.count(i) > 1)
    duplicates_in_rows = frozenset(i for i in row_set if row_ids.count(i) > 1)

    if missing or extra or duplicates_in_assistant or duplicates_in_rows:
        raise ToolCallIDMismatchError(
            missing=missing,
            extra=extra,
            duplicates_in_assistant=duplicates_in_assistant,
            duplicates_in_rows=duplicates_in_rows,
        )


def _enveloped_state_column(value: Any) -> Any:
    """Return the JSON envelope stored by composition_states JSON columns.

    Existing ``save_composition_state`` and ``fork_session`` each carried a
    local ``_enveloped`` helper. ``_insert_composition_state`` is module/class
    scope, so the envelope rule must be extracted before the helper can
    call it. Do not duplicate the helper back into individual methods.

    The envelope shape ``{"_version": 1, "data": <raw>}`` is the on-disk
    format for ``composition_states``' JSON columns; the ``_version``
    field is reserved for schema evolution. ``deep_thaw()`` handles
    ``MappingProxyType``/``frozenset``/tuple unwrap from ``freeze_fields()``.
    """
    raw = deep_thaw(value)
    if raw is None:
        return None
    return {"_version": 1, "data": raw}


def _strip_guided_profile_in_meta(composer_meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Reset a forked GuidedSession's WorkflowProfile to the empty profile.

    ``composer_meta`` is copied verbatim on fork. A tutorial profile must not
    leak into an ordinary forked session, so the strip lives inside
    ``fork_session`` where the composer_meta copies happen.
    """
    from elspeth.web.composer.guided.profile import EMPTY_PROFILE

    if composer_meta is None:
        return None
    thawed: dict[str, Any] = dict(deep_thaw(composer_meta))
    guided_raw = thawed.get("guided_session")
    if not isinstance(guided_raw, dict) or "profile" not in guided_raw:
        return thawed
    guided_copy = dict(guided_raw)
    guided_copy["profile"] = EMPTY_PROFILE.to_dict()
    thawed["guided_session"] = guided_copy
    return thawed


def _current_adr019_counter_subsets_hold(
    *,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> bool:
    return rows_routed_success <= rows_succeeded and rows_routed_failure <= rows_failed and rows_quarantined <= rows_failed


def _legacy_disjoint_counter_shape_holds(
    *,
    status: SessionRunStatus,
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> bool:
    """Return true for the pre-ADR-019 disjoint session-counter contract."""
    if status not in SESSION_TERMINAL_RUN_STATUS_VALUES:
        return False
    legacy_total = rows_succeeded + rows_failed + rows_routed_success + rows_routed_failure + rows_quarantined
    if rows_processed < legacy_total:
        return False

    success_indicator = rows_succeeded > 0 or rows_routed_success > 0
    failure_indicator = rows_failed > 0 or rows_routed_failure > 0 or rows_quarantined > 0
    if status == "completed":
        return success_indicator and not failure_indicator
    if status == "completed_with_failures":
        return success_indicator and failure_indicator
    if status == "failed":
        return True
    if status == "empty":
        return rows_processed == 0 and not success_indicator and not failure_indicator
    return status == "cancelled"


def _normalize_pre_adr019_session_counters(
    *,
    status: SessionRunStatus,
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> tuple[int, int]:
    """Fold unambiguous legacy disjoint counters into ADR-019 base counters.

    ADR-019 did not change the session DB schema, so historical rows have no
    version marker. Only shapes that fail the current subset invariant but pass
    the previous disjoint predicate are normalized; current rows and ambiguous
    data are left untouched for the response/schema guards to validate.
    """
    if _current_adr019_counter_subsets_hold(
        rows_succeeded=rows_succeeded,
        rows_failed=rows_failed,
        rows_routed_success=rows_routed_success,
        rows_routed_failure=rows_routed_failure,
        rows_quarantined=rows_quarantined,
    ):
        return rows_succeeded, rows_failed
    if not _legacy_disjoint_counter_shape_holds(
        status=status,
        rows_processed=rows_processed,
        rows_succeeded=rows_succeeded,
        rows_failed=rows_failed,
        rows_routed_success=rows_routed_success,
        rows_routed_failure=rows_routed_failure,
        rows_quarantined=rows_quarantined,
    ):
        return rows_succeeded, rows_failed
    return rows_succeeded + rows_routed_success, rows_failed + rows_routed_failure + rows_quarantined


def _proposal_record_from_row(row: Any) -> CompositionProposalRecord:
    return CompositionProposalRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        tool_call_id=row.tool_call_id,
        user_message_id=UUID(row.user_message_id) if row.user_message_id is not None else None,
        composer_model_identifier=row.composer_model_identifier,
        composer_model_version=row.composer_model_version,
        composer_provider=row.composer_provider,
        composer_skill_hash=row.composer_skill_hash,
        tool_arguments_hash=row.tool_arguments_hash,
        tool_name=row.tool_name,
        status=row.status,
        summary=row.summary,
        rationale=row.rationale,
        affects=tuple(row.affects),
        arguments_json=row.arguments_json,
        arguments_redacted_json=row.arguments_redacted_json,
        base_state_id=UUID(row.base_state_id) if row.base_state_id else None,
        committed_state_id=UUID(row.committed_state_id) if row.committed_state_id else None,
        audit_event_id=UUID(row.audit_event_id) if row.audit_event_id else None,
        created_at=SessionServiceImpl._ensure_utc(row.created_at),
        updated_at=SessionServiceImpl._ensure_utc(row.updated_at),
    )


def _proposal_event_record_from_row(row: Any) -> ProposalEventRecord:
    return ProposalEventRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        proposal_id=UUID(row.proposal_id) if row.proposal_id else None,
        event_type=row.event_type,
        actor=row.actor,
        payload=row.payload,
        created_at=SessionServiceImpl._ensure_utc(row.created_at),
    )


def _restore_authoritative_pipeline_proposal(
    *,
    conn: Connection,
    row: CompositionProposalRecord,
    creation_event: ProposalEventRecord,
    reviewed_facts: Mapping[str, Any] | None,
) -> AuthoritativePipelineProposal:
    payload = deep_thaw(creation_event.payload)
    if type(payload) is not dict:
        raise AuditIntegrityError("pipeline proposal creation event payload is malformed")
    if set(payload) != _PIPELINE_CREATED_FIELDS:
        raise AuditIntegrityError("pipeline proposal creation event fields are malformed")
    if payload["schema"] != _PIPELINE_CREATED_SCHEMA:
        raise AuditIntegrityError("pipeline proposal creation event schema is malformed")
    if creation_event.session_id != row.session_id or creation_event.proposal_id != row.id:
        raise AuditIntegrityError("pipeline proposal creation event ownership mismatch")
    if payload["tool_call_id"] != row.tool_call_id or payload["tool_name"] != row.tool_name:
        raise AuditIntegrityError("pipeline proposal creation event tool binding mismatch")
    if row.tool_name != "set_pipeline" or payload["status"] != "pending":
        raise AuditIntegrityError("pipeline proposal creation event status/tool is malformed")
    if payload["custody_result"] not in {"not_required", "ready"}:
        raise AuditIntegrityError("pipeline proposal custody result is malformed")

    private_hash = _pipeline_private_arguments_hash(row.arguments_json)
    if payload["private_arguments_hash"] != private_hash:
        raise AuditIntegrityError("pipeline proposal private arguments binding mismatch")
    if payload["audit_payload_hash"] != _pipeline_audit_payload_hash(
        summary=row.summary,
        rationale=row.rationale,
        affects=row.affects,
        arguments_redacted_json=row.arguments_redacted_json,
    ):
        raise AuditIntegrityError("pipeline proposal audit payload binding mismatch")
    if None in {
        row.composer_model_identifier,
        row.composer_model_version,
        row.composer_provider,
        row.composer_skill_hash,
        row.tool_arguments_hash,
    }:
        raise AuditIntegrityError("pipeline proposal composer provenance is incomplete")
    assert row.composer_model_identifier is not None
    assert row.composer_model_version is not None
    assert row.composer_provider is not None
    assert row.composer_skill_hash is not None
    assert row.tool_arguments_hash is not None
    expected_provenance_hash = _pipeline_provenance_hash(
        user_message_id=row.user_message_id,
        composer_model_identifier=row.composer_model_identifier,
        composer_model_version=row.composer_model_version,
        composer_provider=row.composer_provider,
        composer_skill_hash=row.composer_skill_hash,
        tool_arguments_hash=row.tool_arguments_hash,
    )
    if payload["provenance_hash"] != expected_provenance_hash:
        raise AuditIntegrityError("pipeline proposal provenance binding mismatch")
    if row.tool_arguments_hash != stable_hash(row.arguments_json):
        raise AuditIntegrityError("pipeline proposal row arguments hash mismatch")
    if row.composer_skill_hash != payload["skill_hash"]:
        raise AuditIntegrityError("pipeline proposal skill provenance mismatch")
    raw_base = payload["base"]
    if type(raw_base) is not dict:
        raise AuditIntegrityError("pipeline proposal base metadata is malformed")
    if raw_base.get("kind") == "absent" and set(raw_base) == {"kind"}:
        base: AbsentBase | PresentBase = AbsentBase()
    elif raw_base.get("kind") == "present" and set(raw_base) == {"kind", "state_id", "composition_content_hash"}:
        raw_state_id = raw_base["state_id"]
        if type(raw_state_id) is not str:
            raise AuditIntegrityError("pipeline proposal base state id is malformed")
        try:
            state_id = UUID(raw_state_id)
        except ValueError as exc:
            raise AuditIntegrityError("pipeline proposal base state id is malformed") from exc
        if str(state_id) != raw_state_id:
            raise AuditIntegrityError("pipeline proposal base state id is not canonical")
        base = PresentBase(state_id=state_id, composition_content_hash=raw_base["composition_content_hash"])
    else:
        raise AuditIntegrityError("pipeline proposal base metadata is malformed")
    raw_surface = payload["surface"]
    if type(raw_surface) is not str:
        raise AuditIntegrityError("pipeline proposal surface is malformed")
    try:
        surface = PlannerSurface(raw_surface)
    except ValueError as exc:
        raise AuditIntegrityError("pipeline proposal surface is malformed") from exc
    covered_ids = payload["covered_deferred_intent_ids"]
    if type(covered_ids) is not list or any(type(value) is not str for value in covered_ids):
        raise AuditIntegrityError("pipeline proposal covered intent ids are malformed")
    proposal = PipelineProposal(
        pipeline=deep_thaw(row.arguments_json),
        draft_hash=payload["draft_hash"],
        base=base,
        reviewed_anchor_hash=payload["reviewed_anchor_hash"],
        surface=surface,
        repair_count=payload["repair_count"],
        skill_hash=payload["skill_hash"],
        covered_deferred_intent_ids=tuple(covered_ids),
        supersedes_draft_hash=payload["supersedes_draft_hash"],
    )
    if reviewed_facts is not None and proposal.reviewed_anchor_hash != reviewed_anchor_hash(reviewed_facts):
        raise AuditIntegrityError("pipeline proposal reviewed anchor does not match current server facts")
    expected_base_state_id = proposal.base.state_id if type(proposal.base) is PresentBase else None
    if row.base_state_id != expected_base_state_id:
        raise AuditIntegrityError("pipeline proposal row/base state binding mismatch")
    supersedes_raw = payload["supersedes_proposal_id"]
    if supersedes_raw is not None:
        if type(supersedes_raw) is not str:
            raise AuditIntegrityError("pipeline proposal supersedes id is malformed")
        try:
            supersedes_proposal_id = UUID(supersedes_raw)
        except ValueError as exc:
            raise AuditIntegrityError("pipeline proposal supersedes id is malformed") from exc
        if str(supersedes_proposal_id) != supersedes_raw:
            raise AuditIntegrityError("pipeline proposal supersedes id is not canonical")
    else:
        supersedes_proposal_id = None
    if (supersedes_proposal_id is None) != (proposal.supersedes_draft_hash is None):
        raise AuditIntegrityError("pipeline proposal supersedes id/draft binding is incomplete")
    if supersedes_proposal_id is not None:
        referenced_row = conn.execute(
            select(composition_proposals_table)
            .where(composition_proposals_table.c.session_id == str(row.session_id))
            .where(composition_proposals_table.c.id == str(supersedes_proposal_id))
        ).one_or_none()
        if referenced_row is None:
            raise AuditIntegrityError("pipeline proposal supersedes target is missing or cross-session")
        referenced_events = conn.execute(
            select(proposal_events_table)
            .where(proposal_events_table.c.session_id == str(row.session_id))
            .where(proposal_events_table.c.proposal_id == str(supersedes_proposal_id))
            .where(proposal_events_table.c.event_type == "proposal.created")
        ).fetchall()
        if len(referenced_events) != 1:
            raise AuditIntegrityError("pipeline proposal supersedes target creation authority is malformed")
        referenced_payload = referenced_events[0].payload
        if (
            type(referenced_payload) is not dict
            or referenced_payload.get("schema") != _PIPELINE_CREATED_SCHEMA
            or referenced_payload.get("draft_hash") != proposal.supersedes_draft_hash
        ):
            raise AuditIntegrityError("pipeline proposal supersedes target draft binding mismatch")
    authority = AuthoritativePipelineProposal(
        row=row,
        proposal=proposal,
        creation_event_id=creation_event.id,
        custody_result=payload["custody_result"],
        supersedes_proposal_id=supersedes_proposal_id,
    )
    return replace(
        authority,
        row=replace(row, pipeline_metadata=_pipeline_public_metadata(authority)),
    )


def _classify_authoritative_composition_proposal(
    *,
    conn: Connection,
    row: CompositionProposalRecord,
    creation_event: ProposalEventRecord,
    reviewed_facts: Mapping[str, Any] | None,
) -> AuthoritativeCompositionProposal:
    """Accept only the exact historical shape or the closed pipeline shape."""
    payload = creation_event.payload
    if not isinstance(payload, Mapping):
        raise AuditIntegrityError("proposal creation event payload must be a mapping")
    if set(payload) == _LEGACY_PROPOSAL_CREATED_FIELDS:
        expected = {
            "tool_call_id": row.tool_call_id,
            "tool_name": row.tool_name,
            "status": "pending",
        }
        if payload != expected or creation_event.session_id != row.session_id or creation_event.proposal_id != row.id:
            raise AuditIntegrityError("legacy proposal creation event binding is malformed")
        return AuthoritativeCompositionProposal(row=row, pipeline=None)
    pipeline = _restore_authoritative_pipeline_proposal(
        conn=conn,
        row=row,
        creation_event=creation_event,
        reviewed_facts=reviewed_facts,
    )
    return AuthoritativeCompositionProposal(row=pipeline.row, pipeline=pipeline)


def _pipeline_public_metadata(authority: AuthoritativePipelineProposal) -> PipelineProposalPublicMetadata:
    payload = authority.proposal
    return PipelineProposalPublicMetadata(
        surface=payload.surface.value,
        draft_hash=payload.draft_hash,
        base=_proposal_base_payload(payload.base),
        reviewed_anchor_hash=payload.reviewed_anchor_hash,
        repair_count=payload.repair_count,
        skill_hash=payload.skill_hash,
        audit_payload_hash=_pipeline_audit_payload_hash(
            summary=authority.row.summary,
            rationale=authority.row.rationale,
            affects=authority.row.affects,
            arguments_redacted_json=authority.row.arguments_redacted_json,
        ),
        custody_result=authority.custody_result,
    )


def _interpretation_event_record_from_row(row: Any) -> InterpretationEventRecord:
    """Convert a SQLAlchemy row to an InterpretationEventRecord.

    Per the Tier-1 audit-trust contract (CLAUDE.md), this conversion crashes
    loudly on any anomaly — the enum constructors raise ValueError on an
    unrecognised string, and the UUID/datetime constructors raise on
    malformed values. The schema CHECK constraints guarantee the closed-
    enum values and source-conditional nullability invariants; this helper
    relies on that guarantee.
    """
    return InterpretationEventRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        composition_state_id=UUID(row.composition_state_id) if row.composition_state_id is not None else None,
        affected_node_id=row.affected_node_id,
        tool_call_id=row.tool_call_id,
        user_term=row.user_term,
        kind=InterpretationKind(row.kind) if row.kind is not None else None,
        llm_draft=row.llm_draft,
        accepted_value=row.accepted_value,
        choice=InterpretationChoice(row.choice),
        created_at=SessionServiceImpl._ensure_utc(row.created_at),
        resolved_at=SessionServiceImpl._ensure_utc(row.resolved_at) if row.resolved_at is not None else None,
        actor=row.actor,
        model_identifier=row.model_identifier,
        model_version=row.model_version,
        provider=row.provider,
        composer_skill_hash=row.composer_skill_hash,
        arguments_hash=row.arguments_hash,
        hash_domain_version=row.hash_domain_version,
        interpretation_source=InterpretationSource(row.interpretation_source),
        runtime_model_identifier_at_resolve=row.runtime_model_identifier_at_resolve,
        runtime_model_version_at_resolve=row.runtime_model_version_at_resolve,
        resolved_prompt_template_hash=row.resolved_prompt_template_hash,
    )


# Sentinel for the trigger ``trg_interpretation_events_immutable_resolved``
# RAISE(ABORT, ...) message. The exact string lives in models.py at the
# CREATE TRIGGER DDL — keep these in sync. F-28: the service-layer error
# classifier MUST match this specific substring so an immutability violation
# is mapped to a 409/400 by the route, NOT conflated with a generic
# constraint violation (which would emit a spurious security telemetry
# signal upstream).
_INTERPRETATION_IMMUTABLE_TRIGGER_MSG: str = "interpretation_events: resolved rows are immutable"


class InterpretationResolveError(ValueError):
    """Base class for expected interpretation-resolution failures."""


class InterpretationEventNotFoundError(InterpretationResolveError):
    """No event exists for the requested ``(session_id, event_id)`` pair."""


class InterpretationEventAlreadyResolvedError(InterpretationResolveError):
    """The event exists but is no longer pending."""


class InterpretationNodeMissingError(InterpretationResolveError):
    """The affected node disappeared from the live composition state."""


class InterpretationNodePluginMutatedError(InterpretationResolveError):
    """The affected node still exists but is no longer an LLM transform."""


class InterpretationPlaceholderConsumedError(InterpretationResolveError):
    """The affected LLM node no longer carries the expected placeholder."""


class InterpretationUnsupportedChoiceError(InterpretationResolveError):
    """The requested choice is valid generally but unsupported for this kind."""


class QuarantineCleanupError(AuditIntegrityError):
    """Session archive committed, but staged blob cleanup failed."""


class _InterpretationHashDomainV2Payload(TypedDict):
    """Closed hash-domain payload for interpretation review events."""

    session_id: str
    composition_state_id: str | None
    affected_node_id: str
    tool_call_id: str
    user_term: str
    kind: str
    llm_draft: str
    accepted_value: str
    actor: str
    model_identifier: str
    model_version: str
    provider: str
    composer_skill_hash: str


def _interpretation_hash_domain_v2(
    *,
    session_id: str,
    composition_state_id: str | None,
    affected_node_id: str,
    tool_call_id: str,
    user_term: str,
    kind: str,
    llm_draft: str,
    accepted_value: str,
    actor: str,
    model_identifier: str,
    model_version: str,
    provider: str,
    composer_skill_hash: str,
    context: str,
) -> _InterpretationHashDomainV2Payload:
    domain_dict: _InterpretationHashDomainV2Payload = {
        "session_id": session_id,
        "composition_state_id": composition_state_id,
        "affected_node_id": affected_node_id,
        "tool_call_id": tool_call_id,
        "user_term": user_term,
        "kind": kind,
        "llm_draft": llm_draft,
        "accepted_value": accepted_value,
        "actor": actor,
        "model_identifier": model_identifier,
        "model_version": model_version,
        "provider": provider,
        "composer_skill_hash": composer_skill_hash,
    }
    if set(domain_dict.keys()) != INTERPRETATION_HASH_DOMAIN_V2:
        raise AssertionError(
            f"{context}: domain dict keys {set(domain_dict.keys())!r} drifted from "
            f"INTERPRETATION_HASH_DOMAIN_V2 {INTERPRETATION_HASH_DOMAIN_V2!r}"
        )
    return domain_dict


def _require_mapping(value: object, *, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpretationPlaceholderConsumedError(message)
    return value


def _find_llm_transform_node(
    state: CompositionStateRecord,
    *,
    affected_node_id: str,
    context: str,
) -> Mapping[str, Any]:
    if state.nodes is None:
        raise InterpretationNodeMissingError(f"{context}: composition state has no nodes; node {affected_node_id!r} is not present")
    for node in state.nodes:
        if node["id"] != affected_node_id:
            continue
        node_type = node["node_type"] if "node_type" in node else None
        if node_type != "transform" or "plugin" not in node:
            raise InterpretationNodePluginMutatedError(
                f"{context}: node {affected_node_id!r} has no LLM discriminator; expected node_type='transform' with plugin='llm'"
            )
        node_plugin = node["plugin"]
        if node_plugin != "llm":
            raise InterpretationNodePluginMutatedError(
                f"{context}: node {affected_node_id!r} has plugin {node_plugin!r}; only llm nodes carry interpretation review state"
            )
        options = _require_mapping(
            node["options"] if "options" in node else None,
            message=f"{context}: node {affected_node_id!r} has no options mapping",
        )
        if "prompt_template" not in options or not isinstance(options["prompt_template"], str):
            raise InterpretationPlaceholderConsumedError(f"{context}: node {affected_node_id!r} options.prompt_template is not a string")
        prompt_template = options["prompt_template"]
        if not prompt_template:
            raise InterpretationPlaceholderConsumedError(
                f"{context}: node {affected_node_id!r} must declare non-empty options.prompt_template"
            )
        return node
    raise InterpretationNodeMissingError(f"{context}: node {affected_node_id!r} is not present in the composition state's nodes")


def _find_interpretation_review_node(
    state: CompositionStateRecord,
    *,
    affected_node_id: str,
    context: str,
) -> Mapping[str, Any]:
    if state.nodes is None:
        raise InterpretationNodeMissingError(f"{context}: composition state has no nodes; node {affected_node_id!r} is not present")
    for node in state.nodes:
        if node["id"] == affected_node_id:
            return node
    raise InterpretationNodeMissingError(f"{context}: node {affected_node_id!r} is not present in the composition state's nodes")


def _node_specs_from_state_record(state_record: CompositionStateRecord) -> tuple[Any, ...]:
    from elspeth.web.composer.state import NodeSpec

    return tuple(NodeSpec.from_dict(dict(n)) for n in state_record.nodes or ())


def _find_node_spec_from_state_record(
    state_record: CompositionStateRecord,
    *,
    affected_node_id: str,
    context: str,
) -> tuple[Any, tuple[Any, ...]]:
    all_nodes_spec = _node_specs_from_state_record(state_record)
    target = next((n for n in all_nodes_spec if n.id == affected_node_id), None)
    if target is None:
        raise InterpretationNodeMissingError(f"{context}: node {affected_node_id!r} is not present in the composition state's nodes")
    return target, all_nodes_spec


def _pipeline_decision_artifact_hash_from_state_record(
    state_record: CompositionStateRecord,
    *,
    affected_node_id: str,
    user_term: str,
) -> str:
    """Compute the canonical pipeline-decision artifact hash from a record DTO.

    Bridge between the persistence layer (dict-shaped nodes on
    :class:`CompositionStateRecord`) and the canonical hash function on
    :class:`NodeSpec`. Both write and read paths use the same projection
    helpers under the hood, so an interpretation_resolve event stores
    exactly the hash that preflight will recompute later.
    """

    target, all_nodes_spec = _find_node_spec_from_state_record(
        state_record,
        affected_node_id=affected_node_id,
        context="_pipeline_decision_artifact_hash_from_state_record",
    )
    return pipeline_decision_artifact_hash(target, all_nodes_spec, user_term=user_term)


def _validate_pipeline_decision_semantics_from_state_record(
    state_record: CompositionStateRecord,
    *,
    affected_node_id: str,
    user_term: str,
    draft: str | None,
    context: str,
) -> None:
    target, all_nodes_spec = _find_node_spec_from_state_record(
        state_record,
        affected_node_id=affected_node_id,
        context=context,
    )
    validate_pipeline_decision_node_semantics(
        node=target,
        all_nodes=all_nodes_spec,
        user_term=user_term,
        draft=draft,
        context=context,
    )


def _matching_pending_requirement_index(
    requirements_value: object,
    *,
    kind: InterpretationKind,
    user_term: str,
    context: str,
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(requirements_value, (list, tuple)):
        raise InterpretationPlaceholderConsumedError(f"{context}: options.interpretation_requirements is not a list")
    normalized_user_term = user_term.strip()
    requirements: list[dict[str, Any]] = []
    matching_indexes: list[int] = []
    for index, requirement_value in enumerate(requirements_value):
        if not isinstance(requirement_value, Mapping):
            raise InterpretationPlaceholderConsumedError(f"{context}: interpretation requirement entry is not a mapping")
        requirement = dict(requirement_value)
        requirement_kind = requirement["kind"] if "kind" in requirement else InterpretationKind.VAGUE_TERM.value
        requirement_term = requirement["user_term"]
        if not isinstance(requirement_term, str):
            raise InterpretationPlaceholderConsumedError(f"{context}: interpretation requirement user_term is invalid")
        requirement_status = requirement["status"] if "status" in requirement else None
        if requirement_term.strip() == normalized_user_term and requirement_status == "pending" and requirement_kind == kind.value:
            matching_indexes.append(index)
        requirements.append(requirement)
    if len(matching_indexes) != 1:
        raise InterpretationPlaceholderConsumedError(
            f"{context}: does not contain exactly one pending {kind.value!r} requirement for {user_term!r}; found {len(matching_indexes)}"
        )
    return requirements, matching_indexes[0]


# Prefixes (case-insensitive) that, when they appear immediately before the
# placeholder, indicate the LLM placed the placeholder inside a structural
# directive rather than in the prompt body. Substituting the user's
# accepted_value into a structural-directive position would produce a
# broken prompt at runtime — fail closed.
#
# CLOSED LIST — extending this set is a governance action for the
# prompt-template patch helper. Any new prefix must be paired with a
# direct-helper unit test and a writer-path audit.
_STRUCTURAL_DIRECTIVE_PREFIXES: tuple[str, ...] = (
    "system:",
    "role:",
    "instructions:",
)


def _patch_structured_interpretation_prompt(
    *,
    options: Mapping[str, Any],
    affected_node_id: str,
    user_term: str,
    accepted_value: str,
) -> dict[str, Any] | None:
    """Resolve structured interpretation metadata, returning patched options.

    ``None`` means the node does not carry structured interpretation state and
    the caller should fall back to the legacy sentinel-string path.
    """

    if INTERPRETATION_REQUIREMENTS_KEY not in options:
        return None
    requirements_value = options[INTERPRETATION_REQUIREMENTS_KEY]
    if not isinstance(requirements_value, (list, tuple)):
        raise InterpretationPlaceholderConsumedError(
            f"_patch_llm_transform_prompt: node {affected_node_id!r} options.interpretation_requirements is not a list"
        )

    matching_indexes: list[int] = []
    normalized_user_term = user_term.strip()
    requirements: list[dict[str, Any]] = []
    requirements_by_id: dict[str, Mapping[str, Any]] = {}
    for index, requirement_value in enumerate(requirements_value):
        if not isinstance(requirement_value, Mapping):
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} interpretation requirement entry is not a mapping"
            )
        requirement = dict(requirement_value)
        requirement_id = requirement["id"]
        requirement_term = requirement["user_term"]
        if not isinstance(requirement_id, str) or not requirement_id:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} interpretation requirement id is invalid"
            )
        if not isinstance(requirement_term, str):
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} interpretation requirement user_term is invalid"
            )
        if requirement_id in requirements_by_id:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: duplicate interpretation requirement id {requirement_id!r}"
            )
        requirements_by_id[requirement_id] = requirement
        requirement_kind = requirement["kind"] if "kind" in requirement else InterpretationKind.VAGUE_TERM.value
        requirement_status = requirement["status"] if "status" in requirement else None
        if (
            requirement_term.strip() == normalized_user_term
            and requirement_status == "pending"
            and requirement_kind == InterpretationKind.VAGUE_TERM.value
        ):
            matching_indexes.append(index)
        requirements.append(requirement)

    # A node can legitimately carry interpretation_requirements for OTHER kinds
    # (the prompt-template and model-choice auto-stagers add one each to every
    # LLM node) while its vague term is wired by a legacy
    # ``{{interpretation:<term>}}`` placeholder. When no pending vague_term
    # requirement matches this user_term, the structured path does not apply —
    # fall back to the legacy placeholder path rather than demanding
    # prompt_template_parts the node never needed.
    if not matching_indexes:
        return None
    if len(matching_indexes) != 1:
        raise InterpretationPlaceholderConsumedError(
            f"_patch_llm_transform_prompt: node {affected_node_id!r} does not contain exactly one pending "
            f"interpretation requirement for {user_term!r}; found {len(matching_indexes)}"
        )

    # The vague term is structured (a matching requirement exists); the prompt
    # parts that carry its substitution are now required.
    parts_value = options[PROMPT_TEMPLATE_PARTS_KEY] if PROMPT_TEMPLATE_PARTS_KEY in options else None
    if not isinstance(parts_value, (list, tuple)):
        raise InterpretationPlaceholderConsumedError(
            f"_patch_llm_transform_prompt: node {affected_node_id!r} options.prompt_template_parts is required for structured interpretation resolution"
        )

    matching_index = matching_indexes[0]
    matching_requirement = requirements[matching_index]
    matching_requirement_id = matching_requirement["id"]

    rendered: list[str] = []
    matched_ref_count = 0
    for part_value in parts_value:
        if not isinstance(part_value, Mapping):
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} prompt_template_parts entry is not a mapping"
            )
        kind = part_value["kind"]
        if kind == "text":
            text = part_value["text"]
            if not isinstance(text, str):
                raise InterpretationPlaceholderConsumedError(
                    f"_patch_llm_transform_prompt: node {affected_node_id!r} text prompt part is not a string"
                )
            rendered.append(text)
            continue
        if kind != "interpretation_ref":
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} unknown prompt part kind {kind!r}"
            )
        requirement_id = part_value["requirement_id"]
        if not isinstance(requirement_id, str) or requirement_id not in requirements_by_id:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} prompt part references unknown interpretation requirement"
            )
        stored_requirement = requirements_by_id[requirement_id]
        if requirement_id == matching_requirement_id:
            prefix_lower = "".join(rendered).rstrip().lower()
            for directive in _STRUCTURAL_DIRECTIVE_PREFIXES:
                if prefix_lower.endswith(directive):
                    raise InterpretationPlaceholderConsumedError(
                        f"_patch_llm_transform_prompt: interpretation requirement {requirement_id!r} in node "
                        f"{affected_node_id!r} is immediately preceded by structural directive {directive!r}; "
                        f"substituting into a directive position would produce a broken prompt"
                    )
            matched_ref_count += 1
            rendered.append(accepted_value)
            continue
        stored_status = stored_requirement["status"] if "status" in stored_requirement else None
        if stored_status == "resolved":
            accepted = stored_requirement["accepted_value"] if "accepted_value" in stored_requirement else None
            if not isinstance(accepted, str):
                raise InterpretationPlaceholderConsumedError(
                    f"_patch_llm_transform_prompt: resolved interpretation requirement {requirement_id!r} has no accepted value"
                )
            rendered.append(accepted)
            continue
        rendered.append(PENDING_INTERPRETATION_AUTHORING_TEXT)

    # Defense-in-depth backstop: the matched requirement must be referenced by
    # at least one ``interpretation_ref`` part, or the accepted value never
    # lands in the rendered prompt — a "resolved" review whose decision silently
    # never reaches the runtime, i.e. the exact audit divergence CLAUDE.md
    # forbids. Unreachable once the staging gate (vague_term_wiring_count) holds;
    # present so a bypass crashes loudly instead of corrupting the prompt.
    if matched_ref_count == 0:
        raise InterpretationPlaceholderConsumedError(
            f"_patch_llm_transform_prompt: node {affected_node_id!r} prompt_template_parts contains no "
            f"interpretation_ref part referencing the resolved requirement {matching_requirement_id!r}; "
            "the accepted interpretation value would be silently dropped from the prompt"
        )

    new_template = "".join(rendered)
    resolved_prompt_template_hash = stable_hash(new_template)
    updated_requirement = dict(matching_requirement)
    updated_requirement["status"] = "resolved"
    updated_requirement["accepted_value"] = accepted_value
    # The requirement-level hash attests THIS requirement's accepted value, not
    # the full render: the render changes again when a sibling vague term
    # resolves, and reconciliation must be able to re-verify every resolved
    # requirement against state that survives those later resolutions. The
    # full-render hash lives at node level (options.resolved_prompt_template_hash).
    updated_requirement["resolved_prompt_template_hash"] = stable_hash(accepted_value)
    requirements[matching_index] = updated_requirement

    patched_options = dict(options)
    patched_options["prompt_template"] = new_template
    patched_options["resolved_prompt_template_hash"] = resolved_prompt_template_hash
    patched_options[INTERPRETATION_REQUIREMENTS_KEY] = requirements
    return patched_options


def _patch_llm_transform_prompt(
    state: CompositionStateRecord,
    *,
    affected_node_id: str,
    user_term: str,
    accepted_value: str,
) -> Sequence[Mapping[str, Any]]:
    """Return a new ``nodes`` JSON sequence with the LLM transform's prompt
    template patched to embed ``accepted_value`` for ``user_term``.

    The prompt-template patch convention: the LLM transform's
    ``options.prompt_template`` field contains exactly one
    ``{{interpretation:<term>}}`` placeholder that the LLM writes when it
    first stages the LLM transform. This helper substitutes the placeholder
    with the user's ``accepted_value`` and writes the result back into
    ``options.prompt_template``.

    The ``prompt_template`` field lives **inside the node's ``options``
    mapping** because that is the shape ``CompositionState.NodeSpec``
    consumes (``node.options["prompt_template"]``) and the shape
    ``yaml_generator.generate_pipeline_dict`` emits to the runtime engine.
    The LLM discriminator is the production ``CompositionState.to_dict()``
    shape: ``node_type == "transform"`` and ``plugin == "llm"``.

    Raises typed :class:`InterpretationResolveError` subclasses when:

    * the affected node is not present in ``state.nodes``;
    * the affected node is not a transform node with ``plugin == 'llm'``;
    * the affected node has no ``options`` mapping;
    * ``options.prompt_template`` is missing or not a string;
    * the prompt template does not contain the expected placeholder;
    * the placeholder appears more than once in the template;
    * the prefix immediately before the placeholder matches (case-insensitive)
      any of :data:`_STRUCTURAL_DIRECTIVE_PREFIXES`.

    The helper is pure (no DB IO). It is called from
    :meth:`SessionServiceImpl.resolve_interpretation_event` BEFORE the
    composition-state UPDATE so any raise short-circuits the resolve
    transaction cleanly.
    """
    if state.nodes is None:
        raise InterpretationNodeMissingError(
            f"_patch_llm_transform_prompt: composition state has no nodes; node {affected_node_id!r} is not present"
        )

    patched_nodes: list[Mapping[str, Any]] = []
    found = False
    for node in state.nodes:
        if node["id"] != affected_node_id:
            patched_nodes.append(node)
            continue

        found = True

        node_type = node["node_type"] if "node_type" in node else None
        if node_type != "transform" or "plugin" not in node:
            raise InterpretationNodePluginMutatedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} has no LLM discriminator; "
                "expected node_type='transform' with plugin='llm'"
            )

        node_plugin = node["plugin"]
        if node_plugin != "llm":
            raise InterpretationNodePluginMutatedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} has plugin "
                f"{node_plugin!r}; only llm nodes carry interpretation placeholders"
            )

        # ``composition_states.nodes`` is Tier-1 (our own audit data) but
        # stored as schemaless JSON. Membership checks here are an
        # offensive pattern: assert the invariant, raise a structured
        # ValueError with a precise message. Direct indexing on the
        # ``Mapping[str, Any]`` annotation lets a wrong type surface as a
        # KeyError/TypeError at the operation site — informative crash
        # rather than fabricated default, per CLAUDE.md offensive
        # programming rules.
        if "options" not in node:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} has no options "
                f"mapping; expected options.prompt_template carrying the placeholder"
            )
        options_value = node["options"]
        if not isinstance(options_value, Mapping):
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} options is not a mapping; "
                "expected options.prompt_template carrying the placeholder"
            )
        options: Mapping[str, Any] = options_value

        if "prompt_template" not in options:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} has no options.prompt_template field"
            )

        template_value = options["prompt_template"]
        if not isinstance(template_value, str):
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} options.prompt_template is not a string"
            )
        template = template_value

        structured_options = _patch_structured_interpretation_prompt(
            options=options,
            affected_node_id=affected_node_id,
            user_term=user_term,
            accepted_value=accepted_value,
        )
        if structured_options is not None:
            patched_node = dict(node)
            patched_node["options"] = structured_options
            patched_nodes.append(patched_node)
            continue

        placeholder_matches = [match for match in INTERPRETATION_PLACEHOLDER_RE.finditer(template) if match.group(1).strip() == user_term]
        placeholder = f"{{{{interpretation:{user_term}}}}}"
        if not placeholder_matches:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: node {affected_node_id!r} options.prompt_template does not contain placeholder {placeholder!r}"
            )

        # Count occurrences. Exactly one is required; more is a structural
        # error (the LLM emitted an ambiguous template; we cannot know which
        # site the user resolution should bind to).
        if len(placeholder_matches) != 1:
            raise InterpretationPlaceholderConsumedError(
                f"_patch_llm_transform_prompt: placeholder {placeholder!r} appears "
                f"{len(placeholder_matches)} times in node {affected_node_id!r}'s options.prompt_template; "
                f"the placeholder must appear exactly once"
            )
        placeholder_match = placeholder_matches[0]

        # Structural-directive guard: the substring ending at the placeholder
        # is the "prefix immediately before". We strip trailing whitespace
        # (so "System: {{...}}" is caught even though there's a space before
        # the placeholder) and compare case-insensitively against the closed
        # prefix list.
        prefix = template[: placeholder_match.start()].rstrip()
        prefix_lower = prefix.lower()
        for directive in _STRUCTURAL_DIRECTIVE_PREFIXES:
            if prefix_lower.endswith(directive):
                raise InterpretationPlaceholderConsumedError(
                    f"_patch_llm_transform_prompt: placeholder {placeholder!r} in node "
                    f"{affected_node_id!r} is immediately preceded by structural "
                    f"directive {directive!r}; substituting into a directive position "
                    f"would produce a broken prompt"
                )

        # Patch is a single span replacement; we already verified exactly one
        # matching placeholder so this is unambiguous. The resolved string is written
        # back into ``options.prompt_template`` so it lands on
        # ``NodeSpec.options`` after ``state_from_record`` and flows into the
        # runtime YAML emitted by ``generate_pipeline_dict``. The same helper
        # also writes the resolved-prompt-template hash into
        # ``options.resolved_prompt_template_hash`` (the cross-DB anchor
        # the LLM transform plugin reads at execution time to populate the
        # Landscape ``calls.resolved_prompt_template_hash`` column).
        new_template = f"{template[: placeholder_match.start()]}{accepted_value}{template[placeholder_match.end() :]}"
        patched_node = dict(node)
        patched_options = dict(options)
        patched_options["prompt_template"] = new_template
        patched_node["options"] = patched_options
        patched_nodes.append(patched_node)

    if not found:
        raise InterpretationNodeMissingError(
            f"_patch_llm_transform_prompt: node {affected_node_id!r} is not present in the composition state's nodes"
        )

    return patched_nodes


def _resolve_vague_term(
    state_record: CompositionStateRecord,
    *,
    affected_node_id: str,
    user_term: str,
    accepted_value: str,
) -> tuple[Mapping[str, Mapping[str, Any]] | None, list[Mapping[str, Any]], str]:
    patched_nodes = _patch_llm_transform_prompt(
        state_record,
        affected_node_id=affected_node_id,
        user_term=user_term,
        accepted_value=accepted_value,
    )
    patched_node = next(n for n in patched_nodes if n["id"] == affected_node_id)
    resolved_template: str = patched_node["options"]["prompt_template"]
    resolved_prompt_template_hash = stable_hash(resolved_template)

    final_nodes: list[Mapping[str, Any]] = []
    for n in patched_nodes:
        if n["id"] == affected_node_id:
            node_with_hash = dict(n)
            options_with_hash = dict(n["options"])
            options_with_hash["resolved_prompt_template_hash"] = resolved_prompt_template_hash
            node_with_hash["options"] = options_with_hash
            final_nodes.append(node_with_hash)
        else:
            final_nodes.append(n)
    # Vague-term review patches only nodes; the sources map is carried forward
    # unchanged. The legacy singular ``source`` column is dead.
    return state_record.sources, final_nodes, resolved_prompt_template_hash


def _surfacing_prompt_structure_hash(
    surfacing_state_record: CompositionStateRecord | None,
    *,
    affected_node_id: str,
) -> str | None:
    """Skeleton hash of the prompt the user reviewed at surfacing time.

    The ``llm_prompt_template`` review approves the LLM-authored prompt
    *skeleton* (text segments + the requirement each slot references), which
    :func:`prompt_structure_hash` deliberately makes invariant under
    interpretation resolution. Comparing this surfacing skeleton to the live
    skeleton at resolve time distinguishes a benign vague-term bake (skeleton
    unchanged → accept) from a genuine prompt edit (skeleton changed → reject as
    stale).

    Returns ``None`` when the surfacing state, the affected node, or its prompt
    parts are unavailable (legacy no-parts nodes) — the caller then falls back
    to rendered-text equality.
    """
    if surfacing_state_record is None:
        return None
    for node in surfacing_state_record.nodes or ():
        if node["id"] == affected_node_id:
            options = node["options"] if "options" in node else None
            if isinstance(options, Mapping):
                return prompt_structure_hash_from_options(options)
            return None
    return None


def _resolve_prompt_template_review(
    state_record: CompositionStateRecord,
    *,
    event_id: str,
    affected_node_id: str,
    user_term: str,
    accepted_value: str,
    surfacing_structure_hash: str | None,
) -> tuple[Mapping[str, Mapping[str, Any]] | None, list[Mapping[str, Any]], str]:
    node = _find_llm_transform_node(
        state_record,
        affected_node_id=affected_node_id,
        context="resolve_interpretation_event",
    )
    options = _require_mapping(
        node["options"],
        message=f"resolve_interpretation_event: node {affected_node_id!r} options is not a mapping",
    )
    prompt_template = options["prompt_template"]
    if not isinstance(prompt_template, str):
        raise InterpretationPlaceholderConsumedError(
            f"resolve_interpretation_event: node {affected_node_id!r} options.prompt_template is not a string"
        )
    # Acceptance gate: the user is approving the prompt SKELETON the LLM authored
    # (text segments + the requirement each slot references). For a structured
    # node (prompt_template_parts present) the skeleton hash is invariant under
    # vague-term resolution — resolving a sibling vague_term first rewrites the
    # rendered options.prompt_template but PRESERVES the parts — so we gate on
    # skeleton equality, NOT rendered-text equality. Gating on rendered text
    # permanently bricked this review whenever a sibling vague_term was resolved
    # first (elspeth-e51216d305: the frozen surfacing draft could never again
    # equal the post-bake template). A genuine prompt edit changes the skeleton
    # and is still rejected as stale. Legacy no-parts nodes have no skeleton;
    # they fall back to the original rendered-text equality.
    live_structure_hash = prompt_structure_hash_from_options(options)
    if live_structure_hash is not None or surfacing_structure_hash is not None:
        if live_structure_hash != surfacing_structure_hash:
            raise InterpretationPlaceholderConsumedError(
                "resolve_interpretation_event: llm_prompt_template prompt skeleton no longer matches the structure the review approved"
            )
    elif accepted_value != prompt_template:
        raise InterpretationPlaceholderConsumedError(
            "resolve_interpretation_event: llm_prompt_template accepted value must equal current options.prompt_template"
        )
    requirements, matching_index = _matching_pending_requirement_index(
        options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None,
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        user_term=user_term,
        context="resolve_interpretation_event",
    )
    # Node-level / returned hash stays the final-prompt-string hash (the runtime
    # LLM plugin reads options.resolved_prompt_template_hash to populate
    # calls.resolved_prompt_template_hash). The REQUIREMENT-level attestation
    # anchor, by contrast, is the prompt *skeleton* for structured nodes: the
    # prompt-template review approves the LLM-authored structure, while the
    # vague-term reviews approve the slot values. Anchoring the requirement to
    # the skeleton keeps it invariant under vague-term resolution (which rewrites
    # the rendered prompt) — see interpretation_state.prompt_structure_hash.
    resolved_prompt_template_hash = stable_hash(prompt_template)
    structure_hash = prompt_structure_hash_from_options(options)
    requirement_anchor_hash = structure_hash if structure_hash is not None else resolved_prompt_template_hash
    requirement = dict(requirements[matching_index])
    requirement["status"] = "resolved"
    requirement["event_id"] = event_id
    requirement["accepted_value"] = accepted_value
    requirement["resolved_prompt_template_hash"] = requirement_anchor_hash
    requirements[matching_index] = requirement

    final_nodes: list[Mapping[str, Any]] = []
    for current_node in state_record.nodes or ():
        if current_node["id"] == affected_node_id:
            patched_node = dict(current_node)
            patched_options = dict(options)
            patched_options["resolved_prompt_template_hash"] = resolved_prompt_template_hash
            patched_options[INTERPRETATION_REQUIREMENTS_KEY] = requirements
            patched_node["options"] = patched_options
            final_nodes.append(patched_node)
        else:
            final_nodes.append(current_node)
    # Prompt-template review patches only node review metadata; the sources map
    # is carried forward unchanged. The legacy singular ``source`` column is dead.
    return state_record.sources, final_nodes, resolved_prompt_template_hash


def _resolve_invented_source(
    state_record: CompositionStateRecord,
    *,
    event_id: str,
    affected_node_id: str,
    user_term: str,
    llm_draft: str,
    accepted_value: str,
) -> tuple[Mapping[str, Mapping[str, Any]], list[Mapping[str, Any]], None]:
    if affected_node_id != SOURCE_COMPONENT_ID:
        raise InterpretationNodeMissingError(
            f"resolve_interpretation_event: invented_source must target affected_node_id {SOURCE_COMPONENT_ID!r}"
        )
    # Multi-source: the default source under review lives in the ``sources`` map
    # keyed by ``SOURCE_COMPONENT_ID`` (interpretation review is scoped to the
    # default component). The legacy singular ``source`` column is dead.
    sources_map = _require_mapping(
        state_record.sources,
        message="resolve_interpretation_event: invented_source requires a persisted sources mapping",
    )
    source = _require_mapping(
        sources_map[SOURCE_COMPONENT_ID] if SOURCE_COMPONENT_ID in sources_map else None,
        message=f"resolve_interpretation_event: invented_source requires a persisted {SOURCE_COMPONENT_ID!r} source mapping",
    )
    options = _require_mapping(
        source["options"] if "options" in source else None,
        message="resolve_interpretation_event: invented_source requires source.options",
    )
    source_authoring = _require_mapping(
        options[SOURCE_AUTHORING_KEY] if SOURCE_AUTHORING_KEY in options else None,
        message=f"resolve_interpretation_event: invented_source requires source.options.{SOURCE_AUTHORING_KEY}",
    )
    content_hash = source_authoring["content_hash"] if "content_hash" in source_authoring else None
    if not isinstance(content_hash, str) or not content_hash:
        raise InterpretationPlaceholderConsumedError(
            f"resolve_interpretation_event: source.options.{SOURCE_AUTHORING_KEY}.content_hash must be populated"
        )
    requirements, matching_index = _matching_pending_requirement_index(
        options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None,
        kind=InterpretationKind.INVENTED_SOURCE,
        user_term=user_term,
        context="resolve_interpretation_event",
    )
    requirement = dict(requirements[matching_index])
    draft = requirement["draft"] if "draft" in requirement else None
    if isinstance(draft, str) and draft != llm_draft:
        raise InterpretationPlaceholderConsumedError(
            "resolve_interpretation_event: invented_source event draft does not match the source review requirement draft"
        )
    requirement["status"] = "resolved"
    requirement["event_id"] = event_id
    requirement["accepted_value"] = accepted_value
    requirement["accepted_artifact_hash"] = content_hash
    requirements[matching_index] = requirement

    patched_authoring = dict(source_authoring)
    patched_authoring["review_event_id"] = event_id
    patched_authoring["resolved_kind"] = InterpretationKind.INVENTED_SOURCE.value
    patched_options = dict(options)
    patched_options[SOURCE_AUTHORING_KEY] = patched_authoring
    patched_options[INTERPRETATION_REQUIREMENTS_KEY] = requirements
    patched_source = dict(source)
    patched_source["options"] = patched_options
    # Splice the patched default source back into the sources map so the patch
    # is persisted (the caller writes the returned map to ``sources``). Other
    # named sources, if any, are carried forward untouched.
    patched_sources = dict(sources_map)
    patched_sources[SOURCE_COMPONENT_ID] = patched_source
    return patched_sources, list(state_record.nodes or ()), None


def _resolve_pipeline_decision_review(
    state_record: CompositionStateRecord,
    *,
    event_id: str,
    affected_node_id: str,
    user_term: str,
    llm_draft: str,
    accepted_value: str,
) -> tuple[Mapping[str, Mapping[str, Any]] | None, list[Mapping[str, Any]], None]:
    node = _find_interpretation_review_node(
        state_record,
        affected_node_id=affected_node_id,
        context="resolve_interpretation_event",
    )
    options = _require_mapping(
        node["options"] if "options" in node else None,
        message=f"resolve_interpretation_event: node {affected_node_id!r} options is not a mapping",
    )
    requirements, matching_index = _matching_pending_requirement_index(
        options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None,
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=user_term,
        context="resolve_interpretation_event",
    )
    requirement = dict(requirements[matching_index])
    draft = requirement["draft"] if "draft" in requirement else None
    if isinstance(draft, str) and draft != llm_draft:
        raise InterpretationPlaceholderConsumedError(
            "resolve_interpretation_event: pipeline_decision event draft does not match the node review requirement draft"
        )
    _validate_pipeline_decision_semantics_from_state_record(
        state_record,
        affected_node_id=affected_node_id,
        user_term=user_term,
        draft=draft,
        context="resolve_interpretation_event",
    )
    decision_hash = _pipeline_decision_artifact_hash_from_state_record(
        state_record,
        affected_node_id=affected_node_id,
        user_term=user_term,
    )
    requirement["status"] = "resolved"
    requirement["event_id"] = event_id
    requirement["accepted_value"] = accepted_value
    requirement["accepted_artifact_hash"] = decision_hash
    requirements[matching_index] = requirement

    final_nodes: list[Mapping[str, Any]] = []
    for current_node in state_record.nodes or ():
        if current_node["id"] == affected_node_id:
            patched_node = dict(current_node)
            patched_options = dict(options)
            patched_options[INTERPRETATION_REQUIREMENTS_KEY] = requirements
            patched_node["options"] = patched_options
            final_nodes.append(patched_node)
        else:
            final_nodes.append(current_node)
    # Pipeline-decision review patches only node review metadata; the sources map
    # is carried forward unchanged. The legacy singular ``source`` column is dead.
    return state_record.sources, final_nodes, None


def _resolve_model_choice_review(
    state_record: CompositionStateRecord,
    *,
    event_id: str,
    affected_node_id: str,
    user_term: str,
    llm_draft: str,
    accepted_value: str,
) -> tuple[Mapping[str, Mapping[str, Any]] | None, list[Mapping[str, Any]], None]:
    """Resolve an ``llm_model_choice`` review on an LLM node.

    Parallel to :func:`_resolve_pipeline_decision_review`. The reviewed
    artifact is the LLM node's ``options.model`` identifier, which the
    composer authored and the mutation-time auto-stager
    (:func:`elspeth.web.interpretation_state._options_with_default_model_choice_review`)
    surfaced for review. Resolving stamps the requirement and writes
    ``accepted_value`` into ``options.model`` so the audit record (what the
    operator approved) and the runnable pipeline (what executes) cannot
    diverge:

    * ``accepted_as_drafted`` — ``accepted_value`` equals the existing
      ``options.model`` (the drafted identifier); the write is idempotent.
    * ``amended`` — ``accepted_value`` is the operator's substituted model
      identifier; the write applies it so the node runs the approved model.

    No prompt-template patch occurs (model choice is a different field than
    the prompt), so the resolved-prompt-template hash is ``None``.
    """
    node = _find_interpretation_review_node(
        state_record,
        affected_node_id=affected_node_id,
        context="resolve_interpretation_event",
    )
    plugin = node["plugin"] if "plugin" in node else None
    if plugin != "llm":
        # Tier 1 invariant: an llm_model_choice requirement is only ever
        # auto-staged on an llm node. A resolve targeting any other plugin
        # means our own state is corrupt — fail loud, do not coerce.
        raise AuditIntegrityError(
            f"resolve_interpretation_event: llm_model_choice review targets node "
            f"{affected_node_id!r} with plugin {plugin!r}; expected 'llm'"
        )
    options = _require_mapping(
        node["options"] if "options" in node else None,
        message=f"resolve_interpretation_event: node {affected_node_id!r} options is not a mapping",
    )
    requirements, matching_index = _matching_pending_requirement_index(
        options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None,
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term=user_term,
        context="resolve_interpretation_event",
    )
    requirement = dict(requirements[matching_index])
    draft = requirement["draft"] if "draft" in requirement else None
    if isinstance(draft, str) and draft != llm_draft:
        raise InterpretationPlaceholderConsumedError(
            "resolve_interpretation_event: llm_model_choice event draft does not match the node review requirement draft"
        )
    requirement["status"] = "resolved"
    requirement["event_id"] = event_id
    requirement["accepted_value"] = accepted_value
    # Read-side drift guard (_validate_model_choice_review) recomputes
    # stable_hash(options.model) and compares it to this field, so the
    # resolved requirement must carry the hash of the accepted model. The
    # field is named for the prompt-template case but is reused here as the
    # model-choice review's anchor hash (mirroring _resolve_prompt_template_review).
    requirement["resolved_prompt_template_hash"] = model_choice_artifact_hash(accepted_value)
    requirements[matching_index] = requirement

    final_nodes: list[Mapping[str, Any]] = []
    for current_node in state_record.nodes or ():
        if current_node["id"] == affected_node_id:
            patched_node = dict(current_node)
            patched_options = dict(options)
            patched_options[INTERPRETATION_REQUIREMENTS_KEY] = requirements
            patched_options["model"] = accepted_value
            patched_node["options"] = patched_options
            final_nodes.append(patched_node)
        else:
            final_nodes.append(current_node)
    # Model-choice review patches only node options; the sources map is
    # carried forward unchanged. The legacy singular ``source`` column is dead.
    return state_record.sources, final_nodes, None


class SessionServiceImpl:
    """Concrete session service backed by SQLAlchemy Core.

    All public methods are async. Database I/O runs through _run_sync() in a
    bounded worker thread so the async event loop is never blocked.
    """

    def __init__(
        self,
        engine: Engine,
        data_dir: Path | None = None,
        *,
        telemetry: _SessionsTelemetry,
        log: structlog.stdlib.BoundLogger,
        plugin_snapshot_factory: Callable[[str], PluginAvailabilitySnapshot] | None = None,
        operator_profile_registry: OperatorProfileRegistry | None = None,
        catalog: CatalogService | None = None,
    ) -> None:
        if (plugin_snapshot_factory is None) != (operator_profile_registry is None):
            raise ValueError("plugin_snapshot_factory and operator_profile_registry must be configured together")
        if plugin_snapshot_factory is not None and catalog is None:
            raise ValueError("profile-aware session validation requires the authoritative catalog")
        self._engine = engine
        self._data_dir = data_dir
        self._telemetry = telemetry
        self._log = log
        self._plugin_snapshot_factory = plugin_snapshot_factory
        self._operator_profile_registry = operator_profile_registry
        self._catalog = catalog

    def _validate_patched_composition_state(
        self,
        state: CompositionState,
        *,
        plugin_snapshot: PluginAvailabilitySnapshot | None,
    ) -> ValidationSummary:
        """Validate a post-review state through its executable profile view."""
        if self._plugin_snapshot_factory is None:
            return state.validate()
        if plugin_snapshot is None:
            raise AuditIntegrityError("Profile-aware composition validation has no principal snapshot")

        from elspeth.web.plugin_policy.validation import validate_authored_composition_state

        assert self._operator_profile_registry is not None
        assert self._catalog is not None
        result = validate_authored_composition_state(
            state,
            snapshot=plugin_snapshot,
            profile_registry=self._operator_profile_registry,
            catalog=self._catalog,
        )
        return result.validation

    async def _plugin_snapshot_for_session(self, session_id: str) -> PluginAvailabilitySnapshot | None:
        """Build a principal snapshot before a session write transaction starts."""
        if self._plugin_snapshot_factory is None:
            return None

        def _sync() -> PluginAvailabilitySnapshot | None:
            with self._engine.connect() as conn:
                user_id = conn.execute(select(sessions_table.c.user_id).where(sessions_table.c.id == session_id)).scalar_one_or_none()
            if user_id is None:
                return None
            assert self._plugin_snapshot_factory is not None
            return self._plugin_snapshot_factory(user_id)

        return cast("PluginAvailabilitySnapshot | None", await self._run_sync(_sync))

    async def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous callable in the thread pool executor."""
        return await run_sync_in_worker(func, *args, **kwargs)

    def _now(self) -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Restore UTC tzinfo stripped by SQLite round-trip.

        SQLite stores DateTime(timezone=True) as ISO-8601 text and drops
        tzinfo on read.  All timestamps in this service originate from
        _now() which uses UTC, so re-attaching UTC is safe.
        """
        if dt.tzinfo is not None:
            return dt
        return dt.replace(tzinfo=UTC)

    def _acquire_session_advisory_lock(self, conn: Connection, session_id: str) -> None:
        """Acquire a session write lock for the duration of the
        current transaction. Released automatically on COMMIT or ROLLBACK.

        SQLite: no-op. SQLite serialization is owned by
        ``_session_write_lock`` below; this helper exists only for the
        PostgreSQL advisory-lock SQL and remains no-op on SQLite so callers
        can test the dialect-specific SQL separately.

        PostgreSQL: pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
        hashtext(session_id)) -- the **two-argument** form
        (B3 from the advisory-lock review synthesis). The classid namespace
        is reserved in src/elspeth/contracts/advisory_locks.py and is
        on-the-wire ABI under change control; do not open-code the literal
        here, always import the constant.

        Hash-function notes:

        * ``pg_advisory_xact_lock(int, int)`` requires two signed int4
          arguments. ``hashtext(text)`` returns int4 directly. Do not use
          ``hashtextextended(... )::int``: PostgreSQL integer casts are
          range-checked and may fail before the lock is acquired.
        * Birthday collisions become probable around ~65k *concurrent*
          sessions hashing to the same classid slot. This
          is benign -- the unique index ix_chat_messages_session_sequence
          is the correctness guarantee; the advisory lock is a
          contention-reducer ahead of it. Collisions cause spurious
          serialisation between two unrelated sessions, never duplicate
          rows or lost writes.
        * The classid value is **NOT** a deployment knob. Two ELSPETH
          instances on the same Postgres cluster (including different
          versions during a rolling deploy) MUST share the same value
          or they will not mutually exclude each other. See
          src/elspeth/contracts/advisory_locks.py for the ABI commitment.
        """
        dialect = self._engine.dialect.name
        if dialect == "sqlite":
            return  # SQLite serialization is owned by _session_write_lock
        if dialect == "postgresql":
            acquire_session_advisory_xact_lock(conn, session_id)
            return
        raise NotImplementedError(f"_acquire_session_advisory_lock not implemented for dialect {dialect}")

    def _sqlite_lock_for_session(self, session_id: str) -> threading.RLock:
        """Return the process-wide SQLite write lock for one DB/session pair."""
        return sqlite_session_mutex(self._engine, session_id)

    @contextlib.contextmanager
    def _session_process_locked_begin(self, session_id: str) -> Iterator[Connection]:
        """Acquire SQLite process exclusion before opening the DB transaction."""
        with process_session_lock(self._engine, session_id), self._engine.begin() as conn:
            yield conn

    def _assert_session_write_lock_held(
        self,
        conn: Connection,
        session_id: str,
        *,
        caller: str,
    ) -> None:
        """Mechanical precondition guard for session-scoped allocators.

        The docstring precondition on _reserve_sequence_range and
        _insert_composition_state is not enough: future callers can forget the
        lock and still pass type checks. _session_write_lock sets a per-thread
        ContextVar token keyed by (id(conn), session_id); lock-requiring helpers
        crash immediately if called without that token in the same transaction.
        """
        if (id(conn), session_id) not in _SESSION_WRITE_LOCK_HELD.get():
            raise RuntimeError(
                f"{caller}: _session_write_lock(conn, {session_id!r}) must be "
                "held in the same transaction before allocating session-scoped "
                "sequence/version values"
            )

    @contextlib.contextmanager
    def _session_write_lock(self, conn: Connection, session_id: str) -> Iterator[None]:
        """Serialize same-session sequence/version allocators.

        PostgreSQL uses the transaction-scoped advisory lock. SQLite uses a
        process-wide per-session RLock held until the surrounding transaction
        commits or rolls back. Every caller that performs ``SELECT MAX(...) +
        1`` for ``chat_messages.sequence_no`` or ``composition_states.version``
        MUST wrap that read and every dependent INSERT in this context.
        """
        key = (id(conn), session_id)
        held = _SESSION_WRITE_LOCK_HELD.get()
        token = _SESSION_WRITE_LOCK_HELD.set(held | {key})
        dialect = self._engine.dialect.name
        try:
            if dialect == "sqlite":
                with sqlite_transaction_session_lock(conn, self._engine, session_id):
                    yield
                return
            if dialect == "postgresql":
                self._acquire_session_advisory_lock(conn, session_id)
                yield
                return
            raise NotImplementedError(f"_session_write_lock not implemented for dialect {dialect}")
        finally:
            _SESSION_WRITE_LOCK_HELD.reset(token)

    def _reserve_sequence_range(self, conn: Connection, session_id: str, *, count: int) -> int:
        """Reserve ``count`` consecutive sequence numbers for ``session_id``.

        PRECONDITION: caller MUST be inside
        ``with self._session_write_lock(conn, session_id):`` in the same
        transaction. That context acquires the PostgreSQL advisory lock or
        the SQLite process-local session lock, making the unilateral
        ``SELECT MAX + 1`` allocation race-free under concurrent writers.

        Inside the same transaction, performs:
            SELECT COALESCE(MAX(sequence_no), 0) FROM chat_messages WHERE session_id = ?
        and returns max+1. The caller writes rows at max+1, max+2, ... max+count.

        The session write lock prevents same-session allocator collisions
        on both PostgreSQL and SQLite. Do not call this helper outside
        the context, even in tests.

        Note: gaps in sequence_no are permitted (transaction rollback after
        reservation leaves the next caller's MAX+1 higher than the first
        successful row's sequence_no). Sequence_no is an ordering key, not a count.

        *Design seam acknowledgement (synthesised review A-F9 / M11).*
        The three-helper protocol (``_acquire_session_advisory_lock`` →
        ``_reserve_sequence_range`` → ``_insert_chat_message``) requires
        callers to invoke them in order. Schedule 1A has the current
        add-message path and ``fork_session`` copy path; Schedule 1B adds
        ``persist_compose_turn``. Every caller must enter
        ``_session_write_lock`` before reserving a sequence range. If a
        later phase adds another invocation site, consider consolidating
        into a single ``_write_chat_messages_atomic(conn, session_id, rows)``
        entry point that hides the protocol. Until that third site appears,
        the consolidation is not justified.
        """
        self._assert_session_write_lock_held(
            conn,
            session_id,
            caller="_reserve_sequence_range",
        )
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        # SQLAlchemy 2.x ``select(func.max(...))`` is the project-standard
        # idiom (see existing ``save_composition_state`` at
        # ``service.py:398``); using it here keeps the pattern uniform
        # and gives mypy a typed ``int | None`` instead of the ``Any``
        # that ``text(...)`` plus ``.first()`` produces. Closes
        # synthesised review finding P-L-1 / L15.
        current_max = conn.execute(
            select(func.coalesce(func.max(chat_messages_table.c.sequence_no), 0)).where(chat_messages_table.c.session_id == session_id)
        ).scalar_one()
        return int(current_max) + 1

    def _insert_chat_message(
        self,
        conn: Connection,
        /,
        *,
        session_id: str,
        role: str,
        content: str,
        raw_content: str | None,
        tool_calls: Any,
        sequence_no: int,
        writer_principal: ChatMessageWriterPrincipal,
        composition_state_id: str | None,
        tool_call_id: str | None,
        parent_assistant_id: str | None,
        created_at: datetime,
    ) -> str:
        """Single-row insert into ``chat_messages`` with the supplied fields.

        PRECONDITIONS (mechanically enforced — see body):

        1. Caller MUST be inside ``self._session_write_lock(conn, session_id)``
           in the same transaction. The session write lock is what makes
           the ``_reserve_sequence_range`` allocation safe against
           same-session concurrent writers; a writer that bypasses the
           lock could persist a sequence_no that another transaction is
           about to allocate.
        2. Caller MUST have already obtained ``sequence_no`` from
           ``_reserve_sequence_range``. This helper does NOT allocate
           sequence numbers — it persists what the caller supplies.

        ``created_at`` is supplied by the caller so multi-row inserts
        in the same transaction (``persist_compose_turn``) and
        same-transaction ``sessions.updated_at`` writes
        (``add_message``) can share a single timestamp. Generating a
        new ``datetime.now(UTC)`` inside this helper would produce
        per-row drift visible in the audit trail.

        ``raw_content`` is the audit-attribution column for assistant
        messages whose visible ``content`` was rewritten by runtime
        preflight redaction. It MUST be persisted as supplied —
        silently discarding it would regress the pre-rev-4
        ``add_message`` behaviour and create audit-data loss (per
        CLAUDE.md, silent wrong results are worse than a crash).

        If ``role == "tool"``, this helper additionally verifies that
        ``parent_assistant_id`` references an assistant row in the
        same session. The DB FK on
        ``(parent_assistant_id, session_id)`` proves same-session
        existence; SQL cannot portably enforce that the referenced
        row's role is ``assistant``. The guard at the service-writer
        boundary closes that gap.

        Returns the newly-allocated UUID-shaped message id (so the
        caller can persist downstream references — e.g.
        ``tool_call_id`` parents — without a follow-up SELECT).
        """
        self._assert_session_write_lock_held(
            conn,
            session_id,
            caller="_insert_chat_message",
        )
        if role == "tool":
            if parent_assistant_id is None:
                raise RuntimeError(f"_insert_chat_message: tool row requires parent_assistant_id (session={session_id!r})")
            _assert_parent_assistant_message(
                conn,
                parent_assistant_id=parent_assistant_id,
                session_id=session_id,
                caller="_insert_chat_message",
            )
        elif role == "assistant":
            _assert_assistant_row_has_audit_content(
                content=content,
                raw_content=raw_content,
                tool_calls=tool_calls,
                caller="_insert_chat_message",
            )
        msg_id = str(uuid.uuid4())
        conn.execute(
            insert(chat_messages_table).values(
                id=msg_id,
                session_id=session_id,
                role=role,
                content=content,
                raw_content=raw_content,
                tool_calls=tool_calls,
                sequence_no=sequence_no,
                writer_principal=writer_principal,
                composition_state_id=composition_state_id,
                tool_call_id=tool_call_id,
                parent_assistant_id=parent_assistant_id,
                created_at=created_at,
            )
        )
        return msg_id

    def _insert_composition_state(
        self,
        conn: Connection,
        *,
        session_id: str,
        payload: StatePayload,
        provenance: str,
        created_at: datetime | None = None,
        state_id: str | None = None,
    ) -> str:
        """Single-row insert into composition_states with per-session
        version allocation under _session_write_lock.

        Takes a single :class:`StatePayload` carrying
        ``data`` (a :class:`CompositionStateData`) and
        ``derived_from_state_id`` rather than two separate keyword
        arguments. Bundling the two coheres with B1 — the payload object
        is the unit of state-advance, so a helper that takes it as a
        unit prevents future callers from passing inconsistent
        ``(state, derived_from_state_id)`` pairs. Critically,
        ``StatePayload`` does NOT carry a caller-supplied ``version``;
        version allocation remains inside this helper, under the held
        lock (see B1 below).

        PRECONDITION: caller MUST be inside
        ``with self._session_write_lock(conn, session_id):`` in the same
        transaction. The context is what makes the
        ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
        WHERE session_id = :sid`` allocation race-free under concurrent
        writers — without it, two callers could both observe MAX = N,
        both pick N+1, and the loser's INSERT would hit
        ``uq_composition_state_version``. The locked-path
        ``IntegrityError`` handler classifies that as a Tier-1
        audit-integrity violation, fabricating a Tier-1 alert from a
        benign contention loss. **Under ELSPETH's auditability standard
        fabricated Tier-1 violations are evidence-tampering-class harm:
        the audit trail asserts a violation that did not occur.** The
        session write lock makes the SELECT-MAX-then-INSERT sequence
        atomic against every other writer for this ``session_id`` on
        both PostgreSQL and SQLite, so the fabrication path is
        structurally unreachable. Closes B1/B3 from the state-persistence
        review synthesis. Also mirrors the precondition contract on
        ``_reserve_sequence_range``.

        Version allocation is per-session: the COALESCE query filters by
        ``session_id`` because ``uq_composition_state_version`` is a
        per-session constraint. A global MAX
        would silently break the per-session monotonic-version contract
        every read path assumes.

        This helper does NOT contain a retry loop. The lock + atomic
        SELECT-INSERT makes one a defensive-programming anti-pattern.

        Writes the real per-column schema (source/nodes/edges/outputs/
        metadata_/is_valid/validation_errors/composer_meta/
        derived_from_state_id), using the shared
        ``_enveloped_state_column(...)`` and ``deep_thaw(...)`` patterns
        the existing inline inserts use today.

        The ``provenance`` argument must satisfy the
        ``ck_composition_states_provenance`` CHECK constraint; passing an
        unknown value raises ``IntegrityError``.

        The ``created_at`` argument is optional. When ``None`` (the
        default), the helper stamps ``datetime.now(UTC)`` at insert
        time. Callers that need cross-table timestamp consistency within
        a single transaction (e.g. ``fork_session`` pre-computes ``now``
        at the top of its sync block and reuses it across the
        ``sessions``, ``composition_states``, and ``chat_messages``
        inserts so all rows share one wall-clock instant) MUST pass an
        explicit ``created_at``. Earlier B1 framing (helper hardcoding
        ``now()`` silently changed fork timestamp semantics) is preserved.

        The optional ``state_id`` exists for ``fork_session``, which
        already precomputes ``copied_state_id_str`` and uses that same
        id for chat-row ``composition_state_id`` FKs and returned
        records. Other callers leave it ``None`` and let the helper
        allocate a fresh UUID.
        """
        self._assert_session_write_lock_held(
            conn,
            session_id,
            caller="_insert_composition_state",
        )
        # Unpack the bundled payload once so the rest of the body refers
        # to ``state`` and ``derived_from_state_id`` exactly as it did
        # pre-1B-refactor. This avoids touching the version-allocation
        # arithmetic, the per-column INSERT, or the IntegrityError
        # handler — all of which Schedule 1A reviewed and merged.
        state = payload.data
        derived_from_state_id = payload.derived_from_state_id
        # B1: allocate version under _session_write_lock. The
        # SELECT-MAX-then-INSERT sequence is atomic against every other
        # writer for this session because the caller is required to be
        # inside ``_session_write_lock(conn, session_id)`` for the full
        # transaction (see PRECONDITION above). The COALESCE pins the
        # first state to version 1; the WHERE clause makes the
        # allocation per-session, matching ``uq_composition_state_version``'s
        # scope.
        next_version = conn.execute(
            select(func.coalesce(func.max(composition_states_table.c.version), 0) + 1).where(
                composition_states_table.c.session_id == session_id
            )
        ).scalar_one()
        allocated_state_id = state_id or str(uuid.uuid4())
        conn.execute(
            insert(composition_states_table).values(
                id=allocated_state_id,
                session_id=session_id,
                version=int(next_version),
                source=None,
                sources=_enveloped_state_column(state.sources),
                nodes=_enveloped_state_column(state.nodes),
                edges=_enveloped_state_column(state.edges),
                outputs=_enveloped_state_column(state.outputs),
                metadata_=_enveloped_state_column(state.metadata_),
                is_valid=state.is_valid,
                validation_errors=deep_thaw(state.validation_errors),
                composer_meta=_enveloped_state_column(state.composer_meta),
                derived_from_state_id=derived_from_state_id,
                provenance=provenance,
                created_at=created_at if created_at is not None else datetime.now(UTC),
            )
        )
        return allocated_state_id

    def persist_compose_turn(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: ChatMessageWriterPrincipal,
        plugin_crash_pending: bool,
    ) -> AuditOutcome:
        """Synchronous, single-transaction persistence of one compose turn.

        Spec §5.2.2. Concrete sync primitive. Production async callers MUST
        invoke ``await self.persist_compose_turn_async(...)`` through
        :class:`SessionServiceProtocol`; that dispatcher uses ``_run_sync``
        under the hood. Calling this sync primitive directly from async land
        would block the event loop because the body opens a synchronous
        SQLAlchemy transaction.

        The async-loop guard below uses ``asyncio.get_running_loop()`` to
        detect misuse: if there is a running loop in the calling thread,
        we are in async land and MUST refuse. ``RuntimeError`` is the
        canonical "you called the wrong API" signal — the call site is
        a bug, not a recoverable user error. Closes synthesised review
        finding SA-7 / M1.

        Order of work (load-bearing):

        1. Pre-DB transcript validation (``_validate_tool_call_id_set_equality``).
           Pure function of caller args; runs BEFORE ``_engine.begin()`` so
           a contract violation cannot leave a half-written audit trail.
        2. Open transaction; acquire session write lock.
        3. Cross-session guard on ``parent_composition_state_id`` (B5).
        4. Stale-state guard on ``expected_current_state_id``.
        5. Reserve sequence range for assistant + N tool rows.
        6. Insert assistant row (with optional ``raw_content``,
           B2 audit-attribution).
        7. For each tool row: optionally insert composition state under
           the held lock, then insert tool chat row referencing it.

        ``raw_content`` is the audit-attribution column for assistant
        messages whose visible ``content`` was rewritten by runtime
        preflight redaction. Routes already pass
        ``raw_content=result.raw_assistant_content`` to ``add_message``;
        compose-loop call sites use ``persist_compose_turn``, so the
        primitive must accept and persist the column today (B2).
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread -- we are in a worker thread
            # or pure sync test context. Proceed.
            pass
        else:
            raise RuntimeError(
                "persist_compose_turn must be dispatched via "
                "await self.persist_compose_turn_async(...) -- "
                "calling it directly from a coroutine blocks the event "
                "loop on synchronous DB I/O."
            )

        # Q-F1 (Step 3c): transcript validation BEFORE _engine.begin().
        # Pre-lock, pre-transaction; pure function of caller args.
        _validate_tool_call_id_set_equality(
            redacted_assistant_tool_calls=redacted_assistant_tool_calls,
            redacted_tool_rows=redacted_tool_rows,
        )

        now = self._now()
        # IntegrityError disposition (spec §4.5): the catch is
        # OUTSIDE ``with self._engine.begin()`` deliberately. Order is
        # load-bearing — the ``with`` context's ``__exit__`` runs first
        # (rolling back the transaction so no partial audit row survives),
        # THEN we increment the operational counter, THEN the exception
        # re-raises so the caller observes the actual constraint
        # violation. Putting the catch INSIDE the ``with`` would fire
        # before rollback completes and could mask the original error.
        #
        # Audit primacy: telemetry signals operational rate; the
        # exception itself is the authoritative signal to the caller.
        # Both channels fire; neither is suppressed. Catch is exactly
        # ``IntegrityError`` -- not ``Exception``, not ``SQLAlchemyError``
        # -- because only constraint violations belong on this counter.
        try:
            with self._session_process_locked_begin(session_id) as conn:
                with self._session_write_lock(conn, session_id):
                    # B5: if a parent
                    # composition state is supplied, it MUST belong to this
                    # session.
                    if parent_composition_state_id is not None:
                        _assert_state_in_session(
                            conn,
                            state_id=parent_composition_state_id,
                            expected_session_id=session_id,
                            caller="persist_compose_turn",
                        )

                    current_state_id = conn.execute(
                        select(composition_states_table.c.id)
                        .where(composition_states_table.c.session_id == session_id)
                        .order_by(composition_states_table.c.version.desc())
                        .limit(1)
                    ).scalar_one_or_none()
                    if current_state_id != expected_current_state_id:
                        raise StaleComposeStateError(
                            "persist_compose_turn: current composition state changed "
                            f"for session_id={session_id!r}; "
                            f"expected={expected_current_state_id!r}, "
                            f"actual={current_state_id!r}. Refusing to persist a "
                            "compose result based on a stale state."
                        )

                    base_seq = self._reserve_sequence_range(conn, session_id, count=1 + len(redacted_tool_rows))

                    assistant_id = self._insert_chat_message(
                        conn,
                        session_id=session_id,
                        role="assistant",
                        content=assistant_content,
                        # B2: raw_content captures pre-redaction LLM output.
                        raw_content=raw_content,
                        # ``deep_thaw`` recursively converts MappingProxyType /
                        # tuple to JSON-serialisable dict / list (closes
                        # P-L-4 / L18). Mirrors ``add_message``'s pattern.
                        tool_calls=deep_thaw(redacted_assistant_tool_calls) if redacted_assistant_tool_calls else None,
                        sequence_no=base_seq,
                        writer_principal=writer_principal,
                        composition_state_id=parent_composition_state_id,
                        tool_call_id=None,
                        parent_assistant_id=None,
                        created_at=now,
                    )

                    for offset, tool_row in enumerate(redacted_tool_rows, start=1):
                        state_id: str | None = None
                        if tool_row.composition_state_payload is not None:
                            state_id = self._insert_composition_state(
                                conn,
                                session_id=session_id,
                                payload=tool_row.composition_state_payload,
                                provenance="tool_call",
                                created_at=now,
                            )
                            current_state_id = state_id
                        self._insert_chat_message(
                            conn,
                            session_id=session_id,
                            role="tool",
                            content=tool_row.content,
                            raw_content=None,
                            tool_calls=None,
                            sequence_no=base_seq + offset,
                            writer_principal=writer_principal,
                            composition_state_id=state_id,
                            tool_call_id=tool_row.tool_call_id,
                            parent_assistant_id=assistant_id,
                            created_at=now,
                        )

                return AuditOutcome(
                    assistant_id=assistant_id,
                    unwind_audit_failed=False,
                    current_state_id=current_state_id,
                )
        except IntegrityError:
            self._telemetry.tool_row_integrity_violation_total.add(1)
            raise
        except OperationalError as audit_exc:
            # OperationalError disposition (spec §5.2.2 / §5.5
            # rows 9-10): the audit insert itself failed (commit-time
            # disk full, fsync failure, network partition, etc.). The
            # ``with self._engine.begin()`` context has already rolled
            # back by the time we enter this handler, so no partial
            # audit row survives.
            #
            # Disposition is asymmetric on ``plugin_crash_pending``:
            #
            # 1. ``plugin_crash_pending=True`` — the tool plugin
            #    already crashed and the caller is on the unwind path
            #    with a captured plugin exception in hand. Surfacing
            #    a separate ``AuditIntegrityError`` here would mask
            #    the original tool failure (which is what the operator
            #    needs to see). Record the audit failure via counter
            #    + slog (the slog call is permitted under CLAUDE.md
            #    primacy because the audit system itself failed —
            #    telemetry has nowhere to write the structured event)
            #    and return ``AuditOutcome(unwind_audit_failed=True)``
            #    so the caller can raise the captured plugin
            #    exception while still surfacing that the unwind
            #    audit row could not be persisted.
            #
            # 2. ``plugin_crash_pending=False`` — the tool succeeded
            #    but the audit insert failed. This is a Tier-1 audit
            #    corruption per CLAUDE.md doctrine: the system did
            #    work that it cannot prove it did. Returning a flag
            #    would let the caller proceed with corrupted audit
            #    state (synthesised review finding H1).
            #    ``AuditIntegrityError`` is registered in
            #    ``TIER_1_ERRORS`` via ``@tier_1_error`` on its
            #    declaration in ``contracts/errors.py``, so
            #    ``except Exception:`` blocks elsewhere cannot
            #    silently swallow it. The original ``OperationalError``
            #    is preserved as the ``__cause__`` via ``from
            #    audit_exc`` so the underlying DB error remains
            #    visible to the operator.
            if plugin_crash_pending:
                self._telemetry.tool_row_persist_failed_during_unwind_total.add(1)
                self._log.warning(
                    "audit_insert_failed_during_tool_failure_unwind",
                    session_id=session_id,
                    audit_exc_class=type(audit_exc).__name__,
                )
                return AuditOutcome(
                    assistant_id=None,
                    unwind_audit_failed=True,
                )
            self._telemetry.tool_row_tier1_violation_total.add(1)
            raise AuditIntegrityError(
                f"persist_compose_turn: audit insert failed for "
                f"session_id={session_id!r} with tool succeeded — "
                f"Tier-1 audit corruption (no recovery)"
            ) from audit_exc
        except SQLAlchemyError as audit_exc:
            # Spec §5.2.2 / §5.5 row 9: any non-Integrity, non-Operational
            # SQLAlchemyError (DataError, DatabaseError, ProgrammingError,
            # InterfaceError, DBAPIError siblings) on the audit insert
            # path is a Tier-1 audit corruption — the audit system
            # itself failed in an unforeseen way that the IntegrityError
            # / OperationalError dispositions above were not designed to
            # handle. The previous narrow catch let these subclasses
            # propagate uncaught past the disposition logic, leaving the
            # tool_row_tier1_violation_total counter dark on the
            # SLO=0 dashboard while the audit row was lost.
            #
            # Disposition matches the OperationalError success-path arm
            # (plugin_crash_pending=False): increment the Tier-1 counter
            # and raise AuditIntegrityError chained through the
            # SQLAlchemyError. Unlike OperationalError, this branch is
            # NOT asymmetric on plugin_crash_pending — there is no
            # established recovery shape for arbitrary SQLAlchemyError
            # subclasses. If a future caller's audit row fails with
            # DataError on the unwind path, masking the audit failure
            # to "preserve" a primary plugin error would be silently
            # losing a Tier-1 corruption signal; the Tier-1 raise is
            # what spec §5.5 row 9 prescribes.
            self._telemetry.tool_row_tier1_violation_total.add(1)
            raise AuditIntegrityError(
                f"persist_compose_turn: audit insert failed for "
                f"session_id={session_id!r} with non-Integrity, "
                f"non-Operational SQLAlchemyError "
                f"({type(audit_exc).__name__}) — Tier-1 audit "
                f"corruption (no recovery)"
            ) from audit_exc

    async def persist_compose_turn_async(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: ChatMessageWriterPrincipal,
        plugin_crash_pending: bool,
    ) -> AuditOutcome:
        """Async dispatcher for :meth:`persist_compose_turn`.

        Bridges to the sync primitive via ``_run_sync``, which dispatches
        to a worker thread. The worker is shielded from caller
        cancellation: a ``CancelledError`` raised in the awaiter does NOT
        interrupt the in-flight sync transaction (see
        ``elspeth.web.async_workers.run_sync_in_worker``).

        **Commit-wins cancellation contract (Q-F2).** When the caller is
        cancelled mid-flight, the underlying worker continues to run to
        completion. Either:

        1. The transaction commits — the assistant + tool rows are durably
           persisted; the caller observes ``CancelledError`` and never
           sees the ``AuditOutcome``. **Callers MUST NOT retry on
           CancelledError** — retrying risks a duplicate tool-call-ID
           INSERT that fires a fabricated Tier-1 counter increment.
        2. The transaction rolls back atomically — DB-level errors
           (``IntegrityError``, ``OperationalError``,
           ``ToolCallIDMismatchError`` raised pre-DB) cause the
           ``engine.begin()`` block to roll back. No rows persisted.
           Retry-on-CancelledError still forbidden.

        The compose loop is the only caller of this method.

        Pinned by ``test_persist_compose_turn_async_caller_cancellation_commits_anyway``.
        """
        return cast(
            AuditOutcome,
            await self._run_sync(
                self.persist_compose_turn,
                session_id=session_id,
                assistant_content=assistant_content,
                raw_content=raw_content,
                redacted_assistant_tool_calls=redacted_assistant_tool_calls,
                redacted_tool_rows=redacted_tool_rows,
                parent_composition_state_id=parent_composition_state_id,
                expected_current_state_id=expected_current_state_id,
                writer_principal=writer_principal,
                plugin_crash_pending=plugin_crash_pending,
            ),
        )

    async def create_session(
        self,
        user_id: str,
        title: str,
        auth_provider_type: AuthProviderType,
        forked_from_session_id: UUID | None = None,
        forked_from_message_id: UUID | None = None,
    ) -> SessionRecord:
        """Create a new session and return its record."""
        session_id = uuid.uuid4()
        now = self._now()

        def _sync() -> None:
            with self._session_process_locked_begin(str(session_id)) as conn:
                conn.execute(
                    insert(sessions_table).values(
                        id=str(session_id),
                        user_id=user_id,
                        auth_provider_type=auth_provider_type,
                        title=title,
                        created_at=now,
                        updated_at=now,
                        forked_from_session_id=str(forked_from_session_id) if forked_from_session_id else None,
                        forked_from_message_id=str(forked_from_message_id) if forked_from_message_id else None,
                    )
                )

        await self._run_sync(_sync)

        return SessionRecord(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=now,
            archived_at=None,
            forked_from_session_id=forked_from_session_id,
            forked_from_message_id=forked_from_message_id,
        )

    async def get_session(self, session_id: UUID) -> SessionRecord:
        """Fetch a session by ID. Raises SessionNotFoundError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise SessionNotFoundError(session_id)

        return SessionRecord(
            id=UUID(row.id),
            user_id=row.user_id,
            auth_provider_type=row.auth_provider_type,
            title=row.title,
            created_at=self._ensure_utc(row.created_at),
            updated_at=self._ensure_utc(row.updated_at),
            archived_at=self._ensure_utc(row.archived_at) if row.archived_at else None,
            forked_from_session_id=UUID(row.forked_from_session_id) if row.forked_from_session_id else None,
            forked_from_message_id=UUID(row.forked_from_message_id) if row.forked_from_message_id else None,
        )

    async def update_session_title(self, session_id: UUID, title: str) -> SessionRecord:
        """Update a session title and return the refreshed record."""
        sid = str(session_id)
        now = self._now()

        def _sync() -> Any:
            with self._engine.begin() as conn:
                result = conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(title=title, updated_at=now))
                if result.rowcount == 0:
                    raise SessionNotFoundError(session_id)
                return conn.execute(select(sessions_table).where(sessions_table.c.id == sid)).one()

        row = await self._run_sync(_sync)
        return SessionRecord(
            id=UUID(row.id),
            user_id=row.user_id,
            auth_provider_type=row.auth_provider_type,
            title=row.title,
            created_at=self._ensure_utc(row.created_at),
            updated_at=self._ensure_utc(row.updated_at),
            archived_at=self._ensure_utc(row.archived_at) if row.archived_at else None,
            forked_from_session_id=UUID(row.forked_from_session_id) if row.forked_from_session_id else None,
            forked_from_message_id=UUID(row.forked_from_message_id) if row.forked_from_message_id else None,
        )

    async def list_sessions(
        self,
        user_id: str,
        auth_provider_type: AuthProviderType,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> list[SessionRecord]:
        """List sessions for a user, ordered by updated_at descending."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                conditions: list[ColumnElement[bool]] = [
                    sessions_table.c.user_id == user_id,
                    sessions_table.c.auth_provider_type == auth_provider_type,
                ]
                if not include_archived:
                    conditions.append(sessions_table.c.archived_at.is_(None))
                return conn.execute(
                    select(sessions_table).where(*conditions).order_by(desc(sessions_table.c.updated_at)).limit(limit).offset(offset)
                ).fetchall()

        rows = await self._run_sync(_sync)

        return [
            SessionRecord(
                id=UUID(row.id),
                user_id=row.user_id,
                auth_provider_type=row.auth_provider_type,
                title=row.title,
                created_at=self._ensure_utc(row.created_at),
                updated_at=self._ensure_utc(row.updated_at),
                archived_at=self._ensure_utc(row.archived_at) if row.archived_at else None,
                forked_from_session_id=UUID(row.forked_from_session_id) if row.forked_from_session_id else None,
                forked_from_message_id=UUID(row.forked_from_message_id) if row.forked_from_message_id else None,
            )
            for row in rows
        ]

    async def archive_session(self, session_id: UUID) -> None:
        """Delete a session and cascade to all related records and files.

        Filesystem cleanup is staged before the DB delete so the service can
        restore the original blob directory if the transaction fails. After the
        DB commit, any purge failure leaves the staged directory in a recoverable
        quarantine path instead of raising after the session is already gone.
        """
        sid = str(session_id)

        def _sync() -> None:
            blob_dir: Path | None = None
            staged_blob_dir: Path | None = None
            if self._data_dir is not None:
                blob_dir = self._data_dir / "blobs" / sid
                if blob_dir.is_dir():
                    staged_blob_dir = self._data_dir / ".archive_quarantine" / sid
                    staged_blob_dir.parent.mkdir(parents=True, exist_ok=True)
                    if staged_blob_dir.exists():
                        raise OSError(
                            f"archive_session({sid}): quarantine path {staged_blob_dir} already exists. "
                            "Manual cleanup of the stale staged blob directory is required before archive can proceed."
                        )
                    blob_dir.rename(staged_blob_dir)

            try:
                with self._engine.begin() as conn:
                    durable_history_exists = (
                        conn.execute(select(runs_table.c.id).where(runs_table.c.session_id == sid).limit(1)).first() is not None
                        or conn.execute(
                            select(composer_completion_events_table.c.id)
                            .where(composer_completion_events_table.c.session_id == sid)
                            .limit(1)
                        ).first()
                        is not None
                    )
                    if durable_history_exists:
                        now = self._now()
                        result = conn.execute(
                            update(sessions_table).where(sessions_table.c.id == sid).values(archived_at=now, updated_at=now)
                        )
                        if result.rowcount == 0:
                            raise SessionNotFoundError(session_id)
                        if staged_blob_dir is not None and blob_dir is not None and staged_blob_dir.exists():
                            blob_dir.parent.mkdir(parents=True, exist_ok=True)
                            staged_blob_dir.rename(blob_dir)
                        return
                    # Session archival is the only supported transcript purge:
                    # delete the parent row and let the schema-owned cascades
                    # remove session-scoped children. Direct chat-message
                    # deletes are blocked by trg_chat_messages_no_delete.
                    conn.execute(delete(sessions_table).where(sessions_table.c.id == sid))
            except Exception as primary_exc:
                if blob_dir is not None and staged_blob_dir is not None and staged_blob_dir.exists():
                    try:
                        blob_dir.parent.mkdir(parents=True, exist_ok=True)
                        staged_blob_dir.rename(blob_dir)
                    except OSError as restore_exc:
                        primary_exc.add_note(
                            f"RecoveryFailed[{type(restore_exc).__name__}]: "
                            f"could not restore staged blob directory for session {sid} "
                            f"from {staged_blob_dir} to {blob_dir} after archive rollback "
                            f"({restore_exc}). Manual cleanup required."
                        )
                raise

            # After commit, the delete succeeded from the caller's perspective.
            # If the final purge fails, keep the staged directory in quarantine
            # for manual cleanup and raise so the operator sees the cleanup
            # failure. The session rows may already be gone, but silently
            # returning here would hide the orphaned quarantine state.
            if staged_blob_dir is not None and staged_blob_dir.exists():
                try:
                    shutil.rmtree(staged_blob_dir)
                except OSError as exc:
                    raise QuarantineCleanupError(
                        f"archive_session({sid}): session delete committed but quarantine cleanup failed for "
                        f"{staged_blob_dir}. Manual cleanup of the staged blob directory is required."
                    ) from exc

                quarantine_root = staged_blob_dir.parent
                with contextlib.suppress(OSError):
                    quarantine_root.rmdir()

        await self._run_sync(_sync)

    async def get_composer_preferences(self, session_id: UUID) -> ComposerSessionPreferencesRecord:
        """Fetch composer trust/scaffolding preferences for a session."""

        def _sync() -> ComposerSessionPreferencesRecord:
            with self._engine.connect() as conn:
                row = conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).one()
                return ComposerSessionPreferencesRecord(
                    session_id=UUID(row.id),
                    trust_mode=row.trust_mode,
                    density_default=row.density_default,
                    interpretation_review_disabled=bool(row.interpretation_review_disabled),
                    updated_at=self._ensure_utc(row.updated_at),
                )

        return cast(ComposerSessionPreferencesRecord, await self._run_sync(_sync))

    async def update_composer_preferences(
        self,
        session_id: UUID,
        *,
        trust_mode: ComposerTrustMode,
        density_default: ComposerDensityDefault,
        actor: str,
    ) -> ComposerSessionPreferencesTransition:
        """Update composer preferences and append the audit event first.

        Returns both the prior and current ``ComposerSessionPreferencesRecord``
        wrapped in a ``ComposerSessionPreferencesTransition``. The prior
        record is loaded **inside the same write transaction** as the
        audit + state writes — no TOCTOU window between read and write.
        Phase 8 plan §"Service signature precondition (B2 — load-bearing)"
        explicitly rejects the route-handler read-before-write
        alternative on these atomicity grounds.

        Audit-primacy ordering: the ``trust_mode.changed`` row is
        inserted before the ``sessions`` UPDATE. The audit payload now
        carries ``prior_trust_mode`` (B1 — see the docstring on
        ``proposal_events_table`` in ``sessions/models.py`` for the
        full payload contract). Telemetry consumers reading
        ``prior.trust_mode`` from this return value are guaranteed to
        find the same value in the audit row, satisfying the
        audit-primacy superset rule.
        """
        now = self._now()
        sid = str(session_id)

        def _sync() -> ComposerSessionPreferencesTransition:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                # B2 (load-bearing): load the prior record inside the same
                # transaction as the audit insert and the state update.
                # A concurrent PATCH cannot interpose because the per-
                # session write lock above serialises writes for this
                # ``sid`` and the SELECT runs inside the same connection
                # as the UPDATE.
                prior_row = conn.execute(select(sessions_table).where(sessions_table.c.id == sid)).one()
                prior_record = ComposerSessionPreferencesRecord(
                    session_id=UUID(prior_row.id),
                    trust_mode=prior_row.trust_mode,
                    density_default=prior_row.density_default,
                    interpretation_review_disabled=bool(prior_row.interpretation_review_disabled),
                    updated_at=self._ensure_utc(prior_row.updated_at),
                )
                # Audit fires before state mutation per CLAUDE.md
                # §"Telemetry and Logging" primacy rule. B1 (load-bearing):
                # the payload now carries ``prior_trust_mode`` so a
                # downstream telemetry counter emitting
                # ``{from_mode, to_mode}`` attributes remains a strict
                # subset of audit-recorded reality.
                conn.execute(
                    insert(proposal_events_table).values(
                        id=str(uuid.uuid4()),
                        session_id=sid,
                        proposal_id=None,
                        event_type="trust_mode.changed",
                        actor=actor,
                        payload={
                            "trust_mode": trust_mode,
                            "prior_trust_mode": prior_record.trust_mode,
                            "density_default": density_default,
                        },
                        created_at=now,
                    )
                )
                conn.execute(
                    update(sessions_table)
                    .where(sessions_table.c.id == sid)
                    .values(
                        trust_mode=trust_mode,
                        density_default=density_default,
                        updated_at=now,
                    )
                )
                row = conn.execute(select(sessions_table).where(sessions_table.c.id == sid)).one()
                current_record = ComposerSessionPreferencesRecord(
                    session_id=UUID(row.id),
                    trust_mode=row.trust_mode,
                    density_default=row.density_default,
                    interpretation_review_disabled=bool(row.interpretation_review_disabled),
                    updated_at=self._ensure_utc(row.updated_at),
                )
                return ComposerSessionPreferencesTransition(prior=prior_record, current=current_record)

        return cast(ComposerSessionPreferencesTransition, await self._run_sync(_sync))

    async def create_composition_proposal(
        self,
        *,
        session_id: UUID,
        tool_call_id: str,
        tool_name: str,
        summary: str,
        rationale: str,
        affects: Sequence[str],
        arguments_json: Mapping[str, Any],
        arguments_redacted_json: Mapping[str, Any],
        base_state_id: UUID | None,
        actor: str,
        user_message_id: UUID | None = None,
        composer_model_identifier: str | None = None,
        composer_model_version: str | None = None,
        composer_provider: str | None = None,
        composer_skill_hash: str | None = None,
        tool_arguments_hash: str | None = None,
    ) -> CompositionProposalRecord:
        """Create a pending composer proposal and its forward audit event."""
        composer_provenance = _normalize_proposal_composer_provenance(
            composer_model_identifier=composer_model_identifier,
            composer_model_version=composer_model_version,
            composer_provider=composer_provider,
            composer_skill_hash=composer_skill_hash,
            tool_arguments_hash=tool_arguments_hash,
        )
        now = self._now()
        sid = str(session_id)
        umid = str(user_message_id) if user_message_id is not None else None
        proposal_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())

        def _sync() -> CompositionProposalRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                validate_proposal_blob_references(
                    conn,
                    session_id=sid,
                    tool_name=tool_name,
                    arguments=arguments_json,
                )
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=proposal_id,
                        event_type="proposal.created",
                        actor=actor,
                        payload={
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "status": "pending",
                        },
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(composition_proposals_table).values(
                        id=proposal_id,
                        session_id=sid,
                        tool_call_id=tool_call_id,
                        user_message_id=umid,
                        composer_model_identifier=composer_provenance["composer_model_identifier"],
                        composer_model_version=composer_provenance["composer_model_version"],
                        composer_provider=composer_provenance["composer_provider"],
                        composer_skill_hash=composer_provenance["composer_skill_hash"],
                        tool_arguments_hash=composer_provenance["tool_arguments_hash"],
                        tool_name=tool_name,
                        status="pending",
                        summary=summary,
                        rationale=rationale,
                        affects=list(affects),
                        arguments_json=deep_thaw(arguments_json),
                        arguments_redacted_json=deep_thaw(arguments_redacted_json),
                        base_state_id=str(base_state_id) if base_state_id else None,
                        committed_state_id=None,
                        audit_event_id=event_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).one()
                return _proposal_record_from_row(row)

        return cast(CompositionProposalRecord, await self._run_sync(_sync))

    async def create_pipeline_composition_proposal(
        self,
        *,
        session_id: UUID,
        plan: PipelinePlanResult,
        summary: str,
        rationale: str,
        affects: Sequence[str],
        arguments_redacted_json: Mapping[str, Any],
        actor: str,
        composer_model_identifier: str,
        composer_model_version: str,
        composer_provider: str,
        user_message_id: UUID | None = None,
        supersedes_proposal_id: UUID | None = None,
    ) -> CompositionProposalRecord:
        """Atomically create one canonical pipeline row + bound event."""
        if type(plan) is not PipelinePlanResult:
            raise TypeError("plan must be an exact PipelinePlanResult")
        expected_redacted_arguments = redact_tool_call_arguments(
            "set_pipeline",
            deep_thaw(plan.proposal.pipeline),
            telemetry=NoopRedactionTelemetry(),
        )
        if deep_thaw(arguments_redacted_json) != expected_redacted_arguments:
            raise AuditIntegrityError("pipeline proposal redacted arguments do not match the manifest projection")
        proposal = plan.proposal
        if proposal.surface in {PlannerSurface.FREEFORM, PlannerSurface.GUIDED_FULL} and proposal.reviewed_anchor_hash != stable_hash(
            {"schema": "guided.reviewed-anchors.v1", "facts": {}}
        ):
            raise AuditIntegrityError("freeform and guided-full pipeline proposals require empty reviewed facts")
        if (supersedes_proposal_id is None) != (proposal.supersedes_draft_hash is None):
            raise AuditIntegrityError("superseded proposal id and draft hash must be supplied together")
        normalized = _normalize_proposal_composer_provenance(
            composer_model_identifier=composer_model_identifier,
            composer_model_version=composer_model_version,
            composer_provider=composer_provider,
            composer_skill_hash=proposal.skill_hash,
            tool_arguments_hash=stable_hash(proposal.pipeline),
        )
        assert all(value is not None for value in normalized.values())
        payload = _pipeline_created_payload(
            plan=plan,
            user_message_id=user_message_id,
            composer_model_identifier=cast(str, normalized["composer_model_identifier"]),
            composer_model_version=cast(str, normalized["composer_model_version"]),
            composer_provider=cast(str, normalized["composer_provider"]),
            summary=summary,
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=arguments_redacted_json,
            supersedes_proposal_id=supersedes_proposal_id,
        )
        now = self._now()
        sid = str(session_id)
        proposal_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())

        def _sync() -> CompositionProposalRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if type(proposal.base) is AbsentBase:
                    if current_row is not None:
                        raise StaleComposeStateError("pipeline proposal absent base conflicts with current state")
                    base_state_id = None
                elif type(proposal.base) is PresentBase:
                    if current_row is None or current_row.id != str(proposal.base.state_id):
                        raise StaleComposeStateError("pipeline proposal present base state changed before creation")
                    current_record = self._row_to_state_record(current_row)
                    if composition_content_hash(state_from_record(current_record)) != proposal.base.composition_content_hash:
                        raise StaleComposeStateError("pipeline proposal present base content changed before creation")
                    base_state_id = str(proposal.base.state_id)
                else:
                    raise AuditIntegrityError("pipeline proposal base is malformed")

                if supersedes_proposal_id is not None:
                    superseded = conn.execute(
                        select(composition_proposals_table)
                        .where(composition_proposals_table.c.id == str(supersedes_proposal_id))
                        .where(composition_proposals_table.c.session_id == sid)
                    ).one_or_none()
                    if superseded is None:
                        raise AuditIntegrityError("superseded pipeline proposal is not owned by this session")
                    superseded_events = conn.execute(
                        select(proposal_events_table)
                        .where(proposal_events_table.c.session_id == sid)
                        .where(proposal_events_table.c.proposal_id == str(supersedes_proposal_id))
                        .where(proposal_events_table.c.event_type == "proposal.created")
                    ).fetchall()
                    if len(superseded_events) != 1:
                        raise AuditIntegrityError("superseded pipeline proposal has invalid creation authority")
                    superseded_payload = superseded_events[0].payload
                    if (
                        not isinstance(superseded_payload, Mapping)
                        or superseded_payload.get("schema") != _PIPELINE_CREATED_SCHEMA
                        or superseded_payload.get("draft_hash") != proposal.supersedes_draft_hash
                    ):
                        raise AuditIntegrityError("superseded pipeline proposal draft binding mismatch")

                validate_proposal_blob_references(
                    conn,
                    session_id=sid,
                    tool_name="set_pipeline",
                    arguments=deep_thaw(proposal.pipeline),
                )
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=proposal_id,
                        event_type="proposal.created",
                        actor=actor,
                        payload=payload,
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(composition_proposals_table).values(
                        id=proposal_id,
                        session_id=sid,
                        tool_call_id=plan.tool_call_id,
                        user_message_id=str(user_message_id) if user_message_id is not None else None,
                        composer_model_identifier=normalized["composer_model_identifier"],
                        composer_model_version=normalized["composer_model_version"],
                        composer_provider=normalized["composer_provider"],
                        composer_skill_hash=normalized["composer_skill_hash"],
                        tool_arguments_hash=normalized["tool_arguments_hash"],
                        tool_name="set_pipeline",
                        status="pending",
                        summary=summary,
                        rationale=rationale,
                        affects=list(affects),
                        arguments_json=deep_thaw(proposal.pipeline),
                        arguments_redacted_json=deep_thaw(arguments_redacted_json),
                        base_state_id=base_state_id,
                        committed_state_id=None,
                        audit_event_id=event_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).one()
                record = _proposal_record_from_row(row)
                authority = AuthoritativePipelineProposal(
                    row=record,
                    proposal=proposal,
                    creation_event_id=UUID(event_id),
                    custody_result=plan.custody_result,
                    supersedes_proposal_id=supersedes_proposal_id,
                )
                return replace(record, pipeline_metadata=_pipeline_public_metadata(authority))

        record = cast(CompositionProposalRecord, await self._run_sync(_sync))
        _PIPELINE_PLANNER_COUNTER.add(1, {"surface": proposal.surface.value, "result": "proposal_created"})
        _PIPELINE_CUSTODY_COUNTER.add(1, {"surface": proposal.surface.value, "result": plan.custody_result})
        return record

    async def get_authoritative_pipeline_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        reviewed_facts: Mapping[str, Any],
    ) -> AuthoritativePipelineProposal:
        """Load one canonical authority, rejecting exact historical proposals."""
        authority = await self.get_authoritative_composition_proposal(
            session_id=session_id,
            proposal_id=proposal_id,
            reviewed_facts=reviewed_facts,
        )
        if authority.pipeline is None:
            raise ValueError("proposal uses the exact legacy lifecycle contract")
        return authority.pipeline

    async def get_authoritative_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        reviewed_facts: Mapping[str, Any] | None,
    ) -> AuthoritativeCompositionProposal:
        """Load exactly one row and one event, then classify without fallback."""
        sid = str(session_id)
        pid = str(proposal_id)

        def _sync() -> AuthoritativeCompositionProposal:
            with self._engine.begin() as conn:
                row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if row is None:
                    raise KeyError(pid)
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("pipeline proposal must have exactly one authoritative creation event")
                authority = _classify_authoritative_composition_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=reviewed_facts,
                )
                if authority.pipeline is not None:
                    _verify_pipeline_lifecycle_authority(
                        conn,
                        service=self,
                        authority=authority.pipeline,
                    )
                return authority

        return cast(AuthoritativeCompositionProposal, await self._run_sync(_sync))

    async def settle_pipeline_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        draft_hash: str,
        reviewed_facts: Mapping[str, Any],
        state: CompositionStateData,
        candidate_content_hash: str,
        executor_content_hash: str,
        final_composer_metadata: Mapping[str, Any] | None,
        dispatch: PipelineDispatchAuditBinding,
        actor: str,
    ) -> PipelineProposalSettlementResult:
        """Atomically publish state and settle a verified pipeline proposal."""
        if type(dispatch) is not PipelineDispatchAuditBinding:
            raise TypeError("dispatch must be an exact PipelineDispatchAuditBinding")
        state_content_hash = _composition_state_data_content_hash(state)
        if candidate_content_hash != executor_content_hash or candidate_content_hash != state_content_hash:
            raise AuditIntegrityError("pipeline candidate/executor/state content hash mismatch")
        settled_state = replace(state, composer_meta=final_composer_metadata)
        sid = str(session_id)
        pid = str(proposal_id)
        now = self._now()

        def _sync() -> tuple[PipelineProposalSettlementResult, bool]:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if row is None:
                    raise KeyError(pid)
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("pipeline proposal must have exactly one authoritative creation event")
                authority = _restore_authoritative_pipeline_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=reviewed_facts,
                )
                _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                if authority.proposal.draft_hash != draft_hash:
                    raise StaleComposeStateError("pipeline proposal draft hash echo is stale or mismatched")
                if dispatch.tool_call_id != authority.row.tool_call_id:
                    raise AuditIntegrityError("pipeline dispatch tool call does not match proposal authority")
                if dispatch.arguments_hash != stable_hash(authority.row.arguments_redacted_json):
                    raise AuditIntegrityError("pipeline dispatch arguments do not match persisted redacted proposal")
                if _persisted_pipeline_dispatch_content_hashes(conn, session_id=sid, dispatch=dispatch) != (state_content_hash,):
                    raise AuditIntegrityError("pipeline settlement requires one durable dispatch audit bound to the exact state content")

                terminal_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type.in_(("proposal.accepted", "proposal.rejected")))
                ).fetchall()
                if authority.row.status == "committed":
                    if len(terminal_rows) != 1 or terminal_rows[0].event_type != "proposal.accepted":
                        raise AuditIntegrityError("committed pipeline proposal terminal event authority is malformed")
                    if authority.row.audit_event_id != UUID(terminal_rows[0].id):
                        raise AuditIntegrityError("committed pipeline proposal terminal event pointer is malformed")
                    if authority.row.committed_state_id is None:
                        raise AuditIntegrityError("committed pipeline proposal is missing committed state")
                    committed_row = conn.execute(
                        select(composition_states_table)
                        .where(composition_states_table.c.session_id == sid)
                        .where(composition_states_table.c.id == str(authority.row.committed_state_id))
                    ).one_or_none()
                    if committed_row is None:
                        raise AuditIntegrityError("committed pipeline proposal state is missing")
                    committed_record = self._row_to_state_record(committed_row)
                    committed_hash = composition_content_hash(state_from_record(committed_record))
                    if committed_hash != state_content_hash:
                        raise AuditIntegrityError("pipeline exact retry state content mismatch")
                    if deep_thaw(committed_record.composer_meta) != deep_thaw(final_composer_metadata):
                        raise AuditIntegrityError("pipeline exact retry final metadata mismatch")
                    expected_terminal = _pipeline_accepted_payload(
                        authority=authority,
                        state_id=str(committed_record.id),
                        state_content_hash=committed_hash,
                        final_composer_metadata=final_composer_metadata,
                        dispatch=dispatch,
                    )
                    if terminal_rows[0].payload != expected_terminal:
                        raise AuditIntegrityError("pipeline exact retry terminal binding mismatch")
                    return (
                        PipelineProposalSettlementResult(
                            proposal=replace(authority.row, pipeline_metadata=_pipeline_public_metadata(authority)),
                            state=committed_record,
                        ),
                        False,
                    )
                if authority.row.status != "pending":
                    raise ValueError(f"Proposal {pid} must be pending to commit; got {authority.row.status!r}")
                if terminal_rows:
                    raise AuditIntegrityError("pending pipeline proposal already has a terminal event")

                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if type(authority.proposal.base) is AbsentBase:
                    if current_row is not None:
                        raise StaleComposeStateError("pipeline proposal absent base conflicts with current state")
                    derived_from_state_id = None
                elif type(authority.proposal.base) is PresentBase:
                    if current_row is None or current_row.id != str(authority.proposal.base.state_id):
                        raise StaleComposeStateError("pipeline proposal present base state changed before settlement")
                    current_record = self._row_to_state_record(current_row)
                    current_hash = composition_content_hash(state_from_record(current_record))
                    if current_hash != authority.proposal.base.composition_content_hash:
                        raise StaleComposeStateError("pipeline proposal present base content changed before settlement")
                    derived_from_state_id = str(authority.proposal.base.state_id)
                else:
                    raise AuditIntegrityError("pipeline proposal base is malformed")

                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=settled_state, derived_from_state_id=derived_from_state_id),
                    provenance="tool_call",
                    created_at=now,
                )
                event_id = str(uuid.uuid4())
                terminal_payload = _pipeline_accepted_payload(
                    authority=authority,
                    state_id=state_id,
                    state_content_hash=state_content_hash,
                    final_composer_metadata=final_composer_metadata,
                    dispatch=dispatch,
                )
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.accepted",
                        actor=actor,
                        payload=terminal_payload,
                        created_at=now,
                    )
                )
                conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.status == "pending")
                    .values(
                        status="committed",
                        committed_state_id=state_id,
                        audit_event_id=event_id,
                        updated_at=now,
                    )
                )
                settled_row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one()
                state_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .where(composition_states_table.c.id == state_id)
                ).one()
                return (
                    PipelineProposalSettlementResult(
                        proposal=replace(
                            _proposal_record_from_row(settled_row),
                            pipeline_metadata=_pipeline_public_metadata(authority),
                        ),
                        state=self._row_to_state_record(state_row),
                    ),
                    True,
                )

        result, transitioned = cast(tuple[PipelineProposalSettlementResult, bool], await self._run_sync(_sync))
        if transitioned:
            assert result.proposal.pipeline_metadata is not None
            _PIPELINE_SETTLEMENT_COUNTER.add(
                1,
                {"surface": result.proposal.pipeline_metadata.surface, "result": "accepted"},
            )
        return result

    async def get_pipeline_dispatch_recovery(
        self,
        *,
        authority: AuthoritativePipelineProposal,
    ) -> PipelineDispatchRecovery | None:
        """Return one content-bound durable dispatch for pending recovery."""
        if type(authority) is not AuthoritativePipelineProposal:
            raise TypeError("authority must be an exact AuthoritativePipelineProposal")
        sid = str(authority.row.session_id)
        expected_arguments_hash = stable_hash(authority.row.arguments_redacted_json)

        def _sync() -> PipelineDispatchRecovery | None:
            with self._engine.begin() as conn:
                rows = conn.execute(select(chat_messages_table.c.tool_calls).where(chat_messages_table.c.session_id == sid)).fetchall()
                matches: list[PipelineDispatchRecovery] = []
                for row in rows:
                    if type(row.tool_calls) is not list:
                        continue
                    for envelope in row.tool_calls:
                        recovery = _pipeline_dispatch_recovery_from_envelope(
                            envelope,
                            expected_tool_call_id=authority.row.tool_call_id,
                        )
                        if recovery is None:
                            continue
                        if recovery.binding.arguments_hash != expected_arguments_hash:
                            raise AuditIntegrityError("pipeline recovery dispatch arguments do not match authority")
                        matches.append(recovery)
                if len(matches) > 1:
                    raise AuditIntegrityError("pipeline recovery has duplicate successful content-bound dispatches")
                return matches[0] if matches else None

        return cast(PipelineDispatchRecovery | None, await self._run_sync(_sync))

    async def reject_pipeline_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        draft_hash: str,
        reviewed_facts: Mapping[str, Any] | None,
        reason: PipelineProposalRejectionReason,
        dispatch: PipelineDispatchAuditBinding | None,
        actor: str,
    ) -> CompositionProposalRecord:
        """Atomically terminalise a pipeline proposal with a closed reason."""
        reason = _validated_pipeline_rejection_reason(reason)
        if reason == "candidate_executor_mismatch" and dispatch is None:
            raise AuditIntegrityError("candidate/executor mismatch rejection requires dispatch evidence")
        if dispatch is not None and type(dispatch) is not PipelineDispatchAuditBinding:
            raise TypeError("dispatch must be an exact PipelineDispatchAuditBinding or None")
        sid = str(session_id)
        pid = str(proposal_id)
        now = self._now()

        def _sync() -> tuple[CompositionProposalRecord, bool]:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if row is None:
                    raise KeyError(pid)
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("pipeline proposal must have exactly one authoritative creation event")
                authority = _restore_authoritative_pipeline_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=reviewed_facts,
                )
                _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                if authority.proposal.draft_hash != draft_hash:
                    raise StaleComposeStateError("pipeline proposal draft hash echo is stale or mismatched")
                if dispatch is not None:
                    if dispatch.tool_call_id != authority.row.tool_call_id:
                        raise AuditIntegrityError("pipeline rejection dispatch tool call does not match proposal authority")
                    if dispatch.arguments_hash != stable_hash(authority.row.arguments_redacted_json):
                        raise AuditIntegrityError("pipeline rejection dispatch arguments do not match persisted redacted proposal")
                    if len(_persisted_pipeline_dispatch_content_hashes(conn, session_id=sid, dispatch=dispatch)) != 1:
                        raise AuditIntegrityError("pipeline rejection requires exactly one matching durable dispatch audit")
                expected_payload = _pipeline_rejected_payload(authority=authority, reason=reason, dispatch=dispatch)
                terminal_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type.in_(("proposal.accepted", "proposal.rejected")))
                ).fetchall()
                if authority.row.status == "rejected":
                    if (
                        len(terminal_rows) != 1
                        or terminal_rows[0].event_type != "proposal.rejected"
                        or authority.row.audit_event_id != UUID(terminal_rows[0].id)
                        or terminal_rows[0].payload != expected_payload
                    ):
                        raise AuditIntegrityError("rejected pipeline proposal terminal binding mismatch")
                    return replace(authority.row, pipeline_metadata=_pipeline_public_metadata(authority)), False
                if authority.row.status != "pending":
                    raise ValueError(f"Proposal {pid} must be pending to reject; got {authority.row.status!r}")
                if terminal_rows:
                    raise AuditIntegrityError("pending pipeline proposal already has a terminal event")
                event_id = str(uuid.uuid4())
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.rejected",
                        actor=actor,
                        payload=expected_payload,
                        created_at=now,
                    )
                )
                conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.status == "pending")
                    .values(status="rejected", audit_event_id=event_id, updated_at=now)
                )
                updated_row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one()
                return (
                    replace(
                        _proposal_record_from_row(updated_row),
                        pipeline_metadata=_pipeline_public_metadata(authority),
                    ),
                    True,
                )

        result, transitioned = cast(tuple[CompositionProposalRecord, bool], await self._run_sync(_sync))
        if transitioned:
            assert result.pipeline_metadata is not None
            _PIPELINE_SETTLEMENT_COUNTER.add(
                1,
                {"surface": result.pipeline_metadata.surface, "result": reason},
            )
        return result

    async def list_composition_proposals(
        self,
        session_id: UUID,
        *,
        status: ProposalLifecycleStatus | None = None,
    ) -> list[CompositionProposalRecord]:
        """List composer proposals for a session in creation order."""
        sid = str(session_id)

        def _sync() -> list[CompositionProposalRecord]:
            stmt = select(composition_proposals_table).where(composition_proposals_table.c.session_id == sid)
            if status is not None:
                stmt = stmt.where(composition_proposals_table.c.status == status)
            stmt = stmt.order_by(composition_proposals_table.c.created_at)
            with self._engine.begin() as conn:
                rows = conn.execute(stmt).fetchall()
                records = [_proposal_record_from_row(row) for row in rows]
                if not records:
                    return records
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                    .where(proposal_events_table.c.proposal_id.in_([str(record.id) for record in records]))
                ).fetchall()
                by_proposal: dict[str, list[Any]] = {}
                for event_row in creation_rows:
                    by_proposal.setdefault(event_row.proposal_id, []).append(event_row)
                enriched: list[CompositionProposalRecord] = []
                for record in records:
                    events = by_proposal.get(str(record.id), [])
                    if len(events) != 1:
                        raise AuditIntegrityError("composition proposal must have exactly one creation event")
                    authority = _classify_authoritative_composition_proposal(
                        conn=conn,
                        row=record,
                        creation_event=_proposal_event_record_from_row(events[0]),
                        reviewed_facts=None,
                    )
                    if authority.pipeline is not None:
                        _verify_pipeline_lifecycle_authority(
                            conn,
                            service=self,
                            authority=authority.pipeline,
                        )
                    metadata = _pipeline_public_metadata(authority.pipeline) if authority.pipeline is not None else None
                    enriched.append(replace(record, pipeline_metadata=metadata))
                return enriched

        return cast(list[CompositionProposalRecord], await self._run_sync(_sync))

    async def reject_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord:
        """Reject a pending proposal by appending an event, then updating status."""
        now = self._now()
        sid = str(session_id)
        pid = str(proposal_id)

        def _sync() -> CompositionProposalRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.session_id == sid)
                ).one_or_none()
                if row is None:
                    raise KeyError(pid)
                if row.status != "pending":
                    raise ValueError(f"Proposal {pid} must be pending to reject; got {row.status!r}")

                conn.execute(
                    insert(proposal_events_table).values(
                        id=str(uuid.uuid4()),
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.rejected",
                        actor=actor,
                        payload={"status": "rejected"},
                        created_at=now,
                    )
                )
                conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.session_id == sid)
                    .values(
                        status="rejected",
                        updated_at=now,
                    )
                )
                updated_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                return _proposal_record_from_row(updated_row)

        return cast(CompositionProposalRecord, await self._run_sync(_sync))

    async def mark_composition_proposal_committed(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        committed_state_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord:
        """Commit a pending proposal by appending an event, then updating status."""
        now = self._now()
        sid = str(session_id)
        pid = str(proposal_id)
        state_id = str(committed_state_id)

        def _sync() -> CompositionProposalRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.session_id == sid)
                ).one_or_none()
                if row is None:
                    raise KeyError(pid)
                if row.status != "pending":
                    raise ValueError(f"Proposal {pid} must be pending to commit; got {row.status!r}")

                event_id = str(uuid.uuid4())
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.accepted",
                        actor=actor,
                        payload={"committed_state_id": state_id},
                        created_at=now,
                    )
                )
                conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.session_id == sid)
                    .values(
                        status="committed",
                        committed_state_id=state_id,
                        audit_event_id=event_id,
                        updated_at=now,
                    )
                )
                updated_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                return _proposal_record_from_row(updated_row)

        return cast(CompositionProposalRecord, await self._run_sync(_sync))

    async def list_proposal_events(
        self,
        session_id: UUID,
    ) -> list[ProposalEventRecord]:
        """List immutable composer proposal lifecycle events for a session."""
        sid = str(session_id)

        def _sync() -> list[ProposalEventRecord]:
            with self._engine.begin() as conn:
                rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .order_by(proposal_events_table.c.created_at, proposal_events_table.c.id)
                ).fetchall()
                return [_proposal_event_record_from_row(row) for row in rows]

        return cast(list[ProposalEventRecord], await self._run_sync(_sync))

    # ----------------------------------------------------------------- #
    # Interpretation-event writer/reader methods.
    #
    # Telemetry: NONE — composition-time user decisions are audit-primary;
    # the Landscape ``interpretation_events_table`` is the source of truth
    # and there is no ephemeral operational signal worth emitting. Do not
    # add slog or telemetry hooks here without a primacy-order review.
    # ----------------------------------------------------------------- #

    async def create_pending_interpretation_event(
        self,
        *,
        session_id: UUID,
        composition_state_id: UUID,
        affected_node_id: str,
        tool_call_id: str,
        user_term: str,
        kind: InterpretationKind,
        llm_draft: str,
        model_identifier: str,
        model_version: str,
        provider: str,
        composer_skill_hash: str,
        created_at: datetime | None = None,
    ) -> InterpretationEventRecord:
        """Insert a PENDING interpretation event.

        Called from the compose-loop tool handler for
        ``request_interpretation_review``. Acquires the session write lock
        for the duration of the insert. Validates ``affected_node_id``
        exists in ``composition_states.nodes`` BEFORE committing the row
        (raises :class:`ValueError` otherwise; the tool handler converts to
        ARG_ERROR).

        Per CLAUDE.md offensive-programming rules, the writer-boundary
        validation reads the parent composition_states row inside the
        locked transaction and inspects its ``nodes`` JSON before INSERT —
        a malformed reference is a Tier-1 audit anomaly we crash on rather
        than fabricating a binding.

        The ``actor`` column on the pending row is set to the sentinel
        ``"composer-llm"`` — the row was created by the composer LLM, not
        a user. ``resolve_interpretation_event`` overwrites this with the
        user identity passed by the route at resolution time, which is
        what :data:`InterpretationEventRecord` ``actor`` documents as
        "user identity at resolution". The closed CHECK on choice and the
        immutability trigger together prevent any other writer from
        appearing on a resolved row.

        Telemetry: NONE — composition-time user decisions are audit-primary;
        no ephemeral operational signal required.
        """
        if not isinstance(kind, InterpretationKind):
            raise ValueError(f"kind must be InterpretationKind, got {type(kind).__name__}: {kind!r}")
        now = self._ensure_utc(created_at) if created_at is not None else self._now()
        sid = str(session_id)
        state_id_str = str(composition_state_id)
        kind_value = kind.value
        event_id = str(uuid.uuid4())
        plugin_snapshot = await self._plugin_snapshot_for_session(sid)

        def _sync() -> InterpretationEventRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                # Writer-boundary validation: resolve the parent state and
                # validate the affected component before any interpretation
                # row is written. ``invented_source`` binds to the synthetic
                # source component; transform kinds bind to real LLM nodes.
                state_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.id == state_id_str)
                    .where(composition_states_table.c.session_id == sid)
                ).one_or_none()
                if state_row is None:
                    raise ValueError(
                        f"create_pending_interpretation_event: composition state {state_id_str!r} not found in session {sid!r}"
                    )
                nodes = self._unwrap_envelope(state_row.nodes)
                # Multi-source: the legacy singular ``source`` column is dead
                # (``_insert_composition_state`` always writes it NULL). The
                # default source under review lives in the ``sources`` map keyed
                # by ``SOURCE_COMPONENT_ID``. Interpretation review is scoped to
                # that default component, so the invented-source writer-boundary
                # validation below reads it from the map. A missing/malformed
                # default source leaves ``source`` None and the existing
                # ``isinstance`` guard raises the same clear error as before.
                sources = self._unwrap_envelope(state_row.sources)
                source = sources[SOURCE_COMPONENT_ID] if isinstance(sources, Mapping) and SOURCE_COMPONENT_ID in sources else None
                if kind is InterpretationKind.INVENTED_SOURCE:
                    if affected_node_id != SOURCE_COMPONENT_ID:
                        raise ValueError(
                            f"create_pending_interpretation_event: invented_source must target affected_node_id "
                            f"{SOURCE_COMPONENT_ID!r}, got {affected_node_id!r}"
                        )
                    if not isinstance(source, Mapping):
                        raise ValueError("create_pending_interpretation_event: invented_source requires a persisted source mapping")
                    source_options = source["options"] if "options" in source else None
                    if not isinstance(source_options, Mapping) or SOURCE_AUTHORING_KEY not in source_options:
                        raise ValueError(
                            f"create_pending_interpretation_event: invented_source requires source.options.{SOURCE_AUTHORING_KEY}"
                        )
                    source_authoring = source_options[SOURCE_AUTHORING_KEY]
                    if not isinstance(source_authoring, Mapping):
                        raise ValueError(f"create_pending_interpretation_event: source.options.{SOURCE_AUTHORING_KEY} must be a mapping")
                    content_hash = source_authoring["content_hash"] if "content_hash" in source_authoring else None
                    if not isinstance(content_hash, str) or not content_hash:
                        raise ValueError(
                            f"create_pending_interpretation_event: source.options.{SOURCE_AUTHORING_KEY}.content_hash must be populated"
                        )
                    try:
                        requirements, matching_index = _matching_pending_requirement_index(
                            source_options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in source_options else None,
                            kind=kind,
                            user_term=user_term,
                            context="create_pending_interpretation_event",
                        )
                    except InterpretationPlaceholderConsumedError as exc:
                        raise ValueError(
                            "create_pending_interpretation_event: source.options.interpretation_requirements "
                            f"must contain exactly one pending {kind.value!r} requirement for {user_term!r}"
                        ) from exc
                    requirement = requirements[matching_index]
                    draft = requirement["draft"] if "draft" in requirement else None
                    if isinstance(draft, str) and draft != llm_draft:
                        raise ValueError(
                            "create_pending_interpretation_event: invented_source event draft does not match the source review requirement draft"
                        )
                elif kind is InterpretationKind.PIPELINE_DECISION:
                    if nodes is None:
                        raise ValueError(
                            f"create_pending_interpretation_event: composition state "
                            f"{state_id_str!r} has no nodes; affected_node_id "
                            f"{affected_node_id!r} is not present"
                        )
                    state_record = self._row_to_state_record(state_row)
                    node = _find_interpretation_review_node(
                        state_record,
                        affected_node_id=affected_node_id,
                        context="create_pending_interpretation_event",
                    )
                    options = _require_mapping(
                        node["options"] if "options" in node else None,
                        message=f"create_pending_interpretation_event: node {affected_node_id!r} options is not a mapping",
                    )
                    try:
                        requirements, matching_index = _matching_pending_requirement_index(
                            options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None,
                            kind=kind,
                            user_term=user_term,
                            context="create_pending_interpretation_event",
                        )
                    except InterpretationPlaceholderConsumedError as exc:
                        raise ValueError(
                            "create_pending_interpretation_event: node options.interpretation_requirements "
                            f"must contain exactly one pending {kind.value!r} requirement for {user_term!r}"
                        ) from exc
                    requirement = requirements[matching_index]
                    draft = requirement["draft"] if "draft" in requirement else None
                    if isinstance(draft, str) and draft != llm_draft:
                        raise ValueError(
                            "create_pending_interpretation_event: pipeline_decision event draft does not match the node review requirement draft"
                        )
                    _validate_pipeline_decision_semantics_from_state_record(
                        state_record,
                        affected_node_id=affected_node_id,
                        user_term=user_term,
                        draft=draft,
                        context="create_pending_interpretation_event",
                    )
                else:
                    if nodes is None:
                        raise ValueError(
                            f"create_pending_interpretation_event: composition state "
                            f"{state_id_str!r} has no nodes; affected_node_id "
                            f"{affected_node_id!r} is not present"
                        )
                    state_record = self._row_to_state_record(state_row)
                    node = _find_llm_transform_node(
                        state_record,
                        affected_node_id=affected_node_id,
                        context="create_pending_interpretation_event",
                    )
                    if kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
                        options = _require_mapping(
                            node["options"],
                            message=f"create_pending_interpretation_event: node {affected_node_id!r} options is not a mapping",
                        )
                        if "prompt_template" not in options or not isinstance(options["prompt_template"], str):
                            raise ValueError(
                                f"create_pending_interpretation_event: node {affected_node_id!r} options.prompt_template is not a string"
                            )
                        prompt_template = options["prompt_template"]
                        if llm_draft != prompt_template:
                            raise ValueError(
                                "create_pending_interpretation_event: llm_prompt_template event draft must match current options.prompt_template"
                            )
                        try:
                            _matching_pending_requirement_index(
                                options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None,
                                kind=kind,
                                user_term=user_term,
                                context="create_pending_interpretation_event",
                            )
                        except InterpretationPlaceholderConsumedError as exc:
                            raise ValueError(
                                "create_pending_interpretation_event: node options.interpretation_requirements "
                                f"must contain exactly one pending {kind.value!r} requirement for {user_term!r}"
                            ) from exc

                session_row = conn.execute(
                    select(sessions_table.c.interpretation_review_disabled).where(sessions_table.c.id == sid)
                ).one_or_none()
                if session_row is not None and bool(session_row.interpretation_review_disabled):
                    marker_row = conn.execute(
                        select(interpretation_events_table)
                        .where(interpretation_events_table.c.session_id == sid)
                        .where(interpretation_events_table.c.interpretation_source == InterpretationSource.AUTO_INTERPRETED_OPT_OUT.value)
                        .where(interpretation_events_table.c.kind.is_(None))
                        .order_by(desc(interpretation_events_table.c.created_at))
                    ).first()
                    if marker_row is None:
                        conn.execute(
                            insert(interpretation_events_table).values(
                                id=str(uuid.uuid4()),
                                session_id=sid,
                                composition_state_id=None,
                                affected_node_id=None,
                                tool_call_id=None,
                                user_term=None,
                                kind=None,
                                llm_draft=None,
                                accepted_value=None,
                                choice=InterpretationChoice.OPTED_OUT.value,
                                created_at=now,
                                resolved_at=now,
                                actor="composer-llm",
                                model_identifier=None,
                                model_version=None,
                                provider=None,
                                composer_skill_hash=None,
                                arguments_hash=None,
                                hash_domain_version=None,
                                interpretation_source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT.value,
                                runtime_model_identifier_at_resolve=None,
                                runtime_model_version_at_resolve=None,
                                resolved_prompt_template_hash=None,
                            )
                        )
                    domain_dict = _interpretation_hash_domain_v2(
                        session_id=sid,
                        composition_state_id=state_id_str,
                        affected_node_id=affected_node_id,
                        tool_call_id=tool_call_id,
                        user_term=user_term,
                        kind=kind_value,
                        llm_draft=llm_draft,
                        accepted_value=llm_draft,
                        actor="composer-llm",
                        model_identifier=model_identifier,
                        model_version=model_version,
                        provider=provider,
                        composer_skill_hash=composer_skill_hash,
                        context="create_pending_interpretation_event",
                    )
                    live_state_row = conn.execute(
                        select(composition_states_table)
                        .where(composition_states_table.c.session_id == sid)
                        .order_by(desc(composition_states_table.c.version))
                        .limit(1)
                    ).one_or_none()
                    if live_state_row is None:
                        raise AuditIntegrityError(f"create_pending_interpretation_event: session {sid!r} has no composition state to patch")
                    live_state_record = self._row_to_state_record(live_state_row)
                    final_sources: Mapping[str, Mapping[str, Any]] | None
                    final_nodes: list[Mapping[str, Any]]
                    resolved_prompt_template_hash: str | None
                    if kind is InterpretationKind.VAGUE_TERM:
                        final_sources, final_nodes, resolved_prompt_template_hash = _resolve_vague_term(
                            live_state_record,
                            affected_node_id=affected_node_id,
                            user_term=user_term,
                            accepted_value=llm_draft,
                        )
                    elif kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
                        # Opt-out auto-resolve fires at create time, so the
                        # surfacing state IS the live state — its skeleton hash
                        # trivially matches the gate.
                        final_sources, final_nodes, resolved_prompt_template_hash = _resolve_prompt_template_review(
                            live_state_record,
                            event_id=event_id,
                            affected_node_id=affected_node_id,
                            user_term=user_term,
                            accepted_value=llm_draft,
                            surfacing_structure_hash=_surfacing_prompt_structure_hash(
                                live_state_record,
                                affected_node_id=affected_node_id,
                            ),
                        )
                    elif kind is InterpretationKind.INVENTED_SOURCE:
                        final_sources, final_nodes, resolved_prompt_template_hash = _resolve_invented_source(
                            live_state_record,
                            event_id=event_id,
                            affected_node_id=affected_node_id,
                            user_term=user_term,
                            llm_draft=llm_draft,
                            accepted_value=llm_draft,
                        )
                    elif kind is InterpretationKind.PIPELINE_DECISION:
                        final_sources, final_nodes, resolved_prompt_template_hash = _resolve_pipeline_decision_review(
                            live_state_record,
                            event_id=event_id,
                            affected_node_id=affected_node_id,
                            user_term=user_term,
                            llm_draft=llm_draft,
                            accepted_value=llm_draft,
                        )
                    elif kind is InterpretationKind.LLM_MODEL_CHOICE:
                        final_sources, final_nodes, resolved_prompt_template_hash = _resolve_model_choice_review(
                            live_state_record,
                            event_id=event_id,
                            affected_node_id=affected_node_id,
                            user_term=user_term,
                            llm_draft=llm_draft,
                            accepted_value=llm_draft,
                        )
                    else:
                        raise AssertionError(f"unhandled InterpretationKind {kind!r}")

                    from elspeth.web.sessions.converters import state_from_record

                    patched_state_record = replace(
                        live_state_record,
                        source=None,
                        sources=final_sources,
                        nodes=final_nodes,
                        is_valid=False,
                        validation_errors=None,
                    )
                    patched_validation = self._validate_patched_composition_state(
                        state_from_record(patched_state_record),
                        plugin_snapshot=plugin_snapshot,
                    )
                    patched_validation_errors = [error.message for error in patched_validation.errors] or None
                    conn.execute(
                        insert(interpretation_events_table).values(
                            id=event_id,
                            session_id=sid,
                            composition_state_id=state_id_str,
                            affected_node_id=affected_node_id,
                            tool_call_id=tool_call_id,
                            user_term=user_term,
                            kind=kind_value,
                            llm_draft=llm_draft,
                            accepted_value=llm_draft,
                            choice=InterpretationChoice.OPTED_OUT.value,
                            created_at=now,
                            resolved_at=now,
                            actor="composer-llm",
                            model_identifier=model_identifier,
                            model_version=model_version,
                            provider=provider,
                            composer_skill_hash=composer_skill_hash,
                            arguments_hash=stable_hash(domain_dict),
                            hash_domain_version="v2",
                            interpretation_source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT.value,
                            runtime_model_identifier_at_resolve=None,
                            runtime_model_version_at_resolve=None,
                            resolved_prompt_template_hash=(
                                resolved_prompt_template_hash if kind is InterpretationKind.LLM_PROMPT_TEMPLATE else None
                            ),
                        )
                    )
                    self._insert_composition_state(
                        conn,
                        session_id=sid,
                        payload=StatePayload(
                            data=CompositionStateData(
                                sources=final_sources,
                                nodes=final_nodes,
                                edges=live_state_record.edges,
                                outputs=live_state_record.outputs,
                                metadata_=live_state_record.metadata_,
                                is_valid=patched_validation.is_valid,
                                validation_errors=patched_validation_errors,
                                composer_meta=live_state_record.composer_meta,
                            ),
                            derived_from_state_id=str(live_state_record.id),
                        ),
                        provenance="interpretation_resolve",
                        created_at=now,
                    )
                    row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == event_id)).one()
                    return _interpretation_event_record_from_row(row)

                # Idempotent re-surface, every kind: an identical pending event
                # (same node/kind/user_term/draft) is returned, never twinned.
                # The resolve path demands exactly ONE pending requirement per
                # (node, kind, user_term), so twin pending events are
                # structurally unresolvable — the first resolve stamps the
                # requirement and the twin 422s placeholder_unavailable forever
                # (elspeth-1fcaec9b63, reachable by importing the same YAML
                # twice). Surfacer read-side dedup cannot enforce this: it
                # crosses transactions, while this SELECT runs under the
                # session advisory lock.
                existing_pending_row = conn.execute(
                    select(interpretation_events_table)
                    .where(interpretation_events_table.c.session_id == sid)
                    .where(interpretation_events_table.c.affected_node_id == affected_node_id)
                    .where(interpretation_events_table.c.kind == kind_value)
                    .where(interpretation_events_table.c.user_term == user_term)
                    .where(interpretation_events_table.c.llm_draft == llm_draft)
                    .where(interpretation_events_table.c.choice == InterpretationChoice.PENDING.value)
                    .where(interpretation_events_table.c.interpretation_source == InterpretationSource.USER_APPROVED.value)
                    .order_by(interpretation_events_table.c.created_at, interpretation_events_table.c.id)
                    .limit(1)
                ).one_or_none()
                if existing_pending_row is not None:
                    return _interpretation_event_record_from_row(existing_pending_row)

                conn.execute(
                    insert(interpretation_events_table).values(
                        id=event_id,
                        session_id=sid,
                        composition_state_id=state_id_str,
                        affected_node_id=affected_node_id,
                        tool_call_id=tool_call_id,
                        user_term=user_term,
                        kind=kind_value,
                        llm_draft=llm_draft,
                        accepted_value=None,
                        choice=InterpretationChoice.PENDING.value,
                        created_at=now,
                        resolved_at=None,
                        actor="composer-llm",
                        model_identifier=model_identifier,
                        model_version=model_version,
                        provider=provider,
                        composer_skill_hash=composer_skill_hash,
                        arguments_hash=None,
                        hash_domain_version=None,
                        interpretation_source=InterpretationSource.USER_APPROVED.value,
                        runtime_model_identifier_at_resolve=None,
                        runtime_model_version_at_resolve=None,
                        resolved_prompt_template_hash=None,
                    )
                )
                row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == event_id)).one()
                return _interpretation_event_record_from_row(row)

        return cast(InterpretationEventRecord, await self._run_sync(_sync))

    async def resolve_interpretation_event(
        self,
        *,
        session_id: UUID,
        event_id: UUID,
        choice: InterpretationChoice,
        amended_value: str | None,
        actor: str,
        resolved_at: datetime | None = None,
        runtime_model_identifier: str | None = None,
        runtime_model_version: str | None = None,
    ) -> tuple[InterpretationEventRecord, CompositionStateRecord]:
        """Commit a resolution AND patch the affected LLM transform's prompt template.

        F-14 (business-rule split): ``accepted_value`` is computed
        internally. When ``choice == ACCEPTED_AS_DRAFTED``, the service
        reads the pending event's ``llm_draft`` from the DB and uses that
        as ``accepted_value``. When ``choice == AMENDED``,
        ``amended_value`` is used directly. The route passes only
        ``choice`` and ``amended_value`` — the computation lives here to
        avoid duplicating the branch across callers.

        Single transaction (F-25: ``_session_write_lock`` acquires the
        per-session write lock; the SQLAlchemy ``begin()`` is implicitly
        BEGIN IMMEDIATE on SQLite when a write occurs inside, but the
        process-wide RLock is the actual serialiser):

            1. SELECT the pending event by id AND session_id AND choice='pending'
               (F-7: session_id in WHERE prevents cross-session IDOR;
               choice='pending' is the TOCTOU guard against double-resolve).
               Raise ValueError if no matching row.
            2. Compute ``accepted_value`` per F-14.
            3. Validate ``accepted_value`` via ``_validate_accepted_value_content``
               (defence-in-depth against future callers that bypass the route).
            4. Call :func:`_patch_llm_transform_prompt` to produce the
               resolved prompt-template string. Keep a local reference; do
               NOT call again in step 5a.
            4a. Compute ``resolved_prompt_template_hash`` via
                :func:`stable_hash` over the resolved prompt string.
                ``CANONICAL_VERSION = "sha256-rfc8785-v1"``. NOT part of
                ``INTERPRETATION_HASH_DOMAIN_V2`` — covers a different
                input.
            5. UPDATE interpretation_events with the settled fields.
            5a. Write the new composition_states row with provenance =
                'interpretation_resolve', version += 1, carrying the
                patched ``prompt_template`` and ``resolved_prompt_template_hash``
                on the affected node JSON.
            6. Return the resolved event + the new state.

        Trigger error note (F-28): if the immutability trigger
        ``trg_interpretation_events_immutable_resolved`` fires, SQLAlchemy
        raises :class:`IntegrityError` carrying the trigger's RAISE(ABORT,
        ...) message. We match that specific substring explicitly so it is
        mapped to a 409/400 by the route, NOT conflated with a generic
        integrity violation. Normal use never reaches the trigger because
        the SELECT-then-UPDATE pattern with ``WHERE choice='pending'``
        short-circuits to a ValueError first.

        Telemetry: NONE — composition-time user decisions are audit-primary;
        no ephemeral operational signal required.
        """
        now = self._ensure_utc(resolved_at) if resolved_at is not None else self._now()
        sid = str(session_id)
        eid = str(event_id)
        plugin_snapshot = await self._plugin_snapshot_for_session(sid)

        def _sync() -> tuple[InterpretationEventRecord, CompositionStateRecord]:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                # Step 1: SELECT pending event, scoped to session.
                event_row = conn.execute(
                    select(interpretation_events_table)
                    .where(interpretation_events_table.c.id == eid)
                    .where(interpretation_events_table.c.session_id == sid)
                    .where(interpretation_events_table.c.choice == InterpretationChoice.PENDING.value)
                ).one_or_none()
                if event_row is None:
                    existing_event_row = conn.execute(
                        select(interpretation_events_table)
                        .where(interpretation_events_table.c.id == eid)
                        .where(interpretation_events_table.c.session_id == sid)
                    ).one_or_none()
                    if existing_event_row is not None:
                        raise InterpretationEventAlreadyResolvedError(
                            f"resolve_interpretation_event: interpretation event {eid!r} in session {sid!r} is already resolved"
                        )
                    raise InterpretationEventNotFoundError(
                        f"resolve_interpretation_event: interpretation event {eid!r} not found in session {sid!r}"
                    )

                # Step 2: compute accepted_value per F-14.
                if choice is InterpretationChoice.ACCEPTED_AS_DRAFTED:
                    accepted_value = event_row.llm_draft
                    if accepted_value is None:
                        raise AuditIntegrityError(
                            f"resolve_interpretation_event: event {eid!r} has no llm_draft to accept; row shape is malformed"
                        )
                elif choice is InterpretationChoice.AMENDED:
                    if amended_value is None:
                        raise ValueError("resolve_interpretation_event: choice=AMENDED requires amended_value to be set")
                    accepted_value = amended_value
                else:
                    raise ValueError(
                        f"resolve_interpretation_event: choice {choice!r} is not "
                        f"a resolution choice; only ACCEPTED_AS_DRAFTED and AMENDED "
                        f"are valid here"
                    )

                kind = InterpretationKind(event_row.kind)
                if choice is InterpretationChoice.AMENDED and kind in {
                    InterpretationKind.INVENTED_SOURCE,
                    InterpretationKind.LLM_PROMPT_TEMPLATE,
                    InterpretationKind.PIPELINE_DECISION,
                }:
                    raise InterpretationUnsupportedChoiceError(
                        f"resolve_interpretation_event: {kind.value} does not support inline amendment in this release"
                    )
                if kind in {
                    InterpretationKind.VAGUE_TERM,
                    InterpretationKind.INVENTED_SOURCE,
                    InterpretationKind.PIPELINE_DECISION,
                    InterpretationKind.LLM_MODEL_CHOICE,
                }:
                    # Step 3: defence-in-depth validation of user/LLM-supplied
                    # content. Prompt-template review carries real Jinja and
                    # deliberately skips this accepted-value validator.
                    _validate_accepted_value_content(accepted_value)

                # Step 4: produce kind-specific patched state. Helpers raise
                # typed interpretation errors on structural anomalies; the
                # raise short-circuits the transaction before any UPDATE/INSERT.
                #
                # Vague-term patches still land on the CURRENT composition
                # state (highest version for the session), not the surfacing
                # state. Prompt-template and invented-source reviews follow
                # the same current-state rule but update only review metadata:
                # prompt-template review stamps the existing LLM prompt hash,
                # and invented-source review stamps source authoring metadata.
                live_state_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if live_state_row is None:
                    raise AuditIntegrityError(f"resolve_interpretation_event: session {sid!r} has no composition state to patch")
                state_record = self._row_to_state_record(live_state_row)
                final_sources: Mapping[str, Mapping[str, Any]] | None
                final_nodes: list[Mapping[str, Any]]
                resolved_prompt_template_hash: str | None
                if kind is InterpretationKind.VAGUE_TERM:
                    final_sources, final_nodes, resolved_prompt_template_hash = _resolve_vague_term(
                        state_record,
                        affected_node_id=event_row.affected_node_id,
                        user_term=event_row.user_term,
                        accepted_value=accepted_value,
                    )
                elif kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
                    # Skeleton the user actually reviewed (the surfacing state's
                    # node). The acceptance gate compares this to the live
                    # skeleton so a sibling vague_term baked between surfacing and
                    # resolve does not invalidate this review (elspeth-e51216d305).
                    surfacing_structure_hash: str | None = None
                    if event_row.composition_state_id is not None:
                        surfacing_state_row = conn.execute(
                            select(composition_states_table)
                            .where(composition_states_table.c.id == event_row.composition_state_id)
                            .where(composition_states_table.c.session_id == sid)
                        ).one_or_none()
                        if surfacing_state_row is not None:
                            surfacing_structure_hash = _surfacing_prompt_structure_hash(
                                self._row_to_state_record(surfacing_state_row),
                                affected_node_id=event_row.affected_node_id,
                            )
                    final_sources, final_nodes, resolved_prompt_template_hash = _resolve_prompt_template_review(
                        state_record,
                        event_id=eid,
                        affected_node_id=event_row.affected_node_id,
                        user_term=event_row.user_term,
                        accepted_value=accepted_value,
                        surfacing_structure_hash=surfacing_structure_hash,
                    )
                elif kind is InterpretationKind.INVENTED_SOURCE:
                    final_sources, final_nodes, resolved_prompt_template_hash = _resolve_invented_source(
                        state_record,
                        event_id=eid,
                        affected_node_id=event_row.affected_node_id,
                        user_term=event_row.user_term,
                        llm_draft=event_row.llm_draft,
                        accepted_value=accepted_value,
                    )
                elif kind is InterpretationKind.PIPELINE_DECISION:
                    final_sources, final_nodes, resolved_prompt_template_hash = _resolve_pipeline_decision_review(
                        state_record,
                        event_id=eid,
                        affected_node_id=event_row.affected_node_id,
                        user_term=event_row.user_term,
                        llm_draft=event_row.llm_draft,
                        accepted_value=accepted_value,
                    )
                elif kind is InterpretationKind.LLM_MODEL_CHOICE:
                    final_sources, final_nodes, resolved_prompt_template_hash = _resolve_model_choice_review(
                        state_record,
                        event_id=eid,
                        affected_node_id=event_row.affected_node_id,
                        user_term=event_row.user_term,
                        llm_draft=event_row.llm_draft,
                        accepted_value=accepted_value,
                    )
                else:
                    raise AssertionError(f"unhandled InterpretationKind {kind!r}")

                # The live state can be invalid solely because an earlier
                # finalisation/runtime-preflight pass saw the unresolved
                # ``{{interpretation:<term>}}`` placeholder. Resolving the
                # event consumes that placeholder, so the new state row must
                # carry validation for the patched state, not stale failure
                # metadata copied from the pre-resolve live row.
                from elspeth.web.sessions.converters import state_from_record

                patched_state_record = replace(
                    state_record,
                    source=None,
                    sources=final_sources,
                    nodes=final_nodes,
                    is_valid=False,
                    validation_errors=None,
                )
                patched_validation = self._validate_patched_composition_state(
                    state_from_record(patched_state_record),
                    plugin_snapshot=plugin_snapshot,
                )
                patched_validation_errors = [error.message for error in patched_validation.errors] or None

                # Compute arguments_hash over the INTERPRETATION_HASH_DOMAIN_V2
                # field set. The closed domain is the source of truth — read
                # from the constant, do not duplicate the field list inline.
                # The composition_state_id in the hash domain is the surfacing
                # anchor (the pending row's composition_state_id), not the
                # live state we patched — the hash identifies the discrete
                # surface-and-decide event, not the state mutation it
                # produced.
                surfacing_state_id_str = event_row.composition_state_id
                domain_dict = _interpretation_hash_domain_v2(
                    session_id=sid,
                    composition_state_id=surfacing_state_id_str,
                    affected_node_id=event_row.affected_node_id,
                    tool_call_id=event_row.tool_call_id,
                    user_term=event_row.user_term,
                    kind=event_row.kind,
                    llm_draft=event_row.llm_draft,
                    accepted_value=accepted_value,
                    actor=actor,
                    model_identifier=event_row.model_identifier,
                    model_version=event_row.model_version,
                    provider=event_row.provider,
                    composer_skill_hash=event_row.composer_skill_hash,
                    context="resolve_interpretation_event",
                )
                arguments_hash = stable_hash(domain_dict)

                # Step 5: UPDATE the interpretation event. The trigger
                # short-circuit (F-28) is unreachable on this branch because
                # the SELECT already filtered to choice='pending'; we catch
                # IntegrityError defensively in case a future refactor
                # reorders the writes.
                try:
                    conn.execute(
                        update(interpretation_events_table)
                        .where(interpretation_events_table.c.id == eid)
                        .where(interpretation_events_table.c.session_id == sid)
                        .where(interpretation_events_table.c.choice == InterpretationChoice.PENDING.value)
                        .values(
                            choice=choice.value,
                            accepted_value=accepted_value,
                            resolved_at=now,
                            actor=actor,
                            arguments_hash=arguments_hash,
                            hash_domain_version="v2",
                            runtime_model_identifier_at_resolve=runtime_model_identifier,
                            runtime_model_version_at_resolve=runtime_model_version,
                            resolved_prompt_template_hash=resolved_prompt_template_hash,
                        )
                    )
                except IntegrityError as exc:
                    # F-28: classify the trigger immutability message
                    # specifically. Any other IntegrityError reraises as-is.
                    if _INTERPRETATION_IMMUTABLE_TRIGGER_MSG in str(exc):
                        raise InterpretationEventAlreadyResolvedError(
                            f"resolve_interpretation_event: event {eid!r} is already resolved (immutability trigger fired)"
                        ) from exc
                    raise

                # Step 5a: write the new composition state row carrying the
                # patched prompt template + hash sibling. The
                # ``interpretation_resolve`` provenance is the load-bearing
                # discriminator that lets backward-direction audit walks
                # identify this row as resulting from an interpretation
                # decision.
                new_state_id_str = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(
                        data=CompositionStateData(
                            sources=final_sources,
                            nodes=final_nodes,
                            edges=state_record.edges,
                            outputs=state_record.outputs,
                            metadata_=state_record.metadata_,
                            is_valid=patched_validation.is_valid,
                            validation_errors=patched_validation_errors,
                            composer_meta=state_record.composer_meta,
                        ),
                        derived_from_state_id=str(state_record.id),
                    ),
                    provenance="interpretation_resolve",
                    created_at=now,
                )

                resolved_event_row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == eid)).one()
                new_state_row = conn.execute(
                    select(composition_states_table).where(composition_states_table.c.id == new_state_id_str)
                ).one()
                return (
                    _interpretation_event_record_from_row(resolved_event_row),
                    self._row_to_state_record(new_state_row),
                )

        return cast(
            tuple[InterpretationEventRecord, CompositionStateRecord],
            await self._run_sync(_sync),
        )

    async def list_interpretation_events(
        self,
        session_id: UUID,
        *,
        status: Literal["pending", "all"] = "all",
        composition_state_id: UUID | None = None,
        sources: Sequence[InterpretationSource] | None = None,
    ) -> list[InterpretationEventRecord]:
        """Read-back of interpretation events for the session.

        Used by the audit-readiness panel (counts), by the frontend on
        reload (rehydrate pending review affordances), and by the
        opt-out audit-summary surface (``sources`` filter — F-22).

        Telemetry: NONE — composition-time user decisions are audit-primary;
        no ephemeral operational signal required.
        """
        sid = str(session_id)
        cs_id = str(composition_state_id) if composition_state_id is not None else None
        # Materialise the source-value list once so the inner _sync closure
        # uses primitive string values, not enum instances captured by
        # closure (cheap; defensive against the iterable being a generator).
        source_values: list[str] | None = [s.value for s in sources] if sources is not None else None

        def _sync() -> list[InterpretationEventRecord]:
            stmt = select(interpretation_events_table).where(interpretation_events_table.c.session_id == sid)
            if status == "pending":
                stmt = stmt.where(interpretation_events_table.c.choice == InterpretationChoice.PENDING.value)
            if cs_id is not None:
                stmt = stmt.where(interpretation_events_table.c.composition_state_id == cs_id)
            if source_values is not None:
                stmt = stmt.where(interpretation_events_table.c.interpretation_source.in_(source_values))
            stmt = stmt.order_by(
                interpretation_events_table.c.created_at,
                interpretation_events_table.c.id,
            )
            with self._engine.begin() as conn:
                rows = conn.execute(stmt).fetchall()
                return [_interpretation_event_record_from_row(row) for row in rows]

        return cast(list[InterpretationEventRecord], await self._run_sync(_sync))

    async def record_session_interpretation_opt_out(
        self,
        *,
        session_id: UUID,
        actor: str,
        opted_out_at: datetime | None = None,
    ) -> InterpretationEventRecord:
        """Mark the session as 'don't surface interpretations any more'.

        F-27 (write-lock annotation): acquires the session write lock for
        the ENTIRE duration of the transaction — both the
        interpretation_events INSERT and the sessions boolean UPDATE are
        inside one ``_session_write_lock`` block to ensure atomicity.

        Idempotency (F-29): if an opted_out row already exists for this
        session, return the existing record without inserting a duplicate.
        The sessions boolean remains true. First opt-out timestamp is
        authoritative.

        Writes a row to ``interpretation_events_table`` with
        ``choice='opted_out'``, ``interpretation_source='auto_interpreted_opt_out'``,
        all nullable interpretation fields NULL, and ``resolved_at`` set
        to ``opted_out_at``. Also sets
        ``sessions.interpretation_review_disabled = true``. Single
        transaction.

        Does NOT write to ``proposal_events_table``. The
        ``interpretation_events`` table is the single source of truth for
        all interpretation-related decisions.

        Telemetry: ``composer.interpretation.opt_out_total`` fires on the
        INSERT path only (B3 cohort b1 — Sub-task 7e in Phase 8 plan). The
        F-29 idempotent re-fire returns the existing row without writing a
        new audit row, so emitting there would over-count and break the
        superset rule (the counter must aggregate over audit rows, not over
        route hits). Emission happens AFTER the transaction commits, in
        line with the ``record_audit_grade_view`` pattern elsewhere in
        this module.
        """
        now = self._ensure_utc(opted_out_at) if opted_out_at is not None else self._now()
        sid = str(session_id)

        def _sync() -> tuple[InterpretationEventRecord, bool]:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                # F-29: idempotency. SELECT inside the lock so the
                # SELECT-then-INSERT sequence is atomic against concurrent
                # writers. If a prior opt-out exists, return it; do not
                # insert a duplicate (the closed enum on choice plus the
                # source-keyed nullability CHECK would not prevent
                # duplicates, only structurally-bad rows).
                existing = conn.execute(
                    select(interpretation_events_table)
                    .where(interpretation_events_table.c.session_id == sid)
                    .where(interpretation_events_table.c.interpretation_source == InterpretationSource.AUTO_INTERPRETED_OPT_OUT.value)
                    .order_by(interpretation_events_table.c.created_at)
                    .limit(1)
                ).one_or_none()
                if existing is not None:
                    return _interpretation_event_record_from_row(existing), False

                event_id = str(uuid.uuid4())
                conn.execute(
                    insert(interpretation_events_table).values(
                        id=event_id,
                        session_id=sid,
                        composition_state_id=None,
                        affected_node_id=None,
                        tool_call_id=None,
                        user_term=None,
                        kind=None,
                        llm_draft=None,
                        accepted_value=None,
                        choice=InterpretationChoice.OPTED_OUT.value,
                        created_at=now,
                        resolved_at=now,
                        actor=actor,
                        model_identifier=None,
                        model_version=None,
                        provider=None,
                        composer_skill_hash=None,
                        arguments_hash=None,
                        hash_domain_version=None,
                        interpretation_source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT.value,
                        runtime_model_identifier_at_resolve=None,
                        runtime_model_version_at_resolve=None,
                        resolved_prompt_template_hash=None,
                    )
                )
                conn.execute(
                    update(sessions_table).where(sessions_table.c.id == sid).values(interpretation_review_disabled=True, updated_at=now)
                )
                row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == event_id)).one()
                return _interpretation_event_record_from_row(row), True

        record, was_inserted = cast(
            "tuple[InterpretationEventRecord, bool]",
            await self._run_sync(_sync),
        )
        # B3 cohort b1 — Phase 5b interpretation opt-out (Sub-task 7e).
        # Helper-based emit. The helper applies W5 wrapping so a broken
        # OTel exporter cannot 500 a POST whose audit row already wrote.
        # Fires only on the INSERT path; idempotent re-fires do not emit
        # (no new audit row → no aggregate increment, per superset rule).
        if was_inserted:
            record_interpretation_opt_out(self._telemetry)
        return record

    async def upsert_skill_markdown_history(
        self,
        *,
        skill_hash: str,
        filename: str,
        content: str,
        first_seen_at: datetime | None = None,
    ) -> bool:
        """Best-effort INSERT-OR-IGNORE into ``skill_markdown_history`` (F-5c).

        Called once per ``(skill_hash, compose_loop_init)`` so a forensic
        auditor can reconstruct the exact composer skill text that was in
        memory when any ``interpretation_events.composer_skill_hash``
        column was populated. The hash is the primary key, so subsequent
        upserts with the same hash collapse via ``INSERT OR IGNORE`` —
        no row is duplicated.

        **Best-effort, not transactional.** This writer is intentionally
        decoupled from the interpretation-event row write (per the spec
        at docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md
        §"skill_markdown_history upsert (F-5c)"). Storage cost is
        bounded — one row per distinct deploy of the skill markdown.

        Returns ``True`` when a row was inserted, ``False`` when the row
        already existed (the upsert was a no-op). Callers MAY use the
        return value for telemetry, but they MUST NOT branch on it for
        correctness: the table's contents are an audit-archive, not a
        coordination surface.

        Trust tier: the ``skill_hash`` / ``filename`` / ``content`` triple
        is operator-supplied (drawn from the on-disk skill file the
        operator deployed). It is Tier-1 data on insert — we crash on any
        DB-side anomaly. The caller's discipline (passing values from
        ``load_skill_with_hash``) ensures atomic consistency.
        """
        now = self._ensure_utc(first_seen_at) if first_seen_at is not None else self._now()

        def _sync() -> bool:
            values = {
                "hash": skill_hash,
                "filename": filename,
                "content": content,
                "first_seen_at": now,
            }
            with self._engine.begin() as conn:
                dialect = conn.dialect.name
                stmt: Any
                if dialect == "sqlite":
                    stmt = sqlite_insert(skill_markdown_history_table).values(**values)
                elif dialect == "postgresql":
                    stmt = postgresql_insert(skill_markdown_history_table).values(**values)
                else:
                    raise NotImplementedError(
                        "skill_markdown_history requires an atomic insert-or-ignore for session database "
                        f"dialect {dialect!r}; supported dialects: sqlite, postgresql"
                    )
                stmt = stmt.on_conflict_do_nothing(index_elements=[skill_markdown_history_table.c.hash]).returning(
                    skill_markdown_history_table.c.hash
                )
                result = conn.execute(stmt)
                # ``RETURNING`` yields the hash only when an INSERT occurred;
                # conflict-ignored writes yield no row. This remains truthful
                # across both supported drivers without relying on rowcount.
                return result.scalar_one_or_none() is not None

        return cast(bool, await self._run_sync(_sync))

    async def record_auto_interpreted_no_surfaces_event(
        self,
        *,
        session_id: UUID,
        actor: str,
        kind: InterpretationKind,
        model_identifier: str,
        model_version: str,
        provider: str,
        composer_skill_hash: str,
        created_at: datetime | None = None,
    ) -> InterpretationEventRecord:
        """Write an AUTO_INTERPRETED_NO_SURFACES row (F-6).

        Triggered by the compose loop when the per-term or per-day
        ``request_interpretation_review`` rate cap is hit and the LLM
        is expected to bake the interpretation directly into the prompt
        template without surfacing it for review. The row records that
        the LLM *was* consulted (provenance fields populated) but no
        surface was produced (interpretation surface fields NULL).

        Validates against ``ck_interpretation_events_no_surfaces_shape``:
        the five interpretation-surface fields (composition_state_id,
        affected_node_id, tool_call_id, user_term, llm_draft) MUST be
        NULL; kind and the four LLM provenance fields (model_identifier,
        model_version, provider, composer_skill_hash) MUST be NOT NULL.

        ``choice`` is set to ``OPTED_OUT`` because the resolve semantics
        are "no further user action required" — the rate cap is the
        resolution. ``resolved_at`` equals ``created_at`` because the
        row is born resolved. ``arguments_hash`` is NULL because no
        user-visible surface was created to resolve.

        Telemetry: NONE — composition-time user decisions are
        audit-primary; no ephemeral operational signal required.
        """
        if not isinstance(kind, InterpretationKind):
            raise ValueError(f"kind must be InterpretationKind, got {type(kind).__name__}: {kind!r}")
        now = self._ensure_utc(created_at) if created_at is not None else self._now()
        sid = str(session_id)
        kind_value = kind.value
        event_id = str(uuid.uuid4())

        def _sync() -> InterpretationEventRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                conn.execute(
                    insert(interpretation_events_table).values(
                        id=event_id,
                        session_id=sid,
                        composition_state_id=None,
                        affected_node_id=None,
                        tool_call_id=None,
                        user_term=None,
                        kind=kind_value,
                        llm_draft=None,
                        accepted_value=None,
                        choice=InterpretationChoice.OPTED_OUT.value,
                        created_at=now,
                        resolved_at=now,
                        actor=actor,
                        model_identifier=model_identifier,
                        model_version=model_version,
                        provider=provider,
                        composer_skill_hash=composer_skill_hash,
                        arguments_hash=None,
                        hash_domain_version=None,
                        interpretation_source=InterpretationSource.AUTO_INTERPRETED_NO_SURFACES.value,
                        runtime_model_identifier_at_resolve=None,
                        runtime_model_version_at_resolve=None,
                        resolved_prompt_template_hash=None,
                    )
                )
                row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == event_id)).one()
                return _interpretation_event_record_from_row(row)

        return cast(InterpretationEventRecord, await self._run_sync(_sync))

    async def add_message(
        self,
        session_id: UUID,
        role: ChatMessageRole,
        content: str,
        *,
        writer_principal: ChatMessageWriterPrincipal,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        composition_state_id: UUID | None = None,
        raw_content: str | None = None,
        tool_call_id: str | None = None,
        parent_assistant_id: UUID | None = None,
    ) -> ChatMessageRecord:
        """Add a chat message and update the session's ``updated_at``.

        BREAKING CHANGE in rev 4: ``writer_principal`` is now a required
        keyword-only argument (must be one of the values listed in the
        ``ck_chat_messages_writer_principal`` CHECK constraint). All
        callers were updated atomically with this signature change per
        the no-legacy single-cut policy.

        Preserved behaviours from pre-rev-4:
        - ``_assert_state_in_session`` cross-session guard fires when
          ``composition_state_id`` is not None.
        - ``sessions_table.updated_at`` is bumped to ``now``.
        - ``raw_content`` is persisted verbatim when supplied.
        - Returns ``ChatMessageRecord``, not just an id string.

        New in rev 4:
        - ``sequence_no`` is allocated under ``_session_write_lock``
          (PostgreSQL advisory lock or SQLite per-session process lock),
          replacing the implicit "last write wins" ordering.
        - ``tool_call_id`` and ``parent_assistant_id`` MUST be set when
          ``role='tool'`` and MUST be ``None`` otherwise; the
          ``ck_chat_messages_tool_call_id_role`` and
          ``ck_chat_messages_parent_role`` CHECK constraints enforce
          this at write time.
        """
        now = self._now()
        sid = str(session_id)
        csid = str(composition_state_id) if composition_state_id else None
        pid = str(parent_assistant_id) if parent_assistant_id else None
        msg_id_holder: dict[str, str] = {}
        sequence_holder: dict[str, int] = {}

        def _sync() -> None:
            with self._session_process_locked_begin(sid) as conn:
                if csid is not None:
                    _assert_state_in_session(
                        conn,
                        state_id=csid,
                        expected_session_id=sid,
                        caller="add_message",
                    )
                with self._session_write_lock(conn, sid):
                    seq = self._reserve_sequence_range(conn, sid, count=1)
                    sequence_holder["sequence_no"] = seq
                    msg_id_holder["id"] = self._insert_chat_message(
                        conn,
                        session_id=sid,
                        role=role,
                        content=content,
                        raw_content=raw_content,
                        # ``deep_thaw`` matches the persist_compose_turn
                        # site: SQLAlchemy JSON serialisation handles raw
                        # dicts/lists, but tool_calls may be a
                        # ``MappingProxyType`` / ``tuple`` after frozen-
                        # dataclass round-trips, which the JSON encoder
                        # rejects.
                        tool_calls=deep_thaw(tool_calls) if tool_calls else None,
                        sequence_no=seq,
                        writer_principal=writer_principal,
                        composition_state_id=csid,
                        tool_call_id=tool_call_id,
                        parent_assistant_id=pid,
                        created_at=now,
                    )
                conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

        await self._run_sync(_sync)

        return ChatMessageRecord(
            id=UUID(msg_id_holder["id"]),
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            tool_calls=tool_calls,
            created_at=now,
            sequence_no=sequence_holder["sequence_no"],
            composition_state_id=composition_state_id,
            writer_principal=writer_principal,
            tool_call_id=tool_call_id,
            parent_assistant_id=parent_assistant_id,
        )

    async def get_messages(
        self,
        session_id: UUID,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[ChatMessageRecord]:
        """Get messages for a session, ordered by ``sequence_no`` ascending.

        Rev-4 (B2): the canonical ordering key is ``sequence_no``, allocated
        under the per-session advisory/process write lock. ``created_at`` is
        informational; on fast SQLite paths multiple rows in one
        ``persist_compose_turn`` share a single timestamp, so ordering by
        ``created_at`` produced arbitrary intra-turn ordering. The
        per-session unique index ``ix_chat_messages_session_sequence``
        makes the new key total within a session.
        """

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(chat_messages_table)
                    .where(chat_messages_table.c.session_id == str(session_id))
                    .order_by(chat_messages_table.c.sequence_no)
                    .limit(limit)
                    .offset(offset)
                ).fetchall()

        rows = await self._run_sync(_sync)

        return [
            ChatMessageRecord(
                id=UUID(row.id),
                session_id=UUID(row.session_id),
                role=row.role,
                content=row.content,
                raw_content=row.raw_content,
                tool_calls=row.tool_calls,
                created_at=self._ensure_utc(row.created_at),
                sequence_no=row.sequence_no,
                composition_state_id=UUID(row.composition_state_id) if row.composition_state_id else None,
                writer_principal=row.writer_principal,
                tool_call_id=row.tool_call_id,
                parent_assistant_id=UUID(row.parent_assistant_id) if row.parent_assistant_id else None,
            )
            for row in rows
        ]

    def count_tool_responses_for_assistant(
        self,
        *,
        session_id: str,
        assistant_message_id: str | None,
    ) -> int:
        """Count role='tool' rows linked to the given assistant message."""

        if assistant_message_id is None:
            return 0
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(chat_messages_table)
                .where(chat_messages_table.c.session_id == session_id)
                .where(chat_messages_table.c.parent_assistant_id == assistant_message_id)
                .where(chat_messages_table.c.role == "tool")
            ).scalar_one()
        return int(result)

    async def count_tool_responses_for_assistant_async(
        self,
        *,
        session_id: str,
        assistant_message_id: str | None,
    ) -> int:
        """Async dispatcher for :meth:`count_tool_responses_for_assistant`."""

        return cast(
            int,
            await self._run_sync(
                self.count_tool_responses_for_assistant,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
            ),
        )

    @staticmethod
    def _validate_audit_grade_query_args(query_args: Mapping[str, str]) -> dict[str, str]:
        """Return an owned dict after enforcing the privacy allowlist."""

        unexpected = frozenset(query_args) - AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST
        if unexpected:
            raise ValueError(f"unallowlisted audit-grade query args: {sorted(unexpected)}")
        return dict(query_args)

    def record_audit_grade_view(
        self,
        *,
        session_id: str,
        requesting_principal: str,
        request_path: str,
        query_args: Mapping[str, str],
        ip_address: str | None,
    ) -> None:
        """Append one row to ``audit_access_log`` before returning tool rows.

        The writer principal is pinned to ``audit_grade_view``; admin-tool
        writes use a separate path. The privacy posture is mechanical:
        ``query_args`` must already be reduced to the closed allowlist in
        ``AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST``, and ``ip_address`` is
        either stored literally or omitted as ``None``.
        """

        allowed_query_args = self._validate_audit_grade_query_args(query_args)
        now = self._now()
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    audit_access_log_table.insert().values(
                        id=str(uuid.uuid4()),
                        timestamp=now,
                        session_id=session_id,
                        requesting_principal=requesting_principal,
                        request_path=request_path,
                        query_args=allowed_query_args,
                        ip_address=ip_address,
                        writer_principal=AUDIT_GRADE_VIEW_WRITER_PRINCIPAL,
                    )
                )
        except SQLAlchemyError as exc:
            self._telemetry.audit_access_log_write_failed_total.add(1)
            raise AuditAccessLogWriteError("audit_access_log write failed for audit-grade messages view") from exc
        self._telemetry.audit_grade_view_total.add(1)

    async def record_audit_grade_view_async(
        self,
        *,
        session_id: str,
        requesting_principal: str,
        request_path: str,
        query_args: Mapping[str, str],
        ip_address: str | None,
    ) -> None:
        """Async dispatcher for :meth:`record_audit_grade_view`."""

        await self._run_sync(
            self.record_audit_grade_view,
            session_id=session_id,
            requesting_principal=requesting_principal,
            request_path=request_path,
            query_args=query_args,
            ip_address=ip_address,
        )

    def list_audit_access_log(self, *, session_id: str) -> list[AuditAccessLogRecord]:
        """Return audit-access log rows for tests and operator diagnostics."""

        with self._engine.connect() as conn:
            rows = conn.execute(
                select(audit_access_log_table)
                .where(audit_access_log_table.c.session_id == session_id)
                .order_by(audit_access_log_table.c.timestamp)
            ).fetchall()
        return [
            AuditAccessLogRecord(
                id=row.id,
                timestamp=self._ensure_utc(row.timestamp),
                session_id=row.session_id,
                requesting_principal=row.requesting_principal,
                request_path=row.request_path,
                query_args=row.query_args,
                ip_address=row.ip_address,
                writer_principal=row.writer_principal,
            )
            for row in rows
        ]

    async def save_composition_state(
        self,
        session_id: UUID,
        state: CompositionStateData,
        *,
        provenance: CompositionStateProvenance,
    ) -> CompositionStateRecord:
        """Save a new immutable composition state snapshot.

        Version is max(existing versions for session) + 1, starting at 1.

        ``provenance`` is required (no default). Earlier revisions hardcoded
        ``"session_seed"`` here, which silently conflated four distinct
        writer paths (session create / branch reseed plus the three
        ``_handle_*`` partial-state captures from
        ``web/sessions/routes.py``). Threading the discriminator through
        the public API restores the audit attribution promised by
        §4.1.2 and the ``ck_composition_states_provenance`` CHECK
        constraint.
        """
        state_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        def _sync() -> int:
            # The per-session write lock makes the SELECT-MAX +
            # INSERT sequence atomic against every other writer for
            # this session_id on both PostgreSQL (advisory lock) and
            # SQLite (process-wide per-session RLock). If the lock
            # invariant is ever broken in a refactor, the
            # IntegrityError on uq_composition_state_version names the
            # constraint directly — no retry layer is permitted to
            # consume that diagnostic before it reaches the operator
            # (CLAUDE.md No Legacy Code Policy: no belt-and-suspenders).
            with self._session_process_locked_begin(sid) as conn:
                with self._session_write_lock(conn, sid):
                    result = conn.execute(
                        select(func.max(composition_states_table.c.version)).where(composition_states_table.c.session_id == sid)
                    ).scalar()
                    version = (result or 0) + 1

                    conn.execute(
                        insert(composition_states_table).values(
                            id=str(state_id),
                            session_id=sid,
                            version=version,
                            source=None,
                            sources=_enveloped_state_column(state.sources),
                            nodes=_enveloped_state_column(state.nodes),
                            edges=_enveloped_state_column(state.edges),
                            outputs=_enveloped_state_column(state.outputs),
                            metadata_=_enveloped_state_column(state.metadata_),
                            is_valid=state.is_valid,
                            validation_errors=deep_thaw(state.validation_errors),
                            composer_meta=_enveloped_state_column(state.composer_meta),
                            derived_from_state_id=None,
                            provenance=provenance,
                            created_at=now,
                        )
                    )
                return version

        version = await self._run_sync(_sync)

        return CompositionStateRecord(
            id=state_id,
            session_id=session_id,
            version=version,
            source=None,
            sources=state.sources,
            nodes=state.nodes,
            edges=state.edges,
            outputs=state.outputs,
            metadata_=state.metadata_,
            is_valid=state.is_valid,
            validation_errors=state.validation_errors,
            created_at=now,
            derived_from_state_id=None,
            composer_meta=state.composer_meta,
        )

    async def get_current_state(
        self,
        session_id: UUID,
    ) -> CompositionStateRecord | None:
        """Return the highest-version state for a session, or None."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == str(session_id))
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            return None

        return self._row_to_state_record(row)

    async def get_state_versions(
        self,
        session_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CompositionStateRecord]:
        """Return state versions for a session, ascending order."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == str(session_id))
                    .order_by(composition_states_table.c.version)
                    .limit(limit)
                    .offset(offset)
                ).fetchall()

        rows = await self._run_sync(_sync)

        return [self._row_to_state_record(row) for row in rows]

    @staticmethod
    def _unwrap_envelope(val: Any) -> Any:
        """Unwrap _version envelope from a JSON column value.

        Seam contract A: JSON columns are stored with {"_version": 1, "data": ...}.
        Raises ValueError on unknown versions. Returns None for NULL columns.
        """
        if val is None:
            return None
        if isinstance(val, dict) and "_version" in val:
            if val["_version"] != 1:
                raise ValueError(f"Unknown composition state envelope version: {val['_version']}")
            return val["data"]
        raise ValueError(
            f"Composition state column has no _version envelope: {val!r}. This indicates a bug in the write path or database corruption."
        )

    def _row_to_state_record(self, row: Any) -> CompositionStateRecord:
        """Convert a SQLAlchemy row to a CompositionStateRecord.

        Seam contract A: metadata_ maps DB column metadata_ back to the
        dataclass field. JSON columns are unwrapped from their _version envelope.
        """
        row_mapping = row._mapping
        return CompositionStateRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            version=row.version,
            source=self._unwrap_envelope(row.source),
            sources=self._unwrap_envelope(row_mapping["sources"] if "sources" in row_mapping else None),
            nodes=self._unwrap_envelope(row.nodes),
            edges=self._unwrap_envelope(row.edges),
            outputs=self._unwrap_envelope(row.outputs),
            metadata_=self._unwrap_envelope(row.metadata_),
            is_valid=row.is_valid,
            validation_errors=row.validation_errors,
            created_at=self._ensure_utc(row.created_at),
            derived_from_state_id=(UUID(row.derived_from_state_id) if row.derived_from_state_id is not None else None),
            composer_meta=self._unwrap_envelope(row.composer_meta),
        )

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord:
        """Create a new pending run, enforcing one active run per session (B6).

        Enforced by partial unique index uq_runs_one_active_per_session
        (at most one row with status IN ('pending','running') per session_id).
        The SELECT is an early-out optimization; the index is the real guard.
        Raises RunAlreadyActiveError if a pending or running run exists.
        """
        run_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)
        state_sid = str(state_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                _assert_state_in_session(
                    conn,
                    state_id=state_sid,
                    expected_session_id=sid,
                    caller="create_run",
                )

                # Early-out: check before INSERT to give a clear error message
                active = conn.execute(
                    select(runs_table.c.id).where(
                        runs_table.c.session_id == sid,
                        runs_table.c.status.in_(["pending", "running"]),
                    )
                ).fetchone()

                if active is not None:
                    raise RunAlreadyActiveError(sid)

                try:
                    conn.execute(
                        insert(runs_table).values(
                            id=str(run_id),
                            session_id=sid,
                            state_id=state_sid,
                            status="pending",
                            started_at=now,
                            rows_processed=0,
                            rows_failed=0,
                            pipeline_yaml=pipeline_yaml,
                        )
                    )
                except IntegrityError as exc:
                    # The pre-check for active runs passed, but a concurrent insert
                    # hit the partial unique index. This is genuinely "run already active."
                    raise RunAlreadyActiveError(sid) from exc

        await self._run_sync(_sync)

        return RunRecord(
            id=run_id,
            session_id=session_id,
            state_id=state_id,
            status="pending",
            started_at=now,
            finished_at=None,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            error=None,
            landscape_run_id=None,
            pipeline_yaml=pipeline_yaml,
        )

    async def get_run(self, run_id: UUID) -> RunRecord:
        """Fetch a run by ID. Raises ValueError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(runs_table).where(runs_table.c.id == str(run_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise ValueError(f"Run not found: {run_id}")

        return self._row_to_run_record(row)

    async def list_runs_for_session(self, session_id: UUID) -> list[RunRecord]:
        """List all runs for a session, newest first."""
        sid = str(session_id)

        def _sync() -> Any:
            with self._engine.connect() as conn:
                return conn.execute(
                    select(runs_table).where(runs_table.c.session_id == sid).order_by(runs_table.c.started_at.desc())
                ).fetchall()

        rows = await self._run_sync(_sync)
        return [self._row_to_run_record(row) for row in rows]

    async def append_run_event(
        self,
        *,
        run_id: UUID,
        timestamp: datetime,
        event_type: SessionRunEventType,
        data: Mapping[str, Any],
    ) -> RunEventRecord:
        """Append a structured run event for websocket replay and audit inspection."""
        if event_type not in SESSION_RUN_EVENT_TYPE_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: run_events.event_type is {event_type!r}, expected one of {sorted(SESSION_RUN_EVENT_TYPE_VALUES)}"
            )
        event_id = uuid.uuid4()
        rid = str(run_id)
        payload = deep_thaw(dict(data))

        def _sync() -> tuple[str, int]:
            with self._engine.connect() as lookup_conn:
                found_session_id = lookup_conn.execute(select(runs_table.c.session_id).where(runs_table.c.id == rid)).scalar_one_or_none()
            if found_session_id is None:
                raise ValueError(f"Run {run_id} not found")
            session_id = str(found_session_id)
            with self._session_process_locked_begin(session_id) as conn:
                with self._session_write_lock(conn, session_id):
                    locked_session_id = conn.execute(select(runs_table.c.session_id).where(runs_table.c.id == rid)).scalar_one_or_none()
                    if locked_session_id != session_id:
                        raise ValueError(f"Run {run_id} not found")
                    sequence = (
                        int(
                            conn.execute(
                                select(func.coalesce(func.max(run_events_table.c.sequence), 0)).where(run_events_table.c.run_id == rid)
                            ).scalar_one()
                        )
                        + 1
                    )
                    conn.execute(
                        insert(run_events_table).values(
                            id=str(event_id),
                            run_id=rid,
                            sequence=sequence,
                            timestamp=timestamp,
                            event_type=event_type,
                            data=payload,
                        )
                    )
                return session_id, sequence

        _session_id, sequence = await self._run_sync(_sync)
        return RunEventRecord(
            id=event_id,
            run_id=run_id,
            sequence=sequence,
            timestamp=timestamp,
            event_type=event_type,
            data=cast(Mapping[str, Any], payload),
        )

    async def list_run_events(self, run_id: UUID) -> list[RunEventRecord]:
        """List persisted run events in durable insertion order."""
        rid = str(run_id)

        def _sync() -> Any:
            with self._engine.connect() as conn:
                return conn.execute(
                    select(run_events_table).where(run_events_table.c.run_id == rid).order_by(run_events_table.c.sequence)
                ).fetchall()

        rows = await self._run_sync(_sync)
        return [self._row_to_run_event_record(row) for row in rows]

    async def update_run_status(
        self,
        run_id: UUID,
        status: SessionRunStatus,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_succeeded: int | None = None,
        rows_failed: int | None = None,
        rows_routed_success: int | None = None,
        rows_routed_failure: int | None = None,
        rows_quarantined: int | None = None,
    ) -> None:
        """Update a run's status and optional fields.

        Enforces LEGAL_RUN_TRANSITIONS (D3). Enforces landscape_run_id
        write-once semantics (D4). Sets finished_at for terminal states
        (completed, failed, cancelled). Optional parameters only update
        the column when not None. Raises ValueError if run not found or
        transition is illegal.
        """
        now = self._now()
        rid = str(run_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                # Read current state for transition + write-once validation
                current = conn.execute(
                    select(
                        runs_table.c.status,
                        runs_table.c.landscape_run_id,
                    ).where(runs_table.c.id == rid)
                ).fetchone()

                if current is None:
                    raise ValueError(f"Run not found: {run_id}")

                # D3: Enforce legal transitions — direct access; KeyError = Tier 1 crash.
                # Use the narrow subclass so the cancelled-race recovery in
                # ExecutionService can match identity, not message.  See
                # IllegalRunTransitionError docstring for the full rationale.
                current_status = current.status
                allowed = LEGAL_RUN_TRANSITIONS[current_status]
                if status not in allowed:
                    raise IllegalRunTransitionError(current_status, status, allowed)

                # D4: landscape_run_id is write-once
                if landscape_run_id is not None and current.landscape_run_id is not None:
                    raise ValueError(f"landscape_run_id already set to {current.landscape_run_id!r}; cannot overwrite")
                if status in OPERATOR_COMPLETION_RUN_STATUS_VALUES and not (landscape_run_id or current.landscape_run_id):
                    raise ValueError(f"{status} status requires landscape_run_id")
                if status == "failed" and not error:
                    raise ValueError("failed status requires error")

                values: dict[str, Any] = {"status": status}
                if status in SESSION_TERMINAL_RUN_STATUS_VALUES:
                    values["finished_at"] = now
                if error is not None:
                    values["error"] = error
                if landscape_run_id is not None:
                    values["landscape_run_id"] = landscape_run_id
                if rows_processed is not None:
                    values["rows_processed"] = rows_processed
                if rows_succeeded is not None:
                    values["rows_succeeded"] = rows_succeeded
                if rows_failed is not None:
                    values["rows_failed"] = rows_failed
                if rows_routed_success is not None:
                    values["rows_routed_success"] = rows_routed_success
                if rows_routed_failure is not None:
                    values["rows_routed_failure"] = rows_routed_failure
                if rows_quarantined is not None:
                    values["rows_quarantined"] = rows_quarantined

                conn.execute(update(runs_table).where(runs_table.c.id == rid).values(**values))

        await self._run_sync(_sync)

    async def record_blob_inline_resolutions(
        self,
        *,
        run_id: UUID,
        resolutions: Sequence[ResolvedBlobContent],
        attempt: int = 1,
    ) -> None:
        """Write audit rows for runtime-resolved inline blob content.

        This is an audit-primary write site: callers must invoke it before
        resolved bytes can reach plugin construction. A DB failure is a
        Tier-1 anomaly and propagates as ``AuditIntegrityError``.
        """
        if not resolutions:
            return

        run_id_str = str(run_id)
        now = self._now()

        def _sync() -> None:
            rows = [
                {
                    "run_id": run_id_str,
                    "attempt": attempt,
                    "field_path": resolution.field_path,
                    "blob_id": str(resolution.blob_id),
                    "content_hash": resolution.content_hash,
                    "byte_length": resolution.byte_length,
                    "mime_type": resolution.mime_type,
                    "encoding": resolution.encoding,
                    "resolved_at": now,
                }
                for resolution in resolutions
            ]
            try:
                with self._engine.begin() as conn:
                    conn.execute(insert(blob_inline_resolutions_table), rows)
            except SQLAlchemyError as exc:
                raise AuditIntegrityError(f"Tier 1: failed to record blob_inline_resolutions for run {run_id_str}: {exc}") from exc

        await self._run_sync(_sync)

    async def get_active_run(
        self,
        session_id: UUID,
    ) -> RunRecord | None:
        """Return the pending/running run for a session, or None."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(runs_table).where(
                        runs_table.c.session_id == str(session_id),
                        runs_table.c.status.in_(["pending", "running"]),
                    )
                ).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            return None

        return self._row_to_run_record(row)

    async def get_state(self, state_id: UUID) -> CompositionStateRecord:
        """Fetch a composition state by its primary key. Raises ValueError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(composition_states_table).where(composition_states_table.c.id == str(state_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise ValueError(f"State not found: {state_id}")

        return self._row_to_state_record(row)

    async def get_state_in_session(
        self,
        state_id: UUID,
        session_id: UUID,
    ) -> CompositionStateRecord:
        """Scoped read: fetch state and verify it belongs to ``session_id``.

        Runtime defence-in-depth complementing the current-schema
        composite foreign key: persisted data cannot create cross-session
        state references at the schema layer, and this method raises
        ``AuditIntegrityError`` on any mismatch it
        encounters. The exception class is chosen deliberately to match
        the cross-session blob-ref rejection in the
        ``fork_from_message`` route handler (web/sessions/routes.py):
        a Tier 1 anomaly in our own data must surface as corruption,
        not as a soft 404. ``ValueError`` on "state does not exist at
        all" is preserved from ``get_state`` for callers that still
        need to distinguish absence from mismatch.

        The ``AuditIntegrityError`` class is the same signal used by
        the ``fork_from_message`` route handler in
        ``sessions/routes.py`` for the cross-session blob-reference
        rejection — the two guards are the read-side and write-side of
        the same Tier 1 invariant.
        """
        record = await self.get_state(state_id)
        if record.session_id != session_id:
            raise AuditIntegrityError(
                f"Tier 1 audit anomaly: composition_state {state_id} "
                f"belongs to session {record.session_id}, not {session_id}. "
                f"Migration 007 composite FK prevents this for post-007 "
                f"data; pre-007 orphans should have been deleted by "
                f"Variant-A repair. Cross-session state reference rejected."
            )
        return record

    async def set_active_state(
        self,
        session_id: UUID,
        state_id: UUID,
    ) -> CompositionStateRecord:
        """Revert to a prior state by copying it as a new version.

        Creates a new version record that is a copy of the specified prior
        version (looked up by state_id). The new record gets
        version = max(existing) + 1. Raises ValueError if state_id not
        found or does not belong to the session.
        """
        sid = str(session_id)
        new_state_id = uuid.uuid4()
        now = self._now()

        def _sync() -> tuple[Any, int]:
            # The per-session write lock makes the prior-row SELECT +
            # SELECT-MAX + INSERT atomic against every other writer for
            # this session_id (advisory lock on Postgres, per-session
            # RLock on SQLite). If the lock invariant is ever broken in
            # a refactor, the IntegrityError on
            # uq_composition_state_version names the constraint
            # directly — no retry layer is permitted to consume that
            # diagnostic before it reaches the operator (CLAUDE.md No
            # Legacy Code Policy: no belt-and-suspenders). Same
            # discipline as save_composition_state._sync above.
            with self._session_process_locked_begin(sid) as conn:
                with self._session_write_lock(conn, sid):
                    prior_row = conn.execute(
                        select(composition_states_table).where(composition_states_table.c.id == str(state_id))
                    ).fetchone()

                    # NOTE: Both branches below raise ValueError (not RuntimeError),
                    # and the HTTP handler at routes.py maps ValueError to 404. This
                    # is INTENTIONAL and distinct from _assert_state_in_session
                    # (module-level) which raises RuntimeError on cross-session
                    # references:
                    #
                    #   * _assert_state_in_session guards internal callers that
                    #     supply BOTH session_id and state_id from the same scope
                    #     (e.g. add_message, create_run). A mismatch there is a
                    #     caller-code contract violation — RuntimeError/500 is
                    #     the correct signal because no legitimate user input
                    #     can produce it.
                    #
                    #   * set_active_state receives state_id from the HTTP body
                    #     while session_id comes from the authenticated URL path.
                    #     A state owned by another user's session is
                    #     indistinguishable from "does not exist" to this user —
                    #     surfacing a RuntimeError/500 would leak the existence
                    #     of that other session's states. Collapsing both cases
                    #     to ValueError -> 404 is the correct information-hiding
                    #     boundary for user-supplied identifiers.
                    #
                    # If you find yourself tempted to consolidate these checks,
                    # reconsider: the exception type is load-bearing because it
                    # encodes WHO is wrong (caller code vs. user) and the HTTP
                    # status depends on it.
                    if prior_row is None:
                        raise ValueError(f"State not found: {state_id}")
                    if prior_row.session_id != sid:
                        raise ValueError(f"State {state_id} does not belong to session {session_id}")

                    max_version = conn.execute(
                        select(func.max(composition_states_table.c.version)).where(composition_states_table.c.session_id == sid)
                    ).scalar()
                    new_version = (max_version or 0) + 1

                    conn.execute(
                        insert(composition_states_table).values(
                            id=str(new_state_id),
                            session_id=sid,
                            version=new_version,
                            # prior_row.* values are already enveloped — copy as-is
                            source=None,
                            sources=prior_row.sources,
                            nodes=prior_row.nodes,
                            edges=prior_row.edges,
                            outputs=prior_row.outputs,
                            metadata_=prior_row.metadata_,
                            is_valid=prior_row.is_valid,
                            validation_errors=prior_row.validation_errors,
                            composer_meta=prior_row.composer_meta,
                            derived_from_state_id=str(state_id),
                            provenance="session_seed",
                            created_at=now,
                        )
                    )
                return prior_row, new_version

        prior_row, new_version = await self._run_sync(_sync)

        return CompositionStateRecord(
            id=new_state_id,
            session_id=session_id,
            version=new_version,
            source=None,
            sources=self._unwrap_envelope(prior_row.sources),
            nodes=self._unwrap_envelope(prior_row.nodes),
            edges=self._unwrap_envelope(prior_row.edges),
            outputs=self._unwrap_envelope(prior_row.outputs),
            metadata_=self._unwrap_envelope(prior_row.metadata_),
            is_valid=prior_row.is_valid,
            validation_errors=prior_row.validation_errors,
            created_at=now,
            derived_from_state_id=state_id,
            composer_meta=self._unwrap_envelope(prior_row.composer_meta),
        )

    async def cancel_orphaned_runs(
        self,
        session_id: UUID,
        max_age_seconds: int = 3600,
    ) -> list[RunRecord]:
        """Force-cancel runs stuck in 'pending' or 'running' beyond max_age_seconds.

        Returns the list of cancelled RunRecords. Called by the execution
        service on startup and periodically to prevent orphaned runs from
        permanently blocking sessions (D5). Includes 'pending' because a
        crash between create_run() and the first update_run_status("running")
        would leave a permanently unblockable session otherwise.
        """
        sid = str(session_id)
        now = self._now()
        cutoff = now - timedelta(seconds=max_age_seconds)

        def _sync() -> list[RunRecord]:
            cancelled: list[RunRecord] = []
            with self._engine.begin() as conn:
                stale_rows = conn.execute(
                    select(runs_table).where(
                        runs_table.c.session_id == sid,
                        runs_table.c.status.in_(["pending", "running"]),
                        runs_table.c.started_at <= cutoff,
                    )
                ).fetchall()

                for row in stale_rows:
                    conn.execute(update(runs_table).where(runs_table.c.id == row.id).values(status="cancelled", finished_at=now))
                    cancelled.append(
                        RunRecord(
                            id=UUID(row.id),
                            session_id=UUID(row.session_id),
                            state_id=UUID(row.state_id),
                            status="cancelled",
                            started_at=self._ensure_utc(row.started_at),
                            finished_at=now,
                            rows_processed=row.rows_processed,
                            rows_succeeded=row.rows_succeeded,
                            rows_failed=row.rows_failed,
                            rows_routed_success=row.rows_routed_success,
                            rows_routed_failure=row.rows_routed_failure,
                            rows_quarantined=row.rows_quarantined,
                            error=row.error,
                            landscape_run_id=row.landscape_run_id,
                            pipeline_yaml=row.pipeline_yaml,
                        )
                    )
            return cancelled

        result: list[RunRecord] = cast(list[RunRecord], await self._run_sync(_sync))
        return result

    async def cancel_all_orphaned_runs(
        self,
        max_age_seconds: int | None = None,
        exclude_run_ids: frozenset[str] = frozenset(),
        reason: str | None = None,
    ) -> int:
        """Force-cancel orphaned runs across all sessions.

        Called on startup to recover sessions blocked by runs orphaned
        during a previous server crash. Returns the count of cancelled runs.

        Args:
            max_age_seconds: Only cancel runs older than this. None cancels
                all non-terminal runs (correct for single-process servers
                where every non-terminal run is orphaned after restart).
            exclude_run_ids: Run IDs known to have active executor threads.
                These are skipped even if they exceed max_age_seconds.
            reason: Written to the error column so operators can distinguish
                orphan-cleanup cancellations from user cancellations.
        """
        cancelled = await self.cancel_all_orphaned_run_records(
            max_age_seconds=max_age_seconds,
            exclude_run_ids=exclude_run_ids,
            reason=reason,
        )
        return len(cancelled)

    async def cancel_all_orphaned_run_records(
        self,
        max_age_seconds: int | None = None,
        exclude_run_ids: frozenset[str] = frozenset(),
        reason: str | None = None,
    ) -> list[RunRecord]:
        """Force-cancel orphaned runs and return the cancelled records.

        Startup reconciliation needs the cancelled run ids and their
        ``landscape_run_id`` anchors so it can terminalize the matching
        Landscape audit rows. ``cancel_all_orphaned_runs`` keeps the older
        integer API by delegating here.
        """
        now = self._now()

        def _sync() -> list[RunRecord]:
            with self._engine.begin() as conn:
                conditions: list[ColumnElement[bool]] = [runs_table.c.status.in_(["pending", "running"])]
                if max_age_seconds is not None:
                    cutoff = now - timedelta(seconds=max_age_seconds)
                    conditions.append(runs_table.c.started_at <= cutoff)
                if exclude_run_ids:
                    conditions.append(runs_table.c.id.not_in(exclude_run_ids))

                stale_rows = conn.execute(select(runs_table.c.id).where(*conditions)).fetchall()

                values: dict[str, Any] = {"status": "cancelled", "finished_at": now}
                if reason is not None:
                    values["error"] = reason

                cancelled: list[RunRecord] = []
                for row in stale_rows:
                    conn.execute(update(runs_table).where(runs_table.c.id == row.id).values(**values))
                    updated = conn.execute(select(runs_table).where(runs_table.c.id == row.id)).one()
                    cancelled.append(self._row_to_run_record(updated))
                return cancelled

        return cast(list[RunRecord], await self._run_sync(_sync))

    async def list_pending_landscape_reconciliations(self) -> list[RunRecord]:
        """Return the durable, exact pending Landscape reconciliation set."""
        from elspeth.web.sessions.protocol import LANDSCAPE_RECONCILIATION_PENDING_SUFFIX

        def _sync() -> list[RunRecord]:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(runs_table)
                    .where(
                        runs_table.c.status == "cancelled",
                        runs_table.c.error.is_not(None),
                        runs_table.c.error.endswith(LANDSCAPE_RECONCILIATION_PENDING_SUFFIX, autoescape=True),
                    )
                    .order_by(runs_table.c.started_at, runs_table.c.id)
                ).fetchall()
            return [self._row_to_run_record(row) for row in rows]

        return cast(list[RunRecord], await self._run_sync(_sync))

    async def mark_landscape_reconciliation_outcomes(
        self,
        *,
        complete_run_ids: frozenset[UUID],
        absent_run_ids: frozenset[UUID],
    ) -> None:
        """Atomically close exact pending markers without rewriting reasons."""
        from elspeth.web.sessions.protocol import (
            LANDSCAPE_RECONCILIATION_ABSENT_SUFFIX,
            LANDSCAPE_RECONCILIATION_COMPLETE_SUFFIX,
            LANDSCAPE_RECONCILIATION_PENDING_SUFFIX,
        )

        overlap = complete_run_ids & absent_run_ids
        if overlap:
            raise ValueError("Landscape reconciliation outcome sets overlap")

        def _sync() -> None:
            with self._engine.begin() as conn:
                outcomes = (
                    (complete_run_ids, LANDSCAPE_RECONCILIATION_COMPLETE_SUFFIX),
                    (absent_run_ids, LANDSCAPE_RECONCILIATION_ABSENT_SUFFIX),
                )
                for run_ids, closed_suffix in outcomes:
                    for run_id in sorted(run_ids, key=str):
                        row = conn.execute(
                            select(runs_table.c.status, runs_table.c.error).where(runs_table.c.id == str(run_id))
                        ).one_or_none()
                        if (
                            row is None
                            or row.status != "cancelled"
                            or not isinstance(row.error, str)
                            or not row.error.endswith(LANDSCAPE_RECONCILIATION_PENDING_SUFFIX)
                        ):
                            raise ValueError("Run is not an exact pending Landscape reconciliation candidate")
                        updated_error = row.error[: -len(LANDSCAPE_RECONCILIATION_PENDING_SUFFIX)] + closed_suffix
                        result = conn.execute(
                            update(runs_table)
                            .where(
                                runs_table.c.id == str(run_id),
                                runs_table.c.status == "cancelled",
                                runs_table.c.error == row.error,
                            )
                            .values(error=updated_error)
                        )
                        if result.rowcount != 1:
                            raise RuntimeError("Landscape reconciliation marker update lost its compare-and-swap")

        await self._run_sync(_sync)

    async def prune_state_versions(
        self,
        session_id: UUID,
        keep_latest: int = 50,
    ) -> int:
        """Delete old composition state versions beyond keep_latest.

        Preserves the most recent `keep_latest` versions and any versions
        referenced by a run (via runs.state_id). Returns the count of
        deleted versions.
        """
        sid = str(session_id)

        def _sync() -> int:
            with self._engine.begin() as conn:
                # Get all version IDs ordered by version DESC
                all_rows = conn.execute(
                    select(
                        composition_states_table.c.id,
                        composition_states_table.c.version,
                    )
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                ).fetchall()

                if len(all_rows) <= keep_latest:
                    return 0

                # IDs to keep: the top keep_latest versions
                keep_ids = {row.id for row in all_rows[:keep_latest]}

                # IDs referenced by runs
                run_referenced = {
                    row.state_id
                    for row in conn.execute(
                        select(runs_table.c.state_id).where(
                            runs_table.c.session_id == sid,
                        )
                    ).fetchall()
                }

                # IDs referenced by chat messages (provenance tracking)
                message_referenced = {
                    row.composition_state_id
                    for row in conn.execute(
                        select(chat_messages_table.c.composition_state_id).where(
                            chat_messages_table.c.session_id == sid,
                            chat_messages_table.c.composition_state_id.isnot(None),
                        )
                    ).fetchall()
                }

                # IDs referenced via derived_from_state_id (revert lineage).
                # Build transitive closure: if v5→v3→v1 and v5 is kept,
                # both v3 and v1 must be protected.
                derived_from_map: dict[str, str | None] = {
                    row.id: row.derived_from_state_id
                    for row in conn.execute(
                        select(
                            composition_states_table.c.id,
                            composition_states_table.c.derived_from_state_id,
                        ).where(composition_states_table.c.session_id == sid)
                    ).fetchall()
                }
                lineage_protected: set[str] = set()
                seeds = (keep_ids | run_referenced | message_referenced) & set(derived_from_map)
                for seed_id in seeds:
                    parent = derived_from_map[seed_id]
                    while parent is not None and parent not in lineage_protected:
                        if parent not in derived_from_map:
                            raise AuditIntegrityError(
                                f"composition state {seed_id} in session {sid} has dangling derived_from_state_id {parent}"
                            )
                        lineage_protected.add(parent)
                        parent = derived_from_map[parent]

                # Candidates for deletion: not kept, not referenced, not in lineage
                protected = keep_ids | run_referenced | message_referenced | lineage_protected
                delete_ids = [row.id for row in all_rows if row.id not in protected]

                if not delete_ids:
                    return 0

                result = conn.execute(delete(composition_states_table).where(composition_states_table.c.id.in_(delete_ids)))
                return result.rowcount

        return cast(int, await self._run_sync(_sync))

    async def fork_session(
        self,
        source_session_id: UUID,
        fork_message_id: UUID,
        new_message_content: str,
        user_id: str,
        auth_provider_type: AuthProviderType,
    ) -> tuple[SessionRecord, list[ChatMessageRecord], CompositionStateRecord | None]:
        """Fork a session from a specific user message.

        All writes happen in a single transaction — if anything fails,
        the entire fork is rolled back with no partial state.

        Creates a new session containing:
        1. Composition state copied from the fork message's pre-send state
        2. All messages BEFORE the fork message (with NULL state provenance)
        3. A synthetic system message noting the fork
        4. The new edited user message (provenance = copied state, not source)

        Returns (new_session, new_messages, copied_state_or_none).
        """
        from elspeth.web.sessions.protocol import InvalidForkTargetError

        # Load source data (read-only, outside the write transaction)
        source_session = await self.get_session(source_session_id)
        source_messages = await self.get_messages(source_session_id, limit=None)

        # Find the fork message — must be a user message
        fork_msg = None
        fork_idx = -1
        for i, msg in enumerate(source_messages):
            if msg.id == fork_message_id:
                fork_msg = msg
                fork_idx = i
                break

        if fork_msg is None:
            raise ValueError(f"Message {fork_message_id} not found in session {source_session_id}")
        if fork_msg.role != "user":
            raise InvalidForkTargetError(str(fork_message_id), fork_msg.role)

        messages_to_copy = source_messages[:fork_idx]
        pre_send_state_id = fork_msg.composition_state_id

        # Load source composition state if it exists (read-only)
        source_state_record: CompositionStateRecord | None = None
        if pre_send_state_id is not None:
            source_state_record = await self.get_state_in_session(
                pre_send_state_id,
                source_session_id,
            )

        # Profile strip (finding 10, rev 4): never let a tutorial WorkflowProfile
        # leak into a forked session. Computed once, used by BOTH verbatim
        # composer_meta copies below (the :5150 persist copy and the :5227 return
        # copy). The route-layer blob-rewrite save preserves composer_meta
        # verbatim and never strips the profile (and is not in this service
        # method's path), so the strip must live here — independent of rewritten.
        forked_composer_meta = _strip_guided_profile_in_meta(source_state_record.composer_meta) if source_state_record is not None else None

        # Prepare IDs and timestamps upfront
        new_session_id = uuid.uuid4()
        new_session_id_str = str(new_session_id)
        now = self._now()
        title = f"{source_session.title} (fork)"

        # Prepare state copy if needed
        copied_state_id = uuid.uuid4() if source_state_record is not None else None
        copied_state_id_str = str(copied_state_id) if copied_state_id else None

        # §14.6: build a source-id → copied-id map for in-slice assistant rows
        # BEFORE building msg_records_data so tool rows in the slice can rewrite
        # their ``parent_assistant_id`` to the copied assistant's new id.
        # Copying the source ``parent_assistant_id`` verbatim would point at
        # the SOURCE session's assistant; the copied tool row lives in a NEW
        # session and the FK ``(parent_assistant_id, session_id)`` would fail.
        source_to_copied_assistant_id: dict[str, str] = {}
        for msg in messages_to_copy:
            if msg.role == "assistant":
                source_to_copied_assistant_id[str(msg.id)] = str(uuid.uuid4())

        # Prepare all message rows upfront — preserve original created_at
        # so get_messages() ordering is deterministic.  Stamping all rows
        # with `now` would make them indistinguishable by timestamp and
        # produce non-deterministic ordering on subsequent reads.
        msg_records_data: list[dict[str, Any]] = []
        for msg in messages_to_copy:
            if msg.role == "assistant":
                copied_msg_id = source_to_copied_assistant_id[str(msg.id)]
            else:
                copied_msg_id = str(uuid.uuid4())

            copied_parent_assistant_id: str | None = None
            if msg.role == "tool":
                # CHECK constraint biconditional: tool rows must carry both
                # ``tool_call_id`` and ``parent_assistant_id``. The source row
                # already had them (the biconditional applies there too);
                # an absent value here means the source row predates the
                # cutover — that's a Tier-1 audit anomaly, crash with a named
                # error rather than letting the FK fire generically.
                if msg.parent_assistant_id is None:
                    raise RuntimeError(f"fork_session: tool message id={msg.id} has no parent assistant")
                parent_key = str(msg.parent_assistant_id)
                if parent_key not in source_to_copied_assistant_id:
                    # Slice ``[:fork_idx]`` excluded the assistant message
                    # this tool row depends on. Detect it pre-batch with a
                    # named error per the offensive-programming policy.
                    raise RuntimeError(f"fork slice excludes parent assistant of tool message id={msg.id}")
                copied_parent_assistant_id = source_to_copied_assistant_id[parent_key]

            msg_records_data.append(
                {
                    "id": copied_msg_id,
                    "session_id": new_session_id_str,
                    "role": msg.role,
                    "content": msg.content,
                    # raw_content preserves audit provenance for intercepted assistant turns
                    "raw_content": msg.raw_content,
                    "tool_calls": deep_thaw(msg.tool_calls) if msg.tool_calls else None,
                    "tool_call_id": msg.tool_call_id,
                    "parent_assistant_id": copied_parent_assistant_id,
                    # Preserve the source row's stored writer; deriving from
                    # role would fabricate provenance for any source row whose
                    # writer differs from the role-keyed default (admin tool,
                    # future re-classifications, etc.).
                    "writer_principal": msg.writer_principal,
                    "created_at": msg.created_at,
                    "composition_state_id": None,  # Don't reference source session states
                }
            )
        # System message — no raw_content (synthetic, not from the LLM).
        # writer_principal="session_fork" because this row is unambiguously
        # authored by the fork operation, not by any route handler.
        system_msg_id = str(uuid.uuid4())
        msg_records_data.append(
            {
                "id": system_msg_id,
                "session_id": new_session_id_str,
                "role": "system",
                "content": "Conversation forked from an earlier point.",
                "raw_content": None,
                "tool_calls": None,
                "tool_call_id": None,
                "parent_assistant_id": None,
                "writer_principal": "session_fork",
                "created_at": now,
                "composition_state_id": None,
            }
        )
        # New edited user message — provenance points to COPIED state, not source.
        # created_at = now is correct here: ordering is enforced by sequence_no
        # (allocated under the new-session write lock inside _sync), not by
        # created_at. The microsecond offset that earlier guarded against
        # SQLite same-microsecond ordering ambiguity is no longer needed.
        # raw_content is None: this is a new user-authored message, not an LLM turn.
        new_user_msg_id = str(uuid.uuid4())
        msg_records_data.append(
            {
                "id": new_user_msg_id,
                "session_id": new_session_id_str,
                "role": "user",
                "content": new_message_content,
                "raw_content": None,
                "tool_calls": None,
                "tool_call_id": None,
                "parent_assistant_id": None,
                "writer_principal": "session_fork",
                "created_at": now,
                "composition_state_id": copied_state_id_str,
            }
        )

        def _sync() -> int | None:
            """Single atomic transaction for the entire fork."""
            with self._session_process_locked_begin(new_session_id_str) as conn:
                # 1. Create session
                conn.execute(
                    insert(sessions_table).values(
                        id=new_session_id_str,
                        user_id=user_id,
                        auth_provider_type=auth_provider_type,
                        title=title,
                        created_at=now,
                        updated_at=now,
                        forked_from_session_id=str(source_session_id),
                        forked_from_message_id=str(fork_message_id),
                    )
                )

                # 2. Copy composition state + reserve chat sequence range +
                # batch-insert chat rows. All under one ``_session_write_lock``
                # context so state-version and chat-sequence allocation share
                # the same SQLite/PostgreSQL serialization boundary. The new
                # session id was minted seconds ago and no other writer can
                # know it yet, so the lock is technically uncontended; we
                # acquire it because the helpers' precondition contract
                # requires it.
                state_version: int | None = None
                with self._session_write_lock(conn, new_session_id_str):
                    if source_state_record is not None and copied_state_id_str is not None:
                        self._insert_composition_state(
                            conn,
                            session_id=new_session_id_str,
                            # B1: no ``version=`` kwarg. The helper allocates
                            # ``COALESCE(MAX(version), 0) + 1`` under the held
                            # lock. For a freshly minted session this is always
                            # 1, but the contract is per-session monotonicity,
                            # not "always 1" — so the literal ``state_version =
                            # 1`` below is correct because of the freshness
                            # invariant, not because the helper is hard-coded.
                            #
                            # State + lineage are bundled into a single
                            # ``StatePayload`` rather than passed as two
                            # separate kwargs (see ``_insert_composition_state``
                            # docstring for rationale).
                            payload=StatePayload(
                                data=CompositionStateData(
                                    sources=source_state_record.sources,
                                    nodes=source_state_record.nodes,
                                    edges=source_state_record.edges,
                                    outputs=source_state_record.outputs,
                                    metadata_=source_state_record.metadata_,
                                    is_valid=source_state_record.is_valid,
                                    validation_errors=source_state_record.validation_errors,
                                    composer_meta=forked_composer_meta,
                                ),
                                derived_from_state_id=None,
                            ),
                            provenance="session_fork",
                            created_at=now,  # cross-table timestamp consistency
                            state_id=copied_state_id_str,
                        )
                        state_version = 1

                    # 3. Reserve the chat sequence range and assign sequence_no
                    # to every row in list order before the batch insert.
                    # ``msg_records_data`` already encodes the intended chat
                    # ordering — copied messages first (in source order), then
                    # the system fork notice, then the new user message.
                    if msg_records_data:
                        base_seq = self._reserve_sequence_range(conn, new_session_id_str, count=len(msg_records_data))
                        for sequence_offset, record in enumerate(msg_records_data):
                            record["sequence_no"] = base_seq + sequence_offset
                        conn.execute(insert(chat_messages_table), msg_records_data)

                return state_version

        state_version = await self._run_sync(_sync)

        # Build return records from the pre-computed data
        new_session = SessionRecord(
            id=new_session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=now,
            forked_from_session_id=source_session_id,
            forked_from_message_id=fork_message_id,
        )

        # §14.6 backstop: copied ``role="audit"`` rows live in the DB for
        # audit fidelity, but the fork response payload is the user-facing
        # ``new_messages`` list and must exclude them. Filter at the
        # response boundary; the underlying batch insert above still
        # persisted them.
        new_messages = [
            ChatMessageRecord(
                id=UUID(d["id"]),
                session_id=new_session_id,
                role=d["role"],
                content=d["content"],
                raw_content=d["raw_content"],
                tool_calls=d["tool_calls"],
                created_at=d["created_at"],
                sequence_no=d["sequence_no"],
                composition_state_id=UUID(d["composition_state_id"]) if d["composition_state_id"] else None,
                writer_principal=d["writer_principal"],
                tool_call_id=d["tool_call_id"],
                parent_assistant_id=UUID(d["parent_assistant_id"]) if d["parent_assistant_id"] else None,
            )
            for d in msg_records_data
            if d["role"] != "audit"
        ]

        copied_state: CompositionStateRecord | None = None
        if source_state_record is not None and copied_state_id is not None and state_version is not None:
            copied_state = CompositionStateRecord(
                id=copied_state_id,
                session_id=new_session_id,
                version=state_version,
                source=None,
                sources=source_state_record.sources,
                nodes=source_state_record.nodes,
                edges=source_state_record.edges,
                outputs=source_state_record.outputs,
                metadata_=source_state_record.metadata_,
                is_valid=source_state_record.is_valid,
                validation_errors=source_state_record.validation_errors,
                created_at=now,
                derived_from_state_id=None,
                composer_meta=forked_composer_meta,
            )

        return new_session, new_messages, copied_state

    async def update_message_composition_state(
        self,
        message_id: UUID,
        composition_state_id: UUID,
    ) -> None:
        """Re-point a message's composition_state_id to a different state.

        Enforces same-session ownership: the target state must belong to
        the same session as the message. Cross-session re-pointing is a
        caller bug and raises RuntimeError.
        """
        mid = str(message_id)
        csid = str(composition_state_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                message_session_id = conn.execute(select(chat_messages_table.c.session_id).where(chat_messages_table.c.id == mid)).scalar()
                if message_session_id is None:
                    raise ValueError(f"Message {message_id} not found")

                _assert_state_in_session(
                    conn,
                    state_id=csid,
                    expected_session_id=str(message_session_id),
                    caller="update_message_composition_state",
                )

                conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == mid).values(composition_state_id=csid))

        await self._run_sync(_sync)

    def _row_to_run_record(self, row: Any) -> RunRecord:
        """Convert a SQLAlchemy row to a RunRecord."""
        rows_succeeded, rows_failed = _normalize_pre_adr019_session_counters(
            status=row.status,
            rows_processed=row.rows_processed,
            rows_succeeded=row.rows_succeeded,
            rows_failed=row.rows_failed,
            rows_routed_success=row.rows_routed_success,
            rows_routed_failure=row.rows_routed_failure,
            rows_quarantined=row.rows_quarantined,
        )
        return RunRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            state_id=UUID(row.state_id),
            status=row.status,
            started_at=self._ensure_utc(row.started_at),
            finished_at=self._ensure_utc(row.finished_at) if row.finished_at is not None else None,
            rows_processed=row.rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed_success=row.rows_routed_success,
            rows_routed_failure=row.rows_routed_failure,
            rows_quarantined=row.rows_quarantined,
            error=row.error,
            landscape_run_id=row.landscape_run_id,
            pipeline_yaml=row.pipeline_yaml,
        )

    def _row_to_run_event_record(self, row: Any) -> RunEventRecord:
        """Convert a SQLAlchemy row to a RunEventRecord."""
        if row.event_type not in SESSION_RUN_EVENT_TYPE_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: run_events.event_type is {row.event_type!r}, expected one of {sorted(SESSION_RUN_EVENT_TYPE_VALUES)}"
            )
        if not isinstance(row.data, Mapping):
            raise AuditIntegrityError(f"Tier 1: run_events.data for event {row.id} is not a JSON object")
        return RunEventRecord(
            id=UUID(row.id),
            run_id=UUID(row.run_id),
            sequence=int(row.sequence),
            timestamp=self._ensure_utc(row.timestamp),
            event_type=cast(SessionRunEventType, row.event_type),
            data=cast(Mapping[str, Any], row.data),
        )
