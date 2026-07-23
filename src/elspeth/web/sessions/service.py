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
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from uuid import UUID

import structlog
from opentelemetry import metrics
from sqlalchemy import ColumnElement, Connection, Engine, delete, desc, exists, func, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from elspeth.contracts.auth import AuthProviderType
from elspeth.contracts.blobs import BlobForkPlanEntry, fork_blob_id
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
from elspeth.contracts.hashing import canonical_json, stable_hash
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
    source_name_from_component_id,
    validate_pipeline_decision_node_semantics,
)
from elspeth.web.sessions._persist_payload import AuditOutcome, RedactedToolRow, StatePayload
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.guided_audit import (
    bind_guided_failure_audit_rows,
    is_authentic_guided_synthetic_invocation,
    prepare_guided_audit_rows,
    validate_guided_audit_payload_references,
)
from elspeth.web.sessions.guided_payloads import verify_guided_json_payloads
from elspeth.web.sessions.guided_replay import (
    guided_response_projection_hash,
    project_composition_proposal,
    project_guided_response,
    response_json,
    validation_errors_for_composer_surface,
    with_guided_response_descriptor,
)
from elspeth.web.sessions.locking import (
    acquire_session_advisory_xact_lock,
    process_session_lock,
    sqlite_session_mutex,
    sqlite_transaction_session_lock,
)
from elspeth.web.sessions.models import (
    audit_access_log_table,
    blob_inline_resolutions_table,
    blobs_table,
    chat_messages_table,
    composer_completion_events_table,
    composition_proposals_table,
    composition_states_table,
    guided_operation_admission_blocks_table,
    guided_operation_events_table,
    guided_operations_table,
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
    GUIDED_FAILURE_AUDIT_LINEAGE_KEY,
    GUIDED_OPERATION_FAILURE_CODE_VALUES,
    GUIDED_OPERATION_KIND_VALUES,
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
    GuidedAuditEvidence,
    GuidedCompositionStateResult,
    GuidedFailureAuditCohort,
    GuidedFailureAuditLineage,
    GuidedForkSettlementCommand,
    GuidedFullPipelineProposalStageCommand,
    GuidedFullPipelineProposalStageSettlement,
    GuidedOperationActive,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationConflictError,
    GuidedOperationFailed,
    GuidedOperationFailureCode,
    GuidedOperationFailureCommand,
    GuidedOperationFence,
    GuidedOperationFenceLostError,
    GuidedOperationKind,
    GuidedOperationOutcome,
    GuidedOperationResult,
    GuidedOperationSettlementConflictError,
    GuidedOperationTakenOver,
    GuidedOriginatingUserMessageDraft,
    GuidedPendingProposalInvalidation,
    GuidedPipelineConfirmationAdmissionCommand,
    GuidedPipelineDispatchRecordCommand,
    GuidedPipelineProposalAcceptCommand,
    GuidedPipelineProposalBackEditCommand,
    GuidedPipelineProposalRejectCommand,
    GuidedPipelineProposalResult,
    GuidedPipelineProposalStageCommand,
    GuidedPipelineProposalStageSettlement,
    GuidedSessionResult,
    GuidedStartStateConverged,
    GuidedStartStateOutcome,
    GuidedStartStateSeeded,
    GuidedStateOperationCommand,
    GuidedStateOperationSettlement,
    IllegalRunTransitionError,
    PipelineDispatchRecovery,
    PipelineProposalPublicMetadata,
    PipelineProposalRejectionReason,
    PipelineProposalSettlementResult,
    PreparedGuidedAuditRow,
    PreparedGuidedJsonPayload,
    ProposalEventRecord,
    ProposalLifecycleStatus,
    RunAlreadyActiveError,
    RunEventRecord,
    RunRecord,
    SessionGuidedOperationInProgressError,
    SessionNotFoundError,
    SessionRecord,
    SessionRunEventType,
    SessionRunStatus,
    StagedForkSession,
    StaleComposeStateError,
    ToolCallIDMismatchError,
    TransitionAssistantDraft,
    TransitionResponseSettlement,
)
from elspeth.web.sessions.telemetry import _SessionsTelemetry
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE, _validate_accepted_value_content

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.web.catalog.protocol import CatalogService
    from elspeth.web.composer.guided.state_machine import DeferredStageIntent, GuidedProposalRef, GuidedSession
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
_TOOL_PROPOSAL_CREATED_SCHEMA = "tool_proposal_created.v1"
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
_TOOL_PROPOSAL_CREATED_FIELDS = frozenset({"schema", "tool_call_id", "tool_name", "status"})
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


def _verify_guided_deferred_message_authority(
    conn: Connection,
    *,
    session_id: str,
    guided: Any,
) -> None:
    """Re-resolve every deferred intent's private user row under settlement lock."""

    message_ids = tuple(intent.originating_message_id for intent in guided.deferred_intents)
    if not message_ids:
        return
    rows = conn.execute(
        select(chat_messages_table.c.id, chat_messages_table.c.role, chat_messages_table.c.content)
        .where(chat_messages_table.c.session_id == session_id)
        .where(chat_messages_table.c.id.in_(message_ids))
    ).fetchall()
    rows_by_id = {row.id: row for row in rows}
    if set(rows_by_id) != set(message_ids):
        raise AuditIntegrityError("guided deferred intent message is missing or cross-session")
    for intent in guided.deferred_intents:
        row = rows_by_id[intent.originating_message_id]
        if row.role != "user":
            raise AuditIntegrityError("guided deferred intent must originate from a user message")
        if stable_hash(row.content) != intent.message_content_hash:
            raise AuditIntegrityError("guided deferred intent message content hash mismatch")


def _verify_guided_correction_message_authority(
    conn: Connection,
    *,
    session_id: str,
    guided: Any,
) -> None:
    """Re-resolve every private wire-correction row under settlement lock."""

    message_ids = tuple(str(reference.message_id) for reference in guided.correction_messages)
    if not message_ids:
        return
    rows = conn.execute(
        select(chat_messages_table.c.id, chat_messages_table.c.role, chat_messages_table.c.content)
        .where(chat_messages_table.c.session_id == session_id)
        .where(chat_messages_table.c.id.in_(message_ids))
    ).fetchall()
    rows_by_id = {row.id: row for row in rows}
    if set(rows_by_id) != set(message_ids):
        raise AuditIntegrityError("guided correction message is missing or cross-session")
    for reference in guided.correction_messages:
        row = rows_by_id[str(reference.message_id)]
        if row.role != "user":
            raise AuditIntegrityError("guided correction must originate from a user message")
        if stable_hash(row.content) != reference.content_hash:
            raise AuditIntegrityError("guided correction message content hash mismatch")


def _verify_guided_root_message_authority(
    conn: Connection,
    *,
    service: Any,
    session_id: str,
    guided: Any,
) -> None:
    """Re-derive the live root message from its immutable start operation."""

    if guided.root_intent_message_id is None:
        return
    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
    from elspeth.web.sessions.schemas import StartGuidedRequest

    message_id = guided.root_intent_message_id
    message_row = conn.execute(
        select(chat_messages_table.c.role, chat_messages_table.c.content, chat_messages_table.c.writer_principal)
        .where(chat_messages_table.c.session_id == session_id)
        .where(chat_messages_table.c.id == message_id)
    ).one_or_none()
    if message_row is None or message_row.role != "user" or message_row.writer_principal != "route_user_message":
        raise AuditIntegrityError("guided root intent row failed session/role/writer custody")
    operations = conn.execute(
        select(guided_operations_table)
        .where(guided_operations_table.c.session_id == session_id)
        .where(guided_operations_table.c.kind == "guided_start")
        .where(guided_operations_table.c.status == "completed")
        .where(guided_operations_table.c.originating_message_id == message_id)
        .where(guided_operations_table.c.result_kind == "composition_state")
    ).fetchall()
    if len(operations) != 1:
        raise AuditIntegrityError("guided root intent has absent or ambiguous start-operation authority")
    operation = operations[0]
    request = StartGuidedRequest.model_validate(
        {"operation_id": operation.operation_id, "profile": "live", "intent": message_row.content},
        strict=True,
    )
    if (
        guided_operation_request_hash(
            session_id=UUID(session_id),
            kind="guided_start",
            request=request,
        )
        != operation.request_hash
    ):
        raise AuditIntegrityError("guided root intent content no longer matches its start request hash")
    state_row = conn.execute(
        select(composition_states_table)
        .where(composition_states_table.c.session_id == session_id)
        .where(composition_states_table.c.id == operation.result_state_id)
    ).one_or_none()
    if state_row is None:
        raise AuditIntegrityError("guided root intent start result state is missing")
    start_guided = state_from_record(service._row_to_state_record(state_row)).guided_session
    if start_guided is None or start_guided.root_intent_message_id != message_id:
        raise AuditIntegrityError("guided root intent differs from its live start checkpoint")


def _require_exact_guided_intent_cancellation_audit(
    command: GuidedStateOperationCommand,
    existing_intent: DeferredStageIntent,
) -> None:
    cancellation_events: list[object] = []
    for invocation in command.audit_evidence.invocations:
        if invocation.tool_name != "guided_intent_cancelled":
            continue
        if not is_authentic_guided_synthetic_invocation(invocation):
            raise AuditIntegrityError("guided intent cancellation audit event is not an authentic server synthetic invocation")
        try:
            arguments = json.loads(invocation.arguments_canonical)
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("guided intent cancellation audit payload is malformed") from exc
        if stable_hash(arguments) != invocation.arguments_hash:
            raise AuditIntegrityError("guided intent cancellation audit payload hash mismatch")
        cancellation_events.append(arguments)
    expected = {
        "intent_id": existing_intent.intent_id,
        "receiving_stage": existing_intent.receiving_stage,
        "target_stage": existing_intent.target_stage,
    }
    if cancellation_events != [expected]:
        raise AuditIntegrityError("guided intent cancellation requires one exact structural audit event")


def _verify_guided_deferred_intent_append(
    command: GuidedStateOperationCommand,
    *,
    prior_guided: GuidedSession,
    candidate_guided: GuidedSession,
) -> DeferredStageIntent:
    if (
        len(candidate_guided.deferred_intents) != len(prior_guided.deferred_intents) + 1
        or candidate_guided.deferred_intents[:-1] != prior_guided.deferred_intents
        or candidate_guided.deferred_intents[-1].intent_id != str(command.retained_deferred_intent_id)
    ):
        raise AuditIntegrityError("retained deferred intent must be one exact terminal append")
    retained = candidate_guided.deferred_intents[-1]
    originating = command.originating_message
    if originating is None:  # pragma: no cover - command type owns this guard
        raise AuditIntegrityError("retained deferred intent lost its originating message")
    if retained.originating_message_id != str(originating.message_id):
        raise AuditIntegrityError("retained deferred intent names the wrong originating message")
    if retained.message_content_hash != stable_hash(originating.content):
        raise AuditIntegrityError("retained deferred intent message content hash mismatch")
    return retained


def _expected_guided_deferred_intents_after_management(
    conn: Connection,
    *,
    session_id: str,
    command: GuidedStateOperationCommand,
    prior_guided: GuidedSession,
) -> tuple[DeferredStageIntent, ...]:
    from elspeth.web.composer.guided.deferred_intents import (
        DeferredIntentCancelAction,
        DeferredIntentEditAction,
        create_deferred_stage_intent,
    )

    action = command.deferred_intent_action
    if action is None:  # pragma: no cover - command shape owns this branch
        raise AuditIntegrityError("deferred intent action sideband is missing")
    matching = [(index, intent) for index, intent in enumerate(prior_guided.deferred_intents) if intent.intent_id == action.intent_id]
    if len(matching) != 1:
        raise AuditIntegrityError("deferred intent action does not name one exact pending intent")
    intent_index, existing = matching[0]
    from elspeth.web.composer.guided.intent_management import (
        deferred_intent_management_option,
        deferred_intent_management_user_authority_matches,
    )

    if action.selection_token != deferred_intent_management_option(existing).selection_token:
        raise AuditIntegrityError("deferred intent action selection token does not bind its exact pending intent")
    originating = command.originating_message
    if originating is None or not deferred_intent_management_user_authority_matches(
        action,
        deferred_intents=prior_guided.deferred_intents,
        originating_message_content=originating.content,
    ):
        raise AuditIntegrityError("deferred intent mutation lacks matching exact action-specific user authority")
    _verify_guided_deferred_message_authority(conn, session_id=session_id, guided=prior_guided)
    if type(action) is DeferredIntentCancelAction:
        _require_exact_guided_intent_cancellation_audit(command, existing)
        replacement: tuple[DeferredStageIntent, ...] = ()
    elif type(action) is DeferredIntentEditAction:
        originating = command.originating_message
        if originating is None:  # pragma: no cover - command guards this
            raise AuditIntegrityError("deferred intent edit lost its originating message")
        replacement = (
            create_deferred_stage_intent(
                action.replacement,
                receiving_stage=existing.receiving_stage,
                intent_id=existing.intent_id,
                originating_message_id=str(originating.message_id),
                originating_message_content=originating.content,
            ),
        )
    else:  # pragma: no cover - command owns the exact union
        raise AuditIntegrityError("deferred intent action type is unsupported")
    return (*prior_guided.deferred_intents[:intent_index], *replacement, *prior_guided.deferred_intents[intent_index + 1 :])


def _verify_guided_deferred_intent_mutation(
    conn: Connection,
    *,
    session_id: str,
    command: GuidedStateOperationCommand,
    prior_guided: GuidedSession,
    candidate_guided: GuidedSession,
) -> DeferredStageIntent | None:
    """Verify the exact append/cancel/edit sideband against both checkpoints."""

    from elspeth.web.composer.guided.deferred_intents import DeferredIntentCancelAction

    cancellation_invocations = tuple(
        invocation for invocation in command.audit_evidence.invocations if invocation.tool_name == "guided_intent_cancelled"
    )
    is_cancel = type(command.deferred_intent_action) is DeferredIntentCancelAction
    if is_cancel != bool(cancellation_invocations):
        raise AuditIntegrityError("guided intent cancellation audit must exist if and only if the typed action is cancel")

    if command.retained_deferred_intent_id is None and command.deferred_intent_action is None:
        if candidate_guided.deferred_intents != prior_guided.deferred_intents:
            raise AuditIntegrityError("deferred intent mutation requires an explicit typed sideband")
        return None
    if command.retained_deferred_intent_id is not None:
        return _verify_guided_deferred_intent_append(command, prior_guided=prior_guided, candidate_guided=candidate_guided)
    expected = _expected_guided_deferred_intents_after_management(
        conn,
        session_id=session_id,
        command=command,
        prior_guided=prior_guided,
    )
    if candidate_guided.deferred_intents != expected:
        raise AuditIntegrityError("deferred intent action candidate differs from its exact typed mutation")
    return None


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


def _strip_guided_profile_in_meta(
    composer_meta: Mapping[str, Any] | None,
    source_to_child_message_id: Mapping[str, str],
    source_messages_by_id: Mapping[str, ChatMessageRecord],
) -> dict[str, Any] | None:
    """Prepare one schema-10 guided checkpoint for a child session.

    A fork preserves reviewed facts and deferred intent, but proposal authority
    and parent-session chat identities cannot cross the session boundary. Parse
    through the strict schema-10 decoder, rebuild the typed checkpoint, and fail
    before the fork transaction if any message reference is outside the copied
    slice.
    """
    from elspeth.core.canonical import stable_hash as _message_content_hash
    from elspeth.web.composer.guided.errors import InvariantError
    from elspeth.web.composer.guided.profile import EMPTY_PROFILE
    from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
    from elspeth.web.composer.guided.state_machine import GuidedSession

    if composer_meta is None:
        return None
    thawed: dict[str, Any] = dict(deep_thaw(composer_meta))
    guided_raw = thawed.get("guided_session")
    if guided_raw is None:
        return thawed
    if type(guided_raw) is not dict:
        raise AuditIntegrityError("fork guided metadata is not an exact schema-10 object")
    try:
        guided = GuidedSession.from_dict(guided_raw)
    except (InvariantError, KeyError, TypeError, ValueError) as exc:
        raise AuditIntegrityError("fork guided schema-10 authority is malformed") from exc

    unanswered_indices = [index for index, record in enumerate(guided.history) if record.response_hash is None]
    trailing = guided.history[-1] if guided.history else None
    has_exact_active_occurrence = bool(
        unanswered_indices == [len(guided.history) - 1]
        and trailing is not None
        and (
            (
                guided.step is GuidedStep.STEP_3_TRANSFORMS
                and trailing.step is GuidedStep.STEP_3_TRANSFORMS
                and trailing.turn_type is TurnType.PROPOSE_PIPELINE
            )
            or (
                guided.step is GuidedStep.STEP_4_WIRE
                and trailing.step is GuidedStep.STEP_4_WIRE
                and trailing.turn_type is TurnType.CONFIRM_WIRING
            )
        )
    )
    has_orphan_authority_occurrence = any(
        guided.history[index].turn_type in {TurnType.PROPOSE_PIPELINE, TurnType.CONFIRM_WIRING} for index in unanswered_indices
    )
    if (guided.active_proposal is not None and not has_exact_active_occurrence) or (
        guided.active_proposal is None and has_orphan_authority_occurrence
    ):
        raise AuditIntegrityError("fork guided proposal reference/history coupling is malformed")

    if set(source_to_child_message_id) != set(source_messages_by_id):
        raise AuditIntegrityError("fork guided message maps have different source keysets")

    def _child_user_message(source_message_id: str, field_name: str) -> tuple[str, ChatMessageRecord]:
        child_message_id = source_to_child_message_id.get(source_message_id)
        source_message = source_messages_by_id.get(source_message_id)
        if child_message_id is None or source_message is None:
            raise AuditIntegrityError(f"fork guided {field_name} references a message outside copied slice")
        if source_message.role != "user":
            raise AuditIntegrityError("fork guided planner lineage must identify user messages")
        return child_message_id, source_message

    remapped_root = (
        _child_user_message(guided.root_intent_message_id, "root_intent_message_id")[0]
        if guided.root_intent_message_id is not None
        else None
    )
    remapped_deferred_list = []
    for intent in guided.deferred_intents:
        child_message_id, source_message = _child_user_message(
            intent.originating_message_id,
            "deferred_intents.originating_message_id",
        )
        if _message_content_hash(source_message.content) != intent.message_content_hash:
            raise AuditIntegrityError("fork guided deferred intent message content hash mismatch")
        remapped_deferred_list.append(
            replace(
                intent,
                originating_message_id=child_message_id,
            )
        )
    remapped_deferred = tuple(remapped_deferred_list)
    remapped_corrections = []
    for reference in guided.correction_messages:
        child_message_id, source_message = _child_user_message(
            str(reference.message_id),
            "correction_messages.message_id",
        )
        if _message_content_hash(source_message.content) != reference.content_hash:
            raise AuditIntegrityError("fork guided correction message content hash mismatch")
        remapped_corrections.append(replace(reference, message_id=UUID(child_message_id)))
    rewinds_to_topology = guided.active_proposal is not None or guided.step in {GuidedStep.STEP_3_TRANSFORMS, GuidedStep.STEP_4_WIRE}
    reconciled_history = guided.history
    if rewinds_to_topology:
        if len(unanswered_indices) > 1 or (unanswered_indices and unanswered_indices[0] != len(guided.history) - 1):
            raise AuditIntegrityError("fork guided topology rewind has malformed unanswered history")
        if unanswered_indices:
            reconciled_history = guided.history[:-1]
    # A proposal cannot cross the session boundary. Rewind to the reviewed
    # output boundary, whose existing ``finish`` action deterministically
    # re-enters the shared planner and stages child-local proposal authority.
    # Leaving the child at Step 3 with no proposal would produce no GET turn
    # and no legal POST, permanently stranding the fork.
    forked_step = GuidedStep.STEP_2_SINK if rewinds_to_topology else guided.step
    forked_guided = replace(
        guided,
        step=forked_step,
        history=reconciled_history,
        profile=EMPTY_PROFILE,
        terminal=None if rewinds_to_topology else guided.terminal,
        transition_consumed=False if rewinds_to_topology else guided.transition_consumed,
        deferred_intents=remapped_deferred,
        correction_messages=tuple(remapped_corrections),
        active_proposal=None,
        active_edit_target=None,
        root_intent_message_id=remapped_root,
    )
    thawed["guided_session"] = forked_guided.to_dict()
    return thawed


_FORK_BLOB_PLAN_SCHEMA = "session-fork-blob-plan.v1"


def _fork_blob_plan_content(
    *,
    source_session_id: UUID,
    child_session_id: UUID,
    operation_id: str,
    entries: tuple[BlobForkPlanEntry, ...],
) -> str:
    return canonical_json(
        {
            "schema": _FORK_BLOB_PLAN_SCHEMA,
            "source_session_id": str(source_session_id),
            "child_session_id": str(child_session_id),
            "operation_id": operation_id,
            "source_blobs": [
                {
                    "source_blob_id": str(entry.source_blob_id),
                    "target_blob_id": str(entry.target_blob_id),
                    "content_hash": entry.content_hash,
                    "size_bytes": entry.size_bytes,
                }
                for entry in entries
            ],
        }
    )


def _fork_blob_plan_from_content(
    content: str,
    *,
    expected_source_session_id: UUID,
    expected_child_session_id: UUID,
    expected_operation_id: str,
) -> tuple[BlobForkPlanEntry, ...]:
    try:
        raw = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AuditIntegrityError("staged fork blob plan is not valid JSON") from exc
    if type(raw) is not dict or set(raw) != {
        "schema",
        "source_session_id",
        "child_session_id",
        "operation_id",
        "source_blobs",
    }:
        raise AuditIntegrityError("staged fork blob plan has malformed keys")
    if (
        raw["schema"] != _FORK_BLOB_PLAN_SCHEMA
        or raw["source_session_id"] != str(expected_source_session_id)
        or raw["child_session_id"] != str(expected_child_session_id)
        or raw["operation_id"] != expected_operation_id
    ):
        raise AuditIntegrityError("staged fork blob plan has malformed custody binding")
    source_blobs = raw["source_blobs"]
    if type(source_blobs) is not list:
        raise AuditIntegrityError("staged fork blob plan source_blobs must be a list")
    entries: list[BlobForkPlanEntry] = []
    for item in source_blobs:
        if type(item) is not dict or set(item) != {"source_blob_id", "target_blob_id", "content_hash", "size_bytes"}:
            raise AuditIntegrityError("staged fork blob plan entry has malformed keys")
        try:
            raw_source_blob_id = item["source_blob_id"]
            raw_target_blob_id = item["target_blob_id"]
            if type(raw_source_blob_id) is not str or type(raw_target_blob_id) is not str:
                raise TypeError("fork blob ids must be exact strings")
            source_blob_id = UUID(raw_source_blob_id)
            target_blob_id = UUID(raw_target_blob_id)
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("staged fork blob plan entry has malformed blob id") from exc
        if str(source_blob_id) != raw_source_blob_id or str(target_blob_id) != raw_target_blob_id:
            raise AuditIntegrityError("staged fork blob plan entry has non-canonical blob id")
        if target_blob_id != fork_blob_id(
            target_session_id=expected_child_session_id,
            source_blob_id=source_blob_id,
        ):
            raise AuditIntegrityError("staged fork blob plan entry has a non-deterministic target blob id")
        try:
            entry = BlobForkPlanEntry(
                source_blob_id=source_blob_id,
                target_blob_id=target_blob_id,
                content_hash=item["content_hash"],
                size_bytes=item["size_bytes"],
            )
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("staged fork blob plan entry is malformed") from exc
        entries.append(entry)
    result = tuple(entries)
    if tuple(sorted(result, key=lambda entry: str(entry.source_blob_id))) != result:
        raise AuditIntegrityError("staged fork blob plan entries are not in canonical id order")
    if len({entry.source_blob_id for entry in result}) != len(result):
        raise AuditIntegrityError("staged fork blob plan repeats a source blob id")
    return result


def _settlement_fork_blob_plan(
    conn: Connection,
    *,
    parent_session_id: UUID,
    child_session_id: UUID,
    operation_id: str,
) -> tuple[BlobForkPlanEntry, ...]:
    candidates: list[tuple[BlobForkPlanEntry, ...]] = []
    for row in conn.execute(
        select(chat_messages_table.c.content).where(
            chat_messages_table.c.session_id == str(child_session_id),
            chat_messages_table.c.role == "audit",
            chat_messages_table.c.writer_principal == "session_fork",
        )
    ).all():
        try:
            decoded = json.loads(row.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            type(decoded) is dict
            and decoded.get("schema") == _FORK_BLOB_PLAN_SCHEMA
            and decoded.get("child_session_id") == str(child_session_id)
            and decoded.get("operation_id") == operation_id
        ):
            candidates.append(
                _fork_blob_plan_from_content(
                    row.content,
                    expected_source_session_id=parent_session_id,
                    expected_child_session_id=child_session_id,
                    expected_operation_id=operation_id,
                )
            )
    if len(candidates) != 1:
        raise AuditIntegrityError("Guided fork settlement requires exactly one retained frozen blob plan")
    return candidates[0]


def _value_references_parent_blob(value: Any, forbidden: frozenset[str]) -> bool:
    if type(value) is str:
        return value in forbidden or (value.startswith("blob:") and value.removeprefix("blob:") in forbidden)
    if isinstance(value, Mapping):
        return any(_value_references_parent_blob(item, forbidden) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_value_references_parent_blob(item, forbidden) for item in value)
    return False


def _verify_fork_settlement_blob_custody(
    conn: Connection,
    *,
    parent_session_id: UUID,
    child_session_id: UUID,
    operation_id: str,
    state_payload: Mapping[str, Any],
) -> None:
    plan = _settlement_fork_blob_plan(
        conn,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        operation_id=operation_id,
    )
    actual_rows = conn.execute(
        select(
            blobs_table.c.id,
            blobs_table.c.status,
            blobs_table.c.content_hash,
            blobs_table.c.size_bytes,
        ).where(blobs_table.c.session_id == str(child_session_id))
    ).all()
    actual = {row.id: row for row in actual_rows}
    expected_ids = {str(entry.target_blob_id) for entry in plan}
    if set(actual) != expected_ids:
        raise AuditIntegrityError("Guided fork settlement child blob ids do not exactly match the frozen plan")
    for entry in plan:
        row = actual[str(entry.target_blob_id)]
        if row.status != "ready" or row.content_hash != entry.content_hash or row.size_bytes != entry.size_bytes:
            raise AuditIntegrityError("Guided fork settlement child blob status, hash, or size does not match the frozen plan")
    planned_parent_rows = (
        conn.execute(
            select(blobs_table.c.id, blobs_table.c.storage_path).where(
                blobs_table.c.session_id == str(parent_session_id),
                blobs_table.c.id.in_([str(entry.source_blob_id) for entry in plan]),
            )
        ).all()
        if plan
        else []
    )
    if len(planned_parent_rows) != len(plan):
        raise AuditIntegrityError("Guided fork settlement parent blob custody no longer matches the frozen plan")
    parent_rows = conn.execute(
        select(blobs_table.c.id, blobs_table.c.storage_path).where(blobs_table.c.session_id == str(parent_session_id))
    ).all()
    forbidden = frozenset(item for row in parent_rows for item in (row.id, row.storage_path))
    if _value_references_parent_blob(state_payload, forbidden):
        raise AuditIntegrityError("Guided fork settlement state retains parent blob custody")


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


def _require_pending_guided_checkpoint_proposal_authority(
    conn: Connection,
    *,
    service: SessionServiceImpl,
    session_id: str,
    checkpoint: CompositionStateRecord,
    guided: GuidedSession,
    role: str,
) -> AuthoritativePipelineProposal | None:
    """Validate one checkpoint's complete pending guided proposal authority."""
    from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
    from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

    unanswered_indices = [index for index, turn in enumerate(guided.history) if turn.response_hash is None]
    trailing = guided.history[-1] if guided.history else None
    has_exact_active_occurrence = bool(
        unanswered_indices == [len(guided.history) - 1]
        and trailing is not None
        and (
            (
                guided.step is GuidedStep.STEP_3_TRANSFORMS
                and trailing.step is GuidedStep.STEP_3_TRANSFORMS
                and trailing.turn_type is TurnType.PROPOSE_PIPELINE
            )
            or (
                guided.step is GuidedStep.STEP_4_WIRE
                and trailing.step is GuidedStep.STEP_4_WIRE
                and trailing.turn_type is TurnType.CONFIRM_WIRING
            )
        )
    )
    has_orphan_authority_occurrence = any(
        guided.history[index].turn_type in {TurnType.PROPOSE_PIPELINE, TurnType.CONFIRM_WIRING} for index in unanswered_indices
    )
    if (guided.active_proposal is not None and not has_exact_active_occurrence) or (
        guided.active_proposal is None and has_orphan_authority_occurrence
    ):
        raise AuditIntegrityError(f"{role} guided proposal reference/history coupling is malformed")
    reference = guided.active_proposal
    if reference is None:
        return None

    proposal_row = conn.execute(
        select(composition_proposals_table)
        .where(composition_proposals_table.c.session_id == session_id)
        .where(composition_proposals_table.c.id == str(reference.proposal_id))
    ).one_or_none()
    if proposal_row is None:
        raise AuditIntegrityError(f"{role} guided proposal authority is missing or cross-session")
    creation_rows = conn.execute(
        select(proposal_events_table)
        .where(proposal_events_table.c.session_id == session_id)
        .where(proposal_events_table.c.proposal_id == str(reference.proposal_id))
        .where(proposal_events_table.c.event_type == "proposal.created")
    ).fetchall()
    if len(creation_rows) != 1:
        raise AuditIntegrityError(f"{role} guided proposal must have exactly one creation event")
    authority = _restore_authoritative_pipeline_proposal(
        conn=conn,
        row=_proposal_record_from_row(proposal_row),
        creation_event=_proposal_event_record_from_row(creation_rows[0]),
        reviewed_facts=guided_private_reviewed_facts(guided),
    )
    _verify_pipeline_lifecycle_authority(conn, service=service, authority=authority)
    proposal = authority.proposal
    if (
        reference.proposal_id != authority.row.id
        or reference.draft_hash != proposal.draft_hash
        or reference.base != proposal.base
        or reference.reviewed_anchor_hash != proposal.reviewed_anchor_hash
        or reference.covered_deferred_intent_ids != proposal.covered_deferred_intent_ids
        or reference.creation_event_schema != _PIPELINE_CREATED_SCHEMA
        or reference.supersedes_proposal_id != authority.supersedes_proposal_id
        or reference.supersedes_draft_hash != proposal.supersedes_draft_hash
    ):
        raise AuditIntegrityError(f"{role} guided proposal reference differs from canonical authority")
    if proposal.surface not in {PlannerSurface.GUIDED_STAGED, PlannerSurface.TUTORIAL_PROFILE}:
        raise AuditIntegrityError(f"{role} guided proposal authority has an invalid surface")
    if type(proposal.base) is not PresentBase:
        raise AuditIntegrityError(f"{role} guided proposal checkpoint base is malformed")
    base_row = conn.execute(
        select(composition_states_table)
        .where(composition_states_table.c.session_id == session_id)
        .where(composition_states_table.c.id == str(proposal.base.state_id))
    ).one_or_none()
    if base_row is None:
        raise AuditIntegrityError(f"{role} guided proposal base is missing or cross-session")
    base_record = service._row_to_state_record(base_row)
    if composition_content_hash(state_from_record(base_record)) != proposal.base.composition_content_hash:
        raise AuditIntegrityError(f"{role} guided proposal base content binding is malformed")
    if proposal.base.composition_content_hash != composition_content_hash(state_from_record(checkpoint)):
        raise AuditIntegrityError(f"{role} guided proposal checkpoint base content binding is malformed")
    if authority.row.status != "pending":
        raise AuditIntegrityError(f"{role} guided checkpoint references a terminal pipeline proposal")
    return authority


@dataclass(frozen=True, slots=True)
class _GuidedPendingProposalInvalidationContext:
    """Frozen checkpoint and database authority for one proposal invalidation."""

    service: SessionServiceImpl
    session_id: str
    current_record: CompositionStateRecord | None
    prior_guided: GuidedSession
    candidate_guided: GuidedSession
    expected_current_content_hash: str | None


def _require_guided_pending_proposal_transition(
    context: _GuidedPendingProposalInvalidationContext,
    invalidation: GuidedPendingProposalInvalidation | None,
) -> GuidedProposalRef | None:
    """Verify that the candidate clears exactly the checkpoint's active ref."""

    if context.prior_guided.active_proposal == context.candidate_guided.active_proposal:
        if invalidation is not None:
            raise AuditIntegrityError("guided proposal invalidation sideband did not clear an active proposal")
        return None
    if invalidation is None:
        raise AuditIntegrityError("clearing guided active proposal requires an exact invalidation sideband")
    active = context.prior_guided.active_proposal
    if active is None or context.candidate_guided.active_proposal is not None:
        raise AuditIntegrityError("guided proposal invalidation must clear one exact active proposal")
    if active.proposal_id != invalidation.proposal_id or active.draft_hash != invalidation.draft_hash:
        raise AuditIntegrityError("guided proposal invalidation sideband differs from checkpoint authority")
    if context.current_record is None:
        raise AuditIntegrityError("guided proposal invalidation requires a current checkpoint")
    return active


def _verify_guided_pending_proposal_invalidation(
    conn: Connection,
    *,
    context: _GuidedPendingProposalInvalidationContext,
    invalidation: GuidedPendingProposalInvalidation | None,
) -> AuthoritativePipelineProposal | None:
    """Verify one exact active-reference clear and restore pending row authority."""

    active = _require_guided_pending_proposal_transition(context, invalidation)
    if active is None:
        return None
    if invalidation is None or context.current_record is None:  # pragma: no cover - transition verifier owns this
        raise AuditIntegrityError("guided proposal invalidation lost its verified authority")
    proposal_row = conn.execute(
        select(composition_proposals_table)
        .where(composition_proposals_table.c.session_id == context.session_id)
        .where(composition_proposals_table.c.id == str(invalidation.proposal_id))
    ).one_or_none()
    if proposal_row is None:
        raise AuditIntegrityError("guided proposal invalidation authority is missing or cross-session")
    creation_rows = conn.execute(
        select(proposal_events_table)
        .where(proposal_events_table.c.session_id == context.session_id)
        .where(proposal_events_table.c.proposal_id == str(invalidation.proposal_id))
        .where(proposal_events_table.c.event_type == "proposal.created")
    ).fetchall()
    if len(creation_rows) != 1:
        raise AuditIntegrityError("guided proposal invalidation requires one creation event")
    authority = _restore_authoritative_pipeline_proposal(
        conn=conn,
        row=_proposal_record_from_row(proposal_row),
        creation_event=_proposal_event_record_from_row(creation_rows[0]),
        reviewed_facts=deep_thaw(invalidation.reviewed_facts),
    )
    _verify_pipeline_lifecycle_authority(conn, service=context.service, authority=authority)
    if authority.row.status != "pending":
        raise AuditIntegrityError("guided proposal invalidation requires a pending proposal")
    proposal = authority.proposal
    if (
        proposal.draft_hash != active.draft_hash
        or proposal.base != active.base
        or proposal.reviewed_anchor_hash != active.reviewed_anchor_hash
        or proposal.covered_deferred_intent_ids != active.covered_deferred_intent_ids
        or authority.supersedes_proposal_id != active.supersedes_proposal_id
        or proposal.supersedes_draft_hash != active.supersedes_draft_hash
    ):
        raise AuditIntegrityError("guided proposal invalidation restored authority differs from checkpoint")
    if type(proposal.base) is not PresentBase:
        raise AuditIntegrityError("guided proposal invalidation base is not a persisted checkpoint")
    base_row = conn.execute(
        select(composition_states_table)
        .where(composition_states_table.c.session_id == context.session_id)
        .where(composition_states_table.c.id == str(proposal.base.state_id))
    ).one_or_none()
    if base_row is None:
        raise AuditIntegrityError("guided proposal invalidation base is missing or cross-session")
    base_record = context.service._row_to_state_record(base_row)
    if composition_content_hash(state_from_record(base_record)) != proposal.base.composition_content_hash:
        raise AuditIntegrityError("guided proposal invalidation base content binding changed")
    if proposal.base.composition_content_hash != context.expected_current_content_hash:
        raise AuditIntegrityError("guided proposal invalidation base content hash changed")
    return authority


def _reject_guided_pending_proposal(
    conn: Connection,
    *,
    authority: AuthoritativePipelineProposal,
    actor: str,
    created_at: datetime,
) -> None:
    """Append one immutable supersession event and terminalize the pending row."""

    session_id = str(authority.row.session_id)
    proposal_id = str(authority.row.id)
    _require_no_active_guided_confirmation_admission(
        conn,
        session_id=session_id,
        proposal_id=proposal_id,
        now=created_at,
    )
    event_id = str(uuid.uuid4())
    conn.execute(
        insert(proposal_events_table).values(
            id=event_id,
            session_id=session_id,
            proposal_id=proposal_id,
            event_type="proposal.rejected",
            actor=actor,
            payload=_pipeline_rejected_payload(authority=authority, reason="superseded", dispatch=None),
            created_at=created_at,
        )
    )
    updated = conn.execute(
        update(composition_proposals_table)
        .where(composition_proposals_table.c.session_id == session_id)
        .where(composition_proposals_table.c.id == proposal_id)
        .where(composition_proposals_table.c.status == "pending")
        .values(
            status="rejected",
            committed_state_id=None,
            audit_event_id=event_id,
            updated_at=created_at,
        )
    )
    if updated.rowcount != 1:
        raise AuditIntegrityError("guided proposal invalidation lost the pending proposal CAS")


def _require_no_active_guided_confirmation_admission(
    conn: Connection,
    *,
    session_id: str,
    proposal_id: str,
    now: datetime,
) -> None:
    """Fail a competing mutation while a live confirmation owns dispatch.

    Expired leases cannot dispatch or persist under their old fence, so their
    proposal locator is released before the active-owner check. This prevents
    an abandoned admission from fencing the proposal forever.
    """

    conn.execute(
        update(guided_operations_table)
        .where(
            guided_operations_table.c.session_id == session_id,
            guided_operations_table.c.proposal_id == proposal_id,
            guided_operations_table.c.status == "in_progress",
            guided_operations_table.c.lease_expires_at <= now,
        )
        .values(proposal_id=None, updated_at=now)
    )
    active = conn.execute(
        select(guided_operations_table.c.operation_id)
        .where(
            guided_operations_table.c.session_id == session_id,
            guided_operations_table.c.proposal_id == proposal_id,
            guided_operations_table.c.status == "in_progress",
            guided_operations_table.c.lease_expires_at > now,
        )
        .limit(1)
    ).one_or_none()
    if active is not None:
        raise GuidedOperationSettlementConflictError()


def _classify_authoritative_composition_proposal(
    *,
    conn: Connection,
    row: CompositionProposalRecord,
    creation_event: ProposalEventRecord,
    reviewed_facts: Mapping[str, Any] | None,
) -> AuthoritativeCompositionProposal:
    """Accept only one of the two closed current proposal event schemas."""
    payload = creation_event.payload
    if not isinstance(payload, Mapping):
        raise AuditIntegrityError("proposal creation event payload must be a mapping")
    if set(payload) == _TOOL_PROPOSAL_CREATED_FIELDS:
        expected = {
            "schema": _TOOL_PROPOSAL_CREATED_SCHEMA,
            "tool_call_id": row.tool_call_id,
            "tool_name": row.tool_name,
            "status": "pending",
        }
        if payload != expected or creation_event.session_id != row.session_id or creation_event.proposal_id != row.id:
            raise AuditIntegrityError("tool proposal creation event binding is malformed")
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
    source_name = source_name_from_component_id(affected_node_id)
    if source_name is None:
        raise InterpretationNodeMissingError(
            "resolve_interpretation_event: invented_source must target a source component "
            f"({SOURCE_COMPONENT_ID!r} or {SOURCE_COMPONENT_ID!r}:<name>)"
        )
    sources_map = _require_mapping(
        state_record.sources,
        message="resolve_interpretation_event: invented_source requires a persisted sources mapping",
    )
    source = _require_mapping(
        sources_map[source_name] if source_name in sources_map else None,
        message=f"resolve_interpretation_event: invented_source requires persisted source {source_name!r}",
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
    # Splice only the reviewed source back into the sources map. Every sibling
    # source carries its own independent review authority and is left untouched.
    patched_sources = dict(sources_map)
    patched_sources[source_name] = patched_source
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

    @contextlib.contextmanager
    def _session_pair_locked_begin(self, first_session_id: str, second_session_id: str) -> Iterator[Connection]:
        """Acquire two session locks in global UUID order before DB work."""
        if first_session_id == second_session_id:
            raise AuditIntegrityError("paired session transaction requires two distinct session ids")
        ordered = tuple(sorted((first_session_id, second_session_id)))
        with contextlib.ExitStack() as process_stack:
            for session_id in ordered:
                process_stack.enter_context(process_session_lock(self._engine, session_id))
            with self._engine.begin() as conn, contextlib.ExitStack() as transaction_stack:
                for session_id in ordered:
                    transaction_stack.enter_context(self._session_write_lock(conn, session_id))
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

    @staticmethod
    def _guided_database_now(conn: Connection) -> datetime:
        """Read one fresh wall-clock timestamp from the operation database."""
        dialect = conn.dialect.name
        if dialect == "sqlite":
            value = conn.exec_driver_sql("SELECT CURRENT_TIMESTAMP").scalar_one()
        elif dialect == "postgresql":
            # CURRENT_TIMESTAMP is transaction-start time on PostgreSQL.
            value = conn.exec_driver_sql("SELECT clock_timestamp()").scalar_one()
        else:
            raise NotImplementedError(f"guided operation database time not implemented for dialect {dialect}")
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise AuditIntegrityError("Guided operation database clock returned a non-datetime value")
        return SessionServiceImpl._ensure_utc(value)

    @staticmethod
    def _validate_guided_hash(value: str, *, label: str) -> None:
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")

    @staticmethod
    def _validate_guided_actor(actor: str) -> None:
        if not isinstance(actor, str) or not actor or len(actor) > 128:
            raise ValueError("guided operation actor must be a non-empty string of at most 128 characters")

    @staticmethod
    def _validate_guided_lease_seconds(lease_seconds: int) -> None:
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 1 <= lease_seconds <= 3600:
            raise ValueError("guided operation lease_seconds must be an integer from 1 through 3600")

    @staticmethod
    def _validate_guided_identity(*, operation_id: str, kind: GuidedOperationKind, request_hash: str) -> None:
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128:
            raise ValueError("guided operation id must be a non-empty string of at most 128 characters")
        if kind not in GUIDED_OPERATION_KIND_VALUES:
            raise ValueError("unsupported guided operation kind")
        SessionServiceImpl._validate_guided_hash(request_hash, label="guided operation request_hash")

    @staticmethod
    def _insert_guided_operation_event(
        conn: Connection,
        *,
        session_id: str,
        operation_id: str,
        event_kind: Literal["claimed", "renewed", "taken_over", "completed", "failed"],
        actor: str,
        attempt: int,
        prior_attempt: int | None,
        lease_expires_at: datetime | None,
        request_hash: str,
        failure_audit_cohort: GuidedFailureAuditCohort | None,
        occurred_at: datetime,
    ) -> None:
        if event_kind == "failed" and type(failure_audit_cohort) is not GuidedFailureAuditCohort:
            raise AuditIntegrityError("failed guided operation event must carry exactly one failure audit cohort commitment")
        if event_kind != "failed" and failure_audit_cohort is not None:
            raise AuditIntegrityError("non-failed guided operation event must not carry a failure audit cohort commitment")
        next_sequence = conn.execute(
            select(func.coalesce(func.max(guided_operation_events_table.c.sequence), 0) + 1).where(
                guided_operation_events_table.c.session_id == session_id,
                guided_operation_events_table.c.operation_id == operation_id,
            )
        ).scalar_one()
        conn.execute(
            insert(guided_operation_events_table).values(
                session_id=session_id,
                operation_id=operation_id,
                sequence=next_sequence,
                event_kind=event_kind,
                actor=actor,
                attempt=attempt,
                prior_attempt=prior_attempt,
                lease_expires_at=lease_expires_at,
                request_hash=request_hash,
                failure_audit_cohort=(failure_audit_cohort.envelope() if failure_audit_cohort is not None else None),
                occurred_at=occurred_at,
            )
        )

    @staticmethod
    def _validate_guided_operation_row(
        row: RowMapping,
        *,
        expected_session_id: str,
        expected_operation_id: str,
    ) -> None:
        """Validate the complete Tier-1 row before replay/conflict classification."""

        if row["session_id"] != expected_session_id or row["operation_id"] != expected_operation_id:
            raise AuditIntegrityError("Tier 1: guided operation persisted identity does not match its lookup key")
        operation_id = row["operation_id"]
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128:
            raise AuditIntegrityError("Tier 1: guided operation operation_id is invalid")
        kind = row["kind"]
        if kind not in GUIDED_OPERATION_KIND_VALUES:
            raise AuditIntegrityError("Tier 1: guided operation kind is invalid")
        status = row["status"]
        if status not in {"in_progress", "completed", "failed"}:
            raise AuditIntegrityError("Tier 1: guided operation status is invalid")
        request_hash = row["request_hash"]
        try:
            SessionServiceImpl._validate_guided_hash(request_hash, label="guided operation request_hash")
        except ValueError as exc:
            raise AuditIntegrityError("Tier 1: guided operation request_hash is invalid") from exc
        attempt = row["attempt"]
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise AuditIntegrityError("Tier 1: guided operation attempt is invalid")

        for field in ("originating_message_id", "proposal_id", "result_state_id", "result_session_id"):
            value = row[field]
            if value is None:
                continue
            try:
                parsed = UUID(value)
            except (AttributeError, TypeError, ValueError) as exc:
                raise AuditIntegrityError(f"Tier 1: guided operation {field} is not a UUID") from exc
            if str(parsed) != value:
                raise AuditIntegrityError(f"Tier 1: guided operation {field} is not a canonical UUID")

        created_at = row["created_at"]
        updated_at = row["updated_at"]
        if not isinstance(created_at, datetime) or not isinstance(updated_at, datetime):
            raise AuditIntegrityError("Tier 1: guided operation timestamps are invalid")
        created_at = SessionServiceImpl._ensure_utc(created_at)
        updated_at = SessionServiceImpl._ensure_utc(updated_at)
        if updated_at < created_at:
            raise AuditIntegrityError("Tier 1: guided operation updated_at predates created_at")
        settled_at = row["settled_at"]
        if settled_at is not None:
            if not isinstance(settled_at, datetime):
                raise AuditIntegrityError("Tier 1: guided operation settled_at is invalid")
            if SessionServiceImpl._ensure_utc(settled_at) < created_at:
                raise AuditIntegrityError("Tier 1: guided operation settled_at predates created_at")

        if status == "in_progress":
            SessionServiceImpl._guided_in_progress_expiry(row)
        else:
            SessionServiceImpl._validate_guided_terminal_bundle(row)

    @staticmethod
    def _validate_guided_operation_admission_block_row(
        row: RowMapping,
        *,
        expected_session_id: str,
        expected_operation_id: str,
    ) -> None:
        """Validate one complete Tier-1 negative admission authority."""

        if row["session_id"] != expected_session_id or row["operation_id"] != expected_operation_id:
            raise AuditIntegrityError("Tier 1: guided operation admission block identity does not match its lookup key")
        operation_id = row["operation_id"]
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128:
            raise AuditIntegrityError("Tier 1: guided operation admission block operation_id is invalid")
        if row["kind"] != "guided_start":
            raise AuditIntegrityError("Tier 1: guided operation admission block kind is invalid")
        if row["failure_code"] != "request_cancelled":
            raise AuditIntegrityError("Tier 1: guided operation admission block failure code is invalid")
        try:
            SessionServiceImpl._validate_guided_actor(row["actor"])
        except ValueError as exc:
            raise AuditIntegrityError("Tier 1: guided operation admission block actor is invalid") from exc
        if not isinstance(row["created_at"], datetime):
            raise AuditIntegrityError("Tier 1: guided operation admission block timestamp is invalid")
        SessionServiceImpl._ensure_utc(row["created_at"])

    @staticmethod
    def _read_guided_operation_authority(
        conn: Connection,
        *,
        session_id: str,
        operation_id: str,
    ) -> tuple[RowMapping | None, RowMapping | None]:
        """Read the mutually exclusive positive and negative authorities."""

        operation = (
            conn.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == operation_id,
                )
            )
            .mappings()
            .one_or_none()
        )
        block = (
            conn.execute(
                select(guided_operation_admission_blocks_table).where(
                    guided_operation_admission_blocks_table.c.session_id == session_id,
                    guided_operation_admission_blocks_table.c.operation_id == operation_id,
                )
            )
            .mappings()
            .one_or_none()
        )
        if operation is not None and block is not None:
            raise AuditIntegrityError("Tier 1: guided operation and admission block coexist")
        return operation, block

    @staticmethod
    def _guided_in_progress_expiry(row: RowMapping) -> datetime:
        lease_token = row["lease_token"]
        lease_expires_at = row["lease_expires_at"]
        if not isinstance(lease_token, str) or not 1 <= len(lease_token) <= 256 or not isinstance(lease_expires_at, datetime):
            raise AuditIntegrityError("Tier 1: in-progress guided operation has an invalid lease bundle")
        if any(row[field] is not None for field in ("settled_at", "result_kind", "response_hash", "failure_code")):
            raise AuditIntegrityError("Tier 1: in-progress guided operation retained terminal residue")
        kind = row["kind"]
        if row["result_session_id"] is not None and kind != "session_fork":
            raise AuditIntegrityError("Tier 1: in-progress guided operation has a mismatched session locator")
        if row["result_state_id"] is not None and kind == "session_fork":
            raise AuditIntegrityError("Tier 1: in-progress fork has a mismatched state locator")
        if row["proposal_id"] is not None and kind not in {"guided_respond", "guided_chat"}:
            raise AuditIntegrityError("Tier 1: in-progress guided operation has a mismatched proposal locator")
        return SessionServiceImpl._ensure_utc(lease_expires_at)

    @staticmethod
    def _validate_guided_terminal_bundle(row: RowMapping) -> None:
        status = row["status"]
        if status == "failed":
            failure_code = row["failure_code"]
            if failure_code not in GUIDED_OPERATION_FAILURE_CODE_VALUES:
                raise AuditIntegrityError("Tier 1: guided operation has an invalid terminal failure code")
            if any(
                row[field] is not None
                for field in (
                    "lease_token",
                    "lease_expires_at",
                    "result_kind",
                    "result_state_id",
                    "result_session_id",
                    "proposal_id",
                    "response_hash",
                )
            ):
                raise AuditIntegrityError("Tier 1: failed guided operation retained terminal failure residue")
            if not isinstance(row["settled_at"], datetime):
                raise AuditIntegrityError("Tier 1: failed guided operation is missing settled_at")
            return
        if status != "completed":
            raise AuditIntegrityError("Tier 1: guided operation terminal decoder received a non-terminal row")
        if row["lease_token"] is not None or row["lease_expires_at"] is not None or row["failure_code"] is not None:
            raise AuditIntegrityError("Tier 1: completed guided operation retained terminal residue")
        if not isinstance(row["settled_at"], datetime):
            raise AuditIntegrityError("Tier 1: completed guided operation is missing settled_at")
        response_hash = row["response_hash"]
        if not isinstance(response_hash, str):
            raise AuditIntegrityError("Tier 1: completed guided operation is missing response_hash")
        try:
            SessionServiceImpl._validate_guided_hash(response_hash, label="guided operation response_hash")
            result_kind = row["result_kind"]
            if result_kind == "composition_state":
                if row["kind"] in {"session_fork", "guided_plan"}:
                    raise AuditIntegrityError("Tier 1: guided operation kind does not match its result locator")
                if row["result_session_id"] is not None:
                    raise AuditIntegrityError("Tier 1: state result retained a session locator")
                if row["proposal_id"] is not None and row["kind"] not in {"guided_respond", "guided_chat"}:
                    raise AuditIntegrityError("Tier 1: state result retained an unsupported proposal locator")
                UUID(row["result_state_id"])
                if row["proposal_id"] is not None:
                    UUID(row["proposal_id"])
            elif result_kind == "pipeline_proposal":
                if (
                    row["kind"] != "guided_plan"
                    or row["result_state_id"] is None
                    or row["proposal_id"] is None
                    or row["result_session_id"] is not None
                ):
                    raise AuditIntegrityError("Tier 1: guided plan operation has a malformed proposal locator")
                UUID(row["result_state_id"])
                UUID(row["proposal_id"])
            elif result_kind == "session":
                if row["kind"] != "session_fork" or row["result_state_id"] is not None or row["proposal_id"] is not None:
                    raise AuditIntegrityError("Tier 1: guided operation kind does not match its result locator")
                UUID(row["result_session_id"])
            else:
                raise AuditIntegrityError("Tier 1: completed guided operation has an invalid result kind")
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("Tier 1: completed guided operation has a malformed replay locator") from exc

    @staticmethod
    def _guided_terminal_outcome(row: RowMapping) -> GuidedOperationCompleted | GuidedOperationFailed:
        SessionServiceImpl._validate_guided_terminal_bundle(row)
        if row["status"] == "failed":
            return GuidedOperationFailed(failure_code=cast("GuidedOperationFailureCode", row["failure_code"]))
        response_hash = cast("str", row["response_hash"])
        result_kind = row["result_kind"]
        if result_kind == "composition_state":
            result: GuidedOperationResult = GuidedCompositionStateResult(
                state_id=UUID(row["result_state_id"]),
                proposal_id=UUID(row["proposal_id"]) if row["proposal_id"] is not None else None,
            )
        elif result_kind == "pipeline_proposal":
            result = GuidedPipelineProposalResult(
                proposal_id=UUID(row["proposal_id"]),
                checkpoint_state_id=UUID(row["result_state_id"]),
            )
        elif result_kind == "session":
            result = GuidedSessionResult(session_id=UUID(row["result_session_id"]))
        else:
            raise AuditIntegrityError("Tier 1: completed guided operation has an invalid result kind")
        return GuidedOperationCompleted(result=result, response_hash=response_hash)

    @staticmethod
    def _guided_conflict(*, session_id: UUID, operation_id: str) -> GuidedOperationConflictError:
        return GuidedOperationConflictError(session_id=session_id, operation_id=operation_id)

    async def reserve_guided_operation(
        self,
        *,
        session_id: UUID,
        operation_id: str,
        kind: GuidedOperationKind,
        request_hash: str,
        actor: str,
        lease_seconds: int,
    ) -> GuidedOperationOutcome:
        """Claim, join, replay, or take over one normalized operation."""
        self._validate_guided_identity(operation_id=operation_id, kind=kind, request_hash=request_hash)
        self._validate_guided_actor(actor)
        self._validate_guided_lease_seconds(lease_seconds)
        sid = str(session_id)

        def _sync() -> GuidedOperationOutcome:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                now = self._guided_database_now(conn)
                row, block = self._read_guided_operation_authority(
                    conn,
                    session_id=sid,
                    operation_id=operation_id,
                )
                if block is not None:
                    self._validate_guided_operation_admission_block_row(
                        block,
                        expected_session_id=sid,
                        expected_operation_id=operation_id,
                    )
                    if kind != "guided_start":
                        raise self._guided_conflict(session_id=session_id, operation_id=operation_id)
                    return GuidedOperationFailed(failure_code="request_cancelled")
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                if row is None:
                    parent = conn.execute(
                        select(sessions_table.c.id, sessions_table.c.archived_at).where(sessions_table.c.id == sid)
                    ).one_or_none()
                    if parent is None or parent.archived_at is not None:
                        raise SessionNotFoundError(session_id)
                    lease_token = uuid.uuid4().hex
                    conn.execute(
                        insert(guided_operations_table).values(
                            session_id=sid,
                            operation_id=operation_id,
                            kind=kind,
                            status="in_progress",
                            request_hash=request_hash,
                            lease_token=lease_token,
                            lease_expires_at=lease_expires_at,
                            attempt=1,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    self._insert_guided_operation_event(
                        conn,
                        session_id=sid,
                        operation_id=operation_id,
                        event_kind="claimed",
                        actor=actor,
                        attempt=1,
                        prior_attempt=None,
                        lease_expires_at=lease_expires_at,
                        request_hash=request_hash,
                        failure_audit_cohort=None,
                        occurred_at=now,
                    )
                    return GuidedOperationClaimed(
                        fence=GuidedOperationFence(session_id, operation_id, lease_token, 1),
                        lease_expires_at=lease_expires_at,
                    )
                self._validate_guided_operation_row(
                    row,
                    expected_session_id=sid,
                    expected_operation_id=operation_id,
                )
                if row["kind"] != kind or row["request_hash"] != request_hash:
                    raise self._guided_conflict(session_id=session_id, operation_id=operation_id)
                if row["status"] in {"completed", "failed"}:
                    return self._guided_terminal_outcome(row)
                if row["status"] != "in_progress":
                    raise AuditIntegrityError("Tier 1: guided operation has an invalid status")
                parent = conn.execute(
                    select(sessions_table.c.id, sessions_table.c.archived_at).where(sessions_table.c.id == sid)
                ).one_or_none()
                if parent is None or parent.archived_at is not None:
                    raise AuditIntegrityError("Tier 1: in-progress guided operation lost its active parent custody")
                prior_expiry = self._guided_in_progress_expiry(row)
                prior_attempt = row["attempt"]
                if not isinstance(prior_attempt, int) or isinstance(prior_attempt, bool) or prior_attempt < 1:
                    raise AuditIntegrityError("Tier 1: guided operation has an invalid attempt")
                if prior_expiry > now:
                    return GuidedOperationActive(attempt=prior_attempt, lease_expires_at=prior_expiry)
                prior_token = row["lease_token"]
                assert isinstance(prior_token, str)
                next_attempt = prior_attempt + 1
                lease_token = uuid.uuid4().hex
                changed = conn.execute(
                    update(guided_operations_table)
                    .where(
                        guided_operations_table.c.session_id == sid,
                        guided_operations_table.c.operation_id == operation_id,
                        guided_operations_table.c.status == "in_progress",
                        guided_operations_table.c.lease_token == prior_token,
                        guided_operations_table.c.attempt == prior_attempt,
                        guided_operations_table.c.lease_expires_at <= now,
                    )
                    .values(
                        lease_token=lease_token,
                        lease_expires_at=lease_expires_at,
                        attempt=next_attempt,
                        updated_at=now,
                    )
                ).rowcount
                if changed != 1:
                    raise AuditIntegrityError("Guided operation takeover lost its locked compare-and-swap")
                self._insert_guided_operation_event(
                    conn,
                    session_id=sid,
                    operation_id=operation_id,
                    event_kind="taken_over",
                    actor=actor,
                    attempt=next_attempt,
                    prior_attempt=prior_attempt,
                    lease_expires_at=lease_expires_at,
                    request_hash=request_hash,
                    failure_audit_cohort=None,
                    occurred_at=now,
                )
                return GuidedOperationTakenOver(
                    fence=GuidedOperationFence(session_id, operation_id, lease_token, next_attempt),
                    prior_attempt=prior_attempt,
                    lease_expires_at=lease_expires_at,
                )

        return cast("GuidedOperationOutcome", await self._run_sync(_sync))

    async def get_guided_operation(
        self,
        *,
        session_id: UUID,
        operation_id: str,
        kind: GuidedOperationKind,
        request_hash: str,
    ) -> GuidedOperationActive | GuidedOperationCompleted | GuidedOperationFailed | None:
        """Read one matching replay descriptor without acquiring its lease."""
        self._validate_guided_identity(operation_id=operation_id, kind=kind, request_hash=request_hash)
        sid = str(session_id)

        def _sync() -> GuidedOperationActive | GuidedOperationCompleted | GuidedOperationFailed | None:
            with self._engine.connect() as conn:
                now = self._guided_database_now(conn)
                row, block = self._read_guided_operation_authority(
                    conn,
                    session_id=sid,
                    operation_id=operation_id,
                )
            if block is not None:
                self._validate_guided_operation_admission_block_row(
                    block,
                    expected_session_id=sid,
                    expected_operation_id=operation_id,
                )
                if kind != "guided_start":
                    raise self._guided_conflict(session_id=session_id, operation_id=operation_id)
                return GuidedOperationFailed(failure_code="request_cancelled")
            if row is None:
                return None
            self._validate_guided_operation_row(
                row,
                expected_session_id=sid,
                expected_operation_id=operation_id,
            )
            if row["kind"] != kind or row["request_hash"] != request_hash:
                raise self._guided_conflict(session_id=session_id, operation_id=operation_id)
            if row["status"] in {"completed", "failed"}:
                return self._guided_terminal_outcome(row)
            if row["status"] != "in_progress":
                raise AuditIntegrityError("Tier 1: guided operation has an invalid status")
            expiry = self._guided_in_progress_expiry(row)
            attempt = row["attempt"]
            if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
                raise AuditIntegrityError("Tier 1: guided operation has an invalid attempt")
            return GuidedOperationActive(attempt=attempt, lease_expires_at=expiry, expired=expiry <= now)

        return cast(
            "GuidedOperationActive | GuidedOperationCompleted | GuidedOperationFailed | None",
            await self._run_sync(_sync),
        )

    async def reconcile_guided_start_operation(
        self,
        *,
        session_id: UUID,
        operation_id: str,
        actor: str,
    ) -> GuidedOperationActive | GuidedOperationCompleted | GuidedOperationFailed:
        """Read one guided-start outcome, abandoning an expired exact attempt."""
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128:
            raise ValueError("guided operation id must be a non-empty string of at most 128 characters")
        self._validate_guided_actor(actor)
        sid = str(session_id)

        def _sync() -> GuidedOperationActive | GuidedOperationCompleted | GuidedOperationFailed:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                now = self._guided_database_now(conn)
                row, block = self._read_guided_operation_authority(
                    conn,
                    session_id=sid,
                    operation_id=operation_id,
                )
                if row is None:
                    if block is not None:
                        self._validate_guided_operation_admission_block_row(
                            block,
                            expected_session_id=sid,
                            expected_operation_id=operation_id,
                        )
                        return GuidedOperationFailed(failure_code="request_cancelled")
                    parent = conn.execute(
                        select(sessions_table.c.id, sessions_table.c.archived_at).where(sessions_table.c.id == sid)
                    ).one_or_none()
                    if parent is None or parent.archived_at is not None:
                        raise SessionNotFoundError(session_id)
                    conn.execute(
                        insert(guided_operation_admission_blocks_table).values(
                            session_id=sid,
                            operation_id=operation_id,
                            kind="guided_start",
                            failure_code="request_cancelled",
                            actor=actor,
                            created_at=now,
                        )
                    )
                    return GuidedOperationFailed(failure_code="request_cancelled")
                self._validate_guided_operation_row(
                    row,
                    expected_session_id=sid,
                    expected_operation_id=operation_id,
                )
                if row["kind"] != "guided_start":
                    raise self._guided_conflict(session_id=session_id, operation_id=operation_id)
                if row["status"] in {"completed", "failed"}:
                    return self._guided_terminal_outcome(row)
                if row["status"] != "in_progress":
                    raise AuditIntegrityError("Tier 1: guided operation has an invalid status")
                expiry = self._guided_in_progress_expiry(row)
                attempt = row["attempt"]
                if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
                    raise AuditIntegrityError("Tier 1: guided operation has an invalid attempt")
                if expiry > now:
                    return GuidedOperationActive(attempt=attempt, lease_expires_at=expiry)

                lease_token = row["lease_token"]
                assert isinstance(lease_token, str)
                changed = conn.execute(
                    update(guided_operations_table)
                    .where(
                        guided_operations_table.c.session_id == sid,
                        guided_operations_table.c.operation_id == operation_id,
                        guided_operations_table.c.status == "in_progress",
                        guided_operations_table.c.lease_token == lease_token,
                        guided_operations_table.c.attempt == attempt,
                        guided_operations_table.c.lease_expires_at <= now,
                    )
                    .values(
                        status="failed",
                        lease_token=None,
                        lease_expires_at=None,
                        proposal_id=None,
                        result_kind=None,
                        result_state_id=None,
                        result_session_id=None,
                        response_hash=None,
                        failure_code="request_cancelled",
                        settled_at=now,
                        updated_at=now,
                    )
                ).rowcount
                if changed != 1:
                    raise AuditIntegrityError("Guided operation reconciliation lost its locked compare-and-swap")
                self._insert_guided_operation_event(
                    conn,
                    session_id=sid,
                    operation_id=operation_id,
                    event_kind="failed",
                    actor=actor,
                    attempt=attempt,
                    prior_attempt=None,
                    lease_expires_at=None,
                    request_hash=row["request_hash"],
                    failure_audit_cohort=GuidedFailureAuditCohort.empty(),
                    occurred_at=now,
                )
                return GuidedOperationFailed(failure_code="request_cancelled")

        return cast(
            "GuidedOperationActive | GuidedOperationCompleted | GuidedOperationFailed",
            await self._run_sync(_sync),
        )

    def require_guided_operation_fence_on_connection(
        self,
        conn: Connection,
        fence: GuidedOperationFence,
    ) -> tuple[RowMapping, datetime]:
        """Verify an exact live fence inside the caller's locked transaction."""
        sid = str(fence.session_id)
        self._assert_session_write_lock_held(conn, sid, caller="require_guided_operation_fence_on_connection")
        now = self._guided_database_now(conn)
        row = (
            conn.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == sid,
                    guided_operations_table.c.operation_id == fence.operation_id,
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is not None:
            self._validate_guided_operation_row(
                row,
                expected_session_id=sid,
                expected_operation_id=fence.operation_id,
            )
        if (
            row is None
            or row["status"] != "in_progress"
            or row["lease_token"] != fence.lease_token
            or row["attempt"] != fence.attempt
            or self._guided_in_progress_expiry(row) <= now
        ):
            raise GuidedOperationFenceLostError(fence)
        return row, now

    async def renew_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        actor: str,
        lease_seconds: int,
    ) -> GuidedOperationFence:
        self._validate_guided_actor(actor)
        self._validate_guided_lease_seconds(lease_seconds)
        sid = str(fence.session_id)

        def _sync() -> GuidedOperationFence:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                row, now = self.require_guided_operation_fence_on_connection(conn, fence)
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                changed = conn.execute(
                    update(guided_operations_table)
                    .where(
                        guided_operations_table.c.session_id == sid,
                        guided_operations_table.c.operation_id == fence.operation_id,
                        guided_operations_table.c.status == "in_progress",
                        guided_operations_table.c.lease_token == fence.lease_token,
                        guided_operations_table.c.attempt == fence.attempt,
                        guided_operations_table.c.lease_expires_at > now,
                    )
                    .values(lease_expires_at=lease_expires_at, updated_at=now)
                ).rowcount
                if changed != 1:
                    raise GuidedOperationFenceLostError(fence)
                self._insert_guided_operation_event(
                    conn,
                    session_id=sid,
                    operation_id=fence.operation_id,
                    event_kind="renewed",
                    actor=actor,
                    attempt=fence.attempt,
                    prior_attempt=None,
                    lease_expires_at=lease_expires_at,
                    request_hash=row["request_hash"],
                    failure_audit_cohort=None,
                    occurred_at=now,
                )
                return fence

        return cast("GuidedOperationFence", await self._run_sync(_sync))

    @staticmethod
    def _merge_guided_binding(*, current: Any, requested: UUID | None, label: str) -> str | None:
        if requested is None:
            return cast("str | None", current)
        requested_value = str(requested)
        if current is not None and current != requested_value:
            raise AuditIntegrityError(f"Guided operation {label} is already bound to a different row")
        return requested_value

    def bind_guided_operation_on_connection(
        self,
        conn: Connection,
        fence: GuidedOperationFence,
        *,
        originating_message_id: UUID | None = None,
        proposal_id: UUID | None = None,
        result_state_id: UUID | None = None,
        result_session_id: UUID | None = None,
    ) -> None:
        """Bind resumable row ids under the exact fence and caller transaction."""
        row, now = self.require_guided_operation_fence_on_connection(conn, fence)
        values = {
            "originating_message_id": self._merge_guided_binding(
                current=row["originating_message_id"], requested=originating_message_id, label="originating message"
            ),
            "proposal_id": self._merge_guided_binding(current=row["proposal_id"], requested=proposal_id, label="proposal"),
            "result_state_id": self._merge_guided_binding(current=row["result_state_id"], requested=result_state_id, label="result state"),
            "result_session_id": self._merge_guided_binding(
                current=row["result_session_id"], requested=result_session_id, label="result session"
            ),
            "updated_at": now,
        }
        changed = conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(fence.session_id),
                guided_operations_table.c.operation_id == fence.operation_id,
                guided_operations_table.c.status == "in_progress",
                guided_operations_table.c.lease_token == fence.lease_token,
                guided_operations_table.c.attempt == fence.attempt,
                guided_operations_table.c.lease_expires_at > now,
            )
            .values(**values)
        ).rowcount
        if changed != 1:
            raise GuidedOperationFenceLostError(fence)

    async def bind_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        originating_message_id: UUID | None = None,
        proposal_id: UUID | None = None,
        result_state_id: UUID | None = None,
        result_session_id: UUID | None = None,
    ) -> None:
        sid = str(fence.session_id)

        def _sync() -> None:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                self.bind_guided_operation_on_connection(
                    conn,
                    fence,
                    originating_message_id=originating_message_id,
                    proposal_id=proposal_id,
                    result_state_id=result_state_id,
                    result_session_id=result_session_id,
                )

        await self._run_sync(_sync)

    @staticmethod
    def _guided_completion_values(
        *,
        row: RowMapping,
        result: GuidedOperationResult,
    ) -> tuple[dict[str, str | None], GuidedOperationResult]:
        kind = row["kind"]
        if isinstance(result, GuidedCompositionStateResult):
            if kind in {"session_fork", "guided_plan"}:
                raise ValueError("guided operation kind requires a different result locator")
            if result.proposal_id is not None and kind not in {"guided_respond", "guided_chat"}:
                raise ValueError("only guided respond/chat state results may carry proposal_id")
            state_id = SessionServiceImpl._merge_guided_binding(
                current=row["result_state_id"], requested=result.state_id, label="result state"
            )
            proposal_id = SessionServiceImpl._merge_guided_binding(
                current=row["proposal_id"], requested=result.proposal_id, label="proposal"
            )
            if row["result_session_id"] is not None:
                raise AuditIntegrityError("State operation has a conflicting result-session binding")
            assert state_id is not None
            normalized: GuidedOperationResult = GuidedCompositionStateResult(
                state_id=UUID(state_id),
                proposal_id=UUID(proposal_id) if proposal_id is not None else None,
            )
            return (
                {
                    "result_kind": "composition_state",
                    "result_state_id": state_id,
                    "result_session_id": None,
                    "proposal_id": proposal_id,
                },
                normalized,
            )
        if isinstance(result, GuidedPipelineProposalResult):
            if kind != "guided_plan":
                raise ValueError("only guided_plan may complete with a pipeline proposal locator")
            proposal_id = SessionServiceImpl._merge_guided_binding(
                current=row["proposal_id"], requested=result.proposal_id, label="proposal"
            )
            checkpoint_state_id = SessionServiceImpl._merge_guided_binding(
                current=row["result_state_id"], requested=result.checkpoint_state_id, label="checkpoint state"
            )
            if row["result_session_id"] is not None:
                raise AuditIntegrityError("Guided plan operation has a conflicting result-session binding")
            assert proposal_id is not None and checkpoint_state_id is not None
            normalized = GuidedPipelineProposalResult(
                proposal_id=UUID(proposal_id),
                checkpoint_state_id=UUID(checkpoint_state_id),
            )
            return (
                {
                    "result_kind": "pipeline_proposal",
                    "result_state_id": checkpoint_state_id,
                    "result_session_id": None,
                    "proposal_id": proposal_id,
                },
                normalized,
            )
        if isinstance(result, GuidedSessionResult):
            if kind != "session_fork":
                raise ValueError("only session_fork may complete with a session locator")
            session_id = SessionServiceImpl._merge_guided_binding(
                current=row["result_session_id"], requested=result.session_id, label="result session"
            )
            if row["result_state_id"] is not None or row["proposal_id"] is not None:
                raise AuditIntegrityError("Fork operation has a conflicting state/proposal binding")
            assert session_id is not None
            return (
                {
                    "result_kind": "session",
                    "result_state_id": None,
                    "result_session_id": session_id,
                    "proposal_id": None,
                },
                GuidedSessionResult(session_id=UUID(session_id)),
            )
        raise TypeError("unsupported guided operation result locator")

    def complete_guided_operation_on_connection(
        self,
        conn: Connection,
        fence: GuidedOperationFence,
        *,
        result: GuidedOperationResult,
        response_hash: str,
        actor: str,
    ) -> GuidedOperationCompleted:
        """Settle a successful result and append its event in one transaction."""
        self._validate_guided_actor(actor)
        self._validate_guided_hash(response_hash, label="guided operation response_hash")
        row, now = self.require_guided_operation_fence_on_connection(conn, fence)
        if isinstance(result, GuidedSessionResult):
            parent = (
                conn.execute(
                    select(
                        sessions_table.c.user_id,
                        sessions_table.c.auth_provider_type,
                    ).where(sessions_table.c.id == str(fence.session_id))
                )
                .mappings()
                .one_or_none()
            )
            child = (
                conn.execute(
                    select(
                        sessions_table.c.user_id,
                        sessions_table.c.auth_provider_type,
                        sessions_table.c.forked_from_session_id,
                    ).where(sessions_table.c.id == str(result.session_id))
                )
                .mappings()
                .one_or_none()
            )
            if (
                parent is None
                or child is None
                or child["forked_from_session_id"] != str(fence.session_id)
                or child["user_id"] != parent["user_id"]
                or child["auth_provider_type"] != parent["auth_provider_type"]
            ):
                raise AuditIntegrityError("Guided fork result session failed lineage or principal custody validation")
        locator_values, normalized = self._guided_completion_values(row=row, result=result)
        changed = conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(fence.session_id),
                guided_operations_table.c.operation_id == fence.operation_id,
                guided_operations_table.c.status == "in_progress",
                guided_operations_table.c.lease_token == fence.lease_token,
                guided_operations_table.c.attempt == fence.attempt,
                guided_operations_table.c.lease_expires_at > now,
            )
            .values(
                status="completed",
                lease_token=None,
                lease_expires_at=None,
                response_hash=response_hash,
                failure_code=None,
                settled_at=now,
                updated_at=now,
                **locator_values,
            )
        ).rowcount
        if changed != 1:
            raise GuidedOperationFenceLostError(fence)
        self._insert_guided_operation_event(
            conn,
            session_id=str(fence.session_id),
            operation_id=fence.operation_id,
            event_kind="completed",
            actor=actor,
            attempt=fence.attempt,
            prior_attempt=None,
            lease_expires_at=None,
            request_hash=row["request_hash"],
            failure_audit_cohort=None,
            occurred_at=now,
        )
        return GuidedOperationCompleted(result=normalized, response_hash=response_hash)

    async def complete_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        result: GuidedOperationResult,
        response_hash: str,
        actor: str,
    ) -> GuidedOperationCompleted:
        sid = str(fence.session_id)

        def _sync() -> GuidedOperationCompleted:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                return self.complete_guided_operation_on_connection(
                    conn,
                    fence,
                    result=result,
                    response_hash=response_hash,
                    actor=actor,
                )

        return cast("GuidedOperationCompleted", await self._run_sync(_sync))

    def fail_guided_operation_on_connection(
        self,
        conn: Connection,
        fence: GuidedOperationFence,
        *,
        failure_code: GuidedOperationFailureCode,
        actor: str,
        failure_audit_cohort: GuidedFailureAuditCohort,
    ) -> GuidedOperationFailed:
        """Clear partial locators and settle one closed safe failure atomically."""
        self._validate_guided_actor(actor)
        if failure_code not in GUIDED_OPERATION_FAILURE_CODE_VALUES:
            raise ValueError("unsupported guided operation failure code")
        row, now = self.require_guided_operation_fence_on_connection(conn, fence)
        changed = conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(fence.session_id),
                guided_operations_table.c.operation_id == fence.operation_id,
                guided_operations_table.c.status == "in_progress",
                guided_operations_table.c.lease_token == fence.lease_token,
                guided_operations_table.c.attempt == fence.attempt,
                guided_operations_table.c.lease_expires_at > now,
            )
            .values(
                status="failed",
                lease_token=None,
                lease_expires_at=None,
                proposal_id=None,
                result_kind=None,
                result_state_id=None,
                result_session_id=None,
                response_hash=None,
                failure_code=failure_code,
                settled_at=now,
                updated_at=now,
            )
        ).rowcount
        if changed != 1:
            raise GuidedOperationFenceLostError(fence)
        self._insert_guided_operation_event(
            conn,
            session_id=str(fence.session_id),
            operation_id=fence.operation_id,
            event_kind="failed",
            actor=actor,
            attempt=fence.attempt,
            prior_attempt=None,
            lease_expires_at=None,
            request_hash=row["request_hash"],
            failure_audit_cohort=failure_audit_cohort,
            occurred_at=now,
        )
        return GuidedOperationFailed(failure_code=failure_code)

    async def fail_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        failure_code: GuidedOperationFailureCode,
        actor: str,
    ) -> GuidedOperationFailed:
        sid = str(fence.session_id)

        def _sync() -> GuidedOperationFailed:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                return self.fail_guided_operation_on_connection(
                    conn,
                    fence,
                    failure_code=failure_code,
                    actor=actor,
                    failure_audit_cohort=GuidedFailureAuditCohort.empty(),
                )

        return cast("GuidedOperationFailed", await self._run_sync(_sync))

    async def fail_guided_operation_with_audit(
        self,
        command: GuidedOperationFailureCommand,
    ) -> GuidedOperationFailed:
        """Atomically persist sanitized evidence and settle one closed failure."""

        if type(command) is not GuidedOperationFailureCommand:
            raise TypeError("command must be an exact GuidedOperationFailureCommand")
        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=command.audit_evidence,
            payloads=(),
            payload_store=None,
        )
        sid = str(command.fence.session_id)
        now = self._now()

        def _sync() -> GuidedOperationFailed:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                audit_records: tuple[ChatMessageRecord, ...] = ()
                if audit_rows:
                    operation, _database_now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                    lineage = GuidedFailureAuditLineage.from_authority(
                        session_id=command.fence.session_id,
                        operation_id=command.fence.operation_id,
                        attempt=command.fence.attempt,
                        request_hash=operation["request_hash"],
                    )
                    bound_audit_rows = bind_guided_failure_audit_rows(audit_rows, lineage=lineage)
                    current_state_id = conn.execute(
                        select(composition_states_table.c.id)
                        .where(composition_states_table.c.session_id == sid)
                        .order_by(desc(composition_states_table.c.version))
                        .limit(1)
                    ).scalar_one_or_none()
                    sequence_no = self._reserve_sequence_range(conn, sid, count=len(bound_audit_rows))
                    audit_records = self._insert_prepared_guided_audit_rows_on_connection(
                        conn,
                        session_id=sid,
                        composition_state_id=UUID(current_state_id) if current_state_id is not None else None,
                        audit_rows=bound_audit_rows,
                        sequence_no=sequence_no,
                        created_at=now,
                    )
                    conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))
                return self.fail_guided_operation_on_connection(
                    conn,
                    command.fence,
                    failure_code=command.failure_code,
                    actor=command.actor,
                    failure_audit_cohort=GuidedFailureAuditCohort.from_records(audit_records),
                )

        return cast("GuidedOperationFailed", await self._run_sync(_sync))

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
        message_id: str | None = None,
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
        msg_id = message_id or str(uuid.uuid4())
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

    def _insert_transition_assistant(
        self,
        conn: Connection,
        *,
        session_id: str,
        state_id: str,
        content: str,
        raw_content: str | None,
        created_at: datetime,
    ) -> ChatMessageRecord:
        """Insert one transition response under the caller's transaction."""
        self._assert_session_write_lock_held(
            conn,
            session_id,
            caller="_insert_transition_assistant",
        )
        sequence_no = self._reserve_sequence_range(conn, session_id, count=1)
        message_id = self._insert_chat_message(
            conn,
            session_id=session_id,
            role="assistant",
            content=content,
            raw_content=raw_content,
            tool_calls=None,
            sequence_no=sequence_no,
            writer_principal="compose_loop",
            composition_state_id=state_id,
            tool_call_id=None,
            parent_assistant_id=None,
            created_at=created_at,
        )
        conn.execute(update(sessions_table).where(sessions_table.c.id == session_id).values(updated_at=created_at))
        message_row = conn.execute(select(chat_messages_table).where(chat_messages_table.c.id == message_id)).one()
        return self._row_to_chat_message_record(message_row)

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
            forked_from_session_id=None,
            forked_from_message_id=None,
        )

    def _row_to_session_record(self, row: Any) -> SessionRecord:
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

    async def get_session(self, session_id: UUID) -> SessionRecord:
        """Fetch a session by ID. Raises SessionNotFoundError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise SessionNotFoundError(session_id)

        return self._row_to_session_record(row)

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
                    or_(
                        sessions_table.c.forked_from_session_id.is_(None),
                        exists(
                            select(guided_operations_table.c.operation_id).where(
                                guided_operations_table.c.kind == "session_fork",
                                guided_operations_table.c.status == "completed",
                                guided_operations_table.c.result_session_id == sessions_table.c.id,
                            )
                        ),
                    ),
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
            blob_dir = self._data_dir / "blobs" / sid if self._data_dir is not None else None
            staged_blob_dir = self._data_dir / ".archive_quarantine" / sid if self._data_dir is not None else None

            try:
                with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                    active_guided_kind = conn.execute(
                        select(guided_operations_table.c.kind)
                        .where(
                            guided_operations_table.c.session_id == sid,
                            guided_operations_table.c.status == "in_progress",
                        )
                        .limit(1)
                    ).scalar_one_or_none()
                    if active_guided_kind is not None:
                        raise SessionGuidedOperationInProgressError(
                            session_id=session_id,
                            kind=cast("GuidedOperationKind", active_guided_kind),
                        )
                    durable_history_exists = (
                        conn.execute(select(runs_table.c.id).where(runs_table.c.session_id == sid).limit(1)).first() is not None
                        or conn.execute(
                            select(composer_completion_events_table.c.id)
                            .where(composer_completion_events_table.c.session_id == sid)
                            .limit(1)
                        ).first()
                        is not None
                        or conn.execute(
                            select(guided_operations_table.c.operation_id)
                            .where(
                                guided_operations_table.c.session_id == sid,
                                guided_operations_table.c.kind == "session_fork",
                                guided_operations_table.c.status.in_(["completed", "failed"]),
                            )
                            .limit(1)
                        ).first()
                        is not None
                        or conn.execute(
                            select(guided_operations_table.c.operation_id)
                            .where(
                                guided_operations_table.c.kind == "session_fork",
                                guided_operations_table.c.status == "completed",
                                guided_operations_table.c.result_session_id == sid,
                            )
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
                        return
                    # The DB guard above and filesystem quarantine happen while
                    # the same session lock is held. A fork cannot freeze plan
                    # bytes and then race an archive that temporarily removes
                    # them before the in-progress operation is observed.
                    if blob_dir is not None and staged_blob_dir is not None and blob_dir.is_dir():
                        staged_blob_dir.parent.mkdir(parents=True, exist_ok=True)
                        if staged_blob_dir.exists():
                            raise OSError(
                                f"archive_session({sid}): quarantine path {staged_blob_dir} already exists. "
                                "Manual cleanup of the stale staged blob directory is required before archive can proceed."
                            )
                        blob_dir.rename(staged_blob_dir)
                    # Session archival is the only supported transcript purge:
                    # delete the parent row and let the schema-owned cascades
                    # remove session-scoped children. Direct chat-message
                    # deletes are blocked by trg_chat_messages_no_delete.
                    deleted = conn.execute(delete(sessions_table).where(sessions_table.c.id == sid))
                    if deleted.rowcount != 1:
                        raise SessionNotFoundError(session_id)
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
                            "schema": _TOOL_PROPOSAL_CREATED_SCHEMA,
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
        """Load one canonical pipeline authority, rejecting current tool proposals."""
        authority = await self.get_authoritative_composition_proposal(
            session_id=session_id,
            proposal_id=proposal_id,
            reviewed_facts=reviewed_facts,
        )
        if authority.pipeline is None:
            raise ValueError("proposal uses the current tool-proposal lifecycle contract")
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
        transition_assistant: TransitionAssistantDraft | None = None,
    ) -> PipelineProposalSettlementResult:
        """Atomically publish state and settle a verified pipeline proposal."""
        if type(dispatch) is not PipelineDispatchAuditBinding:
            raise TypeError("dispatch must be an exact PipelineDispatchAuditBinding")
        state_content_hash = _composition_state_data_content_hash(state)
        if candidate_content_hash != executor_content_hash or candidate_content_hash != state_content_hash:
            raise AuditIntegrityError("pipeline candidate/executor/state content hash mismatch")
        settled_state = replace(state, composer_meta=final_composer_metadata)
        if transition_assistant is not None:
            if type(transition_assistant) is not TransitionAssistantDraft:
                raise TypeError("transition_assistant must be an exact TransitionAssistantDraft")
            composer_meta = deep_thaw(settled_state.composer_meta)
            guided_session = composer_meta.get("guided_session") if type(composer_meta) is dict else None
            if type(guided_session) is not dict or guided_session.get("transition_consumed") is not True:
                raise AuditIntegrityError("transition assistant requires guided_session.transition_consumed=true")
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
                    transition_message = None
                    if transition_assistant is not None:
                        message_rows = conn.execute(
                            select(chat_messages_table)
                            .where(chat_messages_table.c.session_id == sid)
                            .where(chat_messages_table.c.role == "assistant")
                            .where(chat_messages_table.c.composition_state_id == str(committed_record.id))
                            .where(chat_messages_table.c.writer_principal == "compose_loop")
                        ).fetchall()
                        matching_messages = [
                            self._row_to_chat_message_record(message_row)
                            for message_row in message_rows
                            if message_row.content == transition_assistant.content
                            and message_row.raw_content == transition_assistant.raw_content
                        ]
                        if len(matching_messages) > 1:
                            raise AuditIntegrityError("committed transition assistant exact retry is ambiguous")
                        if matching_messages:
                            transition_message = matching_messages[0]
                        else:
                            transition_message = self._insert_transition_assistant(
                                conn,
                                session_id=sid,
                                state_id=str(committed_record.id),
                                content=transition_assistant.content,
                                raw_content=transition_assistant.raw_content,
                                created_at=now,
                            )
                    return (
                        PipelineProposalSettlementResult(
                            proposal=replace(authority.row, pipeline_metadata=_pipeline_public_metadata(authority)),
                            state=committed_record,
                            transition_message=transition_message,
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
                transition_message = None
                if transition_assistant is not None:
                    transition_message = self._insert_transition_assistant(
                        conn,
                        session_id=sid,
                        state_id=state_id,
                        content=transition_assistant.content,
                        raw_content=transition_assistant.raw_content,
                        created_at=now,
                    )
                return (
                    PipelineProposalSettlementResult(
                        proposal=replace(
                            _proposal_record_from_row(settled_row),
                            pipeline_metadata=_pipeline_public_metadata(authority),
                        ),
                        state=self._row_to_state_record(state_row),
                        transition_message=transition_message,
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

        def _sync() -> PipelineDispatchRecovery | None:
            with self._engine.begin() as conn:
                return self._pipeline_dispatch_recovery_on_connection(conn, authority=authority)

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
        """Insert one checked pending event in its own locked transaction."""

        result = await self._prepare_or_create_pending_interpretation_event(
            session_id=session_id,
            composition_state_id=composition_state_id,
            affected_node_id=affected_node_id,
            tool_call_id=tool_call_id,
            user_term=user_term,
            kind=kind,
            llm_draft=llm_draft,
            model_identifier=model_identifier,
            model_version=model_version,
            provider=provider,
            composer_skill_hash=composer_skill_hash,
            created_at=created_at,
        )
        return cast(InterpretationEventRecord, result)

    async def _prepare_or_create_pending_interpretation_event(
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
        _event_id: UUID | None = None,
        _prepare_only: bool = False,
    ) -> InterpretationEventRecord | Callable[[Connection], InterpretationEventRecord]:
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
        event_id = str(_event_id if _event_id is not None else uuid.uuid4())
        plugin_snapshot = await self._plugin_snapshot_for_session(sid)

        def _sync(connection: Connection | None = None) -> InterpretationEventRecord:
            if connection is None:
                with self._session_process_locked_begin(sid) as conn:
                    return _sync(conn)
            conn = connection
            # Preserve one shared, lexically lock-proven writer body for
            # standalone and settlement-cohort transactions. The cohort owns
            # the same re-entrant session lock at its outer boundary.
            with self._session_write_lock(conn, sid):
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
                sources = self._unwrap_envelope(state_row.sources)
                if kind is InterpretationKind.INVENTED_SOURCE:
                    source_name = source_name_from_component_id(affected_node_id)
                    if source_name is None:
                        raise ValueError(
                            "create_pending_interpretation_event: invented_source must target a source component "
                            f"({SOURCE_COMPONENT_ID!r} or {SOURCE_COMPONENT_ID!r}:<name>), got {affected_node_id!r}"
                        )
                    source = sources[source_name] if isinstance(sources, Mapping) and source_name in sources else None
                    if not isinstance(source, Mapping):
                        raise ValueError(f"create_pending_interpretation_event: invented_source requires persisted source {source_name!r}")
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
                    # Free-form validator text can echo filesystem paths,
                    # credentials, and provider diagnostics. The state keeps
                    # the structured validity bit, not those raw messages.
                    raw_validation_errors = [error.message for error in patched_validation.errors] or None
                    patched_validation_errors = validation_errors_for_composer_surface(
                        composer_meta=live_state_record.composer_meta,
                        is_valid=patched_validation.is_valid,
                        validation_errors=raw_validation_errors,
                    )
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

        if _prepare_only:
            return _sync
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
                raw_validation_errors = [error.message for error in patched_validation.errors] or None
                patched_validation_errors = validation_errors_for_composer_surface(
                    composer_meta=state_record.composer_meta,
                    is_valid=patched_validation.is_valid,
                    validation_errors=raw_validation_errors,
                )

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

    def _verify_guided_failure_audit_cohort(
        self,
        message_rows: Sequence[Any],
        failed_event_rows: Sequence[Any],
    ) -> None:
        """Fail closed unless every failed event commits its exact evidence cohort."""

        event_commitments: dict[GuidedFailureAuditLineage, list[GuidedFailureAuditCohort]] = {}
        for event in failed_event_rows:
            try:
                lineage = GuidedFailureAuditLineage.from_authority(
                    session_id=UUID(event.session_id),
                    operation_id=event.operation_id,
                    attempt=event.attempt,
                    request_hash=event.request_hash,
                )
            except (TypeError, ValueError, AuditIntegrityError) as exc:
                raise AuditIntegrityError("guided failure audit lineage has malformed terminal-event authority") from exc
            try:
                commitment = GuidedFailureAuditCohort.from_envelope(event.failure_audit_cohort)
            except (TypeError, ValueError, AuditIntegrityError) as exc:
                raise AuditIntegrityError("guided failure audit cohort has malformed terminal-event commitment") from exc
            event_commitments.setdefault(lineage, []).append(commitment)

        records_by_lineage: dict[GuidedFailureAuditLineage, list[ChatMessageRecord]] = {}
        for row in message_rows:
            try:
                content = json.loads(row.content)
            except (TypeError, ValueError):
                content = None
            content_has_lineage = type(content) is dict and GUIDED_FAILURE_AUDIT_LINEAGE_KEY in content
            tool_calls = row.tool_calls
            envelope_has_lineage = type(tool_calls) is list and any(
                type(envelope) is dict and GUIDED_FAILURE_AUDIT_LINEAGE_KEY in envelope for envelope in tool_calls
            )
            if not content_has_lineage and not envelope_has_lineage:
                continue
            if (
                row.role != "audit"
                or row.writer_principal != "compose_loop"
                or type(content) is not dict
                or type(tool_calls) is not list
                or len(tool_calls) != 1
                or type(tool_calls[0]) is not dict
                or GUIDED_FAILURE_AUDIT_LINEAGE_KEY not in content
                or GUIDED_FAILURE_AUDIT_LINEAGE_KEY not in tool_calls[0]
            ):
                raise AuditIntegrityError("guided failure audit lineage is partial or attached to a malformed row")
            content_lineage = GuidedFailureAuditLineage.from_envelope(content[GUIDED_FAILURE_AUDIT_LINEAGE_KEY])
            envelope_lineage = GuidedFailureAuditLineage.from_envelope(tool_calls[0][GUIDED_FAILURE_AUDIT_LINEAGE_KEY])
            if content_lineage != envelope_lineage:
                raise AuditIntegrityError("guided failure audit lineage content and envelope disagree")
            if content_lineage.session_id != UUID(row.session_id):
                raise AuditIntegrityError("guided failure audit lineage names a different session")
            if len(event_commitments.get(content_lineage, ())) != 1:
                raise AuditIntegrityError(
                    "guided failure audit lineage has absent or ambiguous terminal-event authority; "
                    "guided failure audit cohort authority is not exact"
                )
            records_by_lineage.setdefault(content_lineage, []).append(self._row_to_chat_message_record(row))

        for lineage, commitments in event_commitments.items():
            if len(commitments) != 1:
                raise AuditIntegrityError(
                    "guided failure audit lineage has ambiguous terminal-event authority; "
                    "guided failure audit cohort authority is not exact"
                )
            records = tuple(records_by_lineage.get(lineage, ()))
            actual = GuidedFailureAuditCohort.from_records(records)
            if actual != commitments[0]:
                raise AuditIntegrityError("guided failure audit cohort does not match the exact durable evidence rows")

    def _row_to_chat_message_record(self, row: Any) -> ChatMessageRecord:
        return ChatMessageRecord(
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

        def _sync() -> tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
            with self._engine.begin() as conn:
                message_rows = conn.execute(
                    select(chat_messages_table)
                    .where(chat_messages_table.c.session_id == str(session_id))
                    .order_by(chat_messages_table.c.sequence_no)
                    .limit(limit)
                    .offset(offset)
                ).fetchall()
                verification_message_rows = conn.execute(
                    select(chat_messages_table)
                    .where(chat_messages_table.c.session_id == str(session_id))
                    .order_by(chat_messages_table.c.sequence_no)
                ).fetchall()
                failed_event_rows = conn.execute(
                    select(
                        guided_operation_events_table.c.session_id,
                        guided_operation_events_table.c.operation_id,
                        guided_operation_events_table.c.attempt,
                        guided_operation_events_table.c.request_hash,
                        guided_operation_events_table.c.failure_audit_cohort,
                    )
                    .where(guided_operation_events_table.c.session_id == str(session_id))
                    .where(guided_operation_events_table.c.event_kind == "failed")
                ).fetchall()
                return message_rows, verification_message_rows, failed_event_rows

        rows, verification_message_rows, failed_event_rows = await self._run_sync(_sync)
        self._verify_guided_failure_audit_cohort(verification_message_rows, failed_event_rows)

        return [self._row_to_chat_message_record(row) for row in rows]

    async def get_verified_guided_root_intent(
        self,
        *,
        session_id: UUID,
        root_message_id: UUID,
    ) -> ChatMessageRecord:
        """Re-derive a live start hash before returning its private root row."""

        from elspeth.web.composer.guided.profile import EMPTY_PROFILE
        from elspeth.web.sessions.guided_operations import guided_operation_request_hash
        from elspeth.web.sessions.schemas import StartGuidedRequest

        sid = str(session_id)
        mid = str(root_message_id)

        def _sync() -> ChatMessageRecord:
            with self._engine.begin() as conn:
                message_row = conn.execute(
                    select(chat_messages_table).where(chat_messages_table.c.session_id == sid).where(chat_messages_table.c.id == mid)
                ).one_or_none()
                if message_row is None or message_row.role != "user" or message_row.writer_principal != "route_user_message":
                    raise AuditIntegrityError("guided root intent row failed session/role/writer custody")
                operations = conn.execute(
                    select(guided_operations_table)
                    .where(guided_operations_table.c.session_id == sid)
                    .where(guided_operations_table.c.kind == "guided_start")
                    .where(guided_operations_table.c.status == "completed")
                    .where(guided_operations_table.c.originating_message_id == mid)
                    .where(guided_operations_table.c.result_kind == "composition_state")
                ).fetchall()
                if len(operations) != 1:
                    raise AuditIntegrityError("guided root intent has absent or ambiguous start-operation authority")
                operation = operations[0]
                request = StartGuidedRequest.model_validate(
                    {
                        "operation_id": operation.operation_id,
                        "profile": "live",
                        "intent": message_row.content,
                    },
                    strict=True,
                )
                if guided_operation_request_hash(session_id=session_id, kind="guided_start", request=request) != operation.request_hash:
                    raise AuditIntegrityError("guided root intent content no longer matches its start request hash")
                state_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .where(composition_states_table.c.id == operation.result_state_id)
                ).one_or_none()
                if state_row is None:
                    raise AuditIntegrityError("guided root intent start result state is missing")
                guided = state_from_record(self._row_to_state_record(state_row)).guided_session
                if guided is None or guided.profile != EMPTY_PROFILE or guided.root_intent_message_id != mid:
                    raise AuditIntegrityError("guided root intent differs from its live start checkpoint")
                return self._row_to_chat_message_record(message_row)

        return cast("ChatMessageRecord", await self._run_sync(_sync))

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

    async def commit_transition_response(
        self,
        *,
        session_id: UUID,
        expected_current_state_id: UUID | None,
        state: CompositionStateData,
        assistant_content: str,
        raw_content: str | None,
    ) -> TransitionResponseSettlement:
        """Persist transition consumption and its assistant response atomically."""
        composer_meta = deep_thaw(state.composer_meta)
        guided_session = composer_meta.get("guided_session") if type(composer_meta) is dict else None
        if type(guided_session) is not dict or guided_session.get("transition_consumed") is not True:
            raise AuditIntegrityError("commit_transition_response requires guided_session.transition_consumed=true")

        sid = str(session_id)
        expected_state_id = str(expected_current_state_id) if expected_current_state_id is not None else None
        now = self._now()

        def _sync() -> tuple[CompositionStateRecord, ChatMessageRecord]:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                current_state_id = conn.execute(
                    select(composition_states_table.c.id)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).scalar_one_or_none()
                if current_state_id != expected_state_id:
                    raise StaleComposeStateError(
                        "commit_transition_response: current composition state changed "
                        f"for session_id={sid!r}; expected={expected_state_id!r}, "
                        f"actual={current_state_id!r}"
                    )

                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=state),
                    provenance="post_compose",
                    created_at=now,
                )
                message_record = self._insert_transition_assistant(
                    conn,
                    session_id=sid,
                    state_id=state_id,
                    content=assistant_content,
                    raw_content=raw_content,
                    created_at=now,
                )
                state_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .where(composition_states_table.c.id == state_id)
                ).one()
                return self._row_to_state_record(state_row), message_record

        state_record, message_record = cast(
            tuple[CompositionStateRecord, ChatMessageRecord],
            await self._run_sync(_sync),
        )
        return TransitionResponseSettlement(state=state_record, message=message_record)

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

    async def revert_state_for_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        state_id: UUID,
        expected_current_state_id: UUID,
        expected_current_state_version: int,
        actor: str,
        response_hash_factory: Callable[[CompositionStateRecord], str],
    ) -> CompositionStateRecord:
        """Copy one checkpoint and settle its retry operation atomically.

        The fence check, state copy, system audit message, replay locator, and
        response-domain hash all share one session lock and one database
        transaction.  In particular this is not implemented as a public
        ``require_fence`` followed by ``set_active_state``: takeover between
        those calls would let a stale worker create a durable version.
        """

        sid = str(fence.session_id)
        target_state_id = str(state_id)
        now = self._now()

        def _sync() -> CompositionStateRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                self.require_guided_operation_fence_on_connection(conn, fence)
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=expected_current_state_id,
                    expected_state_version=expected_current_state_version,
                )
                prior_row = conn.execute(
                    select(composition_states_table).where(composition_states_table.c.id == target_state_id)
                ).one_or_none()
                # User-supplied state ids deliberately collapse absence and
                # cross-session ownership to the route's same 404 boundary.
                if prior_row is None or prior_row.session_id != sid:
                    raise ValueError("State not found")

                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if current_row is None:
                    raise AuditIntegrityError("state revert session unexpectedly has no current checkpoint")

                from elspeth.web.composer.guided.errors import InvariantError
                from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
                from elspeth.web.composer.guided.state_machine import GuidedSession
                from elspeth.web.sessions.guided_replay import GUIDED_REPLAY_META_KEY

                def _guided_checkpoint(row: Any, *, role: str) -> tuple[CompositionStateRecord, GuidedSession | None]:
                    checkpoint = self._row_to_state_record(row)
                    metadata = deep_thaw(checkpoint.composer_meta)
                    if metadata is None:
                        return checkpoint, None
                    if type(metadata) is not dict:
                        raise AuditIntegrityError(f"{role} checkpoint composer metadata is malformed")
                    if "guided_session" not in metadata or metadata["guided_session"] is None:
                        return checkpoint, None
                    try:
                        guided = GuidedSession.from_dict(metadata["guided_session"])
                    except (InvariantError, KeyError, TypeError, ValueError) as exc:
                        raise AuditIntegrityError(f"{role} checkpoint guided schema-10 authority is malformed") from exc
                    return checkpoint, guided

                target_record, target_guided = _guided_checkpoint(prior_row, role="target")
                current_record, current_guided = _guided_checkpoint(current_row, role="current")

                referenced_authorities: dict[UUID, AuthoritativePipelineProposal] = {}
                for role, checkpoint, guided in (
                    ("current", current_record, current_guided),
                    ("target", target_record, target_guided),
                ):
                    if guided is None:
                        continue
                    authority = _require_pending_guided_checkpoint_proposal_authority(
                        conn,
                        service=self,
                        session_id=sid,
                        checkpoint=checkpoint,
                        guided=guided,
                        role=role,
                    )
                    if authority is not None:
                        referenced_authorities[authority.row.id] = authority

                def _creation_event_for(proposal_id: UUID) -> ProposalEventRecord:
                    creation_rows = conn.execute(
                        select(proposal_events_table)
                        .where(proposal_events_table.c.session_id == sid)
                        .where(proposal_events_table.c.proposal_id == str(proposal_id))
                        .where(proposal_events_table.c.event_type == "proposal.created")
                    ).fetchall()
                    if len(creation_rows) != 1:
                        raise AuditIntegrityError("state revert pipeline proposal must have exactly one creation event")
                    return _proposal_event_record_from_row(creation_rows[0])

                pending_rows = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.tool_name == "set_pipeline")
                    .where(composition_proposals_table.c.status == "pending")
                    .order_by(composition_proposals_table.c.created_at, composition_proposals_table.c.id)
                ).fetchall()
                pending_authorities: list[AuthoritativePipelineProposal] = []
                for proposal_row in pending_rows:
                    proposal_id = UUID(proposal_row.id)
                    authority = referenced_authorities.get(proposal_id)
                    if authority is None:
                        authority = _restore_authoritative_pipeline_proposal(
                            conn=conn,
                            row=_proposal_record_from_row(proposal_row),
                            creation_event=_creation_event_for(proposal_id),
                            reviewed_facts=None,
                        )
                        _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                        if authority.proposal.surface is not PlannerSurface.FREEFORM:
                            raise AuditIntegrityError("state revert found a dangling pending guided pipeline proposal")
                    if authority.row.status != "pending":
                        raise AuditIntegrityError("state revert pending proposal query restored terminal authority")
                    pending_authorities.append(authority)

                # Verify the complete affected proposal set before the first
                # write.  A malformed ref, event, row, or reviewed-facts
                # binding therefore rolls the whole revert back untouched.
                for authority in pending_authorities:
                    _require_no_active_guided_confirmation_admission(
                        conn,
                        session_id=sid,
                        proposal_id=str(authority.row.id),
                        now=now,
                    )
                    event_id = str(uuid.uuid4())
                    conn.execute(
                        insert(proposal_events_table).values(
                            id=event_id,
                            session_id=sid,
                            proposal_id=str(authority.row.id),
                            event_type="proposal.rejected",
                            actor=actor,
                            payload=_pipeline_rejected_payload(
                                authority=authority,
                                reason="superseded",
                                dispatch=None,
                            ),
                            created_at=now,
                        )
                    )
                    updated = conn.execute(
                        update(composition_proposals_table)
                        .where(composition_proposals_table.c.session_id == sid)
                        .where(composition_proposals_table.c.id == str(authority.row.id))
                        .where(composition_proposals_table.c.status == "pending")
                        .values(
                            status="rejected",
                            committed_state_id=None,
                            audit_event_id=event_id,
                            updated_at=now,
                        )
                    )
                    if updated.rowcount != 1:
                        raise AuditIntegrityError("state revert lost pending pipeline proposal authority")

                reverted_composer_meta = deep_thaw(target_record.composer_meta)
                if type(reverted_composer_meta) is dict:
                    reverted_composer_meta.pop(GUIDED_REPLAY_META_KEY, None)
                if target_guided is not None:
                    assert type(reverted_composer_meta) is dict
                    rewinds_to_topology = target_guided.terminal is None and (
                        target_guided.active_proposal is not None
                        or target_guided.step in {GuidedStep.STEP_3_TRANSFORMS, GuidedStep.STEP_4_WIRE}
                    )
                    restored_guided = target_guided
                    if rewinds_to_topology:
                        unanswered = [index for index, turn in enumerate(target_guided.history) if turn.response_hash is None]
                        if len(unanswered) > 1 or (unanswered and unanswered[0] != len(target_guided.history) - 1):
                            raise AuditIntegrityError("state revert guided topology rewind has malformed unanswered history")
                        history = target_guided.history
                        if unanswered:
                            final_turn = history[-1]
                            if final_turn.step not in {GuidedStep.STEP_3_TRANSFORMS, GuidedStep.STEP_4_WIRE}:
                                raise AuditIntegrityError("state revert guided topology rewind found a non-topology unanswered turn")
                            legal_turn_type = (
                                TurnType.PROPOSE_PIPELINE if final_turn.step is GuidedStep.STEP_3_TRANSFORMS else TurnType.CONFIRM_WIRING
                            )
                            if target_guided.active_proposal is not None and final_turn.turn_type is not legal_turn_type:
                                raise AuditIntegrityError("state revert guided proposal ref lacks its unanswered authority turn")
                            history = history[:-1]
                        restored_guided = replace(
                            target_guided,
                            step=GuidedStep.STEP_3_TRANSFORMS,
                            history=history,
                            terminal=None,
                            transition_consumed=False,
                            active_proposal=None,
                            active_edit_target=None,
                        )
                    reverted_composer_meta["guided_session"] = restored_guided.to_dict()

                new_state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(
                        data=CompositionStateData(
                            sources=self._unwrap_envelope(prior_row.sources),
                            nodes=self._unwrap_envelope(prior_row.nodes),
                            edges=self._unwrap_envelope(prior_row.edges),
                            outputs=self._unwrap_envelope(prior_row.outputs),
                            metadata_=self._unwrap_envelope(prior_row.metadata_),
                            is_valid=prior_row.is_valid,
                            validation_errors=prior_row.validation_errors,
                            composer_meta=reverted_composer_meta,
                        ),
                        derived_from_state_id=target_state_id,
                    ),
                    provenance="session_seed",
                    created_at=now,
                )
                state_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == new_state_id)).one()
                record = self._row_to_state_record(state_row)

                sequence_no = self._reserve_sequence_range(conn, sid, count=1)
                self._insert_chat_message(
                    conn,
                    session_id=sid,
                    role="system",
                    content=f"Pipeline reverted to version {prior_row.version}.",
                    raw_content=None,
                    tool_calls=None,
                    sequence_no=sequence_no,
                    writer_principal="route_system_message",
                    composition_state_id=None,
                    tool_call_id=None,
                    parent_assistant_id=None,
                    created_at=now,
                )
                conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

                response_hash = response_hash_factory(record)
                self.bind_guided_operation_on_connection(
                    conn,
                    fence,
                    result_state_id=record.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    fence,
                    result=GuidedCompositionStateResult(state_id=record.id),
                    response_hash=response_hash,
                    actor=actor,
                )
                return record

        return cast("CompositionStateRecord", await self._run_sync(_sync))

    @staticmethod
    def _require_guided_expected_current_state_on_connection(
        conn: Connection,
        *,
        session_id: str,
        expected_state_id: UUID | None,
        expected_state_version: int | None,
    ) -> None:
        """Fence a state-producing operation to the route-observed head.

        The comparison runs under the same session write lock and transaction
        as the state/message/settlement writes. An empty observation is encoded
        as ``(None, None)``; a present observation requires both exact UUID and
        positive version. No stale worker may attach a breadcrumb or locator to
        a different session head.
        """

        if (expected_state_id is None) != (expected_state_version is None):
            raise ValueError("expected guided current state id and version must be both present or both absent")
        if expected_state_id is not None and type(expected_state_id) is not UUID:
            raise ValueError("expected guided current state id must be a UUID")
        if expected_state_version is not None and (
            not isinstance(expected_state_version, int) or isinstance(expected_state_version, bool) or expected_state_version < 1
        ):
            raise ValueError("expected guided current state version must be a positive integer")
        current = conn.execute(
            select(composition_states_table.c.id, composition_states_table.c.version)
            .where(composition_states_table.c.session_id == session_id)
            .order_by(desc(composition_states_table.c.version))
            .limit(1)
        ).one_or_none()
        if expected_state_id is None:
            if current is not None:
                raise GuidedOperationSettlementConflictError()
            return
        if current is None or current.id != str(expected_state_id) or current.version != expected_state_version:
            raise GuidedOperationSettlementConflictError()

    @staticmethod
    def _prepare_guided_audit_cohort(
        *,
        audit_evidence: GuidedAuditEvidence,
        payloads: tuple[PreparedGuidedJsonPayload, ...],
        payload_store: PayloadStore | None,
    ) -> tuple[PreparedGuidedAuditRow, ...]:
        """Validate one CAS-bound evidence cohort before opening SQL."""

        if type(audit_evidence) is not GuidedAuditEvidence:
            raise TypeError("audit_evidence must be exact GuidedAuditEvidence")
        if type(payloads) is not tuple or any(type(payload) is not PreparedGuidedJsonPayload for payload in payloads):
            raise TypeError("payloads must be an exact prepared-payload tuple")
        audit_rows = prepare_guided_audit_rows(
            invocations=audit_evidence.invocations,
            llm_calls=audit_evidence.llm_calls,
            chat_turns=audit_evidence.chat_turns,
        )
        validate_guided_audit_payload_references(audit_rows, payloads)
        verify_guided_json_payloads(payload_store, payloads)
        return audit_rows

    def _insert_prepared_guided_audit_rows_on_connection(
        self,
        conn: Connection,
        *,
        session_id: str,
        composition_state_id: UUID | None,
        audit_rows: tuple[PreparedGuidedAuditRow, ...],
        sequence_no: int | None,
        created_at: datetime,
    ) -> tuple[ChatMessageRecord, ...]:
        """Insert a prevalidated guided evidence cohort in the caller's transaction."""

        self._assert_session_write_lock_held(
            conn,
            session_id,
            caller="_insert_prepared_guided_audit_rows_on_connection",
        )
        if audit_rows and sequence_no is None:
            raise AuditIntegrityError("Guided audit cohort has no reserved sequence")
        records: list[ChatMessageRecord] = []
        for audit_row in audit_rows:
            if sequence_no is None:  # pragma: no cover - guarded above
                raise AuditIntegrityError("Guided audit row has no reserved sequence")
            message_id = self._insert_chat_message(
                conn,
                session_id=session_id,
                role="audit",
                content=audit_row.content,
                raw_content=None,
                tool_calls=[deep_thaw(audit_row.envelope)],
                sequence_no=sequence_no,
                writer_principal="compose_loop",
                composition_state_id=str(composition_state_id) if composition_state_id is not None else None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=created_at,
            )
            records.append(
                ChatMessageRecord(
                    id=UUID(message_id),
                    session_id=UUID(session_id),
                    role="audit",
                    content=audit_row.content,
                    raw_content=None,
                    tool_calls=[deep_thaw(audit_row.envelope)],
                    created_at=created_at,
                    sequence_no=sequence_no,
                    composition_state_id=composition_state_id,
                    writer_principal="compose_loop",
                )
            )
            sequence_no += 1
        return tuple(records)

    async def seed_or_complete_guided_start_operation(
        self,
        fence: GuidedOperationFence,
        *,
        state: CompositionStateData,
        provenance: CompositionStateProvenance,
        actor: str,
        response_hash_factory: Callable[[CompositionStateRecord], str],
        payloads: tuple[PreparedGuidedJsonPayload, ...] = (),
        audit_evidence: GuidedAuditEvidence | None = None,
        originating_message: GuidedOriginatingUserMessageDraft | None = None,
        payload_store: PayloadStore | None = None,
    ) -> GuidedStartStateOutcome:
        """Atomically seed an empty session or settle its exact guided head.

        The response-hash callback is the guided-state validator for an
        existing head: it must fail closed when the record is freeform or
        otherwise cannot produce the strict start response. No generic
        integrity failure is interpreted as convergence.
        """

        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=audit_evidence if audit_evidence is not None else GuidedAuditEvidence(),
            payloads=payloads,
            payload_store=payload_store,
        )
        sid = str(fence.session_id)
        now = self._now()
        if originating_message is not None and type(originating_message) is not GuidedOriginatingUserMessageDraft:
            raise TypeError("originating_message must be an exact GuidedOriginatingUserMessageDraft or None")

        def _sync() -> GuidedStartStateOutcome:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                self.require_guided_operation_fence_on_connection(conn, fence)
                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if current_row is None:
                    state_id = self._insert_composition_state(
                        conn,
                        session_id=sid,
                        payload=StatePayload(data=state, derived_from_state_id=None),
                        provenance=provenance,
                        created_at=now,
                    )
                    inserted_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == state_id)).one()
                    record = self._row_to_state_record(inserted_row)
                    row_count = len(audit_rows) + (1 if originating_message is not None else 0)
                    if row_count:
                        sequence_no = self._reserve_sequence_range(conn, sid, count=row_count)
                        if originating_message is not None:
                            self._insert_chat_message(
                                conn,
                                session_id=sid,
                                role="user",
                                content=originating_message.content,
                                raw_content=None,
                                tool_calls=None,
                                sequence_no=sequence_no,
                                writer_principal="route_user_message",
                                composition_state_id=state_id,
                                tool_call_id=None,
                                parent_assistant_id=None,
                                created_at=now,
                                message_id=str(originating_message.message_id),
                            )
                            sequence_no += 1
                        self._insert_prepared_guided_audit_rows_on_connection(
                            conn,
                            session_id=sid,
                            composition_state_id=record.id,
                            audit_rows=audit_rows,
                            sequence_no=sequence_no,
                            created_at=now,
                        )
                        conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))
                    outcome: GuidedStartStateOutcome = GuidedStartStateSeeded(state=record)
                else:
                    record = self._row_to_state_record(current_row)
                    outcome = GuidedStartStateConverged(state=record)

                if originating_message is not None:
                    guided = state_from_record(record).guided_session
                    if guided is None or guided.root_intent_message_id != str(originating_message.message_id):
                        raise AuditIntegrityError("guided live start checkpoint does not bind its exact root intent")
                    message_row = conn.execute(
                        select(
                            chat_messages_table.c.session_id,
                            chat_messages_table.c.role,
                            chat_messages_table.c.content,
                            chat_messages_table.c.writer_principal,
                        ).where(chat_messages_table.c.id == str(originating_message.message_id))
                    ).one_or_none()
                    if (
                        message_row is None
                        or message_row.session_id != sid
                        or message_row.role != "user"
                        or message_row.content != originating_message.content
                        or message_row.writer_principal != "route_user_message"
                    ):
                        raise AuditIntegrityError("guided live start root intent row failed custody verification")

                response_hash = response_hash_factory(record)
                self.bind_guided_operation_on_connection(
                    conn,
                    fence,
                    originating_message_id=(originating_message.message_id if originating_message is not None else None),
                    result_state_id=record.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    fence,
                    result=GuidedCompositionStateResult(state_id=record.id),
                    response_hash=response_hash,
                    actor=actor,
                )
                return outcome

        return cast("GuidedStartStateOutcome", await self._run_sync(_sync))

    async def save_state_for_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        expected_current_state_id: UUID | None,
        expected_current_state_version: int | None,
        state: CompositionStateData,
        provenance: CompositionStateProvenance,
        actor: str,
        response_hash_factory: Callable[[CompositionStateRecord], str],
        system_message: str | None = None,
        payloads: tuple[PreparedGuidedJsonPayload, ...] = (),
        audit_evidence: GuidedAuditEvidence | None = None,
        payload_store: PayloadStore | None = None,
    ) -> CompositionStateRecord:
        """Persist one guided checkpoint and its replay settlement atomically."""

        if system_message is not None and (type(system_message) is not str or not system_message):
            raise ValueError("guided operation system_message must be a non-empty string or None")
        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=audit_evidence if audit_evidence is not None else GuidedAuditEvidence(),
            payloads=payloads,
            payload_store=payload_store,
        )
        sid = str(fence.session_id)
        now = self._now()

        def _sync() -> CompositionStateRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                self.require_guided_operation_fence_on_connection(conn, fence)
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=expected_current_state_id,
                    expected_state_version=expected_current_state_version,
                )
                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=state, derived_from_state_id=None),
                    provenance=provenance,
                    created_at=now,
                )
                state_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == state_id)).one()
                record = self._row_to_state_record(state_row)
                row_count = len(audit_rows) + (1 if system_message is not None else 0)
                sequence_no = self._reserve_sequence_range(conn, sid, count=row_count) if row_count else None
                if system_message is not None:
                    if sequence_no is None:
                        raise AuditIntegrityError("Guided system message has no reserved sequence")
                    self._insert_chat_message(
                        conn,
                        session_id=sid,
                        role="system",
                        content=system_message,
                        raw_content=None,
                        tool_calls=None,
                        sequence_no=sequence_no,
                        writer_principal="route_system_message",
                        composition_state_id=None,
                        tool_call_id=None,
                        parent_assistant_id=None,
                        created_at=now,
                    )
                    sequence_no += 1
                if audit_rows:
                    self._insert_prepared_guided_audit_rows_on_connection(
                        conn,
                        session_id=sid,
                        composition_state_id=record.id,
                        audit_rows=audit_rows,
                        sequence_no=sequence_no,
                        created_at=now,
                    )
                if row_count:
                    conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))
                response_hash = response_hash_factory(record)
                self.bind_guided_operation_on_connection(conn, fence, result_state_id=record.id)
                self.complete_guided_operation_on_connection(
                    conn,
                    fence,
                    result=GuidedCompositionStateResult(state_id=record.id),
                    response_hash=response_hash,
                    actor=actor,
                )
                return record

        return cast("CompositionStateRecord", await self._run_sync(_sync))

    async def settle_guided_state_operation(
        self,
        command: GuidedStateOperationCommand,
        *,
        payload_store: PayloadStore | None = None,
    ) -> GuidedStateOperationSettlement:
        """Commit one RESPOND/CHAT state and its evidence under one live fence."""

        if type(command) is not GuidedStateOperationCommand:
            raise TypeError("command must be an exact GuidedStateOperationCommand")
        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=command.audit_evidence,
            payloads=command.payloads,
            payload_store=payload_store,
        )
        prepared_state = with_guided_response_descriptor(command.state, command.response)
        sid = str(command.fence.session_id)
        now = self._now()
        interpretation_writers: list[Callable[[Connection], InterpretationEventRecord]] = []
        for draft in command.interpretations:
            writer = await self._prepare_or_create_pending_interpretation_event(
                session_id=command.fence.session_id,
                composition_state_id=command.state_id,
                affected_node_id=draft.affected_node_id,
                tool_call_id=draft.tool_call_id,
                user_term=draft.user_term,
                kind=draft.kind,
                llm_draft=draft.llm_draft,
                model_identifier=draft.model_identifier,
                model_version=draft.model_version,
                provider=draft.provider,
                composer_skill_hash=draft.composer_skill_hash,
                created_at=now,
                _event_id=draft.event_id,
                _prepare_only=True,
            )
            interpretation_writers.append(cast("Callable[[Connection], InterpretationEventRecord]", writer))

        def _sync() -> GuidedStateOperationSettlement:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation_row, _database_now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation_row["kind"] != command.response.kind:
                    raise AuditIntegrityError("Guided response descriptor kind does not match the reserved operation")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                current_record: CompositionStateRecord | None = None
                if current_row is not None:
                    current_record = self._row_to_state_record(current_row)
                if command.expected_current_content_hash is not None:
                    if current_row is None:  # pragma: no cover - expected-current guard owns absence
                        raise AuditIntegrityError("Guided operation expected content hash without a current state")
                    if current_record is None:  # pragma: no cover - paired with current_row
                        raise AuditIntegrityError("Guided operation current state conversion failed")
                    if composition_content_hash(state_from_record(current_record)) != command.expected_current_content_hash:
                        raise AuditIntegrityError("Guided operation current state content changed before settlement")

                from elspeth.web.composer.guided.errors import InvariantError
                from elspeth.web.composer.guided.state_machine import GuidedSession

                def _guided_checkpoint(composer_meta: object, *, role: str) -> GuidedSession:
                    metadata = deep_thaw(composer_meta)
                    if type(metadata) is not dict or type(metadata.get("guided_session")) is not dict:
                        raise AuditIntegrityError(f"{role} deferred intent state has no exact guided checkpoint")
                    try:
                        return GuidedSession.from_dict(metadata["guided_session"])
                    except (InvariantError, KeyError, TypeError, ValueError) as exc:
                        raise AuditIntegrityError(f"{role} deferred intent guided checkpoint is malformed") from exc

                candidate_guided = _guided_checkpoint(command.state.composer_meta, role="candidate")
                prior_guided = (
                    GuidedSession.initial()
                    if current_row is None
                    else _guided_checkpoint(self._row_to_state_record(current_row).composer_meta, role="prior")
                )
                retained_deferred_intent = _verify_guided_deferred_intent_mutation(
                    conn,
                    session_id=sid,
                    command=command,
                    prior_guided=prior_guided,
                    candidate_guided=candidate_guided,
                )

                invalidated_authority = _verify_guided_pending_proposal_invalidation(
                    conn,
                    context=_GuidedPendingProposalInvalidationContext(
                        service=self,
                        session_id=sid,
                        current_record=current_record,
                        prior_guided=prior_guided,
                        candidate_guided=candidate_guided,
                        expected_current_content_hash=command.expected_current_content_hash,
                    ),
                    invalidation=command.invalidated_pending_proposal,
                )

                inserted_state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(
                        data=prepared_state,
                        derived_from_state_id=(
                            str(command.expected_current_state_id) if command.expected_current_state_id is not None else None
                        ),
                    ),
                    provenance=command.provenance,
                    created_at=now,
                    state_id=str(command.state_id),
                )
                primary_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == inserted_state_id)).one()
                primary_state = self._row_to_state_record(primary_row)

                if invalidated_authority is not None:
                    _reject_guided_pending_proposal(
                        conn,
                        authority=invalidated_authority,
                        actor=command.actor,
                        created_at=now,
                    )

                row_count = len(audit_rows) + (1 if command.originating_message is not None else 0)
                sequence_no = self._reserve_sequence_range(conn, sid, count=row_count) if row_count else None
                originating_record: ChatMessageRecord | None = None
                if command.originating_message is not None:
                    if sequence_no is None:
                        raise AuditIntegrityError("Guided originating message has no reserved sequence")
                    originating = command.originating_message
                    existing_origin = conn.execute(
                        select(
                            chat_messages_table.c.session_id,
                            chat_messages_table.c.role,
                            chat_messages_table.c.content,
                        ).where(chat_messages_table.c.id == str(originating.message_id))
                    ).one_or_none()
                    if existing_origin is not None:
                        raise AuditIntegrityError("Guided originating message id already belongs to a persisted row")
                    self._insert_chat_message(
                        conn,
                        session_id=sid,
                        role="user",
                        content=originating.content,
                        raw_content=None,
                        tool_calls=None,
                        sequence_no=sequence_no,
                        writer_principal="route_user_message",
                        composition_state_id=inserted_state_id,
                        tool_call_id=None,
                        parent_assistant_id=None,
                        created_at=now,
                        message_id=str(originating.message_id),
                    )
                    if retained_deferred_intent is not None:
                        persisted_origin = conn.execute(
                            select(
                                chat_messages_table.c.session_id,
                                chat_messages_table.c.role,
                                chat_messages_table.c.content,
                            ).where(chat_messages_table.c.id == str(originating.message_id))
                        ).one_or_none()
                        if (
                            persisted_origin is None
                            or persisted_origin.session_id != sid
                            or persisted_origin.role != "user"
                            or stable_hash(persisted_origin.content) != retained_deferred_intent.message_content_hash
                        ):
                            raise AuditIntegrityError("retained deferred intent originating message failed session/role/content custody")
                    originating_record = ChatMessageRecord(
                        id=originating.message_id,
                        session_id=command.fence.session_id,
                        role="user",
                        content=originating.content,
                        raw_content=None,
                        tool_calls=None,
                        created_at=now,
                        sequence_no=sequence_no,
                        composition_state_id=primary_state.id,
                        writer_principal="route_user_message",
                    )
                    sequence_no += 1

                audit_records = self._insert_prepared_guided_audit_rows_on_connection(
                    conn,
                    session_id=sid,
                    composition_state_id=primary_state.id,
                    audit_rows=audit_rows,
                    sequence_no=sequence_no,
                    created_at=now,
                )

                if row_count:
                    conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

                interpretation_records = tuple(writer(conn) for writer in interpretation_writers)
                result_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one()
                result_state = self._row_to_state_record(result_row)
                response = project_guided_response(result_state, payloads=command.payloads)
                projected_json = response_json(response)
                response_hash = guided_response_projection_hash(response)
                self.bind_guided_operation_on_connection(
                    conn,
                    command.fence,
                    originating_message_id=(command.originating_message.message_id if command.originating_message is not None else None),
                    result_state_id=result_state.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedCompositionStateResult(
                        state_id=result_state.id,
                    ),
                    response_hash=response_hash,
                    actor=command.actor,
                )
                return GuidedStateOperationSettlement(
                    primary_state=primary_state,
                    result_state=result_state,
                    audit_messages=audit_records,
                    originating_message=originating_record,
                    interpretations=interpretation_records,
                    response_json=projected_json,
                    response_hash=response_hash,
                )

        return cast("GuidedStateOperationSettlement", await self._run_sync(_sync))

    async def stage_guided_full_pipeline_proposal(
        self,
        command: GuidedFullPipelineProposalStageCommand,
    ) -> GuidedFullPipelineProposalStageSettlement:
        """Publish one guided-full proposal and its replay locator atomically."""

        if type(command) is not GuidedFullPipelineProposalStageCommand:
            raise TypeError("command must be an exact GuidedFullPipelineProposalStageCommand")
        proposal = command.plan.proposal
        if proposal.surface is not PlannerSurface.GUIDED_FULL:
            raise AuditIntegrityError("guided-full stage requires a guided_full planner proposal")
        if type(proposal.base) is not PresentBase or proposal.base.state_id != command.checkpoint_state_id:
            raise AuditIntegrityError("guided-full proposal base must name its checkpoint")
        checkpoint_content_hash = _composition_state_data_content_hash(command.state)
        if proposal.base.composition_content_hash != checkpoint_content_hash:
            raise AuditIntegrityError("guided-full proposal base hash differs from its checkpoint")
        if command.expected_current_content_hash is not None and command.expected_current_content_hash != checkpoint_content_hash:
            raise AuditIntegrityError("guided-full checkpoint content differs from the observed composition head")
        if proposal.reviewed_anchor_hash != reviewed_anchor_hash({}):
            raise AuditIntegrityError("guided-full proposal carries reviewed-stage authority")
        if proposal.covered_deferred_intent_ids or proposal.supersedes_draft_hash is not None:
            raise AuditIntegrityError("guided-full proposal carries staged-only authority")
        expected_redacted = redact_tool_call_arguments(
            "set_pipeline",
            deep_thaw(proposal.pipeline),
            telemetry=NoopRedactionTelemetry(),
        )
        if deep_thaw(command.arguments_redacted_json) != expected_redacted:
            raise AuditIntegrityError("guided-full redacted arguments differ from the manifest projection")

        normalized = _normalize_proposal_composer_provenance(
            composer_model_identifier=command.plan.model_identifier,
            composer_model_version=command.plan.model_version,
            composer_provider=command.plan.provider,
            composer_skill_hash=proposal.skill_hash,
            tool_arguments_hash=stable_hash(proposal.pipeline),
        )
        assert all(value is not None for value in normalized.values())
        creation_payload = _pipeline_created_payload(
            plan=command.plan,
            user_message_id=command.originating_message.message_id,
            composer_model_identifier=cast(str, normalized["composer_model_identifier"]),
            composer_model_version=cast(str, normalized["composer_model_version"]),
            composer_provider=cast(str, normalized["composer_provider"]),
            summary=command.summary,
            rationale=command.rationale,
            affects=command.affects,
            arguments_redacted_json=command.arguments_redacted_json,
            supersedes_proposal_id=None,
        )
        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=command.audit_evidence,
            payloads=(),
            payload_store=None,
        )
        sid = str(command.fence.session_id)
        pid = str(command.proposal_id)
        event_id = str(uuid.uuid4())
        now = self._now()

        def _sync() -> GuidedFullPipelineProposalStageSettlement:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation_row, _database_now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation_row["kind"] != "guided_plan":
                    raise AuditIntegrityError("guided-full stage requires a guided_plan operation")
                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if command.expected_current_state_id is None:
                    if current_row is not None:
                        raise GuidedOperationSettlementConflictError()
                else:
                    if (
                        current_row is None
                        or current_row.id != str(command.expected_current_state_id)
                        or current_row.version != command.expected_current_state_version
                    ):
                        raise GuidedOperationSettlementConflictError()
                    current_record = self._row_to_state_record(current_row)
                    if composition_content_hash(state_from_record(current_record)) != command.expected_current_content_hash:
                        raise AuditIntegrityError("guided-full observed composition content changed before staging")

                validate_proposal_blob_references(
                    conn,
                    session_id=sid,
                    tool_name="set_pipeline",
                    arguments=deep_thaw(proposal.pipeline),
                )
                checkpoint_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(
                        data=command.state,
                        derived_from_state_id=(
                            str(command.expected_current_state_id) if command.expected_current_state_id is not None else None
                        ),
                    ),
                    provenance="convergence_persist",
                    created_at=now,
                    state_id=str(command.checkpoint_state_id),
                )
                checkpoint_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == checkpoint_id)).one()
                checkpoint = self._row_to_state_record(checkpoint_row)

                sequence_no = self._reserve_sequence_range(conn, sid, count=1 + len(audit_rows))
                self._insert_chat_message(
                    conn,
                    session_id=sid,
                    role="user",
                    content=command.originating_message.content,
                    raw_content=None,
                    tool_calls=None,
                    sequence_no=sequence_no,
                    writer_principal="route_user_message",
                    composition_state_id=checkpoint_id,
                    tool_call_id=None,
                    parent_assistant_id=None,
                    created_at=now,
                    message_id=str(command.originating_message.message_id),
                )
                originating_message = ChatMessageRecord(
                    id=command.originating_message.message_id,
                    session_id=command.fence.session_id,
                    role="user",
                    content=command.originating_message.content,
                    raw_content=None,
                    tool_calls=None,
                    created_at=now,
                    sequence_no=sequence_no,
                    composition_state_id=checkpoint.id,
                    writer_principal="route_user_message",
                )
                audit_messages = self._insert_prepared_guided_audit_rows_on_connection(
                    conn,
                    session_id=sid,
                    composition_state_id=checkpoint.id,
                    audit_rows=audit_rows,
                    sequence_no=sequence_no + 1,
                    created_at=now,
                )
                conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.created",
                        actor=command.actor,
                        payload=creation_payload,
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(composition_proposals_table).values(
                        id=pid,
                        session_id=sid,
                        tool_call_id=command.plan.tool_call_id,
                        user_message_id=str(command.originating_message.message_id),
                        composer_model_identifier=normalized["composer_model_identifier"],
                        composer_model_version=normalized["composer_model_version"],
                        composer_provider=normalized["composer_provider"],
                        composer_skill_hash=normalized["composer_skill_hash"],
                        tool_arguments_hash=normalized["tool_arguments_hash"],
                        tool_name="set_pipeline",
                        status="pending",
                        summary=command.summary,
                        rationale=command.rationale,
                        affects=list(command.affects),
                        arguments_json=deep_thaw(proposal.pipeline),
                        arguments_redacted_json=deep_thaw(command.arguments_redacted_json),
                        base_state_id=checkpoint_id,
                        committed_state_id=None,
                        audit_event_id=event_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                proposal_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                stored = _proposal_record_from_row(proposal_row)
                authority = AuthoritativePipelineProposal(
                    row=stored,
                    proposal=proposal,
                    creation_event_id=UUID(event_id),
                    custody_result=command.plan.custody_result,
                    supersedes_proposal_id=None,
                )
                proposal_record = replace(stored, pipeline_metadata=_pipeline_public_metadata(authority))
                response = project_composition_proposal(proposal_record)
                projected_json = response_json(response)
                response_hash = guided_response_projection_hash(response)
                self.bind_guided_operation_on_connection(
                    conn,
                    command.fence,
                    originating_message_id=command.originating_message.message_id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedPipelineProposalResult(
                        proposal_id=command.proposal_id,
                        checkpoint_state_id=checkpoint.id,
                    ),
                    response_hash=response_hash,
                    actor=command.actor,
                )
                return GuidedFullPipelineProposalStageSettlement(
                    checkpoint_state=checkpoint,
                    proposal=proposal_record,
                    originating_message=originating_message,
                    audit_messages=audit_messages,
                    response_json=projected_json,
                    response_hash=response_hash,
                )

        settlement = cast("GuidedFullPipelineProposalStageSettlement", await self._run_sync(_sync))
        _PIPELINE_PLANNER_COUNTER.add(1, {"surface": "guided_full", "result": "proposal_created"})
        _PIPELINE_CUSTODY_COUNTER.add(1, {"surface": "guided_full", "result": command.plan.custody_result})
        return settlement

    async def stage_guided_pipeline_proposal(
        self,
        command: GuidedPipelineProposalStageCommand,
        *,
        payload_store: PayloadStore | None = None,
    ) -> GuidedPipelineProposalStageSettlement:
        """Atomically publish one pending guided checkpoint and its authority."""

        if type(command) is not GuidedPipelineProposalStageCommand:
            raise TypeError("command must be an exact GuidedPipelineProposalStageCommand")
        if type(command.plan) is not PipelinePlanResult:
            raise TypeError("command.plan must be an exact PipelinePlanResult")
        proposal = command.plan.proposal
        if proposal.surface not in {PlannerSurface.GUIDED_FULL, PlannerSurface.GUIDED_STAGED, PlannerSurface.TUTORIAL_PROFILE}:
            raise AuditIntegrityError("guided proposal stage requires a guided planner surface")
        expected_redacted = redact_tool_call_arguments(
            "set_pipeline",
            deep_thaw(proposal.pipeline),
            telemetry=NoopRedactionTelemetry(),
        )
        if deep_thaw(command.arguments_redacted_json) != expected_redacted:
            raise AuditIntegrityError("guided pipeline redacted arguments differ from the manifest projection")
        if type(proposal.base) is not PresentBase:
            raise AuditIntegrityError("guided pipeline proposal must name its future checkpoint base")
        if proposal.base.state_id != command.checkpoint_state_id:
            raise AuditIntegrityError("guided pipeline proposal base does not name the staged checkpoint")
        checkpoint_content_hash = _composition_state_data_content_hash(command.state)
        if proposal.base.composition_content_hash != checkpoint_content_hash:
            raise AuditIntegrityError("guided pipeline proposal base hash does not bind the staged checkpoint")
        if checkpoint_content_hash != command.expected_current_content_hash:
            raise AuditIntegrityError("guided proposal checkpoint unexpectedly changes authored composition content")
        if (command.supersedes_proposal_id is None) != (proposal.supersedes_draft_hash is None):
            raise AuditIntegrityError("guided proposal supersession id/hash binding is incomplete")

        metadata = deep_thaw(command.state.composer_meta)
        if type(metadata) is not dict or set(metadata) != {"guided_session"}:
            raise AuditIntegrityError("guided proposal checkpoint metadata is malformed")
        from elspeth.web.composer.guided.planning import (
            guided_candidate_state,
            guided_private_reviewed_facts,
            verified_remaining_deferred_intents,
            verify_guided_proposal_projection,
        )
        from elspeth.web.composer.guided.protocol import GuidedStep, Turn, TurnType, validate_current_turn
        from elspeth.web.composer.guided.state_machine import GuidedSession

        guided = GuidedSession.from_dict(metadata["guided_session"])
        checkpoint_reviewed_facts = guided_private_reviewed_facts(guided)
        if reviewed_anchor_hash(checkpoint_reviewed_facts) != proposal.reviewed_anchor_hash:
            raise AuditIntegrityError("guided proposal checkpoint reviewed authority differs from the immutable proposal")
        active = guided.active_proposal
        if (
            active is None
            or active.proposal_id != command.proposal_id
            or active.draft_hash != proposal.draft_hash
            or active.base != proposal.base
            or active.reviewed_anchor_hash != proposal.reviewed_anchor_hash
            or active.covered_deferred_intent_ids != proposal.covered_deferred_intent_ids
            or active.creation_event_schema != _PIPELINE_CREATED_SCHEMA
            or active.supersedes_proposal_id != command.supersedes_proposal_id
            or active.supersedes_draft_hash != proposal.supersedes_draft_hash
        ):
            raise AuditIntegrityError("guided proposal reference does not match private proposal authority")
        next_turn = command.response.next_turn
        if next_turn is None:  # pragma: no cover - command type guards this
            raise AuditIntegrityError("guided proposal stage response lost its proposal turn")
        turn_payloads = [
            payload for payload in command.payloads if payload.payload_id == next_turn.payload_id and payload.purpose == "turn"
        ]
        if len(turn_payloads) != 1:
            raise AuditIntegrityError("guided proposal stage requires one exact durable response turn")
        if not guided.history:
            raise AuditIntegrityError("guided proposal checkpoint has no durable proposal turn")
        durable_turn = guided.history[-1]
        expected_step = GuidedStep.STEP_3_TRANSFORMS if next_turn.turn_type is TurnType.PROPOSE_PIPELINE else GuidedStep.STEP_4_WIRE
        expected_step_index = 2 if expected_step is GuidedStep.STEP_3_TRANSFORMS else 3
        if (
            guided.step is not expected_step
            or durable_turn.step is not expected_step
            or durable_turn.turn_type is not next_turn.turn_type
            or durable_turn.payload_hash != next_turn.payload_id
            or durable_turn.response_hash is not None
            or durable_turn.emitter != "server"
            or next_turn.step_index != expected_step_index
        ):
            raise AuditIntegrityError("guided proposal checkpoint turn differs from the durable response")
        durable_payload_json = cast(Mapping[str, Any], deep_thaw(turn_payloads[0].payload))
        proposal_projection_json = cast(Mapping[str, Any], deep_thaw(command.proposal_projection))
        verify_guided_proposal_projection(
            payload=proposal_projection_json,
            proposal_id=command.proposal_id,
            proposal=proposal,
            guided=guided,
            catalog_plugin_ids=command.catalog_plugin_ids,
        )
        if next_turn.turn_type is TurnType.PROPOSE_PIPELINE:
            if durable_payload_json != proposal_projection_json:
                raise AuditIntegrityError("guided proposal turn differs from its verified public projection")
        else:
            from elspeth.web.composer.guided.emitters import build_step_4_wire_turn

            candidate = guided_candidate_state(proposal)
            validation_summary = candidate.validate()
            expected_wire = build_step_4_wire_turn(
                candidate,
                proposal_projection=cast("Any", proposal_projection_json),
                guided=guided,
                catalog=None,
                validation_state=candidate,
                validation_summary=validation_summary,
            )
            try:
                validate_current_turn(
                    GuidedStep.STEP_4_WIRE,
                    Turn(type=TurnType.CONFIRM_WIRING.value, step_index=3, payload=cast("Any", durable_payload_json)),
                )
            except ValueError as exc:
                raise AuditIntegrityError("guided correction produced an invalid wire-review turn") from exc
            for key in (
                "proposal_id",
                "draft_hash",
                "sources",
                "nodes",
                "outputs",
                "connections",
                "semantic_contracts",
            ):
                if durable_payload_json.get(key) != expected_wire["payload"].get(key):
                    raise AuditIntegrityError(f"guided correction wire projection differs at {key}")
        verified_remaining_deferred_intents(
            guided=guided,
            proposal=proposal,
        )
        if command.originating_message is not None:
            if not guided.correction_messages:
                raise AuditIntegrityError("guided correction checkpoint lost its private message reference")
            correction_ref = guided.correction_messages[-1]
            if correction_ref.message_id != command.originating_message.message_id or correction_ref.content_hash != stable_hash(
                command.originating_message.content
            ):
                raise AuditIntegrityError("guided correction checkpoint does not bind its originating message")

        normalized = _normalize_proposal_composer_provenance(
            composer_model_identifier=command.plan.model_identifier,
            composer_model_version=command.plan.model_version,
            composer_provider=command.plan.provider,
            composer_skill_hash=proposal.skill_hash,
            tool_arguments_hash=stable_hash(proposal.pipeline),
        )
        assert all(value is not None for value in normalized.values())
        creation_payload = _pipeline_created_payload(
            plan=command.plan,
            user_message_id=command.user_message_id,
            composer_model_identifier=cast(str, normalized["composer_model_identifier"]),
            composer_model_version=cast(str, normalized["composer_model_version"]),
            composer_provider=cast(str, normalized["composer_provider"]),
            summary=command.summary,
            rationale=command.rationale,
            affects=command.affects,
            arguments_redacted_json=command.arguments_redacted_json,
            supersedes_proposal_id=command.supersedes_proposal_id,
        )
        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=command.audit_evidence,
            payloads=command.payloads,
            payload_store=payload_store,
        )
        prepared_state = with_guided_response_descriptor(command.state, command.response)
        sid = str(command.fence.session_id)
        pid = str(command.proposal_id)
        event_id = str(uuid.uuid4())
        now = self._now()

        def _sync() -> GuidedPipelineProposalStageSettlement:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation_row, _database_now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation_row["kind"] != "guided_respond":
                    raise AuditIntegrityError("guided proposal stage requires a guided_respond operation")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one()
                current_record = self._row_to_state_record(current_row)
                if composition_content_hash(state_from_record(current_record)) != command.expected_current_content_hash:
                    raise AuditIntegrityError("Guided operation current state content changed before settlement")
                current_guided = state_from_record(current_record).guided_session
                if current_guided is None:
                    raise AuditIntegrityError("guided proposal predecessor lost its guided checkpoint")
                if current_guided.deferred_intents != guided.deferred_intents:
                    raise AuditIntegrityError("guided proposal deferred authority changed before staging")
                _verify_guided_deferred_message_authority(
                    conn,
                    session_id=sid,
                    guided=current_guided,
                )
                _verify_guided_correction_message_authority(
                    conn,
                    session_id=sid,
                    guided=current_guided,
                )
                _verify_guided_root_message_authority(
                    conn,
                    service=self,
                    session_id=sid,
                    guided=current_guided,
                )
                verified_remaining_deferred_intents(
                    guided=current_guided,
                    proposal=proposal,
                )

                if command.user_message_id is not None and command.originating_message is None:
                    message_row = conn.execute(
                        select(chat_messages_table.c.role, chat_messages_table.c.content)
                        .where(chat_messages_table.c.session_id == sid)
                        .where(chat_messages_table.c.id == str(command.user_message_id))
                    ).one_or_none()
                    if message_row is None or message_row.role != "user":
                        raise AuditIntegrityError("guided proposal originating message is missing or cross-session")
                    if stable_hash(message_row.content) != command.user_message_content_hash:
                        raise AuditIntegrityError("guided proposal originating message content changed")

                if command.originating_message is None:
                    if guided.correction_messages != current_guided.correction_messages:
                        raise AuditIntegrityError("guided proposal staging changed private correction custody")
                else:
                    expected_corrections = (
                        *current_guided.correction_messages,
                        guided.correction_messages[-1],
                    )
                    if guided.correction_messages != expected_corrections:
                        raise AuditIntegrityError("guided correction staging did not append exactly one custody reference")
                    existing_correction = conn.execute(
                        select(chat_messages_table.c.id).where(chat_messages_table.c.id == str(command.originating_message.message_id))
                    ).one_or_none()
                    if existing_correction is not None:
                        raise AuditIntegrityError("guided correction message id already belongs to a persisted row")

                if command.supersedes_proposal_id is not None:
                    if current_guided is None or current_guided.active_proposal is None:
                        raise AuditIntegrityError("guided proposal revision predecessor reference is missing")
                    if guided_private_reviewed_facts(current_guided) != checkpoint_reviewed_facts:
                        raise AuditIntegrityError("guided proposal revision reviewed authority changed from its predecessor")
                    if current_guided.deferred_intents != guided.deferred_intents:
                        raise AuditIntegrityError("guided proposal revision deferred authority changed from its predecessor")
                    predecessor_ref = current_guided.active_proposal
                    if (
                        predecessor_ref.proposal_id != command.supersedes_proposal_id
                        or predecessor_ref.draft_hash != proposal.supersedes_draft_hash
                    ):
                        raise AuditIntegrityError("guided proposal revision predecessor reference drifted")
                    current_reviewed_facts = guided_private_reviewed_facts(current_guided)
                    if reviewed_anchor_hash(current_reviewed_facts) != proposal.reviewed_anchor_hash:
                        raise AuditIntegrityError("guided proposal revision reviewed authority drifted")
                    superseded = conn.execute(
                        select(composition_proposals_table)
                        .where(composition_proposals_table.c.session_id == sid)
                        .where(composition_proposals_table.c.id == str(command.supersedes_proposal_id))
                    ).one_or_none()
                    if superseded is None:
                        raise AuditIntegrityError("guided proposal revision predecessor is missing or cross-session")
                    superseded_record = _proposal_record_from_row(superseded)
                    superseded_created_rows = conn.execute(
                        select(proposal_events_table)
                        .where(proposal_events_table.c.session_id == sid)
                        .where(proposal_events_table.c.proposal_id == str(command.supersedes_proposal_id))
                        .where(proposal_events_table.c.event_type == "proposal.created")
                    ).fetchall()
                    if len(superseded_created_rows) != 1:
                        raise AuditIntegrityError("guided proposal supersession draft authority is malformed")
                    superseded_authority = _restore_authoritative_pipeline_proposal(
                        conn=conn,
                        row=superseded_record,
                        creation_event=_proposal_event_record_from_row(superseded_created_rows[0]),
                        reviewed_facts=current_reviewed_facts,
                    )
                    _verify_pipeline_lifecycle_authority(conn, service=self, authority=superseded_authority)
                    if (
                        superseded_authority.row.status != "pending"
                        or superseded_authority.proposal.draft_hash != proposal.supersedes_draft_hash
                    ):
                        raise AuditIntegrityError("guided proposal revision predecessor is no longer pending")
                    _require_no_active_guided_confirmation_admission(
                        conn,
                        session_id=sid,
                        proposal_id=str(command.supersedes_proposal_id),
                        now=now,
                    )
                    superseded_event_id = str(uuid.uuid4())
                    conn.execute(
                        insert(proposal_events_table).values(
                            id=superseded_event_id,
                            session_id=sid,
                            proposal_id=str(command.supersedes_proposal_id),
                            event_type="proposal.rejected",
                            actor=command.actor,
                            payload=_pipeline_rejected_payload(
                                authority=superseded_authority,
                                reason="superseded",
                                dispatch=None,
                            ),
                            created_at=now,
                        )
                    )
                    updated_predecessor = conn.execute(
                        update(composition_proposals_table)
                        .where(composition_proposals_table.c.session_id == sid)
                        .where(composition_proposals_table.c.id == str(command.supersedes_proposal_id))
                        .where(composition_proposals_table.c.status == "pending")
                        .values(
                            status="rejected",
                            committed_state_id=None,
                            audit_event_id=superseded_event_id,
                            updated_at=now,
                        )
                    )
                    if updated_predecessor.rowcount != 1:
                        raise AuditIntegrityError("guided proposal revision predecessor changed during settlement")

                validate_proposal_blob_references(
                    conn,
                    session_id=sid,
                    tool_name="set_pipeline",
                    arguments=deep_thaw(proposal.pipeline),
                )
                checkpoint_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=prepared_state, derived_from_state_id=str(command.expected_current_state_id)),
                    provenance="convergence_persist",
                    created_at=now,
                    state_id=str(command.checkpoint_state_id),
                )
                correction_sequence_no: int | None = None
                if command.originating_message is not None:
                    correction_sequence_no = self._reserve_sequence_range(conn, sid, count=1 + len(audit_rows))
                    self._insert_chat_message(
                        conn,
                        session_id=sid,
                        role="user",
                        content=command.originating_message.content,
                        raw_content=None,
                        tool_calls=None,
                        sequence_no=correction_sequence_no,
                        writer_principal="route_user_message",
                        composition_state_id=checkpoint_id,
                        tool_call_id=None,
                        parent_assistant_id=None,
                        created_at=now,
                        message_id=str(command.originating_message.message_id),
                    )
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.created",
                        actor=command.actor,
                        payload=creation_payload,
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(composition_proposals_table).values(
                        id=pid,
                        session_id=sid,
                        tool_call_id=command.plan.tool_call_id,
                        user_message_id=str(command.user_message_id) if command.user_message_id is not None else None,
                        composer_model_identifier=normalized["composer_model_identifier"],
                        composer_model_version=normalized["composer_model_version"],
                        composer_provider=normalized["composer_provider"],
                        composer_skill_hash=normalized["composer_skill_hash"],
                        tool_arguments_hash=normalized["tool_arguments_hash"],
                        tool_name="set_pipeline",
                        status="pending",
                        summary=command.summary,
                        rationale=command.rationale,
                        affects=list(command.affects),
                        arguments_json=deep_thaw(proposal.pipeline),
                        arguments_redacted_json=deep_thaw(command.arguments_redacted_json),
                        base_state_id=checkpoint_id,
                        committed_state_id=None,
                        audit_event_id=event_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                state_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == checkpoint_id)).one()
                result_state = self._row_to_state_record(state_row)
                proposal_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                record = _proposal_record_from_row(proposal_row)
                authority = AuthoritativePipelineProposal(
                    row=record,
                    proposal=proposal,
                    creation_event_id=UUID(event_id),
                    custody_result=command.plan.custody_result,
                    supersedes_proposal_id=command.supersedes_proposal_id,
                )
                proposal_record = replace(record, pipeline_metadata=_pipeline_public_metadata(authority))

                sequence_no = (
                    correction_sequence_no + 1
                    if correction_sequence_no is not None
                    else (self._reserve_sequence_range(conn, sid, count=len(audit_rows)) if audit_rows else None)
                )
                audit_messages = self._insert_prepared_guided_audit_rows_on_connection(
                    conn,
                    session_id=sid,
                    composition_state_id=result_state.id,
                    audit_rows=audit_rows,
                    sequence_no=sequence_no,
                    created_at=now,
                )
                if audit_rows or command.originating_message is not None:
                    conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

                response = project_guided_response(result_state, payloads=command.payloads)
                projected_json = response_json(response)
                response_hash = guided_response_projection_hash(response)
                self.bind_guided_operation_on_connection(
                    conn,
                    command.fence,
                    originating_message_id=(command.originating_message.message_id if command.originating_message is not None else None),
                    proposal_id=command.proposal_id,
                    result_state_id=result_state.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedCompositionStateResult(
                        state_id=result_state.id,
                        proposal_id=command.proposal_id,
                    ),
                    response_hash=response_hash,
                    actor=command.actor,
                )
                return GuidedPipelineProposalStageSettlement(
                    result_state=result_state,
                    proposal=proposal_record,
                    audit_messages=audit_messages,
                    response_json=projected_json,
                    response_hash=response_hash,
                )

        settlement = cast("GuidedPipelineProposalStageSettlement", await self._run_sync(_sync))
        _PIPELINE_PLANNER_COUNTER.add(1, {"surface": proposal.surface.value, "result": "proposal_created"})
        _PIPELINE_CUSTODY_COUNTER.add(1, {"surface": proposal.surface.value, "result": command.plan.custody_result})
        return settlement

    async def reconcile_rejected_guided_pipeline_proposal(
        self,
        *,
        session_id: UUID,
        expected_current_state_id: UUID,
        proposal_id: UUID,
        draft_hash: str,
        reviewed_facts: Mapping[str, Any],
    ) -> CompositionStateRecord:
        """Clear an explicitly terminal guided reference under one DB lock."""

        sid = str(session_id)
        pid = str(proposal_id)
        now = self._now()

        def _sync() -> CompositionStateRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                current_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).one_or_none()
                if current_row is None or current_row.id != str(expected_current_state_id):
                    raise AuditIntegrityError("guided rejected-proposal reconciliation current state changed")
                proposal_row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if proposal_row is None:
                    raise AuditIntegrityError("guided rejected-proposal authority is missing or cross-session")
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("guided rejected proposal must have one creation event")
                authority = _restore_authoritative_pipeline_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(proposal_row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=reviewed_facts,
                )
                _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                if authority.row.status != "rejected" or authority.proposal.draft_hash != draft_hash:
                    raise AuditIntegrityError("guided rejected-proposal authority is not the expected terminal draft")
                terminal_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.rejected")
                ).fetchall()
                if len(terminal_rows) != 1 or terminal_rows[0].payload.get("reason_code") not in {
                    "operator_rejected",
                    "superseded",
                }:
                    raise AuditIntegrityError("guided proposal terminal reason is not reconcilable")

                current_record = self._row_to_state_record(current_row)
                current_state = state_from_record(current_record)
                guided = current_state.guided_session
                if guided is None or guided.active_proposal is None:
                    raise AuditIntegrityError("guided rejected-proposal reference is missing")
                active = guided.active_proposal
                if (
                    active.proposal_id != proposal_id
                    or active.draft_hash != draft_hash
                    or active.base != authority.proposal.base
                    or active.reviewed_anchor_hash != authority.proposal.reviewed_anchor_hash
                    or active.covered_deferred_intent_ids != authority.proposal.covered_deferred_intent_ids
                ):
                    raise AuditIntegrityError("guided rejected-proposal reference differs from authority")
                from elspeth.web.composer.guided.protocol import TurnType
                from elspeth.web.composer.guided.state_machine import GuidedSession

                if (
                    not guided.history
                    or guided.history[-1].turn_type is not TurnType.PROPOSE_PIPELINE
                    or guided.history[-1].response_hash is not None
                ):
                    raise AuditIntegrityError("guided rejected proposal has no active proposal occurrence")

                cleared = replace(
                    guided,
                    history=guided.history[:-1],
                    active_proposal=None,
                    active_edit_target=None,
                )
                if type(cleared) is not GuidedSession:  # pragma: no cover - replace contract
                    raise AuditIntegrityError("guided rejected-proposal reconciliation lost checkpoint type")
                state_dict = current_state.to_dict()
                data = CompositionStateData(
                    sources=state_dict["sources"],
                    nodes=state_dict["nodes"],
                    edges=state_dict["edges"],
                    outputs=state_dict["outputs"],
                    metadata_=state_dict["metadata"],
                    is_valid=current_record.is_valid,
                    validation_errors=current_record.validation_errors,
                    composer_meta={"guided_session": cleared.to_dict()},
                )
                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=data, derived_from_state_id=str(current_record.id)),
                    provenance="convergence_persist",
                    created_at=now,
                )
                row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == state_id)).one()
                return self._row_to_state_record(row)

        return cast("CompositionStateRecord", await self._run_sync(_sync))

    async def back_edit_guided_pipeline_proposal(
        self,
        command: GuidedPipelineProposalBackEditCommand,
        *,
        payload_store: PayloadStore | None = None,
    ) -> GuidedPipelineProposalStageSettlement:
        """Atomically supersede one proposal and rewind to a component edit."""

        if type(command) is not GuidedPipelineProposalBackEditCommand:
            raise TypeError("command must be an exact GuidedPipelineProposalBackEditCommand")

        from elspeth.web.composer.guided.protocol import BLOB_REF_PATH_PREFIX, GuidedStep, Turn, TurnType, validate_current_turn
        from elspeth.web.composer.guided.state_machine import GuidedSession, TurnRecord

        payloads_by_purpose = {payload.purpose: payload for payload in command.payloads}
        response_payload = payloads_by_purpose["turn_response"]
        turn_payload = payloads_by_purpose["turn"]
        expected_response_payload = {
            "action": "revise",
            "proposal_id": str(command.proposal_id),
            "draft_hash": command.draft_hash,
            "edit_target": command.edit_target.to_dict(),
        }
        if deep_thaw(response_payload.payload) != expected_response_payload:
            raise AuditIntegrityError("guided back-edit response payload differs from its exact authority")
        next_turn = command.response.next_turn
        if next_turn is None:  # pragma: no cover - command guards this
            raise AuditIntegrityError("guided back-edit response lost its edit form")
        if next_turn.payload_id != turn_payload.payload_id:
            raise AuditIntegrityError("guided back-edit response does not bind its edit form payload")
        target_step = GuidedStep.STEP_1_SOURCE if command.edit_target.kind == "source" else GuidedStep.STEP_2_SINK
        try:
            validate_current_turn(
                target_step,
                Turn(
                    type=TurnType.SCHEMA_FORM.value,
                    step_index=next_turn.step_index,
                    payload=cast("Any", deep_thaw(turn_payload.payload)),
                ),
            )
        except ValueError as exc:
            raise AuditIntegrityError("guided back-edit form payload is malformed") from exc

        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=command.audit_evidence,
            payloads=command.payloads,
            payload_store=payload_store,
        )
        prepared_state = with_guided_response_descriptor(command.state, command.response)
        sid = str(command.fence.session_id)
        pid = str(command.proposal_id)
        now = self._now()

        def _sync() -> GuidedPipelineProposalStageSettlement:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation_row, _ = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation_row["kind"] != "guided_respond":
                    raise AuditIntegrityError("guided proposal back-edit requires guided_respond")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                current_row = conn.execute(
                    select(composition_states_table).where(
                        composition_states_table.c.session_id == sid,
                        composition_states_table.c.id == str(command.expected_current_state_id),
                    )
                ).one()
                current_record = self._row_to_state_record(current_row)
                current_state = state_from_record(current_record)
                current_content_hash = composition_content_hash(current_state)
                if current_content_hash != command.expected_current_content_hash:
                    raise AuditIntegrityError("guided back-edit current composition content changed")

                proposal_row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if proposal_row is None:
                    raise AuditIntegrityError("guided back-edit proposal authority is missing or cross-session")
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("guided back-edit proposal must have one creation event")
                authority = _restore_authoritative_pipeline_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(proposal_row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=deep_thaw(command.reviewed_facts),
                )
                _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                if authority.row.status != "pending" or authority.proposal.draft_hash != command.draft_hash:
                    raise AuditIntegrityError("guided back-edit does not name the active pending draft")
                if type(authority.proposal.base) is not PresentBase or authority.proposal.base.state_id != current_record.id:
                    raise AuditIntegrityError("guided back-edit proposal base differs from current checkpoint")
                if authority.proposal.base.composition_content_hash != current_content_hash:
                    raise AuditIntegrityError("guided back-edit proposal base content hash changed")

                guided = current_state.guided_session
                if guided is None or guided.active_proposal is None:
                    raise AuditIntegrityError("guided back-edit current checkpoint has no active proposal")
                active = guided.active_proposal
                if (
                    active.proposal_id != command.proposal_id
                    or active.draft_hash != authority.proposal.draft_hash
                    or active.base != authority.proposal.base
                    or active.reviewed_anchor_hash != authority.proposal.reviewed_anchor_hash
                    or active.covered_deferred_intent_ids != authority.proposal.covered_deferred_intent_ids
                    or active.creation_event_schema != "pipeline_proposal_created.v1"
                    or active.supersedes_proposal_id != authority.supersedes_proposal_id
                    or active.supersedes_draft_hash != authority.proposal.supersedes_draft_hash
                ):
                    raise AuditIntegrityError("guided back-edit active proposal reference differs from authority")
                turn_payload_json = deep_thaw(turn_payload.payload)
                if command.edit_target.kind == "source":
                    source_target = guided.reviewed_sources.get(command.edit_target.stable_id)
                    if source_target is None:
                        raise AuditIntegrityError("guided back-edit target is not an exact reviewed component")
                    expected_plugin = source_target.plugin
                    expected_prefill = {"schema": {"mode": "observed"}, **dict(deep_thaw(source_target.options))}
                    blob_ref = source_target.options.get("blob_ref")
                    if blob_ref is not None and type(expected_prefill.get("path")) is str:
                        expected_prefill["path"] = f"{BLOB_REF_PATH_PREFIX}{blob_ref}"
                    expected_prefill["on_validation_failure"] = source_target.on_validation_failure
                else:
                    output_target = guided.reviewed_outputs.get(command.edit_target.stable_id)
                    if output_target is None:
                        raise AuditIntegrityError("guided back-edit target is not an exact reviewed component")
                    expected_plugin = output_target.plugin
                    expected_prefill = {"schema": {"mode": "observed"}, **dict(deep_thaw(output_target.options))}
                    expected_prefill["on_write_failure"] = output_target.on_write_failure
                if turn_payload_json["plugin"] != expected_plugin or deep_thaw(turn_payload_json["prefilled"]) != expected_prefill:
                    raise AuditIntegrityError("guided back-edit form differs from server-held reviewed custody")
                if (
                    guided.step is not GuidedStep.STEP_3_TRANSFORMS
                    or not guided.history
                    or guided.history[-1].step is not GuidedStep.STEP_3_TRANSFORMS
                    or guided.history[-1].turn_type is not TurnType.PROPOSE_PIPELINE
                    or guided.history[-1].response_hash is not None
                    or any(record.response_hash is None for record in guided.history[:-1])
                ):
                    raise AuditIntegrityError("guided back-edit has no exact active proposal occurrence")

                metadata = deep_thaw(command.state.composer_meta)
                if type(metadata) is not dict or set(metadata) != {"guided_session"}:
                    raise AuditIntegrityError("guided back-edit candidate metadata is malformed")
                try:
                    candidate_guided = GuidedSession.from_dict(metadata["guided_session"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise AuditIntegrityError("guided back-edit candidate checkpoint is malformed") from exc
                answered = replace(
                    guided.history[-1],
                    response_hash=response_payload.payload_id,
                    summary="Guided pipeline proposal revision requested.",
                )
                emitted = TurnRecord(
                    step=target_step,
                    turn_type=TurnType.SCHEMA_FORM,
                    payload_hash=turn_payload.payload_id,
                    response_hash=None,
                    emitter="server",
                )
                expected_guided = replace(
                    guided,
                    step=target_step,
                    history=(*guided.history[:-1], answered, emitted),
                    active_proposal=None,
                    active_edit_target=command.edit_target,
                )
                if candidate_guided != expected_guided:
                    raise AuditIntegrityError("guided back-edit candidate differs from the exact rewind transition")
                if (
                    _composition_state_data_content_hash(command.state) != command.expected_current_content_hash
                    or command.state.is_valid is not current_record.is_valid
                    or deep_thaw(command.state.validation_errors) != deep_thaw(current_record.validation_errors)
                ):
                    raise AuditIntegrityError("guided back-edit candidate changed authored composition or validation authority")

                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=prepared_state, derived_from_state_id=str(current_record.id)),
                    provenance="convergence_persist",
                    created_at=now,
                )
                event_id = str(uuid.uuid4())
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.rejected",
                        actor=command.actor,
                        payload=_pipeline_rejected_payload(
                            authority=authority,
                            reason="superseded",
                            dispatch=None,
                        ),
                        created_at=now,
                    )
                )
                updated = conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.status == "pending")
                    .values(
                        status="rejected",
                        committed_state_id=None,
                        audit_event_id=event_id,
                        updated_at=now,
                    )
                )
                if updated.rowcount != 1:
                    raise AuditIntegrityError("guided back-edit proposal changed during settlement")

                result_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == state_id)).one()
                result_state = self._row_to_state_record(result_row)
                sequence_no = self._reserve_sequence_range(conn, sid, count=len(audit_rows)) if audit_rows else None
                audit_messages = self._insert_prepared_guided_audit_rows_on_connection(
                    conn,
                    session_id=sid,
                    composition_state_id=result_state.id,
                    audit_rows=audit_rows,
                    sequence_no=sequence_no,
                    created_at=now,
                )
                if audit_rows:
                    conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

                response = project_guided_response(result_state, payloads=command.payloads)
                projected_json = response_json(response)
                response_hash = guided_response_projection_hash(response)
                self.bind_guided_operation_on_connection(
                    conn,
                    command.fence,
                    proposal_id=command.proposal_id,
                    result_state_id=result_state.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedCompositionStateResult(state_id=result_state.id, proposal_id=command.proposal_id),
                    response_hash=response_hash,
                    actor=command.actor,
                )
                updated_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                proposal_record = replace(
                    _proposal_record_from_row(updated_row),
                    pipeline_metadata=_pipeline_public_metadata(authority),
                )
                return GuidedPipelineProposalStageSettlement(
                    result_state=result_state,
                    proposal=proposal_record,
                    audit_messages=audit_messages,
                    response_json=projected_json,
                    response_hash=response_hash,
                )

        settlement = cast("GuidedPipelineProposalStageSettlement", await self._run_sync(_sync))
        _PIPELINE_SETTLEMENT_COUNTER.add(1, {"surface": "guided_staged", "result": "superseded_for_component_edit"})
        return settlement

    async def reject_guided_pipeline_proposal(
        self,
        command: GuidedPipelineProposalRejectCommand,
    ) -> GuidedPipelineProposalStageSettlement:
        """Atomically reject one pending guided proposal and clear its ref."""

        if type(command) is not GuidedPipelineProposalRejectCommand:
            raise TypeError("command must be an exact GuidedPipelineProposalRejectCommand")
        sid = str(command.fence.session_id)
        pid = str(command.proposal_id)
        now = self._now()

        def _sync() -> GuidedPipelineProposalStageSettlement:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation_row, _ = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation_row["kind"] != "guided_respond":
                    raise AuditIntegrityError("guided proposal rejection requires guided_respond")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                current_row = conn.execute(
                    select(composition_states_table).where(composition_states_table.c.id == str(command.expected_current_state_id))
                ).one()
                current_record = self._row_to_state_record(current_row)
                proposal_row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if proposal_row is None:
                    raise AuditIntegrityError("guided proposal rejection authority is missing")
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("guided proposal rejection requires one creation event")
                authority = _restore_authoritative_pipeline_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(proposal_row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=deep_thaw(command.reviewed_facts),
                )
                _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                if authority.row.status != "pending" or authority.proposal.draft_hash != command.draft_hash:
                    raise AuditIntegrityError("guided proposal rejection does not name the active pending draft")
                current_state = state_from_record(current_record)
                guided = current_state.guided_session
                if guided is None or guided.active_proposal is None:
                    raise AuditIntegrityError("guided proposal rejection has no active reference")
                active = guided.active_proposal
                if active.proposal_id != command.proposal_id or active.draft_hash != command.draft_hash:
                    raise AuditIntegrityError("guided proposal rejection reference differs from authority")
                from elspeth.web.composer.guided.protocol import TurnType

                if (
                    not guided.history
                    or guided.history[-1].turn_type is not TurnType.PROPOSE_PIPELINE
                    or guided.history[-1].response_hash is not None
                ):
                    raise AuditIntegrityError("guided proposal rejection has no active proposal occurrence")
                cleared = replace(
                    guided,
                    history=guided.history[:-1],
                    active_proposal=None,
                    active_edit_target=None,
                )
                state_dict = current_state.to_dict()
                state_data = with_guided_response_descriptor(
                    CompositionStateData(
                        sources=state_dict["sources"],
                        nodes=state_dict["nodes"],
                        edges=state_dict["edges"],
                        outputs=state_dict["outputs"],
                        metadata_=state_dict["metadata"],
                        is_valid=current_record.is_valid,
                        validation_errors=current_record.validation_errors,
                        composer_meta={"guided_session": cleared.to_dict()},
                    ),
                    command.response,
                )
                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=state_data, derived_from_state_id=str(current_record.id)),
                    provenance="convergence_persist",
                    created_at=now,
                )
                event_id = str(uuid.uuid4())
                terminal_payload = _pipeline_rejected_payload(
                    authority=authority,
                    reason="operator_rejected",
                    dispatch=None,
                )
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.rejected",
                        actor=command.actor,
                        payload=terminal_payload,
                        created_at=now,
                    )
                )
                conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.status == "pending")
                    .values(status="rejected", committed_state_id=None, audit_event_id=event_id, updated_at=now)
                )
                result_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == state_id)).one()
                result_state = self._row_to_state_record(result_row)
                updated_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                proposal_record = replace(
                    _proposal_record_from_row(updated_row),
                    pipeline_metadata=_pipeline_public_metadata(authority),
                )
                response = project_guided_response(result_state, payloads=())
                projected_json = response_json(response)
                response_hash = guided_response_projection_hash(response)
                self.bind_guided_operation_on_connection(
                    conn,
                    command.fence,
                    proposal_id=command.proposal_id,
                    result_state_id=result_state.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedCompositionStateResult(state_id=result_state.id, proposal_id=command.proposal_id),
                    response_hash=response_hash,
                    actor=command.actor,
                )
                return GuidedPipelineProposalStageSettlement(
                    result_state=result_state,
                    proposal=proposal_record,
                    audit_messages=(),
                    response_json=projected_json,
                    response_hash=response_hash,
                )

        settlement = cast("GuidedPipelineProposalStageSettlement", await self._run_sync(_sync))
        _PIPELINE_SETTLEMENT_COUNTER.add(1, {"surface": "guided_staged", "result": "rejected"})
        return settlement

    @staticmethod
    def _pipeline_dispatch_recovery_on_connection(
        conn: Connection,
        *,
        authority: AuthoritativePipelineProposal,
    ) -> PipelineDispatchRecovery | None:
        sid = str(authority.row.session_id)
        expected_arguments_hash = stable_hash(authority.row.arguments_redacted_json)
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

    def _restore_guided_confirmation_authority_on_connection(
        self,
        conn: Connection,
        *,
        session_id: str,
        proposal_id: UUID,
        draft_hash: str,
        reviewed_facts: Mapping[str, Any],
    ) -> AuthoritativePipelineProposal:
        proposal_row = conn.execute(
            select(composition_proposals_table)
            .where(composition_proposals_table.c.session_id == session_id)
            .where(composition_proposals_table.c.id == str(proposal_id))
        ).one_or_none()
        if proposal_row is None:
            raise AuditIntegrityError("guided confirmation proposal authority is missing")
        creation_rows = conn.execute(
            select(proposal_events_table)
            .where(proposal_events_table.c.session_id == session_id)
            .where(proposal_events_table.c.proposal_id == str(proposal_id))
            .where(proposal_events_table.c.event_type == "proposal.created")
        ).fetchall()
        if len(creation_rows) != 1:
            raise AuditIntegrityError("guided confirmation requires one creation event")
        authority = _restore_authoritative_pipeline_proposal(
            conn=conn,
            row=_proposal_record_from_row(proposal_row),
            creation_event=_proposal_event_record_from_row(creation_rows[0]),
            reviewed_facts=deep_thaw(reviewed_facts),
        )
        _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
        if authority.row.status != "pending" or authority.proposal.draft_hash != draft_hash:
            raise GuidedOperationSettlementConflictError()
        return authority

    async def admit_guided_pipeline_confirmation(
        self,
        command: GuidedPipelineConfirmationAdmissionCommand,
    ) -> PipelineDispatchRecovery | None:
        """Acquire the proposal fence before any set_pipeline dispatch."""

        if type(command) is not GuidedPipelineConfirmationAdmissionCommand:
            raise TypeError("command must be an exact GuidedPipelineConfirmationAdmissionCommand")
        sid = str(command.fence.session_id)

        def _sync() -> PipelineDispatchRecovery | None:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation, now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation["kind"] != "guided_respond":
                    raise AuditIntegrityError("guided confirmation admission requires guided_respond")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                authority = self._restore_guided_confirmation_authority_on_connection(
                    conn,
                    session_id=sid,
                    proposal_id=command.proposal_id,
                    draft_hash=command.draft_hash,
                    reviewed_facts=command.reviewed_facts,
                )
                current_row = conn.execute(
                    select(composition_states_table).where(
                        composition_states_table.c.session_id == sid,
                        composition_states_table.c.id == str(command.expected_current_state_id),
                    )
                ).one()
                guided = state_from_record(self._row_to_state_record(current_row)).guided_session
                if (
                    guided is None
                    or guided.active_proposal is None
                    or guided.active_proposal.proposal_id != command.proposal_id
                    or guided.active_proposal.draft_hash != command.draft_hash
                ):
                    raise GuidedOperationSettlementConflictError()

                conn.execute(
                    update(guided_operations_table)
                    .where(
                        guided_operations_table.c.session_id == sid,
                        guided_operations_table.c.proposal_id == str(command.proposal_id),
                        guided_operations_table.c.status == "in_progress",
                        guided_operations_table.c.lease_expires_at <= now,
                    )
                    .values(proposal_id=None, updated_at=now)
                )
                owner = conn.execute(
                    select(guided_operations_table.c.operation_id)
                    .where(
                        guided_operations_table.c.session_id == sid,
                        guided_operations_table.c.proposal_id == str(command.proposal_id),
                        guided_operations_table.c.status == "in_progress",
                        guided_operations_table.c.lease_expires_at > now,
                    )
                    .limit(1)
                ).scalar_one_or_none()
                if owner is not None and owner != command.fence.operation_id:
                    raise GuidedOperationSettlementConflictError()
                self.bind_guided_operation_on_connection(conn, command.fence, proposal_id=command.proposal_id)
                return self._pipeline_dispatch_recovery_on_connection(conn, authority=authority)

        try:
            return cast(PipelineDispatchRecovery | None, await self._run_sync(_sync))
        except IntegrityError as exc:
            raise GuidedOperationSettlementConflictError() from exc

    async def record_guided_pipeline_dispatch(
        self,
        command: GuidedPipelineDispatchRecordCommand,
    ) -> PipelineDispatchRecovery:
        """Durably record the successful dispatch before final publication."""

        if type(command) is not GuidedPipelineDispatchRecordCommand:
            raise TypeError("command must be an exact GuidedPipelineDispatchRecordCommand")
        prepared = prepare_guided_audit_rows(invocations=(command.invocation,), llm_calls=(), chat_turns=())
        if len(prepared) != 1 or prepared[0].kind != "tool":
            raise AuditIntegrityError("guided dispatch record requires one prepared tool audit row")
        invocation_binding = PipelineDispatchAuditBinding.from_invocation(command.invocation)
        expected_binding = PipelineDispatchAuditBinding.from_persisted_envelope(cast(dict[str, Any], deep_thaw(prepared[0].envelope)))
        sid = str(command.fence.session_id)
        now = self._now()

        def _sync() -> PipelineDispatchRecovery:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation, _database_now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation["kind"] != "guided_respond" or operation["proposal_id"] != str(command.proposal_id):
                    raise GuidedOperationSettlementConflictError()
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                authority = self._restore_guided_confirmation_authority_on_connection(
                    conn,
                    session_id=sid,
                    proposal_id=command.proposal_id,
                    draft_hash=command.draft_hash,
                    reviewed_facts=command.reviewed_facts,
                )
                if (
                    invocation_binding.tool_call_id != authority.row.tool_call_id
                    or invocation_binding.arguments_hash != authority.row.tool_arguments_hash
                    or expected_binding.tool_call_id != authority.row.tool_call_id
                    or expected_binding.arguments_hash != stable_hash(authority.row.arguments_redacted_json)
                ):
                    raise AuditIntegrityError("guided dispatch record differs from proposal authority")
                existing = self._pipeline_dispatch_recovery_on_connection(conn, authority=authority)
                if existing is not None:
                    if existing.binding != expected_binding:
                        raise AuditIntegrityError("guided dispatch retry differs from durable dispatch")
                    return existing
                sequence_no = self._reserve_sequence_range(conn, sid, count=1)
                self._insert_prepared_guided_audit_rows_on_connection(
                    conn,
                    session_id=sid,
                    composition_state_id=command.expected_current_state_id,
                    audit_rows=prepared,
                    sequence_no=sequence_no,
                    created_at=now,
                )
                conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))
                recovery = self._pipeline_dispatch_recovery_on_connection(conn, authority=authority)
                if recovery is None or recovery.binding != expected_binding:
                    raise AuditIntegrityError("guided dispatch did not become durably recoverable")
                return recovery

        return cast(PipelineDispatchRecovery, await self._run_sync(_sync))

    async def accept_guided_pipeline_proposal(
        self,
        command: GuidedPipelineProposalAcceptCommand,
        *,
        payload_store: PayloadStore | None = None,
    ) -> GuidedPipelineProposalStageSettlement:
        """Atomically publish accepted state, terminal event, and operation."""

        if type(command) is not GuidedPipelineProposalAcceptCommand:
            raise TypeError("command must be an exact GuidedPipelineProposalAcceptCommand")
        if command.candidate_content_hash != command.executor_content_hash:
            raise AuditIntegrityError("guided proposal candidate/executor hashes differ")
        if _composition_state_data_content_hash(command.state) != command.candidate_content_hash:
            raise AuditIntegrityError("guided accepted state differs from prepared candidate")
        audit_rows = self._prepare_guided_audit_cohort(
            audit_evidence=command.audit_evidence,
            payloads=command.payloads,
            payload_store=payload_store,
        )
        dispatch = command.dispatch
        sid = str(command.fence.session_id)
        pid = str(command.proposal_id)
        now = self._now()

        def _sync() -> GuidedPipelineProposalStageSettlement:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                operation_row, _ = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation_row["kind"] != "guided_respond" or operation_row["proposal_id"] != pid:
                    raise AuditIntegrityError("guided proposal acceptance requires guided_respond")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=command.expected_current_state_id,
                    expected_state_version=command.expected_current_state_version,
                )
                current_row = conn.execute(
                    select(composition_states_table).where(composition_states_table.c.id == str(command.expected_current_state_id))
                ).one()
                current_record = self._row_to_state_record(current_row)
                proposal_row = conn.execute(
                    select(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                ).one_or_none()
                if proposal_row is None:
                    raise AuditIntegrityError("guided proposal acceptance authority is missing")
                creation_rows = conn.execute(
                    select(proposal_events_table)
                    .where(proposal_events_table.c.session_id == sid)
                    .where(proposal_events_table.c.proposal_id == pid)
                    .where(proposal_events_table.c.event_type == "proposal.created")
                ).fetchall()
                if len(creation_rows) != 1:
                    raise AuditIntegrityError("guided proposal acceptance requires one creation event")
                authority = _restore_authoritative_pipeline_proposal(
                    conn=conn,
                    row=_proposal_record_from_row(proposal_row),
                    creation_event=_proposal_event_record_from_row(creation_rows[0]),
                    reviewed_facts=deep_thaw(command.reviewed_facts),
                )
                _verify_pipeline_lifecycle_authority(conn, service=self, authority=authority)
                if authority.row.status != "pending" or authority.proposal.draft_hash != command.draft_hash:
                    raise AuditIntegrityError("guided proposal acceptance does not name the active pending draft")
                if type(authority.proposal.base) is not PresentBase:
                    raise AuditIntegrityError("guided proposal acceptance base is not present")
                base_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == sid)
                    .where(composition_states_table.c.id == str(authority.proposal.base.state_id))
                ).one_or_none()
                if base_row is None:
                    raise AuditIntegrityError("guided proposal acceptance base checkpoint is missing or cross-session")
                base_record = self._row_to_state_record(base_row)
                if composition_content_hash(state_from_record(base_record)) != authority.proposal.base.composition_content_hash:
                    raise AuditIntegrityError("guided proposal acceptance base checkpoint hash changed")
                if composition_content_hash(state_from_record(current_record)) != authority.proposal.base.composition_content_hash:
                    raise AuditIntegrityError("guided proposal acceptance review checkpoint changed authored content")
                if dispatch.tool_call_id != authority.row.tool_call_id:
                    raise AuditIntegrityError("guided proposal acceptance dispatch differs from authority")
                if dispatch.arguments_hash != stable_hash(authority.row.arguments_redacted_json):
                    raise AuditIntegrityError("guided proposal acceptance dispatch arguments differ from authority")

                current_guided = state_from_record(current_record).guided_session
                metadata = deep_thaw(command.state.composer_meta)
                if current_guided is None or current_guided.active_proposal is None or type(metadata) is not dict:
                    raise AuditIntegrityError("guided proposal acceptance checkpoint metadata is malformed")
                _verify_guided_deferred_message_authority(
                    conn,
                    session_id=sid,
                    guided=current_guided,
                )
                _verify_guided_correction_message_authority(
                    conn,
                    session_id=sid,
                    guided=current_guided,
                )
                _verify_guided_root_message_authority(
                    conn,
                    service=self,
                    session_id=sid,
                    guided=current_guided,
                )
                from elspeth.web.composer.guided.planning import verified_remaining_deferred_intents
                from elspeth.web.composer.guided.state_machine import GuidedSession

                remaining = verified_remaining_deferred_intents(
                    guided=current_guided,
                    proposal=authority.proposal,
                )
                final_guided = GuidedSession.from_dict(metadata["guided_session"])
                if (
                    final_guided.active_proposal is not None
                    or final_guided.active_edit_target is not None
                    or final_guided.deferred_intents != remaining
                    or final_guided.reviewed_sources != current_guided.reviewed_sources
                    or final_guided.reviewed_outputs != current_guided.reviewed_outputs
                    or final_guided.correction_messages != current_guided.correction_messages
                    or final_guided.root_intent_message_id != current_guided.root_intent_message_id
                ):
                    raise AuditIntegrityError("guided accepted checkpoint did not clear and consume exact authority")

                prepared_state = with_guided_response_descriptor(command.state, command.response)
                state_id = self._insert_composition_state(
                    conn,
                    session_id=sid,
                    payload=StatePayload(data=prepared_state, derived_from_state_id=str(current_record.id)),
                    provenance="tool_call",
                    created_at=now,
                )
                result_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == state_id)).one()
                result_state = self._row_to_state_record(result_row)
                if audit_rows:
                    sequence_no = self._reserve_sequence_range(conn, sid, count=len(audit_rows))
                    audit_messages = self._insert_prepared_guided_audit_rows_on_connection(
                        conn,
                        session_id=sid,
                        composition_state_id=result_state.id,
                        audit_rows=audit_rows,
                        sequence_no=sequence_no,
                        created_at=now,
                    )
                else:
                    audit_messages = ()
                if _persisted_pipeline_dispatch_content_hashes(
                    conn,
                    session_id=sid,
                    dispatch=dispatch,
                ) != (command.executor_content_hash,):
                    raise AuditIntegrityError("guided proposal acceptance requires one durable exact dispatch")
                event_id = str(uuid.uuid4())
                terminal_payload = _pipeline_accepted_payload(
                    authority=authority,
                    state_id=state_id,
                    state_content_hash=command.executor_content_hash,
                    final_composer_metadata=prepared_state.composer_meta,
                    dispatch=dispatch,
                )
                conn.execute(
                    insert(proposal_events_table).values(
                        id=event_id,
                        session_id=sid,
                        proposal_id=pid,
                        event_type="proposal.accepted",
                        actor=command.actor,
                        payload=terminal_payload,
                        created_at=now,
                    )
                )
                updated = conn.execute(
                    update(composition_proposals_table)
                    .where(composition_proposals_table.c.session_id == sid)
                    .where(composition_proposals_table.c.id == pid)
                    .where(composition_proposals_table.c.status == "pending")
                    .values(status="committed", committed_state_id=state_id, audit_event_id=event_id, updated_at=now)
                )
                if updated.rowcount != 1:
                    raise AuditIntegrityError("guided proposal acceptance lost the pending proposal CAS")
                updated_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == pid)).one()
                proposal_record = replace(
                    _proposal_record_from_row(updated_row),
                    pipeline_metadata=_pipeline_public_metadata(authority),
                )
                response = project_guided_response(result_state, payloads=command.payloads)
                projected_json = response_json(response)
                response_hash = guided_response_projection_hash(response)
                self.bind_guided_operation_on_connection(
                    conn,
                    command.fence,
                    proposal_id=command.proposal_id,
                    result_state_id=result_state.id,
                )
                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedCompositionStateResult(state_id=result_state.id, proposal_id=command.proposal_id),
                    response_hash=response_hash,
                    actor=command.actor,
                )
                return GuidedPipelineProposalStageSettlement(
                    result_state=result_state,
                    proposal=proposal_record,
                    audit_messages=audit_messages,
                    response_json=projected_json,
                    response_hash=response_hash,
                )

        settlement = cast("GuidedPipelineProposalStageSettlement", await self._run_sync(_sync))
        _PIPELINE_SETTLEMENT_COUNTER.add(1, {"surface": "guided_staged", "result": "accepted"})
        return settlement

    async def complete_existing_state_guided_operation(
        self,
        fence: GuidedOperationFence,
        *,
        state_id: UUID,
        expected_current_state_id: UUID,
        expected_current_state_version: int,
        actor: str,
        response_hash_factory: Callable[[CompositionStateRecord], str],
    ) -> CompositionStateRecord:
        """Settle a no-op/idempotent surface against one immutable state."""

        sid = str(fence.session_id)

        def _sync() -> CompositionStateRecord:
            with self._session_process_locked_begin(sid) as conn, self._session_write_lock(conn, sid):
                self.require_guided_operation_fence_on_connection(conn, fence)
                if state_id != expected_current_state_id:
                    raise ValueError("existing guided operation result must be the expected current state")
                self._require_guided_expected_current_state_on_connection(
                    conn,
                    session_id=sid,
                    expected_state_id=expected_current_state_id,
                    expected_state_version=expected_current_state_version,
                )
                row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.id == str(state_id))
                    .where(composition_states_table.c.session_id == sid)
                ).one_or_none()
                if row is None:
                    raise AuditIntegrityError("Guided operation result state is absent from its session")
                record = self._row_to_state_record(row)
                response_hash = response_hash_factory(record)
                self.bind_guided_operation_on_connection(conn, fence, result_state_id=record.id)
                self.complete_guided_operation_on_connection(
                    conn,
                    fence,
                    result=GuidedCompositionStateResult(state_id=record.id),
                    response_hash=response_hash,
                    actor=actor,
                )
                return record

        return cast("CompositionStateRecord", await self._run_sync(_sync))

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

    def _load_staged_fork_on_connection(
        self,
        conn: Connection,
        *,
        parent_session_id: UUID,
        child_session_id: UUID,
        operation_id: str,
        fork_message_id: UUID,
        new_message_content: str,
    ) -> StagedForkSession:
        parent_row = conn.execute(
            select(sessions_table.c.user_id, sessions_table.c.auth_provider_type).where(sessions_table.c.id == str(parent_session_id))
        ).one_or_none()
        child_row = conn.execute(select(sessions_table).where(sessions_table.c.id == str(child_session_id))).one_or_none()
        if (
            child_row is None
            or child_row.archived_at is None
            or child_row.forked_from_session_id != str(parent_session_id)
            or child_row.forked_from_message_id != str(fork_message_id)
            or parent_row is None
            or child_row.user_id != parent_row.user_id
            or child_row.auth_provider_type != parent_row.auth_provider_type
        ):
            raise AuditIntegrityError("Guided fork bound child failed archived lineage, principal, or fork message validation")

        rows = conn.execute(
            select(chat_messages_table)
            .where(chat_messages_table.c.session_id == str(child_session_id))
            .order_by(chat_messages_table.c.sequence_no)
        ).all()
        plan_candidates: list[tuple[BlobForkPlanEntry, ...]] = []
        public_messages: list[ChatMessageRecord] = []
        for row in rows:
            if row.role == "audit" and row.writer_principal == "session_fork":
                try:
                    decoded = json.loads(row.content)
                except (TypeError, json.JSONDecodeError):
                    decoded = None
                if (
                    type(decoded) is dict
                    and decoded.get("schema") == _FORK_BLOB_PLAN_SCHEMA
                    and decoded.get("child_session_id") == str(child_session_id)
                    and decoded.get("operation_id") == operation_id
                ):
                    plan_candidates.append(
                        _fork_blob_plan_from_content(
                            row.content,
                            expected_source_session_id=parent_session_id,
                            expected_child_session_id=child_session_id,
                            expected_operation_id=operation_id,
                        )
                    )
            if row.role != "audit":
                public_messages.append(self._row_to_chat_message_record(row))
        if len(plan_candidates) != 1:
            raise AuditIntegrityError("Guided fork bound child must retain exactly one strict blob plan audit row")
        if not public_messages:
            raise AuditIntegrityError("Guided fork bound child has no public messages")
        edited_message = public_messages[-1]
        if (
            edited_message.role != "user"
            or edited_message.writer_principal != "session_fork"
            or edited_message.content != new_message_content
        ):
            raise AuditIntegrityError("Guided fork bound child edited message validation failed")

        state_row = conn.execute(
            select(composition_states_table)
            .where(composition_states_table.c.session_id == str(child_session_id))
            .order_by(composition_states_table.c.version.desc())
            .limit(1)
        ).one_or_none()
        state = self._row_to_state_record(state_row) if state_row is not None else None
        if edited_message.composition_state_id != (state.id if state is not None else None):
            raise AuditIntegrityError("Guided fork bound child edited message does not reference its staged current state")
        return StagedForkSession(
            session=self._row_to_session_record(child_row),
            messages=tuple(public_messages),
            state=state,
            blob_plan=plan_candidates[0],
        )

    async def fork_session(
        self,
        fence: GuidedOperationFence,
        *,
        fork_message_id: UUID,
        new_message_content: str,
    ) -> StagedForkSession:
        """Stage or resume the exact child bound to a fenced fork operation.

        Initial staging atomically creates and binds one archived child with a
        frozen blob plan. Repeated calls under the current fence, including an
        expired-lease takeover, reload that same hidden cohort. Blob realization
        and final activation happen later at the route/coordinator settlement
        boundary; a failed operation retains the archived child and plan as
        integrity evidence while compensating only authorized partial blobs.

        The staged child contains:
        1. Composition state copied from the fork message's pre-send state
        2. All messages BEFORE the fork message (with NULL state provenance)
        3. A synthetic system message noting the fork
        4. The new edited user message (provenance = copied state, not source)

        Returns a ``StagedForkSession`` with the child, public messages, copied
        state when present, and the immutable plan used by later blob custody.
        """
        from elspeth.web.sessions.protocol import InvalidForkTargetError

        # Load source data (read-only, outside the write transaction)
        if type(fence) is not GuidedOperationFence:
            raise TypeError("fork_session fence must be an exact GuidedOperationFence")
        source_session_id = fence.session_id
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

        # Prepare IDs and timestamps upfront
        new_session_id = uuid.uuid4()
        new_session_id_str = str(new_session_id)
        now = self._now()
        # Prepare state copy if needed
        copied_state_id = uuid.uuid4() if source_state_record is not None else None
        copied_state_id_str = str(copied_state_id) if copied_state_id else None

        # Allocate every copied chat id before preparing metadata or inserts.
        # Guided references and tool parents use this one exact map.
        source_messages_by_id = {str(msg.id): msg for msg in messages_to_copy}
        source_to_copied_message_id = {str(msg.id): str(uuid.uuid4()) for msg in messages_to_copy}
        source_assistant_ids = {str(msg.id) for msg in messages_to_copy if msg.role == "assistant"}

        # Prepare all message rows upfront — preserve original created_at
        # so get_messages() ordering is deterministic.  Stamping all rows
        # with `now` would make them indistinguishable by timestamp and
        # produce non-deterministic ordering on subsequent reads.
        msg_records_data: list[dict[str, Any]] = []
        for msg in messages_to_copy:
            copied_msg_id = source_to_copied_message_id[str(msg.id)]

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
                if parent_key not in source_to_copied_message_id or parent_key not in source_assistant_ids:
                    # Slice ``[:fork_idx]`` excluded the assistant message
                    # this tool row depends on. Detect it pre-batch with a
                    # named error per the offensive-programming policy.
                    raise RuntimeError(f"fork slice excludes parent assistant of tool message id={msg.id}")
                copied_parent_assistant_id = source_to_copied_message_id[parent_key]

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

        def _sync() -> StagedForkSession:
            """Create+bind or reload exactly one child under the parent fence."""
            parent_session_id_str = str(source_session_id)
            with self._session_process_locked_begin(parent_session_id_str) as conn, self._session_write_lock(conn, parent_session_id_str):
                operation, _database_now = self.require_guided_operation_fence_on_connection(conn, fence)
                if operation["kind"] != "session_fork":
                    raise AuditIntegrityError("fork_session fence is not bound to session_fork")
                bound_child_id = operation["result_session_id"]
                if bound_child_id is not None:
                    if operation["originating_message_id"] != str(fork_message_id):
                        raise AuditIntegrityError("Guided fork bound child has a different fork message binding")
                    return self._load_staged_fork_on_connection(
                        conn,
                        parent_session_id=source_session_id,
                        child_session_id=UUID(bound_child_id),
                        operation_id=fence.operation_id,
                        fork_message_id=fork_message_id,
                        new_message_content=new_message_content,
                    )

                parent_row = conn.execute(select(sessions_table).where(sessions_table.c.id == parent_session_id_str)).one_or_none()
                fork_row = conn.execute(
                    select(chat_messages_table).where(
                        chat_messages_table.c.id == str(fork_message_id),
                        chat_messages_table.c.session_id == parent_session_id_str,
                    )
                ).one_or_none()
                if parent_row is None or parent_row.archived_at is not None:
                    raise AuditIntegrityError("Guided fork parent failed active custody validation")
                if (
                    fork_row is None
                    or fork_row.role != "user"
                    or fork_row.composition_state_id != (str(source_state_record.id) if source_state_record is not None else None)
                ):
                    raise AuditIntegrityError("Guided fork message changed before staging")

                locked_source_state: CompositionStateRecord | None = None
                forked_composer_meta: dict[str, Any] | None = None
                if source_state_record is not None:
                    locked_source_row = conn.execute(
                        select(composition_states_table)
                        .where(composition_states_table.c.id == str(source_state_record.id))
                        .where(composition_states_table.c.session_id == parent_session_id_str)
                    ).one_or_none()
                    if locked_source_row is None:
                        raise AuditIntegrityError("Guided fork source checkpoint is missing or cross-session")
                    locked_source_state = self._row_to_state_record(locked_source_row)
                    forked_composer_meta = _strip_guided_profile_in_meta(
                        locked_source_state.composer_meta,
                        source_to_copied_message_id,
                        source_messages_by_id,
                    )
                    source_meta = deep_thaw(locked_source_state.composer_meta)
                    if type(source_meta) is dict and source_meta.get("guided_session") is not None:
                        from elspeth.web.composer.guided.errors import InvariantError
                        from elspeth.web.composer.guided.state_machine import GuidedSession

                        try:
                            source_guided = GuidedSession.from_dict(source_meta["guided_session"])
                        except (InvariantError, KeyError, TypeError, ValueError) as exc:
                            raise AuditIntegrityError("fork guided schema-10 authority is malformed") from exc
                        _require_pending_guided_checkpoint_proposal_authority(
                            conn,
                            service=self,
                            session_id=parent_session_id_str,
                            checkpoint=locked_source_state,
                            guided=source_guided,
                            role="fork source",
                        )
                        if source_guided.root_intent_message_id is not None:
                            _verify_guided_root_message_authority(
                                conn,
                                service=self,
                                session_id=parent_session_id_str,
                                guided=source_guided,
                            )

                plan = tuple(
                    BlobForkPlanEntry(
                        source_blob_id=UUID(row.id),
                        target_blob_id=fork_blob_id(
                            target_session_id=new_session_id,
                            source_blob_id=UUID(row.id),
                        ),
                        content_hash=row.content_hash,
                        size_bytes=row.size_bytes,
                    )
                    for row in conn.execute(
                        select(blobs_table.c.id, blobs_table.c.content_hash, blobs_table.c.size_bytes)
                        .where(
                            blobs_table.c.session_id == parent_session_id_str,
                            blobs_table.c.status == "ready",
                        )
                        .order_by(blobs_table.c.id)
                    ).all()
                )
                plan_message = {
                    "id": str(uuid.uuid4()),
                    "session_id": new_session_id_str,
                    "role": "audit",
                    "content": _fork_blob_plan_content(
                        source_session_id=source_session_id,
                        child_session_id=new_session_id,
                        operation_id=fence.operation_id,
                        entries=plan,
                    ),
                    "raw_content": None,
                    "tool_calls": None,
                    "tool_call_id": None,
                    "parent_assistant_id": None,
                    "writer_principal": "session_fork",
                    "created_at": now,
                    "composition_state_id": None,
                }
                rows_to_insert = [dict(record) for record in msg_records_data]
                rows_to_insert.append(plan_message)

                conn.execute(
                    insert(sessions_table).values(
                        id=new_session_id_str,
                        user_id=parent_row.user_id,
                        auth_provider_type=parent_row.auth_provider_type,
                        title=f"{parent_row.title} (fork)",
                        created_at=now,
                        updated_at=now,
                        archived_at=now,
                        forked_from_session_id=parent_session_id_str,
                        forked_from_message_id=str(fork_message_id),
                    )
                )
                with self._session_write_lock(conn, new_session_id_str):
                    if locked_source_state is not None and copied_state_id_str is not None:
                        self._insert_composition_state(
                            conn,
                            session_id=new_session_id_str,
                            payload=StatePayload(
                                data=CompositionStateData(
                                    sources=locked_source_state.sources,
                                    nodes=locked_source_state.nodes,
                                    edges=locked_source_state.edges,
                                    outputs=locked_source_state.outputs,
                                    metadata_=locked_source_state.metadata_,
                                    is_valid=locked_source_state.is_valid,
                                    validation_errors=locked_source_state.validation_errors,
                                    composer_meta=forked_composer_meta,
                                ),
                                derived_from_state_id=None,
                            ),
                            provenance="session_fork",
                            created_at=now,
                            state_id=copied_state_id_str,
                        )
                    base_seq = self._reserve_sequence_range(conn, new_session_id_str, count=len(rows_to_insert))
                    for sequence_offset, record in enumerate(rows_to_insert):
                        record["sequence_no"] = base_seq + sequence_offset
                    conn.execute(insert(chat_messages_table), rows_to_insert)

                self.bind_guided_operation_on_connection(
                    conn,
                    fence,
                    originating_message_id=fork_message_id,
                    result_session_id=new_session_id,
                )
                return self._load_staged_fork_on_connection(
                    conn,
                    parent_session_id=source_session_id,
                    child_session_id=new_session_id,
                    operation_id=fence.operation_id,
                    fork_message_id=fork_message_id,
                    new_message_content=new_message_content,
                )

        return cast("StagedForkSession", await self._run_sync(_sync))

    async def settle_guided_fork_operation(
        self,
        command: GuidedForkSettlementCommand,
    ) -> SessionRecord:
        """Atomically rewrite, activate, and complete one staged fork."""
        if type(command) is not GuidedForkSettlementCommand:
            raise TypeError("settle_guided_fork_operation command must be exact")
        parent_session_id_str = str(command.fence.session_id)
        child_session_id_str = str(command.child_session_id)

        def _sync() -> SessionRecord:
            with self._session_pair_locked_begin(parent_session_id_str, child_session_id_str) as conn:
                operation, now = self.require_guided_operation_fence_on_connection(conn, command.fence)
                if operation["kind"] != "session_fork" or operation["result_session_id"] != child_session_id_str:
                    raise AuditIntegrityError("Guided fork settlement child is not bound to the exact operation fence")
                parent = conn.execute(
                    select(
                        sessions_table.c.user_id,
                        sessions_table.c.auth_provider_type,
                    ).where(sessions_table.c.id == parent_session_id_str)
                ).one_or_none()
                if parent is None:
                    raise AuditIntegrityError("Guided fork settlement parent is missing")

                child = conn.execute(
                    select(sessions_table).where(sessions_table.c.id == child_session_id_str).with_for_update()
                ).one_or_none()
                if (
                    child is None
                    or child.archived_at is None
                    or child.forked_from_session_id != parent_session_id_str
                    or child.forked_from_message_id != operation["originating_message_id"]
                    or child.user_id != parent.user_id
                    or child.auth_provider_type != parent.auth_provider_type
                ):
                    raise AuditIntegrityError("Guided fork settlement child failed staged custody validation")

                current_state_row = conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == child_session_id_str)
                    .order_by(composition_states_table.c.version.desc())
                    .limit(1)
                    .with_for_update()
                ).one_or_none()
                current_state_id = UUID(current_state_row.id) if current_state_row is not None else None
                if current_state_id != command.expected_current_state_id:
                    raise AuditIntegrityError("Guided fork settlement staged current state changed")
                child_guided_root: tuple[str, str] | None = None
                if current_state_row is not None:
                    current_record = self._row_to_state_record(current_state_row)
                    current_meta = deep_thaw(current_record.composer_meta)
                    if type(current_meta) is dict and current_meta.get("guided_session") is not None:
                        from elspeth.web.composer.guided.errors import InvariantError
                        from elspeth.web.composer.guided.state_machine import GuidedSession

                        try:
                            child_guided = GuidedSession.from_dict(current_meta["guided_session"])
                        except (InvariantError, KeyError, TypeError, ValueError) as exc:
                            raise AuditIntegrityError("Guided fork settlement child checkpoint is malformed") from exc
                    else:
                        child_guided = None
                    if child_guided is not None and child_guided.root_intent_message_id is not None:
                        existing_starts = conn.execute(
                            select(guided_operations_table.c.operation_id).where(
                                guided_operations_table.c.session_id == child_session_id_str,
                                guided_operations_table.c.kind == "guided_start",
                            )
                        ).all()
                        if existing_starts:
                            raise AuditIntegrityError("Guided fork staged child has premature start authority")
                        child_root = conn.execute(
                            select(
                                chat_messages_table.c.role,
                                chat_messages_table.c.content,
                                chat_messages_table.c.writer_principal,
                            ).where(
                                chat_messages_table.c.session_id == child_session_id_str,
                                chat_messages_table.c.id == child_guided.root_intent_message_id,
                            )
                        ).one_or_none()
                        if child_root is None or child_root.role != "user" or child_root.writer_principal != "route_user_message":
                            raise AuditIntegrityError("Guided fork staged root intent failed child custody")
                        child_guided_root = (child_guided.root_intent_message_id, child_root.content)

                edited_message = conn.execute(
                    select(chat_messages_table)
                    .where(
                        chat_messages_table.c.id == str(command.edited_message_id),
                        chat_messages_table.c.session_id == child_session_id_str,
                    )
                    .with_for_update()
                ).one_or_none()
                if (
                    edited_message is None
                    or edited_message.role != "user"
                    or edited_message.writer_principal != "session_fork"
                    or edited_message.composition_state_id
                    != (str(command.expected_current_state_id) if command.expected_current_state_id is not None else None)
                ):
                    raise AuditIntegrityError("Guided fork settlement edited message failed staged custody validation")

                if command.rewritten_state is not None:
                    settlement_state: Mapping[str, Any] = {
                        "sources": deep_thaw(command.rewritten_state.sources),
                        "nodes": deep_thaw(command.rewritten_state.nodes),
                        "edges": deep_thaw(command.rewritten_state.edges),
                        "outputs": deep_thaw(command.rewritten_state.outputs),
                        "metadata": deep_thaw(command.rewritten_state.metadata_),
                        "composer_meta": deep_thaw(command.rewritten_state.composer_meta),
                    }
                elif current_state_row is not None:
                    settlement_state = {
                        "source": current_state_row.source,
                        "sources": current_state_row.sources,
                        "nodes": current_state_row.nodes,
                        "edges": current_state_row.edges,
                        "outputs": current_state_row.outputs,
                        "metadata": current_state_row.metadata_,
                        "composer_meta": current_state_row.composer_meta,
                    }
                else:
                    settlement_state = {}
                _verify_fork_settlement_blob_custody(
                    conn,
                    parent_session_id=command.fence.session_id,
                    child_session_id=command.child_session_id,
                    operation_id=command.fence.operation_id,
                    state_payload=settlement_state,
                )

                if command.rewritten_state is not None and command.rewritten_state_id is not None:
                    # Detach the edited message from the staged state and delete
                    # that staged state BEFORE inserting the rewritten replacement.
                    # ``_insert_composition_state`` allocates ``MAX(version)+1``
                    # per session, so the replacement's version depends on whether
                    # the staged row (version 1) is still present when it runs.
                    # Removing the staged row first lets the replacement reclaim
                    # version 1 — a fresh forked-and-edited child settles at its
                    # first version rather than leaking a version-2 gap whose only
                    # meaning is "settlement happened to rewrite blob custody"
                    # (a non-rewriting settlement already lands the child at
                    # version 1). The composite FK on
                    # ``chat_messages(composition_state_id, session_id)`` is
                    # RESTRICT, so the message must be detached to NULL before the
                    # staged state can be removed; the insert then repoints it.
                    # All four writes share this transaction under the
                    # parent+child pair lock, so the transient NULL provenance is
                    # never externally observable.
                    detached = conn.execute(
                        update(chat_messages_table)
                        .where(
                            chat_messages_table.c.id == str(command.edited_message_id),
                            chat_messages_table.c.session_id == child_session_id_str,
                            chat_messages_table.c.composition_state_id == str(command.expected_current_state_id),
                        )
                        .values(composition_state_id=None)
                    )
                    if detached.rowcount != 1:
                        raise AuditIntegrityError("Guided fork settlement lost edited-message compare-and-swap")
                    removed_staged_state = conn.execute(
                        delete(composition_states_table).where(
                            composition_states_table.c.id == str(command.expected_current_state_id),
                            composition_states_table.c.session_id == child_session_id_str,
                        )
                    )
                    if removed_staged_state.rowcount != 1:
                        raise AuditIntegrityError("Guided fork settlement could not remove superseded staged state")
                    self._insert_composition_state(
                        conn,
                        session_id=child_session_id_str,
                        payload=StatePayload(
                            data=command.rewritten_state,
                            derived_from_state_id=None,
                        ),
                        provenance="session_fork",
                        created_at=now,
                        state_id=str(command.rewritten_state_id),
                    )
                    repointed = conn.execute(
                        update(chat_messages_table)
                        .where(
                            chat_messages_table.c.id == str(command.edited_message_id),
                            chat_messages_table.c.session_id == child_session_id_str,
                            chat_messages_table.c.composition_state_id.is_(None),
                        )
                        .values(composition_state_id=str(command.rewritten_state_id))
                    )
                    if repointed.rowcount != 1:
                        raise AuditIntegrityError("Guided fork settlement could not bind replacement state")

                final_state_id = command.rewritten_state_id or command.expected_current_state_id
                if child_guided_root is not None:
                    if final_state_id is None:
                        raise AuditIntegrityError("Guided fork start authority has no final child state")
                    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
                    from elspeth.web.sessions.schemas import StartGuidedRequest

                    child_root_id, child_root_content = child_guided_root
                    child_start_operation_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"elspeth:fork-guided-start:{child_session_id_str}:{child_root_id}",
                        )
                    )
                    child_start_request = StartGuidedRequest.model_validate(
                        {
                            "operation_id": child_start_operation_id,
                            "profile": "live",
                            "intent": child_root_content,
                        },
                        strict=True,
                    )
                    child_request_hash = guided_operation_request_hash(
                        session_id=command.child_session_id,
                        kind="guided_start",
                        request=child_start_request,
                    )
                    child_response_hash = stable_hash(
                        {
                            "schema": "fork-guided-start-lineage.v1",
                            "session_id": child_session_id_str,
                            "operation_id": child_start_operation_id,
                            "root_intent_message_id": child_root_id,
                            "result_state_id": str(final_state_id),
                        }
                    )
                    conn.execute(
                        insert(guided_operations_table).values(
                            session_id=child_session_id_str,
                            operation_id=child_start_operation_id,
                            kind="guided_start",
                            status="completed",
                            request_hash=child_request_hash,
                            lease_token=None,
                            lease_expires_at=None,
                            attempt=1,
                            originating_message_id=child_root_id,
                            proposal_id=None,
                            result_kind="composition_state",
                            result_state_id=str(final_state_id),
                            result_session_id=None,
                            response_hash=child_response_hash,
                            failure_code=None,
                            created_at=now,
                            updated_at=now,
                            settled_at=now,
                        )
                    )
                    self._insert_guided_operation_event(
                        conn,
                        session_id=child_session_id_str,
                        operation_id=child_start_operation_id,
                        event_kind="completed",
                        actor="session_fork",
                        attempt=1,
                        prior_attempt=None,
                        lease_expires_at=None,
                        request_hash=child_request_hash,
                        failure_audit_cohort=None,
                        occurred_at=now,
                    )
                    final_state_row = conn.execute(
                        select(composition_states_table).where(
                            composition_states_table.c.session_id == child_session_id_str,
                            composition_states_table.c.id == str(final_state_id),
                        )
                    ).one()
                    final_guided = state_from_record(self._row_to_state_record(final_state_row)).guided_session
                    if final_guided is None:
                        raise AuditIntegrityError("Guided fork final state lost guided authority")
                    _verify_guided_root_message_authority(
                        conn,
                        service=self,
                        session_id=child_session_id_str,
                        guided=final_guided,
                    )

                activated = conn.execute(
                    update(sessions_table)
                    .where(
                        sessions_table.c.id == child_session_id_str,
                        sessions_table.c.archived_at.is_not(None),
                    )
                    .values(archived_at=None, updated_at=now)
                )
                if activated.rowcount != 1:
                    raise AuditIntegrityError("Guided fork settlement lost archived-to-active compare-and-swap")

                self.complete_guided_operation_on_connection(
                    conn,
                    command.fence,
                    result=GuidedSessionResult(session_id=command.child_session_id),
                    response_hash=command.response_hash,
                    actor=command.actor,
                )
                settled_row = conn.execute(select(sessions_table).where(sessions_table.c.id == child_session_id_str)).one()
                return self._row_to_session_record(settled_row)

        return cast("SessionRecord", await self._run_sync(_sync))

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
