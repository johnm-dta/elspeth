"""Typed audit-record projection helpers for Landscape export."""

from __future__ import annotations

from typing import Literal

from elspeth.contracts import Artifact, NodeStateCompleted, NodeStateFailed, NodeStateOpen, NodeStatePending
from elspeth.contracts.audit import SinkEffect, SinkEffectAttempt, SinkEffectMemberRecord, SinkEffectStream
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.export_records import (
    ArtifactExportRecord,
    NodeStateExportRecord,
    SinkEffectAttemptExportRecord,
    SinkEffectExportRecord,
    SinkEffectMemberExportRecord,
    SinkEffectStreamExportRecord,
)


def artifact_producer_kind(
    *,
    produced_by_state_id: str | None,
    sink_effect_id: str | None,
) -> Literal["node_state", "sink_effect"]:
    """Validate producer XOR and return its explicit wire discriminator."""
    if (produced_by_state_id is None) == (sink_effect_id is None):
        raise AuditIntegrityError("Artifact requires exactly one producer link — audit integrity violation")
    return "node_state" if produced_by_state_id is not None else "sink_effect"


def validate_artifact_publication_projection(
    *,
    producer_kind: Literal["node_state", "sink_effect"],
    publication_performed: object,
    publication_evidence_kind: object,
) -> None:
    """Reject ambiguous raw artifact publication evidence before serialization."""
    if type(publication_performed) is not bool:
        raise AuditIntegrityError("Artifact publication_performed must be bool — audit integrity violation")
    expected: dict[str, bool]
    if producer_kind == "node_state":
        expected = {"legacy_returned": True}
    else:
        expected = {"returned": True, "reconciled": True, "inherited": False, "virtual": False}
    if not isinstance(publication_evidence_kind, str) or expected.get(publication_evidence_kind) is not publication_performed:
        raise AuditIntegrityError("Artifact publication evidence is invalid or contradictory — audit integrity violation")


def artifact_to_export_record(run_id: str, artifact: Artifact) -> ArtifactExportRecord:
    """Project one validated artifact to its complete epoch-26 wire shape."""
    return {
        "record_type": "artifact",
        "run_id": run_id,
        "artifact_id": artifact.artifact_id,
        "sink_node_id": artifact.sink_node_id,
        "producer_kind": artifact.producer_kind,
        "produced_by_state_id": artifact.produced_by_state_id,
        "sink_effect_id": artifact.sink_effect_id,
        "artifact_type": artifact.artifact_type,
        "path_or_uri": artifact.path_or_uri,
        "content_hash": artifact.content_hash,
        "size_bytes": artifact.size_bytes,
        "idempotency_key": artifact.idempotency_key,
        "publication_performed": artifact.publication_performed,
        "publication_evidence_kind": artifact.publication_evidence_kind,
        "created_at": artifact.created_at.isoformat(),
    }


def sink_effect_stream_to_export_record(stream: SinkEffectStream) -> SinkEffectStreamExportRecord:
    """Project stream ordering without exposing its resolved target."""
    return {
        "record_type": "sink_effect_stream",
        "run_id": stream.run_id,
        "stream_id": stream.stream_id,
        "sink_node_id": stream.sink_node_id,
        "role": stream.role.value,
        "requested_target_hash": stream.requested_target_hash,
        "next_sequence": stream.next_sequence,
        "tail_effect_id": stream.tail_effect_id,
        "head_effect_id": stream.head_effect_id,
        "head_descriptor_hash": stream.head_descriptor_hash,
    }


def sink_effect_to_export_record(effect: SinkEffect) -> SinkEffectExportRecord:
    """Project the complete safe effect lifecycle; omit raw target and plan JSON."""
    return {
        "record_type": "sink_effect",
        "run_id": effect.run_id,
        "effect_id": effect.effect_id,
        "sink_node_id": effect.sink_node_id,
        "role": effect.role.value,
        "state": effect.state.value,
        "protocol_version": effect.protocol_version,
        "input_kind": effect.input_kind.value,
        "config_hash": effect.config_hash,
        "membership_or_manifest_hash": effect.membership_or_manifest_hash,
        "group_payload_hash": effect.group_payload_hash,
        "artifact_id": effect.artifact_id,
        "artifact_idempotency_key": effect.artifact_idempotency_key,
        "inspection_mode": None if effect.inspection_mode is None else effect.inspection_mode.value,
        "inspection_attempt_id": effect.inspection_attempt_id,
        "plan_hash": effect.plan_hash,
        "descriptor_mode": None if effect.descriptor_mode is None else effect.descriptor_mode.value,
        "expected_descriptor_hash": effect.expected_descriptor_hash,
        "precondition_hash": effect.precondition_hash,
        "prepared_at": None if effect.prepared_at is None else effect.prepared_at.isoformat(),
        "lease_owner": effect.lease_owner,
        "generation": effect.generation,
        "lease_expires_at": None if effect.lease_expires_at is None else effect.lease_expires_at.isoformat(),
        "lease_heartbeat_at": None if effect.lease_heartbeat_at is None else effect.lease_heartbeat_at.isoformat(),
        "reconcile_kind": None if effect.reconcile_kind is None else effect.reconcile_kind.value,
        "reconcile_evidence_hash": effect.reconcile_evidence_hash,
        "result_descriptor_hash": effect.result_descriptor_hash,
        "publication_performed": effect.publication_performed,
        "publication_evidence_kind": effect.publication_evidence_kind,
        "primary_effect_id": effect.primary_effect_id,
        "stream_id": effect.stream_id,
        "stream_sequence": effect.stream_sequence,
        "predecessor_effect_id": effect.predecessor_effect_id,
        "created_at": effect.created_at.isoformat(),
        "updated_at": effect.updated_at.isoformat(),
        "finalized_at": None if effect.finalized_at is None else effect.finalized_at.isoformat(),
    }


def sink_effect_member_to_export_record(member: SinkEffectMemberRecord) -> SinkEffectMemberExportRecord:
    """Project member state while omitting raw lineage JSON and row payloads."""
    return {
        "record_type": "sink_effect_member",
        "run_id": member.run_id,
        "effect_id": member.effect_id,
        "ordinal": member.ordinal,
        "sink_node_id": member.sink_node_id,
        "role": member.role.value,
        "token_id": member.token_id,
        "row_id": member.row_id,
        "ingest_sequence": member.ingest_sequence,
        "lineage_hash": member.lineage_hash,
        "payload_hash": member.payload_hash,
        "primary_effect_id": member.primary_effect_id,
        "prepared_disposition": member.prepared_disposition,
        "reason_hash": member.reason_hash,
        "member_effect_id": member.member_effect_id,
        "member_state": None if member.member_state is None else member.member_state.value,
        "descriptor_hash": member.descriptor_hash,
        "evidence_hash": member.evidence_hash,
    }


def sink_effect_attempt_to_export_record(
    run_id: str,
    attempt: SinkEffectAttempt,
    *,
    attempt_index: int,
) -> SinkEffectAttemptExportRecord:
    """Project an ordered call witness while omitting raw provider evidence."""
    return {
        "record_type": "sink_effect_attempt",
        "run_id": run_id,
        "attempt_id": attempt.attempt_id,
        "effect_id": attempt.effect_id,
        "attempt_index": attempt_index,
        "member_ordinal": attempt.member_ordinal,
        "generation": attempt.generation,
        "action": attempt.action.value,
        "call_kind": attempt.call_kind,
        "request_hash": attempt.request_hash,
        "state": attempt.state.value,
        "evidence_hash": attempt.evidence_hash,
        "started_at": attempt.started_at.isoformat(),
        "completed_at": None if attempt.completed_at is None else attempt.completed_at.isoformat(),
        "latency_ms": attempt.latency_ms,
    }


def node_state_to_export_record(run_id: str, state: object) -> NodeStateExportRecord:
    """Project a NodeState variant to its flat export record."""
    if isinstance(state, NodeStateOpen):
        return {
            "record_type": "node_state",
            "run_id": run_id,
            "state_id": state.state_id,
            "token_id": state.token_id,
            "node_id": state.node_id,
            "step_index": state.step_index,
            "attempt": state.attempt,
            "status": state.status.value,
            "input_hash": state.input_hash,
            "output_hash": None,
            "duration_ms": None,
            "started_at": state.started_at.isoformat(),
            "completed_at": None,
            "context_before_json": state.context_before_json,
            "context_after_json": None,
            "error_json": None,
            "success_reason_json": None,
        }
    if isinstance(state, NodeStatePending):
        return {
            "record_type": "node_state",
            "run_id": run_id,
            "state_id": state.state_id,
            "token_id": state.token_id,
            "node_id": state.node_id,
            "step_index": state.step_index,
            "attempt": state.attempt,
            "status": state.status.value,
            "input_hash": state.input_hash,
            "output_hash": None,
            "duration_ms": state.duration_ms,
            "started_at": state.started_at.isoformat(),
            "completed_at": state.completed_at.isoformat(),
            "context_before_json": state.context_before_json,
            "context_after_json": state.context_after_json,
            "error_json": None,
            "success_reason_json": None,
        }
    if isinstance(state, NodeStateCompleted):
        return {
            "record_type": "node_state",
            "run_id": run_id,
            "state_id": state.state_id,
            "token_id": state.token_id,
            "node_id": state.node_id,
            "step_index": state.step_index,
            "attempt": state.attempt,
            "status": state.status.value,
            "input_hash": state.input_hash,
            "output_hash": state.output_hash,
            "duration_ms": state.duration_ms,
            "started_at": state.started_at.isoformat(),
            "completed_at": state.completed_at.isoformat(),
            "context_before_json": state.context_before_json,
            "context_after_json": state.context_after_json,
            "error_json": None,
            "success_reason_json": state.success_reason_json,
        }
    if isinstance(state, NodeStateFailed):
        return {
            "record_type": "node_state",
            "run_id": run_id,
            "state_id": state.state_id,
            "token_id": state.token_id,
            "node_id": state.node_id,
            "step_index": state.step_index,
            "attempt": state.attempt,
            "status": state.status.value,
            "input_hash": state.input_hash,
            "output_hash": state.output_hash,
            "duration_ms": state.duration_ms,
            "started_at": state.started_at.isoformat(),
            "completed_at": state.completed_at.isoformat(),
            "context_before_json": state.context_before_json,
            "context_after_json": state.context_after_json,
            "error_json": state.error_json,
            "success_reason_json": None,
        }

    raise AuditIntegrityError(f"Unknown NodeState variant for export: {type(state).__name__}")
