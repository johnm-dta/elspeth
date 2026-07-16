"""Typed audit-record projection helpers for Landscape export."""

from __future__ import annotations

from typing import Literal

from elspeth.contracts import Artifact, NodeStateCompleted, NodeStateFailed, NodeStateOpen, NodeStatePending
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.export_records import ArtifactExportRecord, NodeStateExportRecord


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
