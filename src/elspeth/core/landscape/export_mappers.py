"""Typed audit-record projection helpers for Landscape export."""

from __future__ import annotations

from elspeth.contracts import NodeStateCompleted, NodeStateFailed, NodeStateOpen, NodeStatePending
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.export_records import NodeStateExportRecord


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

    state_id = getattr(state, "state_id", "<missing>")
    raise AuditIntegrityError(f"Unknown NodeState variant for export: {type(state).__name__} state_id={state_id!r}")
