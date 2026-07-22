"""Node detail panel widget for displaying node state information."""

import json
from typing import Any, cast

import structlog

from elspeth.tui.types import (
    ArtifactDisplay,
    CoalesceErrorDisplay,
    ExecutionErrorDisplay,
    NodeStateInfo,
    SelectionDetailInfo,
    TransformErrorDisplay,
)

logger = structlog.get_logger(__name__)


def _validate_execution_error(data: dict[str, Any]) -> ExecutionErrorDisplay:
    """Validate and cast a dict to ExecutionErrorDisplay.

    ExecutionError has required fields: exception, type
    Raises KeyError if required fields are missing (Tier 1 - crash on corruption).
    """
    result: ExecutionErrorDisplay = {
        "exception": data["exception"],
        "type": data["type"],
    }
    # Add optional fields only if present (NotRequired semantics)
    if "traceback" in data:
        result["traceback"] = data["traceback"]
    if "phase" in data:
        result["phase"] = data["phase"]
    return result


def _validate_transform_error(data: dict[str, Any]) -> TransformErrorDisplay:
    """Validate and cast a dict to TransformErrorDisplay.

    TransformErrorReason has required field: reason
    Raises KeyError if required field is missing (Tier 1 - crash on corruption).
    """
    result: TransformErrorDisplay = {
        "reason": data["reason"],
    }
    # Add optional fields only if present (NotRequired semantics)
    if "error" in data:
        result["error"] = data["error"]
    if "message" in data:
        result["message"] = data["message"]
    if "error_type" in data:
        result["error_type"] = data["error_type"]
    if "field" in data:
        result["field"] = data["field"]
    return result


def _validate_coalesce_error(data: dict[str, Any]) -> CoalesceErrorDisplay:
    """Validate and cast a dict to CoalesceErrorDisplay.

    CoalesceFailureReason has required fields: failure_reason, expected_branches,
    branches_arrived, merge_policy.

    Raises AuditIntegrityError if required fields are missing (Tier 1 — crash
    on corruption with diagnostic context, not a bare KeyError).
    """
    from elspeth.contracts.errors import AuditIntegrityError

    required = ("failure_reason", "expected_branches", "branches_arrived", "merge_policy")
    missing = [k for k in required if k not in data]
    if missing:
        raise AuditIntegrityError(
            f"Corrupt coalesce error record: missing required fields {missing}. "
            f"Available keys: {sorted(data.keys())}. "
            f"This indicates database corruption (Tier 1 violation)."
        )
    result: CoalesceErrorDisplay = {
        "failure_reason": data["failure_reason"],
        "expected_branches": data["expected_branches"],
        "branches_arrived": data["branches_arrived"],
        "merge_policy": data["merge_policy"],
    }
    if "timeout_ms" in data:
        result["timeout_ms"] = data["timeout_ms"]
    if "select_branch" in data:
        result["select_branch"] = data["select_branch"]
    return result


def _validate_artifact(data: dict[str, Any]) -> ArtifactDisplay:
    """Validate and cast a dict to ArtifactDisplay.

    Artifact has required fields: artifact_id, path_or_uri, content_hash, size_bytes
    Raises KeyError if required fields are missing (Tier 1 - crash on corruption).
    """
    return {
        "artifact_id": data["artifact_id"],
        "path_or_uri": data["path_or_uri"],
        "content_hash": data["content_hash"],
        "size_bytes": data["size_bytes"],
    }


def _parse_json_object(field_name: str, value: Any, state_id: str) -> dict[str, object]:
    """Parse a JSON audit field that must contain an object."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__} - audit integrity violation in state {state_id}")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{field_name} must parse to dict, got {type(parsed).__name__} - audit integrity violation in state {state_id}")
    return cast(dict[str, object], parsed)


def _format_json_value(value: Any) -> str:
    """Render a JSON-compatible value compactly for operator display."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


class NodeDetailPanel:
    """Panel displaying detailed information about a selected node.

    Shows:
    - Node identity (plugin name, type, IDs) - REQUIRED fields
    - Status and timing - optional, depends on execution state
    - Input/output hashes - optional
    - Errors (if failed) - optional
    - Artifacts (if sink) - optional

    Field access follows the three-tier trust model:
    - Required fields (node_id, plugin_name, node_type): Direct access.
      Missing = bug in _load_node_state, should crash.
    - Optional fields: Use .get() without default, then explicit None check
      for display (e.g., `value if value is not None else 'N/A'`).
    """

    def __init__(self, node_state: NodeStateInfo | SelectionDetailInfo | None) -> None:
        """Initialize with node state data.

        Args:
            node_state: NodeStateInfo with required fields, or None if nothing selected
        """
        self._state = node_state

    def render_content(self) -> str:
        """Render panel content as formatted string.

        Returns:
            Formatted string for display
        """
        if self._state is None:
            return "No node selected. Select a node from the tree to view details."

        if "detail_kind" in self._state:
            return self._render_selection_detail(cast(SelectionDetailInfo, self._state))

        lines: list[str] = []

        # Header - REQUIRED fields, direct access (crash if missing = bug)
        plugin_name = self._state["plugin_name"]
        node_type = self._state["node_type"]
        lines.append(f"=== {plugin_name} ({node_type}) ===")
        lines.append("")

        # Identity - node_id is required, others are optional
        lines.append("Identity:")
        state_id = self._state["state_id"] if "state_id" in self._state else None
        lines.append(f"  State ID:  {state_id if state_id is not None else 'N/A'}")
        lines.append(f"  Node ID:   {self._state['node_id']}")  # Required
        token_id = self._state["token_id"] if "token_id" in self._state else None
        lines.append(f"  Token ID:  {token_id if token_id is not None else 'N/A'}")
        lines.append("")

        # Status - all optional (may not have execution state yet)
        status = self._state["status"] if "status" in self._state else None
        lines.append("Status:")
        lines.append(f"  Status:     {status if status is not None else 'N/A'}")
        started_at = self._state["started_at"] if "started_at" in self._state else None
        lines.append(f"  Started:    {started_at if started_at is not None else 'N/A'}")
        completed_at = self._state["completed_at"] if "completed_at" in self._state else None
        lines.append(f"  Completed:  {completed_at if completed_at is not None else 'N/A'}")
        duration = self._state["duration_ms"] if "duration_ms" in self._state else None
        if duration is not None:
            lines.append(f"  Duration:   {duration} ms")
        lines.append("")

        # Hashes - optional
        lines.append("Data Hashes:")
        input_hash = self._state["input_hash"] if "input_hash" in self._state else None
        output_hash = self._state["output_hash"] if "output_hash" in self._state else None
        lines.append(f"  Input:   {input_hash if input_hash is not None else '(none)'}")
        lines.append(f"  Output:  {output_hash if output_hash is not None else '(none)'}")
        lines.append("")

        # Error (if present) - optional field
        # error_json is Tier 1 (our audit data) - if malformed, that's a bug
        audit_context = self._state["state_id"] if "state_id" in self._state else self._state["node_id"]
        if "error_json" in self._state and self._state["error_json"] is not None:
            error_json = self._state["error_json"]
            lines.append("Error:")
            error = _parse_json_object("error_json", error_json, audit_context)

            # Discriminated union: determine error variant by field presence
            # ExecutionError has "type" + "exception"
            # CoalesceFailureReason has "failure_reason"
            # TransformErrorReason has "reason"
            if "type" in error and "exception" in error:
                # ExecutionError variant
                validated = _validate_execution_error(cast(dict[str, Any], error))
                lines.append(f"  Type:    {validated['type']}")
                lines.append(f"  Message: {validated['exception']}")
                if "phase" in validated and validated["phase"]:
                    lines.append(f"  Phase:   {validated['phase']}")
            elif "failure_reason" in error:
                # CoalesceFailureReason variant
                validated_coalesce = _validate_coalesce_error(cast(dict[str, Any], error))
                lines.append(f"  Failure: {validated_coalesce['failure_reason']}")
                lines.append(f"  Policy:  {validated_coalesce['merge_policy']}")
                lines.append(f"  Expected branches: {', '.join(validated_coalesce['expected_branches'])}")
                lines.append(f"  Arrived branches:  {', '.join(validated_coalesce['branches_arrived']) or '(none)'}")
                if "timeout_ms" in validated_coalesce and validated_coalesce["timeout_ms"] is not None:
                    lines.append(f"  Timeout: {validated_coalesce['timeout_ms']} ms")
                if "select_branch" in validated_coalesce and validated_coalesce["select_branch"]:
                    lines.append(f"  Select branch: {validated_coalesce['select_branch']}")
            elif "reason" in error:
                # TransformErrorReason variant
                validated_transform = _validate_transform_error(cast(dict[str, Any], error))
                lines.append(f"  Reason:  {validated_transform['reason']}")
                # Display message from either 'error' or 'message' field
                error_message = validated_transform["error"] if "error" in validated_transform else None
                fallback_message = validated_transform["message"] if "message" in validated_transform else None
                msg = error_message or fallback_message
                if msg:
                    lines.append(f"  Message: {msg}")
                if "field" in validated_transform and validated_transform["field"]:
                    lines.append(f"  Field:   {validated_transform['field']}")
            else:
                # Unknown error format - this is a bug in our recording code
                raise ValueError(
                    f"error_json has unknown format (no 'type'+'exception' or 'reason') - "
                    f"audit integrity violation in state {audit_context}: "
                    f"keys={list(error.keys())}"
                )
            lines.append("")

        # Success reason (if present) - optional field
        # success_reason_json is Tier 1 (our audit data) - if malformed, crash
        if "success_reason_json" in self._state:
            success_reason_json = self._state["success_reason_json"]
            lines.append("Success Reason:")
            success_reason = _parse_json_object("success_reason_json", success_reason_json, audit_context)
            lines.append(f"  Action: {success_reason['action']}")
            for key, label in (
                ("fields_modified", "Fields modified"),
                ("fields_added", "Fields added"),
                ("fields_removed", "Fields removed"),
                ("validation_warnings", "Validation warnings"),
                ("queries_completed", "Queries completed"),
                ("metadata", "Metadata"),
            ):
                if key in success_reason:
                    lines.append(f"  {label}: {_format_json_value(success_reason[key])}")
            lines.append("")

        # Context after (if present) - optional field
        # context_after_json is Tier 1 (our audit data) - if malformed, crash
        if "context_after_json" in self._state:
            context_after_json = self._state["context_after_json"]
            lines.append("Context After:")
            context_after = _parse_json_object("context_after_json", context_after_json, audit_context)
            for line in json.dumps(context_after, indent=2, sort_keys=True).splitlines():
                lines.append(f"  {line}")
            lines.append("")

        # Artifact (if sink) - optional field
        # artifact is Tier 1 (our audit data) - if malformed, that's a bug
        if "artifact" in self._state and self._state["artifact"] is not None:
            artifact = self._state["artifact"]
            lines.append("Artifact:")
            # artifact MUST be a dict (schema contract)
            if not isinstance(artifact, dict):
                raise TypeError(
                    f"artifact must be dict, got {type(artifact).__name__} - audit integrity violation in state {audit_context}"
                )
            # Validate and access fields directly (Tier 1 - crash on missing)
            validated_artifact = _validate_artifact(artifact)
            lines.append(f"  ID:      {validated_artifact['artifact_id']}")
            lines.append(f"  Path:    {validated_artifact['path_or_uri']}")
            lines.append(f"  Hash:    {validated_artifact['content_hash']}")
            lines.append(f"  Size:    {self._format_size(validated_artifact['size_bytes'])}")
            lines.append("")

        return "\n".join(lines)

    def _render_selection_detail(self, detail: SelectionDetailInfo) -> str:
        """Render non-node tree selections."""
        lines = [f"=== {detail['title']} ===", ""]
        detail_kind = detail["detail_kind"]
        if detail_kind == "run":
            lines.append("Run:")
            lines.append(f"  Run ID: {detail['run_id']}")
        elif detail_kind == "token":
            lines.append("Token:")
            lines.append(f"  Token ID: {detail['token_id']}")
            lines.append(f"  Row ID:   {detail['row_id']}")
            if "sink" in detail:
                lines.append(f"  Sink:     {detail['sink']}")
        elif detail_kind == "edge":
            lines.append("Branch:")
            lines.append(f"  Label: {detail['edge_label']}")
            lines.append(f"  From:  {detail['from_node_id']}")
            lines.append(f"  To:    {detail['to_node_id']}")
            if "edge_id" in detail:
                lines.append(f"  Edge ID: {detail['edge_id']}")
        elif detail_kind == "outcome":
            lines.append("Outcome:")
            lines.append(f"  Outcome:   {detail['outcome']}")
            lines.append(f"  Path:      {detail['outcome_path']}")
            lines.append(f"  Completed: {detail['completed']}")
            if "state_id" in detail:
                lines.append(f"  State ID: {detail['state_id']}")
            if "sink_effect_id" in detail:
                lines.append(f"  Sink effect ID: {detail['sink_effect_id']}")
            if "token_id" in detail:
                lines.append(f"  Token ID: {detail['token_id']}")
            if "row_id" in detail:
                lines.append(f"  Row ID:    {detail['row_id']}")
            if "sink" in detail:
                lines.append(f"  Sink:      {detail['sink']}")
            if "error_hash" in detail:
                lines.append(f"  Error hash: {detail['error_hash']}")
            if "artifact_id" in detail:
                lines.append("")
                lines.append("Artifact:")
                lines.append(f"  ID:   {detail['artifact_id']}")
                if "artifact_type" in detail:
                    lines.append(f"  Type: {detail['artifact_type']}")
                if "artifact_path_or_uri" in detail:
                    lines.append(f"  Path: {detail['artifact_path_or_uri']}")
                if "artifact_hash" in detail:
                    lines.append(f"  Hash: {detail['artifact_hash']}")
                if "artifact_size_bytes" in detail:
                    lines.append(f"  Size: {self._format_size(detail['artifact_size_bytes'])}")
                if "artifact_producer_kind" in detail:
                    lines.append(f"  Producer: {detail['artifact_producer_kind']}")
                if "artifact_publication_performed" in detail and "artifact_publication_evidence_kind" in detail:
                    publication = "performed" if detail["artifact_publication_performed"] else "not performed"
                    lines.append(f"  Publication: {publication} ({detail['artifact_publication_evidence_kind']})")
        elif detail_kind == "status":
            lines.append("Status:")
            if "run_id" in detail:
                lines.append(f"  Run ID: {detail['run_id']}")
            if "message" in detail:
                lines.append(f"  Message: {detail['message']}")
        else:
            raise ValueError(f"Unsupported detail kind: {detail_kind}")

        if "message" in detail:
            lines.append("")
            lines.append(detail["message"])
        return "\n".join(lines)

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size in human-readable form.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string like "1.5 KB" or "2.3 MB"
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def update_state(self, node_state: NodeStateInfo | SelectionDetailInfo | None) -> None:
        """Update the displayed node state.

        Args:
            node_state: New node state to display
        """
        self._state = node_state
