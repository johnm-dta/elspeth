"""Node detail panel widget for displaying node state information."""

import json
from typing import Any

import structlog

from elspeth.tui.types import (
    ArtifactDisplay,
    ExecutionErrorDisplay,
    NodeStateInfo,
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
    - Optional fields: Use .get() without default, then explicit fallback
      for display (e.g., `value or 'N/A'`).
    """

    def __init__(self, node_state: NodeStateInfo | None) -> None:
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

        lines: list[str] = []

        # Header - REQUIRED fields, direct access (crash if missing = bug)
        plugin_name = self._state["plugin_name"]
        node_type = self._state["node_type"]
        lines.append(f"=== {plugin_name} ({node_type}) ===")
        lines.append("")

        # Identity - node_id is required, others are optional
        lines.append("Identity:")
        state_id = self._state.get("state_id")
        lines.append(f"  State ID:  {state_id or 'N/A'}")
        lines.append(f"  Node ID:   {self._state['node_id']}")  # Required
        token_id = self._state.get("token_id")
        lines.append(f"  Token ID:  {token_id or 'N/A'}")
        lines.append("")

        # Status - all optional (may not have execution state yet)
        status = self._state.get("status")
        lines.append("Status:")
        lines.append(f"  Status:     {status or 'N/A'}")
        started_at = self._state.get("started_at")
        lines.append(f"  Started:    {started_at or 'N/A'}")
        completed_at = self._state.get("completed_at")
        lines.append(f"  Completed:  {completed_at or 'N/A'}")
        duration = self._state.get("duration_ms")
        if duration is not None:
            lines.append(f"  Duration:   {duration} ms")
        lines.append("")

        # Hashes - optional
        lines.append("Data Hashes:")
        input_hash = self._state.get("input_hash")
        output_hash = self._state.get("output_hash")
        lines.append(f"  Input:   {input_hash or '(none)'}")
        lines.append(f"  Output:  {output_hash or '(none)'}")
        lines.append("")

        # Error (if present) - optional field
        # error_json is Tier 1 (our audit data) - if malformed, that's a bug
        error_json = self._state.get("error_json")
        if error_json:
            lines.append("Error:")
            # error_json MUST be a string (schema contract)
            if not isinstance(error_json, str):
                raise TypeError(
                    f"error_json must be str, got {type(error_json).__name__} - "
                    f"audit integrity violation in state {self._state.get('state_id')}"
                )
            error = json.loads(error_json)  # Let JSONDecodeError crash - it's our data
            if not isinstance(error, dict):
                raise TypeError(
                    f"error_json must parse to dict, got {type(error).__name__} - "
                    f"audit integrity violation in state {self._state.get('state_id')}"
                )

            # Discriminated union: determine error variant by field presence
            # ExecutionError has "type" + "exception", TransformErrorReason has "reason"
            if "type" in error and "exception" in error:
                # ExecutionError variant
                validated = _validate_execution_error(error)
                lines.append(f"  Type:    {validated['type']}")
                lines.append(f"  Message: {validated['exception']}")
                if validated.get("phase"):
                    lines.append(f"  Phase:   {validated['phase']}")
            elif "reason" in error:
                # TransformErrorReason variant
                validated_transform = _validate_transform_error(error)
                lines.append(f"  Reason:  {validated_transform['reason']}")
                # Display message from either 'error' or 'message' field
                msg = validated_transform.get("error") or validated_transform.get("message")
                if msg:
                    lines.append(f"  Message: {msg}")
                if validated_transform.get("field"):
                    lines.append(f"  Field:   {validated_transform['field']}")
            else:
                # Unknown error format - this is a bug in our recording code
                raise ValueError(
                    f"error_json has unknown format (no 'type'+'exception' or 'reason') - "
                    f"audit integrity violation in state {self._state.get('state_id')}: "
                    f"keys={list(error.keys())}"
                )
            lines.append("")

        # Artifact (if sink) - optional field
        # artifact is Tier 1 (our audit data) - if malformed, that's a bug
        artifact = self._state.get("artifact")
        if artifact:
            lines.append("Artifact:")
            # artifact MUST be a dict (schema contract)
            if not isinstance(artifact, dict):
                raise TypeError(
                    f"artifact must be dict, got {type(artifact).__name__} - "
                    f"audit integrity violation in state {self._state.get('state_id')}"
                )
            # Validate and access fields directly (Tier 1 - crash on missing)
            validated_artifact = _validate_artifact(artifact)
            lines.append(f"  ID:      {validated_artifact['artifact_id']}")
            lines.append(f"  Path:    {validated_artifact['path_or_uri']}")
            lines.append(f"  Hash:    {validated_artifact['content_hash']}")
            lines.append(f"  Size:    {self._format_size(validated_artifact['size_bytes'])}")
            lines.append("")

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

    def update_state(self, node_state: NodeStateInfo | None) -> None:
        """Update the displayed node state.

        Args:
            node_state: New node state to display
        """
        self._state = node_state
