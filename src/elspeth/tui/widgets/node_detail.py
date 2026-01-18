"""Node detail panel widget for displaying node state information."""

import json

import structlog

from elspeth.tui.types import NodeStateInfo

logger = structlog.get_logger(__name__)


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
        # Trust boundary: error_json may be malformed or wrong type from Landscape
        error_json = self._state.get("error_json")
        if error_json:
            lines.append("Error:")
            try:
                # Runtime check: Landscape DB may contain corrupted/malformed data
                if not isinstance(error_json, str):  # type: ignore[unreachable]
                    # Non-string error_json - display as-is
                    lines.append(f"  {error_json}")  # type: ignore[unreachable]
                else:
                    error = json.loads(error_json)
                    if isinstance(error, dict):
                        # Error dict fields are external data - use .get() + fallback
                        error_type = error.get("type")
                        error_message = error.get("message")
                        lines.append(f"  Type:    {error_type or 'unknown'}")
                        lines.append(f"  Message: {error_message or 'unknown'}")
                    else:
                        # Parsed but not a dict (e.g., JSON string/number)
                        lines.append(f"  {error}")
            except json.JSONDecodeError as e:
                # Trust boundary: error_json from Landscape DB may be malformed.
                # Log for debugging but display raw - don't crash the TUI.
                logger.warning(
                    "Failed to parse error_json from Landscape",
                    state_id=self._state.get("state_id"),
                    error_json_preview=error_json[:200]
                    if len(error_json) > 200
                    else error_json,
                    decode_error=str(e),
                )
                lines.append(f"  {error_json}")
            lines.append("")

        # Artifact (if sink) - optional field
        # Trust boundary: artifact may be malformed or wrong type from Landscape
        artifact = self._state.get("artifact")
        if artifact:
            lines.append("Artifact:")
            # Runtime check: Landscape DB may contain corrupted/malformed data
            if isinstance(artifact, dict):
                # Artifact dict fields are external data - use .get() + fallback
                artifact_id = artifact.get("artifact_id")
                path_or_uri = artifact.get("path_or_uri")
                content_hash = artifact.get("content_hash")
                lines.append(f"  ID:      {artifact_id or 'N/A'}")
                lines.append(f"  Path:    {path_or_uri or 'N/A'}")
                lines.append(f"  Hash:    {content_hash or 'N/A'}")
                size_bytes = artifact.get("size_bytes")
                if size_bytes is not None and isinstance(size_bytes, int | float):
                    lines.append(f"  Size:    {self._format_size(int(size_bytes))}")
            else:  # type: ignore[unreachable]
                # Non-dict artifact - display as-is
                lines.append(f"  {artifact}")  # type: ignore[unreachable]
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
