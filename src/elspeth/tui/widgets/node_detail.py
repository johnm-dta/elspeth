"""Node detail panel widget for displaying node state information."""

import json
from typing import Any


class NodeDetailPanel:
    """Panel displaying detailed information about a selected node.

    Shows:
    - Node identity (plugin name, type, IDs)
    - Status and timing
    - Input/output hashes
    - Errors (if failed)
    - Artifacts (if sink)
    """

    def __init__(self, node_state: dict[str, Any] | None) -> None:
        """Initialize with node state data.

        Args:
            node_state: Dict containing node state fields, or None if nothing selected
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

        # Header
        plugin_name = self._state.get("plugin_name", "unknown")
        node_type = self._state.get("node_type", "unknown")
        lines.append(f"=== {plugin_name} ({node_type}) ===")
        lines.append("")

        # Identity
        lines.append("Identity:")
        lines.append(f"  State ID:  {self._state.get('state_id', 'N/A')}")
        lines.append(f"  Node ID:   {self._state.get('node_id', 'N/A')}")
        lines.append(f"  Token ID:  {self._state.get('token_id', 'N/A')}")
        lines.append("")

        # Status
        status = self._state.get("status", "unknown")
        lines.append("Status:")
        lines.append(f"  Status:     {status}")
        lines.append(f"  Started:    {self._state.get('started_at', 'N/A')}")
        lines.append(f"  Completed:  {self._state.get('completed_at', 'N/A')}")
        duration = self._state.get("duration_ms")
        if duration is not None:
            lines.append(f"  Duration:   {duration} ms")
        lines.append("")

        # Hashes
        lines.append("Data Hashes:")
        input_hash = self._state.get("input_hash")
        output_hash = self._state.get("output_hash")
        lines.append(f"  Input:   {input_hash or '(none)'}")
        lines.append(f"  Output:  {output_hash or '(none)'}")
        lines.append("")

        # Error (if present)
        error_json = self._state.get("error_json")
        if error_json:
            lines.append("Error:")
            try:
                error = json.loads(error_json)
                lines.append(f"  Type:    {error.get('type', 'unknown')}")
                lines.append(f"  Message: {error.get('message', 'unknown')}")
            except json.JSONDecodeError:
                lines.append(f"  {error_json}")
            lines.append("")

        # Artifact (if sink)
        artifact = self._state.get("artifact")
        if artifact:
            lines.append("Artifact:")
            lines.append(f"  ID:      {artifact.get('artifact_id', 'N/A')}")
            lines.append(f"  Path:    {artifact.get('path_or_uri', 'N/A')}")
            lines.append(f"  Hash:    {artifact.get('content_hash', 'N/A')}")
            size_bytes = artifact.get("size_bytes")
            if size_bytes is not None:
                lines.append(f"  Size:    {self._format_size(size_bytes)}")
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

    def update_state(self, node_state: dict[str, Any] | None) -> None:
        """Update the displayed node state.

        Args:
            node_state: New node state to display
        """
        self._state = node_state
