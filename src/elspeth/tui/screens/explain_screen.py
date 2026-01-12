"""Explain screen for lineage visualization."""

from typing import Any

from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.tui.widgets.lineage_tree import LineageTree
from elspeth.tui.widgets.node_detail import NodeDetailPanel


class ExplainScreen:
    """Screen for visualizing pipeline lineage.

    Combines LineageTree and NodeDetailPanel widgets to provide
    an interactive exploration of run lineage.

    Layout:
        ┌─────────────────┬──────────────────┐
        │                 │                  │
        │  Lineage Tree   │   Detail Panel   │
        │                 │                  │
        │                 │                  │
        └─────────────────┴──────────────────┘
    """

    def __init__(
        self,
        db: LandscapeDB | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize explain screen.

        Args:
            db: Landscape database connection
            run_id: Run ID to explain
        """
        self._db = db
        self._run_id = run_id
        self._lineage_data: dict[str, Any] | None = None
        self._selected_node_id: str | None = None

        # Initialize widgets
        self._tree: LineageTree | None = None
        self._detail_panel = NodeDetailPanel(None)

        # Load data if available
        if db and run_id:
            self._load_pipeline_structure()

    def _load_pipeline_structure(self) -> None:
        """Load pipeline structure from database."""
        if not self._db or not self._run_id:
            return

        try:
            recorder = LandscapeRecorder(self._db)
            nodes = recorder.get_nodes(self._run_id)

            # Organize nodes by type
            source_nodes = [n for n in nodes if n.node_type == "source"]
            transform_nodes = [n for n in nodes if n.node_type == "transform"]
            sink_nodes = [n for n in nodes if n.node_type == "sink"]

            self._lineage_data = {
                "run_id": self._run_id,
                "source": {
                    "name": source_nodes[0].plugin_name if source_nodes else "unknown",
                    "node_id": source_nodes[0].node_id if source_nodes else None,
                } if source_nodes else {"name": "unknown", "node_id": None},
                "transforms": [
                    {"name": n.plugin_name, "node_id": n.node_id}
                    for n in transform_nodes
                ],
                "sinks": [
                    {"name": n.plugin_name, "node_id": n.node_id}
                    for n in sink_nodes
                ],
                "tokens": [],  # Tokens loaded separately when needed
            }
            self._tree = LineageTree(self._lineage_data)
        except Exception:
            # Handle missing run or other errors gracefully
            self._lineage_data = None
            self._tree = None

    def get_widget_types(self) -> list[str]:
        """Get list of widget types in this screen.

        Returns:
            List of widget type names
        """
        return ["LineageTree", "NodeDetailPanel"]

    def get_lineage_data(self) -> dict[str, Any] | None:
        """Get current lineage data.

        Returns:
            Lineage data dict or None
        """
        return self._lineage_data

    def on_tree_select(self, node_id: str) -> None:
        """Handle tree node selection.

        Args:
            node_id: Selected node ID
        """
        self._selected_node_id = node_id

        # Load node state from database if available
        if self._db and self._run_id and node_id:
            node_state = self._load_node_state(node_id)
            self._detail_panel.update_state(node_state)
        else:
            self._detail_panel.update_state(None)

    def _load_node_state(self, node_id: str) -> dict[str, Any] | None:
        """Load node state from database.

        Args:
            node_id: Node ID to load

        Returns:
            Node state dict or None
        """
        if not self._db:
            return None

        try:
            recorder = LandscapeRecorder(self._db)
            nodes = recorder.get_nodes(self._run_id or "")

            # Find the node
            for node in nodes:
                if node.node_id == node_id:
                    return {
                        "node_id": node.node_id,
                        "plugin_name": node.plugin_name,
                        "node_type": node.node_type,
                        "status": "registered",  # Node exists but state depends on execution
                    }
            return None
        except Exception:
            return None

    def get_detail_panel_state(self) -> dict[str, Any] | None:
        """Get current detail panel state.

        Returns:
            Node state being displayed or None
        """
        return self._detail_panel._state

    def render(self) -> str:
        """Render the screen as text.

        Returns:
            Rendered screen content
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"  ELSPETH Lineage Explorer - Run: {self._run_id or '(none)'}")
        lines.append("=" * 60)
        lines.append("")

        if self._tree:
            lines.append("--- Lineage Tree ---")
            for node in self._tree.get_tree_nodes():
                indent = "  " * node["depth"]
                lines.append(f"{indent}{node['label']}")
            lines.append("")

        lines.append("--- Node Details ---")
        lines.append(self._detail_panel.render_content())

        return "\n".join(lines)
