"""Lineage tree widget for displaying pipeline lineage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    """Node in the lineage tree."""

    label: str
    node_id: str | None = None
    node_type: str = ""
    children: list["TreeNode"] = field(default_factory=list)
    expanded: bool = True


class LineageTree:
    """Widget for displaying pipeline lineage as a tree.

    Structure:
        Run: <run_id>
        └── Source: <source_name>
            └── Transform: <transform_1>
                └── Transform: <transform_2>
                    ├── Sink: <sink_a>
                    │   └── Token: <token_id>
                    └── Sink: <sink_b>
                        └── Token: <token_id>

    The tree shows the flow of data through the pipeline,
    with tokens as leaves showing which rows went where.
    """

    def __init__(self, lineage_data: dict[str, Any]) -> None:
        """Initialize with lineage data.

        Args:
            lineage_data: Dict containing run_id, source, transforms, sinks, tokens
        """
        self._data = lineage_data
        self._root = self._build_tree()

    def _build_tree(self) -> TreeNode:
        """Build tree structure from lineage data.

        Trust boundary: lineage_data comes from Landscape and may have missing
        or malformed fields for failed/partial runs. All field access uses
        graceful defaults.

        Returns:
            Root TreeNode
        """
        run_id = self._data.get("run_id", "unknown")
        root = TreeNode(label=f"Run: {run_id}", node_type="run")

        # Add source - handle None or non-dict gracefully
        source = self._data.get("source") or {}
        if isinstance(source, dict):
            source_name = source.get("name", "unknown")
            source_node_id = source.get("node_id")
        else:
            source_name = str(source)
            source_node_id = None
        source_node = TreeNode(
            label=f"Source: {source_name}",
            node_id=source_node_id,
            node_type="source",
        )
        root.children.append(source_node)

        # Build transform chain - handle None or non-list gracefully
        transforms = self._data.get("transforms") or []
        current_parent = source_node

        for transform in transforms:
            if isinstance(transform, dict):
                transform_name = transform.get("name", "unknown")
                transform_node_id = transform.get("node_id")
            else:
                transform_name = str(transform) if transform else "unknown"
                transform_node_id = None
            transform_node = TreeNode(
                label=f"Transform: {transform_name}",
                node_id=transform_node_id,
                node_type="transform",
            )
            current_parent.children.append(transform_node)
            current_parent = transform_node

        # Add sinks as children of last transform (or source if no transforms)
        sinks = self._data.get("sinks") or []
        sink_nodes: dict[str, TreeNode] = {}

        for sink in sinks:
            if isinstance(sink, dict):
                sink_name = sink.get("name", "unknown")
                raw_sink_node_id = sink.get("node_id")
                # Only use node_id if it's a hashable string
                sink_node_id = (
                    raw_sink_node_id if isinstance(raw_sink_node_id, str) else None
                )
            else:
                sink_name = str(sink) if sink else "unknown"
                sink_node_id = None
            sink_node = TreeNode(
                label=f"Sink: {sink_name}",
                node_id=sink_node_id,
                node_type="sink",
            )
            current_parent.children.append(sink_node)
            if sink_node_id:
                sink_nodes[sink_node_id] = sink_node

        # Add tokens under their terminal nodes
        tokens = self._data.get("tokens") or []
        for token in tokens:
            if isinstance(token, dict):
                token_id = token.get("token_id", "unknown")
                row_id = token.get("row_id", "unknown")
                path = token.get("path") or []
            else:
                token_id = str(token) if token else "unknown"
                row_id = "unknown"
                path = []
            token_node = TreeNode(
                label=f"Token: {token_id} (row: {row_id})",
                node_id=token_id if isinstance(token_id, str) else None,
                node_type="token",
            )
            # Find which sink this token ended at
            if path and isinstance(path, list) and len(path) > 0:
                terminal_node_id = path[-1]
                if terminal_node_id in sink_nodes:
                    sink_nodes[terminal_node_id].children.append(token_node)

        return root

    def get_tree_nodes(self) -> list[dict[str, Any]]:
        """Get flat list of tree nodes for rendering.

        Returns:
            List of dicts with label, node_id, node_type, depth, has_children
        """
        nodes: list[dict[str, Any]] = []
        self._flatten_tree(self._root, 0, nodes)
        return nodes

    def _flatten_tree(
        self, node: TreeNode, depth: int, result: list[dict[str, Any]]
    ) -> None:
        """Recursively flatten tree to list.

        Args:
            node: Current node
            depth: Current depth level
            result: List to append to
        """
        result.append(
            {
                "label": node.label,
                "node_id": node.node_id,
                "node_type": node.node_type,
                "depth": depth,
                "has_children": len(node.children) > 0,
                "expanded": node.expanded,
            }
        )

        if node.expanded:
            for child in node.children:
                self._flatten_tree(child, depth + 1, result)

    def get_node_by_id(self, node_id: str) -> TreeNode | None:
        """Find a node by its ID.

        Args:
            node_id: Node ID to find

        Returns:
            TreeNode if found, None otherwise
        """
        return self._find_node(self._root, node_id)

    def _find_node(self, node: TreeNode, node_id: str) -> TreeNode | None:
        """Recursively search for node.

        Args:
            node: Current node
            node_id: ID to find

        Returns:
            TreeNode if found, None otherwise
        """
        if node.node_id == node_id:
            return node
        for child in node.children:
            found = self._find_node(child, node_id)
            if found:
                return found
        return None

    def toggle_node(self, node_id: str) -> bool:
        """Toggle expansion state of a node.

        Args:
            node_id: Node ID to toggle

        Returns:
            New expansion state
        """
        node = self.get_node_by_id(node_id)
        if node:
            node.expanded = not node.expanded
            return node.expanded
        return False
