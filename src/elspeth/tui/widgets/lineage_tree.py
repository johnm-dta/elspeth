"""Lineage tree widget for displaying pipeline lineage."""

from dataclasses import dataclass, field
from typing import Any

from elspeth.tui.types import LineageData


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

    Data Contract:
        This widget requires valid LineageData. Callers must ensure
        data conforms to the contract BEFORE passing it here. Missing
        or malformed fields will raise KeyError, not silently degrade.
    """

    def __init__(self, lineage_data: LineageData) -> None:
        """Initialize with lineage data.

        Args:
            lineage_data: LineageData containing run_id, source, transforms,
                          sinks, tokens. All fields are required.
        """
        self._data = lineage_data
        self._root = self._build_tree()

    def _build_tree(self) -> TreeNode:
        """Build tree structure from lineage data.

        Returns:
            Root TreeNode
        """
        run_id = self._data["run_id"]
        root = TreeNode(label=f"Run: {run_id}", node_type="run")

        # Add source
        source = self._data["source"]
        source_name = source["name"]
        source_node_id = source["node_id"]
        source_node = TreeNode(
            label=f"Source: {source_name}",
            node_id=source_node_id,
            node_type="source",
        )
        root.children.append(source_node)

        # Build transform chain
        transforms = self._data["transforms"]
        current_parent = source_node

        for transform in transforms:
            transform_name = transform["name"]
            transform_node_id = transform["node_id"]
            transform_node = TreeNode(
                label=f"Transform: {transform_name}",
                node_id=transform_node_id,
                node_type="transform",
            )
            current_parent.children.append(transform_node)
            current_parent = transform_node

        # Add sinks as children of last transform (or source if no transforms)
        sinks = self._data["sinks"]
        sink_nodes: dict[str, TreeNode] = {}

        for sink in sinks:
            sink_name = sink["name"]
            sink_node_id = sink["node_id"]
            sink_node = TreeNode(
                label=f"Sink: {sink_name}",
                node_id=sink_node_id,
                node_type="sink",
            )
            current_parent.children.append(sink_node)
            if sink_node_id:
                sink_nodes[sink_node_id] = sink_node

        # Add tokens under their terminal nodes
        tokens = self._data["tokens"]
        for token in tokens:
            token_id = token["token_id"]
            row_id = token["row_id"]
            path = token["path"]
            token_node = TreeNode(
                label=f"Token: {token_id} (row: {row_id})",
                node_id=token_id,
                node_type="token",
            )
            # Find which sink this token ended at
            if path and len(path) > 0:
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
