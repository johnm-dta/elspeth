"""Lineage tree widget for displaying pipeline lineage."""

from dataclasses import dataclass

from elspeth.contracts.freeze import freeze_fields
from elspeth.tui.types import LineageData, TreeNodeDict


@dataclass(frozen=True, slots=True)
class TreeNode:
    """Immutable node in the lineage tree.

    Frozen to prevent external mutation via get_node_by_id().
    Children is a tuple for deep immutability.
    """

    label: str
    node_id: str | None = None
    node_type: str = ""
    children: tuple["TreeNode", ...] = ()
    expanded: bool = True

    def __post_init__(self) -> None:
        """Validate construction invariants."""
        if not isinstance(self.label, str):
            raise TypeError(f"label must be str, got {type(self.label).__name__}")
        for i, child in enumerate(self.children):
            if not isinstance(child, TreeNode):
                raise TypeError(f"children[{i}] must be TreeNode, got {type(child).__name__}")
        # deep_freeze is idempotent on already-frozen values (tuple of TreeNode
        # instances), so no type guard on `children` is needed — it will be a
        # tuple of opaque frozen dataclasses and deep_freeze returns identity.
        freeze_fields(self, "children")


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

        Builds bottom-up since TreeNode is frozen (children must be known
        at construction time).

        Returns:
            Root TreeNode
        """
        # Label map for processing node types
        _TYPE_LABELS = {
            "transform": "Transform",
            "gate": "Gate",
            "aggregation": "Aggregation",
            "coalesce": "Coalesce",
        }

        # Step 1: Group tokens by their terminal sink
        tokens_by_sink: dict[str, list[TreeNode]] = {}
        for token in self._data["tokens"]:
            token_id = token["token_id"]
            row_id = token["row_id"]
            path = token["path"]
            token_node = TreeNode(
                label=f"Token: {token_id} (row: {row_id})",
                node_id=token_id,
                node_type="token",
            )
            if path:
                terminal_node_id = path[-1]
                if terminal_node_id not in tokens_by_sink:
                    tokens_by_sink[terminal_node_id] = []
                tokens_by_sink[terminal_node_id].append(token_node)

        # Step 2: Build sink nodes with their token children
        sink_nodes: list[TreeNode] = []
        for sink in self._data["sinks"]:
            sink_name = sink["name"]
            sink_node_id = sink["node_id"]
            # Sink without node_id can't have tokens attached
            token_children = tuple(tokens_by_sink.get(sink_node_id, [])) if sink_node_id is not None else ()
            sink_node = TreeNode(
                label=f"Sink: {sink_name}",
                node_id=sink_node_id,
                node_type="sink",
                children=token_children,
            )
            sink_nodes.append(sink_node)

        # Step 3: Build transform chain backwards
        # KNOWN LIMITATION: DAG pipelines with fork/coalesce are rendered as a
        # linear chain — parallel branches and merge points are not shown.
        # All data for DAG display is available (branch_name, fork_group_id,
        # token_parents) but not yet consumed here. For full DAG exploration,
        # use the Landscape MCP server: elspeth-mcp → explain_token().
        transforms = self._data["transforms"]

        # Start with sinks as children of the last transform
        current_children: tuple[TreeNode, ...] = tuple(sink_nodes)

        # Build transforms in reverse order (last transform wraps sinks,
        # second-to-last wraps last, etc.)
        for transform in reversed(transforms):
            transform_name = transform["name"]
            transform_node_id = transform["node_id"]
            raw_type = transform["node_type"]
            try:
                type_label = _TYPE_LABELS[raw_type]
            except KeyError:
                raise ValueError(
                    f"Unknown node type {raw_type!r} for transform {transform_name!r}. "
                    f"Update _TYPE_LABELS in lineage_tree.py to include this type. "
                    f"Known types: {sorted(_TYPE_LABELS)}"
                ) from None
            transform_node = TreeNode(
                label=f"{type_label}: {transform_name}",
                node_id=transform_node_id,
                node_type=raw_type,
                children=current_children,
            )
            # This transform becomes the child of the previous one
            current_children = (transform_node,)

        # Step 4: Build source node
        source = self._data["source"]
        source_name = source["name"]
        source_node_id = source["node_id"]
        source_node = TreeNode(
            label=f"Source: {source_name or '(unknown)'}",
            node_id=source_node_id,
            node_type="source",
            children=current_children,
        )

        # Step 5: Build root
        run_id = self._data["run_id"]
        return TreeNode(
            label=f"Run: {run_id}",
            node_type="run",
            children=(source_node,),
        )

    def get_tree_nodes(self) -> list[TreeNodeDict]:
        """Get flat list of tree nodes for rendering.

        Returns:
            List of TreeNodeDict with label, node_id, node_type, depth,
            has_children, expanded.
        """
        nodes: list[TreeNodeDict] = []
        self._flatten_tree(self._root, 0, nodes)
        return nodes

    def _flatten_tree(self, node: TreeNode, depth: int, result: list[TreeNodeDict]) -> None:
        """Recursively flatten tree to list.

        Args:
            node: Current node
            depth: Current depth level
            result: List to append to
        """
        result.append(
            TreeNodeDict(
                label=node.label,
                node_id=node.node_id,
                node_type=node.node_type,
                depth=depth,
                has_children=len(node.children) > 0,
                expanded=node.expanded,
            )
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
