"""Graph-backed lineage view model for the explain TUI."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from elspeth.contracts import NodeType
from elspeth.tui.types import TokenDisplayInfo, TreeSelection


class _NodeLike(Protocol):
    @property
    def node_id(self) -> str: ...

    @property
    def plugin_name(self) -> str: ...

    @property
    def node_type(self) -> NodeType: ...

    @property
    def sequence_in_pipeline(self) -> int | None: ...


class _EdgeLike(Protocol):
    @property
    def edge_id(self) -> str: ...

    @property
    def from_node_id(self) -> str: ...

    @property
    def to_node_id(self) -> str: ...

    @property
    def label(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TuiLineageItem:
    """One flattened row in the explain lineage tree."""

    label: str
    selection: TreeSelection | None
    depth: int
    has_children: bool
    expanded: bool = True
    node_id: str | None = None
    node_type: str = ""
    token_id: str | None = None
    edge_label: str | None = None


@dataclass(frozen=True, slots=True)
class TuiLineageView:
    """Flattened graph view consumed by the Textual tree adapter."""

    run_id: str
    items: tuple[TuiLineageItem, ...]


_TYPE_LABELS = {
    NodeType.SOURCE: "Source",
    NodeType.QUEUE: "Queue",
    NodeType.TRANSFORM: "Transform",
    NodeType.GATE: "Gate",
    NodeType.AGGREGATION: "Aggregation",
    NodeType.COALESCE: "Coalesce",
    NodeType.SINK: "Sink",
}
_HIDDEN_EDGE_LABELS = {"continue", "default", "on_success"}


def build_lineage_view_model(
    *,
    run_id: str,
    nodes: Sequence[_NodeLike],
    edges: Sequence[_EdgeLike],
    tokens: Sequence[TokenDisplayInfo] = (),
) -> TuiLineageView:
    """Build a deterministic tree view from recorded graph nodes and edges."""
    node_by_id = {node.node_id: node for node in nodes}
    outgoing_by_node: dict[str, list[_EdgeLike]] = defaultdict(list)
    incoming_node_ids: set[str] = set()
    for edge in edges:
        if edge.from_node_id not in node_by_id or edge.to_node_id not in node_by_id:
            continue
        outgoing_by_node[edge.from_node_id].append(edge)
        incoming_node_ids.add(edge.to_node_id)

    token_nodes_by_terminal: dict[str, list[TokenDisplayInfo]] = defaultdict(list)
    for token in tokens:
        path = token["path"]
        if path:
            token_nodes_by_terminal[path[-1]].append(token)

    items: list[TuiLineageItem] = [
        TuiLineageItem(
            label=f"Run: {run_id}",
            selection={"kind": "run", "run_id": run_id},
            depth=0,
            has_children=bool(nodes),
            node_type="run",
        )
    ]
    if not nodes:
        items.append(
            TuiLineageItem(
                label="No recorded nodes",
                selection={"kind": "status", "run_id": run_id, "message": "No recorded nodes"},
                depth=1,
                has_children=False,
                expanded=False,
                node_type="status",
            )
        )
        return TuiLineageView(run_id=run_id, items=tuple(items))

    source_roots = [node for node in nodes if node.node_type is NodeType.SOURCE]
    orphan_roots = [node for node in nodes if node.node_type is not NodeType.SOURCE and node.node_id not in incoming_node_ids]
    roots = _sort_nodes(source_roots) + _sort_nodes(orphan_roots)
    if not roots:
        roots = _sort_nodes(nodes)

    expanded_node_ids: set[str] = set()
    for root in roots:
        _append_node(
            items,
            run_id=run_id,
            node=root,
            node_by_id=node_by_id,
            outgoing_by_node=outgoing_by_node,
            token_nodes_by_terminal=token_nodes_by_terminal,
            expanded_node_ids=expanded_node_ids,
            path_node_ids=frozenset(),
            depth=1,
        )

    return TuiLineageView(run_id=run_id, items=tuple(items))


def _append_node(
    items: list[TuiLineageItem],
    *,
    run_id: str,
    node: _NodeLike,
    node_by_id: dict[str, _NodeLike],
    outgoing_by_node: dict[str, list[_EdgeLike]],
    token_nodes_by_terminal: dict[str, list[TokenDisplayInfo]],
    expanded_node_ids: set[str],
    path_node_ids: frozenset[str],
    depth: int,
) -> None:
    label = _node_label(node)
    selection = _node_selection(run_id, node)
    if node.node_id in path_node_ids or node.node_id in expanded_node_ids:
        items.append(
            TuiLineageItem(
                label=f"Repeated: {label} (already shown)",
                selection=selection,
                depth=depth,
                has_children=False,
                expanded=False,
                node_id=node.node_id,
                node_type=node.node_type.value,
            )
        )
        return

    outgoing_edges = _sort_edges(outgoing_by_node.get(node.node_id, []))
    token_children = sorted(token_nodes_by_terminal.get(node.node_id, []), key=lambda token: (token["row_id"], token["token_id"]))
    items.append(
        TuiLineageItem(
            label=label,
            selection=selection,
            depth=depth,
            has_children=bool(outgoing_edges or token_children),
            node_id=node.node_id,
            node_type=node.node_type.value,
        )
    )
    expanded_node_ids.add(node.node_id)

    next_path_node_ids = path_node_ids | {node.node_id}
    for edge in outgoing_edges:
        child_depth = depth + 1
        if _should_show_edge_row(edge, outgoing_edges):
            items.append(
                TuiLineageItem(
                    label=f"Branch: {edge.label}",
                    selection=_edge_selection(run_id, edge),
                    depth=child_depth,
                    has_children=True,
                    node_type="edge",
                    edge_label=edge.label,
                )
            )
            child_depth += 1
        child = node_by_id[edge.to_node_id]
        _append_node(
            items,
            run_id=run_id,
            node=child,
            node_by_id=node_by_id,
            outgoing_by_node=outgoing_by_node,
            token_nodes_by_terminal=token_nodes_by_terminal,
            expanded_node_ids=expanded_node_ids,
            path_node_ids=next_path_node_ids,
            depth=child_depth,
        )

    for token in token_children:
        items.append(
            TuiLineageItem(
                label=f"Token: {token['token_id']} (row: {token['row_id']})",
                selection={
                    "kind": "token",
                    "run_id": run_id,
                    "token_id": token["token_id"],
                    "row_id": token["row_id"],
                },
                depth=depth + 1,
                has_children=False,
                expanded=False,
                node_type="token",
                token_id=token["token_id"],
            )
        )


def _sort_nodes(nodes: Sequence[_NodeLike]) -> list[_NodeLike]:
    return sorted(
        nodes,
        key=lambda node: (
            node.sequence_in_pipeline is None,
            node.sequence_in_pipeline if node.sequence_in_pipeline is not None else 0,
            node.plugin_name,
            node.node_id,
        ),
    )


def _sort_edges(edges: Sequence[_EdgeLike]) -> list[_EdgeLike]:
    return sorted(edges, key=lambda edge: (edge.label, edge.to_node_id, edge.edge_id))


def _node_label(node: _NodeLike) -> str:
    type_label = _TYPE_LABELS[node.node_type]
    return f"{type_label}: {node.plugin_name}"


def _node_selection(run_id: str, node: _NodeLike) -> TreeSelection:
    return {
        "kind": "node",
        "run_id": run_id,
        "node_id": node.node_id,
        "node_type": node.node_type.value,
    }


def _edge_selection(run_id: str, edge: _EdgeLike) -> TreeSelection:
    return {
        "kind": "edge",
        "run_id": run_id,
        "edge_id": edge.edge_id,
        "from_node_id": edge.from_node_id,
        "to_node_id": edge.to_node_id,
        "edge_label": edge.label,
    }


def _should_show_edge_row(edge: _EdgeLike, siblings: Sequence[_EdgeLike]) -> bool:
    return len(siblings) > 1 or edge.label not in _HIDDEN_EDGE_LABELS
