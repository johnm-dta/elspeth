"""Tests for lineage tree widget."""

from datetime import UTC, datetime

import pytest

from elspeth.contracts import Determinism, NodeType, RoutingMode
from elspeth.contracts.audit import Edge, Node
from elspeth.tui.types import LineageData, NodeInfo, SourceInfo, TokenDisplayInfo


def _node(node_id: str, plugin_name: str, node_type: NodeType, *, sequence: int) -> Node:
    return Node(
        node_id=node_id,
        run_id="run-1",
        plugin_name=plugin_name,
        node_type=node_type,
        plugin_version="1.0",
        determinism=Determinism.DETERMINISTIC,
        config_hash="cfg",
        config_json="{}",
        registered_at=datetime(2026, 1, 1, tzinfo=UTC),
        sequence_in_pipeline=sequence,
    )


def _edge(edge_id: str, from_node_id: str, to_node_id: str, label: str) -> Edge:
    return Edge(
        edge_id=edge_id,
        run_id="run-1",
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        label=label,
        default_mode=RoutingMode.MOVE,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestLineageTreeWidget:
    """Tests for LineageTree widget."""

    def test_widget_accepts_lineage_data(self) -> None:
        """Widget can be initialized with lineage data."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        # Sample lineage structure
        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [
                NodeInfo(name="passthrough", node_id="node-002", node_type="transform"),
                NodeInfo(name="filter", node_id="node-003", node_type="transform"),
            ],
            "sinks": [
                NodeInfo(name="output", node_id="node-004", node_type="sink"),
            ],
            "tokens": [
                TokenDisplayInfo(
                    token_id="token-001",
                    row_id="row-001",
                    path=["node-001", "node-002", "node-003", "node-004"],
                ),
            ],
        }

        tree = LineageTree(lineage_data)
        assert tree is not None

    def test_widget_builds_tree_structure(self) -> None:
        """Widget builds correct tree structure from lineage."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [NodeInfo(name="filter", node_id="node-002", node_type="transform")],
            "sinks": [NodeInfo(name="output", node_id="node-003", node_type="sink")],
            "tokens": [
                TokenDisplayInfo(
                    token_id="token-001",
                    row_id="row-001",
                    path=["node-001", "node-002", "node-003"],
                ),
            ],
        }

        tree = LineageTree(lineage_data)
        nodes = tree.get_tree_nodes()

        # Should have root, source, transforms, sinks sections
        node_labels = [n["label"] for n in nodes]
        assert any("csv_source" in label for label in node_labels)
        assert any("filter" in label for label in node_labels)
        assert any("output" in label for label in node_labels)

    def test_widget_with_empty_transforms(self) -> None:
        """Widget handles pipeline with no transforms."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [],
            "sinks": [NodeInfo(name="output", node_id="node-002", node_type="sink")],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)
        assert tree is not None

    def test_widget_with_forked_tokens(self) -> None:
        """Widget handles tokens that forked to multiple paths."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [NodeInfo(name="threshold_gate", node_id="node-002", node_type="gate")],
            "sinks": [
                NodeInfo(name="high", node_id="node-003", node_type="sink"),
                NodeInfo(name="low", node_id="node-004", node_type="sink"),
            ],
            "tokens": [
                TokenDisplayInfo(
                    token_id="token-001",
                    row_id="row-001",
                    path=["node-001", "node-002", "node-003"],
                ),
                TokenDisplayInfo(
                    token_id="token-002",
                    row_id="row-002",
                    path=["node-001", "node-002", "node-004"],
                ),
            ],
        }

        tree = LineageTree(lineage_data)
        nodes = tree.get_tree_nodes()

        # Should show both sink paths
        node_labels = [n["label"] for n in nodes]
        assert any("high" in label for label in node_labels)
        assert any("low" in label for label in node_labels)

    def test_get_node_by_id(self) -> None:
        """Can find nodes by their ID."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [NodeInfo(name="filter", node_id="node-002", node_type="transform")],
            "sinks": [NodeInfo(name="output", node_id="node-003", node_type="sink")],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)

        node = tree.get_node_by_id("node-002")
        assert node is not None
        assert "filter" in node.label

        # Non-existent node
        missing = tree.get_node_by_id("nonexistent")
        assert missing is None

    def test_gate_aggregation_coalesce_labels(self) -> None:
        """Gate, aggregation, and coalesce nodes display with correct type labels."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [
                NodeInfo(name="field_mapper", node_id="node-002", node_type="transform"),
                NodeInfo(name="threshold_check", node_id="node-003", node_type="gate"),
                NodeInfo(name="batch_agg", node_id="node-004", node_type="aggregation"),
                NodeInfo(name="merge_results", node_id="node-005", node_type="coalesce"),
            ],
            "sinks": [NodeInfo(name="output", node_id="node-006", node_type="sink")],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)
        nodes = tree.get_tree_nodes()
        labels = [n["label"] for n in nodes]

        # Each node type should use its specific label prefix
        assert "Transform: field_mapper" in labels
        assert "Gate: threshold_check" in labels
        assert "Aggregation: batch_agg" in labels
        assert "Coalesce: merge_results" in labels
        assert "Sink: output" in labels

        # Verify node_type is propagated to tree nodes
        node_types = {n["label"]: n["node_type"] for n in nodes}
        assert node_types["Gate: threshold_check"] == "gate"
        assert node_types["Aggregation: batch_agg"] == "aggregation"
        assert node_types["Coalesce: merge_results"] == "coalesce"

    def test_graph_view_renders_multiple_sources_without_dropping_second_source(self) -> None:
        """Graph-backed view preserves multi-source topology and branch labels."""
        from elspeth.tui.lineage_view import build_lineage_view_model
        from elspeth.tui.widgets.lineage_tree import LineageTree

        view = build_lineage_view_model(
            run_id="run-1",
            nodes=[
                _node("src-a", "csv", NodeType.SOURCE, sequence=0),
                _node("src-b", "json", NodeType.SOURCE, sequence=1),
                _node("merge", "coalesce", NodeType.COALESCE, sequence=2),
                _node("sink", "json_sink", NodeType.SINK, sequence=3),
            ],
            edges=[
                _edge("edge-a", "src-a", "merge", "a"),
                _edge("edge-b", "src-b", "merge", "b"),
                _edge("edge-sink", "merge", "sink", "default"),
            ],
        )

        labels = [node["label"] for node in LineageTree(view).get_tree_nodes()]

        assert "Source: csv" in labels
        assert "Source: json" in labels
        assert "Coalesce: coalesce" in labels
        assert "Branch: a" in labels
        assert "Branch: b" in labels

    def test_graph_view_orders_unsorted_edges_deterministically(self) -> None:
        """Graph renderer sorts outgoing edges by label then destination."""
        from elspeth.tui.lineage_view import build_lineage_view_model

        view = build_lineage_view_model(
            run_id="run-1",
            nodes=[
                _node("src", "csv", NodeType.SOURCE, sequence=0),
                _node("branch-b", "b_mapper", NodeType.TRANSFORM, sequence=2),
                _node("branch-a", "a_mapper", NodeType.TRANSFORM, sequence=1),
            ],
            edges=[
                _edge("edge-b", "src", "branch-b", "b"),
                _edge("edge-a", "src", "branch-a", "a"),
            ],
        )

        labels = [item.label for item in view.items]

        assert labels.index("Branch: a") < labels.index("Branch: b")
        assert labels.index("Transform: a_mapper") < labels.index("Transform: b_mapper")

    def test_graph_view_diamond_join_terminates_with_repeated_marker(self) -> None:
        """Diamond DAG joins render once and then show a repeated-node marker."""
        from elspeth.tui.lineage_view import build_lineage_view_model

        view = build_lineage_view_model(
            run_id="run-1",
            nodes=[
                _node("src", "csv", NodeType.SOURCE, sequence=0),
                _node("left", "left_map", NodeType.TRANSFORM, sequence=1),
                _node("right", "right_map", NodeType.TRANSFORM, sequence=2),
                _node("merge", "merge", NodeType.COALESCE, sequence=3),
                _node("sink", "json_sink", NodeType.SINK, sequence=4),
            ],
            edges=[
                _edge("edge-right", "src", "right", "right"),
                _edge("edge-left", "src", "left", "left"),
                _edge("edge-right-merge", "right", "merge", "continue"),
                _edge("edge-left-merge", "left", "merge", "continue"),
                _edge("edge-sink", "merge", "sink", "default"),
            ],
        )

        labels = [item.label for item in view.items]

        assert labels.count("Coalesce: merge") == 1
        assert labels.count("Sink: json_sink") == 1
        assert "Repeated: Coalesce: merge (already shown)" in labels
        assert len(labels) < 20

    def test_focused_token_uses_exact_diamond_branch_path(self) -> None:
        """Focused token rows attach to the actual traversed branch instance."""
        from elspeth.tui.lineage_view import build_lineage_view_model

        view = build_lineage_view_model(
            run_id="run-1",
            nodes=[
                _node("src", "csv", NodeType.SOURCE, sequence=0),
                _node("left", "left_map", NodeType.TRANSFORM, sequence=1),
                _node("right", "right_map", NodeType.TRANSFORM, sequence=2),
                _node("merge", "merge", NodeType.COALESCE, sequence=3),
                _node("sink", "json_sink", NodeType.SINK, sequence=4),
            ],
            edges=[
                _edge("edge-left", "src", "left", "left"),
                _edge("edge-right", "src", "right", "right"),
                _edge("edge-left-merge", "left", "merge", "continue"),
                _edge("edge-right-merge", "right", "merge", "continue"),
                _edge("edge-sink", "merge", "sink", "default"),
            ],
            tokens=[
                TokenDisplayInfo(
                    token_id="token-right",
                    row_id="row-1",
                    path=["src", "right", "merge", "sink"],
                )
            ],
        )

        labels = [item.label for item in view.items]
        token_index = labels.index("Token: token-right (row: row-1)")

        assert labels.index("Branch: right") < token_index
        assert labels.index("Transform: right_map") < token_index
        assert labels.index("Branch: left") < labels.index("Transform: left_map") < labels.index("Branch: right")

    def test_focused_token_outcome_renders_artifact_selection(self) -> None:
        """Focused token outcome rows carry sink and artifact evidence."""
        from elspeth.tui.lineage_view import build_lineage_view_model

        view = build_lineage_view_model(
            run_id="run-1",
            nodes=[
                _node("src", "csv", NodeType.SOURCE, sequence=0),
                _node("sink", "json_sink", NodeType.SINK, sequence=1),
            ],
            edges=[_edge("edge-sink", "src", "sink", "default")],
            tokens=[
                TokenDisplayInfo(
                    token_id="token-1",
                    row_id="row-1",
                    path=["src", "sink"],
                    outcome={
                        "outcome": "success",
                        "path": "default_flow",
                        "completed": True,
                        "sink": "json_sink",
                        "artifact": {
                            "artifact_id": "artifact-1",
                            "artifact_type": "json",
                            "path_or_uri": "/tmp/out.json",
                            "content_hash": "sha256:abc",
                            "size_bytes": 12,
                            "sink_node_id": "sink",
                            "produced_by_state_id": "state-sink",
                        },
                    },
                )
            ],
        )

        outcome = next(item for item in view.items if item.node_type == "outcome")

        assert outcome.label == "Outcome: success / default_flow -> json_sink"
        assert outcome.selection == {
            "kind": "outcome",
            "run_id": "run-1",
            "token_id": "token-1",
            "row_id": "row-1",
            "outcome": "success",
            "outcome_path": "default_flow",
            "completed": True,
            "sink": "json_sink",
            "artifact_id": "artifact-1",
            "artifact_type": "json",
            "artifact_path_or_uri": "/tmp/out.json",
            "artifact_hash": "sha256:abc",
            "artifact_size_bytes": 12,
            "state_id": "state-sink",
        }


class TestTreeNodeImmutability:
    """Tests for TreeNode frozen dataclass invariants."""

    def test_tree_node_is_frozen(self) -> None:
        """TreeNode attributes cannot be reassigned."""
        from dataclasses import FrozenInstanceError

        from elspeth.tui.widgets.lineage_tree import TreeNode

        node = TreeNode(label="test", node_type="test")
        with pytest.raises(FrozenInstanceError):
            node.label = "modified"  # type: ignore[misc]

    def test_tree_node_children_is_tuple(self) -> None:
        """TreeNode.children is immutable tuple, not list."""
        from elspeth.tui.widgets.lineage_tree import TreeNode

        child = TreeNode(label="child", node_type="token")
        parent = TreeNode(label="parent", node_type="sink", children=(child,))

        assert isinstance(parent.children, tuple)
        # Tuple doesn't have .append()
        assert not hasattr(parent.children, "append")

    def test_get_node_by_id_returns_immutable_node(self) -> None:
        """Nodes returned by get_node_by_id() cannot be mutated."""
        from dataclasses import FrozenInstanceError

        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [NodeInfo(name="filter", node_id="node-002", node_type="transform")],
            "sinks": [NodeInfo(name="output", node_id="node-003", node_type="sink")],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)
        node = tree.get_node_by_id("node-002")
        assert node is not None

        # Cannot mutate returned node
        with pytest.raises(FrozenInstanceError):
            node.expanded = False  # type: ignore[misc]

    def test_tree_node_normalizes_list_children_to_tuple(self) -> None:
        """TreeNode deep-freezes container fields in __post_init__, so a list
        of children (if the caller's type annotation was ignored) is normalised
        to a tuple rather than rejected.

        This is the correct behaviour per the freeze contract: deep_freeze
        converts list → tuple, preserving the deep-immutability invariant
        without requiring defensive isinstance guards on the container type.
        The per-element isinstance(child, TreeNode) check remains the
        relevant Tier 1 invariant.
        """
        from elspeth.tui.widgets.lineage_tree import TreeNode

        node = TreeNode(label="test", node_type="test", children=[])  # type: ignore[arg-type]
        assert node.children == ()
        assert isinstance(node.children, tuple)

    def test_tree_node_rejects_non_tree_node_children(self) -> None:
        """TreeNode raises TypeError for children that aren't TreeNodes."""
        from elspeth.tui.widgets.lineage_tree import TreeNode

        with pytest.raises(TypeError, match=r"children\[0\] must be TreeNode"):
            TreeNode(label="test", node_type="test", children=("not a node",))  # type: ignore[arg-type]

    def test_tree_node_rejects_non_string_label(self) -> None:
        """TreeNode raises TypeError for non-string label."""
        from elspeth.tui.widgets.lineage_tree import TreeNode

        with pytest.raises(TypeError, match="label must be str"):
            TreeNode(label=123, node_type="test")  # type: ignore[arg-type]
