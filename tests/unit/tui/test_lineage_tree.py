"""Tests for lineage tree widget."""

import pytest

from elspeth.tui.types import LineageData, NodeInfo, SourceInfo, TokenDisplayInfo


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
