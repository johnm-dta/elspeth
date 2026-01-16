"""Tests for lineage tree widget."""

from elspeth.tui.types import LineageData, NodeInfo, SourceInfo, TokenDisplayInfo


class TestLineageTreeWidget:
    """Tests for LineageTree widget."""

    def test_can_import_widget(self) -> None:
        """LineageTree widget can be imported."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        assert LineageTree is not None

    def test_widget_accepts_lineage_data(self) -> None:
        """Widget can be initialized with lineage data."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        # Sample lineage structure
        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [
                NodeInfo(name="passthrough", node_id="node-002"),
                NodeInfo(name="filter", node_id="node-003"),
            ],
            "sinks": [
                NodeInfo(name="output", node_id="node-004"),
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
            "transforms": [NodeInfo(name="filter", node_id="node-002")],
            "sinks": [NodeInfo(name="output", node_id="node-003")],
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
            "sinks": [NodeInfo(name="output", node_id="node-002")],
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
            "transforms": [NodeInfo(name="gate", node_id="node-002")],
            "sinks": [
                NodeInfo(name="high", node_id="node-003"),
                NodeInfo(name="low", node_id="node-004"),
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

    def test_toggle_node_expansion(self) -> None:
        """Can toggle node expansion state."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [],
            "sinks": [NodeInfo(name="output", node_id="node-002")],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)

        # Toggle source node
        new_state = tree.toggle_node("node-001")
        assert new_state is False  # Was expanded, now collapsed

        # Toggle again
        new_state = tree.toggle_node("node-001")
        assert new_state is True  # Back to expanded

    def test_get_node_by_id(self) -> None:
        """Can find nodes by their ID."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data: LineageData = {
            "run_id": "run-001",
            "source": SourceInfo(name="csv_source", node_id="node-001"),
            "transforms": [NodeInfo(name="filter", node_id="node-002")],
            "sinks": [NodeInfo(name="output", node_id="node-003")],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)

        node = tree.get_node_by_id("node-002")
        assert node is not None
        assert "filter" in node.label

        # Non-existent node
        missing = tree.get_node_by_id("nonexistent")
        assert missing is None
