"""Tests for TUI type contracts."""

import pytest

from elspeth.tui.types import LineageData, NodeInfo, SourceInfo, TokenDisplayInfo


class TestLineageDataContract:
    """Tests for LineageData TypedDict."""

    def test_valid_lineage_data(self) -> None:
        """Valid data should pass type checking."""
        data: LineageData = {
            "run_id": "run-123",
            "source": {"name": "csv", "node_id": "node-1"},
            "transforms": [{"name": "mapper", "node_id": "node-2"}],
            "sinks": [{"name": "output", "node_id": "node-3"}],
            "tokens": [],
        }
        assert data["run_id"] == "run-123"

    def test_source_info_contract(self) -> None:
        """SourceInfo requires name and node_id."""
        source: SourceInfo = {"name": "csv_source", "node_id": "src-001"}
        assert source["name"] == "csv_source"
        assert source["node_id"] == "src-001"

    def test_source_info_with_none_node_id(self) -> None:
        """SourceInfo allows None for node_id."""
        source: SourceInfo = {"name": "csv_source", "node_id": None}
        assert source["name"] == "csv_source"
        assert source["node_id"] is None

    def test_node_info_contract(self) -> None:
        """NodeInfo requires name and node_id."""
        node: NodeInfo = {"name": "filter_transform", "node_id": "tfm-001"}
        assert node["name"] == "filter_transform"
        assert node["node_id"] == "tfm-001"

    def test_node_info_with_none_node_id(self) -> None:
        """NodeInfo allows None for node_id."""
        node: NodeInfo = {"name": "filter_transform", "node_id": None}
        assert node["name"] == "filter_transform"
        assert node["node_id"] is None

    def test_token_info_contract(self) -> None:
        """TokenInfo requires token_id, row_id, and path."""
        token: TokenDisplayInfo = {
            "token_id": "tok-001",
            "row_id": "row-001",
            "path": ["node-1", "node-2", "node-3"],
        }
        assert token["token_id"] == "tok-001"
        assert token["row_id"] == "row-001"
        assert token["path"] == ["node-1", "node-2", "node-3"]

    def test_token_info_with_empty_path(self) -> None:
        """TokenInfo allows empty path list."""
        token: TokenDisplayInfo = {
            "token_id": "tok-001",
            "row_id": "row-001",
            "path": [],
        }
        assert token["path"] == []

    def test_lineage_data_complete_example(self) -> None:
        """Complete LineageData with all components."""
        data: LineageData = {
            "run_id": "run-abc-123",
            "source": {"name": "api_source", "node_id": "src-001"},
            "transforms": [
                {"name": "normalize", "node_id": "tfm-001"},
                {"name": "validate", "node_id": "tfm-002"},
            ],
            "sinks": [
                {"name": "database", "node_id": "sink-001"},
                {"name": "archive", "node_id": "sink-002"},
            ],
            "tokens": [
                {
                    "token_id": "tok-001",
                    "row_id": "row-001",
                    "path": ["src-001", "tfm-001", "tfm-002", "sink-001"],
                },
                {
                    "token_id": "tok-002",
                    "row_id": "row-002",
                    "path": ["src-001", "tfm-001", "tfm-002", "sink-002"],
                },
            ],
        }
        assert data["run_id"] == "run-abc-123"
        assert data["source"]["name"] == "api_source"
        assert len(data["transforms"]) == 2
        assert len(data["sinks"]) == 2
        assert len(data["tokens"]) == 2

    def test_lineage_data_empty_transforms_and_tokens(self) -> None:
        """LineageData with empty transforms and tokens lists."""
        data: LineageData = {
            "run_id": "run-minimal",
            "source": {"name": "simple", "node_id": "src-001"},
            "transforms": [],
            "sinks": [{"name": "output", "node_id": "sink-001"}],
            "tokens": [],
        }
        assert data["transforms"] == []
        assert data["tokens"] == []


class TestLineageDataWithLineageTree:
    """Integration tests for LineageData with LineageTree widget."""

    def test_lineage_tree_accepts_lineage_data(self) -> None:
        """LineageTree accepts properly formed LineageData."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        data: LineageData = {
            "run_id": "run-123",
            "source": {"name": "csv", "node_id": "node-1"},
            "transforms": [{"name": "mapper", "node_id": "node-2"}],
            "sinks": [{"name": "output", "node_id": "node-3"}],
            "tokens": [],
        }
        tree = LineageTree(data)
        nodes = tree.get_tree_nodes()
        assert len(nodes) > 0

    def test_lineage_tree_rejects_missing_run_id(self) -> None:
        """LineageTree raises KeyError for missing run_id."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        # Deliberately malformed data - missing run_id
        data = {
            "source": {"name": "csv", "node_id": "node-1"},
            "transforms": [],
            "sinks": [],
            "tokens": [],
        }
        with pytest.raises(KeyError, match="run_id"):
            LineageTree(data)  # type: ignore[arg-type]

    def test_lineage_tree_rejects_missing_source(self) -> None:
        """LineageTree raises KeyError for missing source."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        # Deliberately malformed data - missing source
        data = {
            "run_id": "run-123",
            "transforms": [],
            "sinks": [],
            "tokens": [],
        }
        with pytest.raises(KeyError, match="source"):
            LineageTree(data)  # type: ignore[arg-type]

    def test_lineage_tree_rejects_malformed_source(self) -> None:
        """LineageTree raises KeyError for malformed source."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        # Deliberately malformed data - source missing 'name'
        data = {
            "run_id": "run-123",
            "source": {"node_id": "node-1"},  # missing 'name'
            "transforms": [],
            "sinks": [],
            "tokens": [],
        }
        with pytest.raises(KeyError, match="name"):
            LineageTree(data)  # type: ignore[arg-type]
