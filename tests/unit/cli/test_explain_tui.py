# tests/unit/cli/test_explain_tui.py
"""Tests for explain command TUI screen state model.

Migrated from tests/cli/test_explain_tui.py.
Tests that require LandscapeDB (screen loading, state transitions with real DB)
are deferred to integration tier. Uses RecorderFactory to access data_flow repository.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from elspeth.contracts import NodeType
from elspeth.tui.screens.explain_screen import (
    ExplainScreen,
    InvalidStateTransitionError,
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)


class TestExplainScreen:
    """Tests for ExplainScreen component."""

    def test_explain_screen_has_detail_panel(self) -> None:
        """Explain screen includes NodeDetailPanel widget."""
        from elspeth.tui.screens.explain_screen import ExplainScreen
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        screen = ExplainScreen()

        assert isinstance(screen.detail_panel, NodeDetailPanel)

    def test_render_without_data(self) -> None:
        """Screen renders gracefully without data."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        content = screen.render()

        assert "ELSPETH" in content
        assert "No node selected" in content or "Select a node" in content


class TestExplainScreenStateModel:
    """Tests for the discriminated union state model (no DB required)."""

    def test_uninitialized_state_without_db(self) -> None:
        """Screen without db/run_id enters UninitializedState."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

    def test_clear_from_uninitialized_is_idempotent(self) -> None:
        """clear() from UninitializedState -> UninitializedState (no-op)."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

        screen.clear()

        assert isinstance(screen.state, UninitializedState)

    def test_retry_from_uninitialized_raises(self) -> None:
        """retry() from UninitializedState raises InvalidStateTransitionError."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.retry()

        assert exc_info.value.method == "retry"
        assert exc_info.value.current_state == "UninitializedState"
        assert "LoadingFailedState" in exc_info.value.allowed_states


class TestExplainScreenLoading:
    """Tests for ExplainScreen loading from mocked RecorderFactory."""

    def _make_mock_node(self, *, node_id: str, plugin_name: str, node_type: NodeType) -> MagicMock:
        """Create a mock node matching the RecorderFactory.data_flow.get_nodes() return shape."""
        node = MagicMock()
        node.node_id = node_id
        node.plugin_name = plugin_name
        node.node_type = node_type
        return node

    def test_load_pipeline_structure_success(self) -> None:
        """Successful load produces LoadedState with correct lineage data."""
        mock_db = MagicMock()
        nodes = [
            self._make_mock_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_mock_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_mock_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.return_value = nodes
            screen = ExplainScreen(db=mock_db, run_id="run-123")

        assert isinstance(screen.state, LoadedState)
        data = screen.get_lineage_data()
        assert data is not None
        assert data["run_id"] == "run-123"
        assert data["source"]["name"] == "csv_source"
        assert data["source"]["node_id"] == "src-1"
        assert len(data["transforms"]) == 1
        assert data["transforms"][0]["name"] == "filter"
        assert data["transforms"][0]["node_type"] == "transform"
        assert len(data["sinks"]) == 1
        assert data["sinks"][0]["name"] == "csv_sink"

    def test_load_pipeline_structure_db_error(self) -> None:
        """Database error during loading produces LoadingFailedState."""
        mock_db = MagicMock()

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.side_effect = OperationalError(
                "connection refused", {}, Exception("connection refused")
            )
            screen = ExplainScreen(db=mock_db, run_id="run-123")

        assert isinstance(screen.state, LoadingFailedState)
        assert screen.state.run_id == "run-123"
        assert screen.state.error is not None
        assert "connection refused" in screen.state.error
        assert screen.state.db is mock_db

    def test_load_classifies_processing_node_types(self) -> None:
        """Gates, aggregations, and coalesces appear in transforms list."""
        mock_db = MagicMock()
        nodes = [
            self._make_mock_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_mock_node(node_id="tfm-1", plugin_name="mapper", node_type=NodeType.TRANSFORM),
            self._make_mock_node(node_id="gate-1", plugin_name="threshold", node_type=NodeType.GATE),
            self._make_mock_node(node_id="agg-1", plugin_name="batch", node_type=NodeType.AGGREGATION),
            self._make_mock_node(node_id="coal-1", plugin_name="merge", node_type=NodeType.COALESCE),
            self._make_mock_node(node_id="sink-1", plugin_name="output", node_type=NodeType.SINK),
        ]

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.return_value = nodes
            screen = ExplainScreen(db=mock_db, run_id="run-456")

        assert isinstance(screen.state, LoadedState)
        data = screen.get_lineage_data()
        assert data is not None
        transform_names = [t["name"] for t in data["transforms"]]
        assert transform_names == ["mapper", "threshold", "batch", "merge"]
        transform_types = [t["node_type"] for t in data["transforms"]]
        assert transform_types == ["transform", "gate", "aggregation", "coalesce"]
        assert len(data["sinks"]) == 1

    def test_load_empty_pipeline(self) -> None:
        """Pipeline with no nodes produces LoadedState with empty fields."""
        mock_db = MagicMock()

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.return_value = []
            screen = ExplainScreen(db=mock_db, run_id="run-empty")

        assert isinstance(screen.state, LoadedState)
        data = screen.get_lineage_data()
        assert data is not None
        assert data["source"]["name"] is None
        assert data["source"]["node_id"] is None
        assert data["transforms"] == []
        assert data["sinks"] == []

    def test_load_transitions_from_uninitialized(self) -> None:
        """load() from UninitializedState transitions to LoadedState."""
        mock_db = MagicMock()
        nodes = [
            self._make_mock_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
        ]

        screen = ExplainScreen()  # Starts in UninitializedState
        assert isinstance(screen.state, UninitializedState)

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.return_value = nodes
            screen.load(mock_db, "run-789")

        assert isinstance(screen.state, LoadedState)  # type: ignore[unreachable]  # load() mutates state; mypy can't track cross-method mutation
        assert screen.get_lineage_data()["run_id"] == "run-789"  # type: ignore[unreachable]

    def test_load_from_loaded_state_raises(self) -> None:
        """load() from LoadedState raises InvalidStateTransitionError."""
        mock_db = MagicMock()
        nodes = [
            self._make_mock_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
        ]

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.return_value = nodes
            screen = ExplainScreen(db=mock_db, run_id="run-loaded")

        assert isinstance(screen.state, LoadedState)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.load(mock_db, "run-other")

        assert exc_info.value.method == "load"
        assert exc_info.value.current_state == "LoadedState"


class TestExplainScreenNodeSelection:
    """Tests for ExplainScreen node selection and detail panel updates."""

    def _make_mock_node(self, *, node_id: str, plugin_name: str, node_type: NodeType) -> MagicMock:
        """Create a mock node matching the RecorderFactory.data_flow return shape."""
        node = MagicMock()
        node.node_id = node_id
        node.plugin_name = plugin_name
        node.node_type = node_type
        return node

    def _make_loaded_screen(self, mock_db: MagicMock) -> ExplainScreen:
        """Create a screen in LoadedState with a simple pipeline."""
        nodes = [
            self._make_mock_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_mock_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_mock_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.return_value = nodes
            screen = ExplainScreen(db=mock_db, run_id="run-sel")
        assert isinstance(screen.state, LoadedState)
        return screen

    def test_select_node_updates_detail_panel(self) -> None:
        """Selecting a node loads its state into the detail panel."""
        mock_db = MagicMock()
        screen = self._make_loaded_screen(mock_db)

        mock_node = self._make_mock_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM)
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_node.return_value = mock_node
            screen.on_tree_select("tfm-1")

        content = screen.detail_panel.render_content()
        assert "filter" in content
        assert "transform" in content

    def test_select_nonexistent_node_clears_panel(self) -> None:
        """Selecting a node that doesn't exist in DB sets panel to None state."""
        mock_db = MagicMock()
        screen = self._make_loaded_screen(mock_db)

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_node.return_value = None
            screen.on_tree_select("nonexistent-node")

        content = screen.detail_panel.render_content()
        assert "No node selected" in content

    def test_select_node_in_uninitialized_state(self) -> None:
        """Selecting a node in UninitializedState clears the detail panel."""
        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

        screen.on_tree_select("any-node")

        content = screen.detail_panel.render_content()
        assert "No node selected" in content

    def test_select_node_in_loading_failed_state(self) -> None:
        """Selecting a node in LoadingFailedState still loads node state from DB."""
        mock_db = MagicMock()

        # Create a screen that enters LoadingFailedState
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_nodes.side_effect = OperationalError(
                "connection refused", {}, Exception("connection refused")
            )
            screen = ExplainScreen(db=mock_db, run_id="run-failed")

        assert isinstance(screen.state, LoadingFailedState)

        # Now select a node — should still work via the LoadingFailedState branch
        mock_node = self._make_mock_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM)
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory") as MockFactory:
            MockFactory.return_value.data_flow.get_node.return_value = mock_node
            screen.on_tree_select("tfm-1")

        content = screen.detail_panel.render_content()
        assert "filter" in content
        assert "transform" in content
