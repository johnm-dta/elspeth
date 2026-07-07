# tests/unit/cli/test_explain_tui.py
"""Tests for explain command TUI screen state model.

Migrated from tests/cli/test_explain_tui.py.
Tests that require LandscapeDB (screen loading, state transitions with real DB)
are deferred to integration tier. Uses RecorderFactory to access data_flow repository.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError

from elspeth.contracts import NodeStateCompleted, NodeStateStatus, NodeType
from elspeth.core.landscape import LandscapeDB
from elspeth.tui.screens.explain_screen import (
    ExplainScreen,
    InvalidStateTransitionError,
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)


def _fake_db() -> LandscapeDB:
    return cast(LandscapeDB, object())


@dataclass(frozen=True, slots=True)
class FakeNode:
    node_id: str
    plugin_name: str
    node_type: NodeType


@dataclass(slots=True)
class FakeDataFlow:
    nodes: list[FakeNode] = field(default_factory=list)
    node_by_id: dict[str, FakeNode] = field(default_factory=dict)
    get_nodes_error: Exception | None = None
    get_node_calls: list[tuple[object, str]] = field(default_factory=list)

    def get_nodes(self, run_id: str) -> list[FakeNode]:
        if self.get_nodes_error is not None:
            raise self.get_nodes_error
        return self.nodes

    def get_node(self, node_id: object, run_id: str) -> FakeNode | None:
        self.get_node_calls.append((node_id, run_id))
        if not isinstance(node_id, str):
            return None
        return self.node_by_id.get(node_id)


@dataclass(slots=True)
class FakeQuery:
    node_states: list[NodeStateCompleted] = field(default_factory=list)

    def get_all_node_states_for_run(self, run_id: str) -> list[NodeStateCompleted]:
        return self.node_states


@dataclass(slots=True)
class FakeRecorderFactory:
    data_flow: FakeDataFlow
    query: FakeQuery = field(default_factory=FakeQuery)


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

    def _make_node(self, *, node_id: str, plugin_name: str, node_type: NodeType) -> FakeNode:
        """Create a node matching the RecorderFactory.data_flow.get_nodes() return shape."""
        return FakeNode(node_id=node_id, plugin_name=plugin_name, node_type=node_type)

    def test_load_pipeline_structure_success(self) -> None:
        """Successful load produces LoadedState with correct lineage data."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-123")

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

    def test_load_pipeline_structure_honors_lineage_selectors(self) -> None:
        """TUI mode must focus the same row/token/sink selector as no-tui mode."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        lineage_result = SimpleNamespace(
            token=SimpleNamespace(token_id="tok-123", row_id="row-456"),
            node_states=(
                SimpleNamespace(node_id="src-1"),
                SimpleNamespace(node_id="tfm-1"),
                SimpleNamespace(node_id="sink-1"),
            ),
        )
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes))

        with (
            patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory),
            patch("elspeth.tui.screens.explain_screen.explain_lineage", autospec=True, return_value=lineage_result) as mock_explain,
        ):
            screen = ExplainScreen(
                db=db,
                run_id="run-123",
                row_id="row-456",
                token_id="tok-123",
                sink="main",
            )

        assert isinstance(screen.state, LoadedState)
        mock_explain.assert_called_once_with(
            factory.query,
            factory.data_flow,
            run_id="run-123",
            token_id="tok-123",
            row_id="row-456",
            sink="main",
        )
        data = screen.get_lineage_data()
        assert data is not None
        assert data["tokens"] == [
            {
                "token_id": "tok-123",
                "row_id": "row-456",
                "path": ["src-1", "tfm-1", "sink-1"],
            }
        ]
        labels = [node["label"] for node in screen.state.tree.get_tree_nodes()]
        assert "Token: tok-123 (row: row-456)" in labels

    def test_load_pipeline_structure_db_error(self) -> None:
        """Database error during loading produces LoadingFailedState."""
        db = _fake_db()
        factory = FakeRecorderFactory(
            data_flow=FakeDataFlow(get_nodes_error=OperationalError("connection refused", {}, Exception("connection refused")))
        )

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-123")

        assert isinstance(screen.state, LoadingFailedState)
        assert screen.state.run_id == "run-123"
        assert screen.state.error is not None
        assert "connection refused" in screen.state.error
        assert screen.state.db is db

    def test_load_classifies_processing_node_types(self) -> None:
        """Gates, aggregations, and coalesces appear in transforms list."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="tfm-1", plugin_name="mapper", node_type=NodeType.TRANSFORM),
            self._make_node(node_id="gate-1", plugin_name="threshold", node_type=NodeType.GATE),
            self._make_node(node_id="agg-1", plugin_name="batch", node_type=NodeType.AGGREGATION),
            self._make_node(node_id="coal-1", plugin_name="merge", node_type=NodeType.COALESCE),
            self._make_node(node_id="sink-1", plugin_name="output", node_type=NodeType.SINK),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-456")

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
        db = _fake_db()
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=[]))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-empty")

        assert isinstance(screen.state, LoadedState)
        data = screen.get_lineage_data()
        assert data is not None
        assert data["source"]["name"] is None
        assert data["source"]["node_id"] is None
        assert data["transforms"] == []
        assert data["sinks"] == []

    def test_load_transitions_from_uninitialized(self) -> None:
        """load() from UninitializedState transitions to LoadedState."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes))

        screen = ExplainScreen()  # Starts in UninitializedState
        assert isinstance(screen.state, UninitializedState)

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen.load(db, "run-789")

        assert isinstance(screen.state, LoadedState)  # type: ignore[unreachable]  # load() mutates state; mypy can't track cross-method mutation
        assert screen.get_lineage_data()["run_id"] == "run-789"  # type: ignore[unreachable]

    def test_load_from_loaded_state_raises(self) -> None:
        """load() from LoadedState raises InvalidStateTransitionError."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-loaded")

        assert isinstance(screen.state, LoadedState)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.load(db, "run-other")

        assert exc_info.value.method == "load"
        assert exc_info.value.current_state == "LoadedState"


class TestExplainScreenNodeSelection:
    """Tests for ExplainScreen node selection and detail panel updates."""

    def _make_node(self, *, node_id: str, plugin_name: str, node_type: NodeType) -> FakeNode:
        """Create a node matching the RecorderFactory.data_flow return shape."""
        return FakeNode(node_id=node_id, plugin_name=plugin_name, node_type=node_type)

    def _make_loaded_screen(self, db: LandscapeDB) -> ExplainScreen:
        """Create a screen in LoadedState with a simple pipeline."""
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes))
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-sel")
        assert isinstance(screen.state, LoadedState)
        return screen

    def test_select_node_updates_detail_panel(self) -> None:
        """Selecting a node loads its state into the detail panel."""
        db = _fake_db()
        screen = self._make_loaded_screen(db)

        node = self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM)
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(node_by_id={"tfm-1": node}))
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen.on_tree_select("tfm-1")

        content = screen.detail_panel.render_content()
        assert "filter" in content
        assert "transform" in content

    def test_select_node_loads_success_reason_and_context_after(self) -> None:
        """Selecting a node includes execution audit context in details."""
        from datetime import UTC, datetime

        db = _fake_db()
        screen = self._make_loaded_screen(db)

        node = self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM)
        state = NodeStateCompleted(
            state_id="state-success",
            token_id="token-001",
            node_id="tfm-1",
            step_index=1,
            attempt=0,
            status=NodeStateStatus.COMPLETED,
            input_hash="abc123",
            output_hash="def456",
            started_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            duration_ms=1000.0,
            context_after_json='{"route_label": "accepted", "result": "True"}',
            success_reason_json='{"action": "mapped", "fields_added": ["normalized_name"]}',
        )
        factory = FakeRecorderFactory(
            data_flow=FakeDataFlow(node_by_id={"tfm-1": node}),
            query=FakeQuery(node_states=[state]),
        )

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen.on_tree_select("tfm-1")

        content = screen.detail_panel.render_content()
        assert "state-success" in content
        assert "token-001" in content
        assert "Success Reason:" in content
        assert "mapped" in content
        assert "Context After:" in content
        assert "accepted" in content

    def test_select_nonexistent_node_clears_panel(self) -> None:
        """Selecting a node that doesn't exist in DB sets panel to None state."""
        db = _fake_db()
        screen = self._make_loaded_screen(db)
        factory = FakeRecorderFactory(data_flow=FakeDataFlow())

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
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
        db = _fake_db()
        failed_factory = FakeRecorderFactory(
            data_flow=FakeDataFlow(get_nodes_error=OperationalError("connection refused", {}, Exception("connection refused")))
        )

        # Create a screen that enters LoadingFailedState
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=failed_factory):
            screen = ExplainScreen(db=db, run_id="run-failed")

        assert isinstance(screen.state, LoadingFailedState)

        # Now select a node — should still work via the LoadingFailedState branch
        node = self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM)
        loaded_factory = FakeRecorderFactory(data_flow=FakeDataFlow(node_by_id={"tfm-1": node}))
        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=loaded_factory):
            screen.on_tree_select("tfm-1")

        content = screen.detail_panel.render_content()
        assert "filter" in content
        assert "transform" in content

    def test_select_token_or_run_payload_does_not_lookup_node(self) -> None:
        """Non-node selections clear details without treating tokens/runs as nodes."""
        from elspeth.tui.types import TreeSelection

        db = _fake_db()
        screen = self._make_loaded_screen(db)
        data_flow = FakeDataFlow(node_by_id={})
        factory = FakeRecorderFactory(data_flow=data_flow)

        token_selection: TreeSelection = {
            "kind": "token",
            "run_id": "run-sel",
            "token_id": "token-001",
            "row_id": "row-001",
        }
        run_selection: TreeSelection = {
            "kind": "run",
            "run_id": "run-sel",
        }

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen.on_tree_select(token_selection)
            screen.on_tree_select(run_selection)

        assert data_flow.get_node_calls == []
        content = screen.detail_panel.render_content()
        assert "No node selected" in content
