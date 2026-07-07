# tests/unit/cli/test_explain_tui.py
"""Tests for explain command TUI screen state model.

Migrated from tests/cli/test_explain_tui.py.
Tests that require LandscapeDB (screen loading, state transitions with real DB)
are deferred to integration tier. Uses RecorderFactory to access data_flow repository.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError

from elspeth.contracts import Artifact, NodeStateCompleted, NodeStateStatus, NodeType, TerminalOutcome, TerminalPath, TokenOutcome
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
    sequence_in_pipeline: int | None = None


@dataclass(frozen=True, slots=True)
class FakeEdge:
    edge_id: str
    from_node_id: str
    to_node_id: str
    label: str


@dataclass(slots=True)
class FakeDataFlow:
    nodes: list[FakeNode] = field(default_factory=list)
    edges: list[FakeEdge] = field(default_factory=list)
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

    def get_edges(self, run_id: str) -> list[FakeEdge]:
        return self.edges


@dataclass(slots=True)
class FakeQuery:
    node_states: list[NodeStateCompleted] = field(default_factory=list)
    get_all_node_states_calls: int = 0

    def get_all_node_states_for_run(self, run_id: str) -> list[NodeStateCompleted]:
        self.get_all_node_states_calls += 1
        return self.node_states


@dataclass(slots=True)
class FakeExecution:
    artifacts: list[Artifact] = field(default_factory=list)

    def get_artifacts(self, run_id: str) -> list[Artifact]:
        return self.artifacts


@dataclass(slots=True)
class FakeRecorderFactory:
    data_flow: FakeDataFlow
    query: FakeQuery = field(default_factory=FakeQuery)
    execution: FakeExecution = field(default_factory=FakeExecution)


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

    def _make_node(
        self,
        *,
        node_id: str,
        plugin_name: str,
        node_type: NodeType,
        sequence: int | None = None,
    ) -> FakeNode:
        """Create a node matching the RecorderFactory.data_flow.get_nodes() return shape."""
        return FakeNode(node_id=node_id, plugin_name=plugin_name, node_type=node_type, sequence_in_pipeline=sequence)

    def test_load_pipeline_structure_success(self) -> None:
        """Successful load produces LoadedState with correct lineage data."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        edges = [
            FakeEdge(edge_id="edge-src-tfm", from_node_id="src-1", to_node_id="tfm-1", label="default"),
            FakeEdge(edge_id="edge-tfm-sink", from_node_id="tfm-1", to_node_id="sink-1", label="default"),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes, edges=edges))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-123")

        assert isinstance(screen.state, LoadedState)
        view = screen.get_lineage_view()
        assert view is not None
        assert view.run_id == "run-123"
        labels = [item.label for item in view.items]
        assert labels == ["Run: run-123", "Source: csv_source", "Transform: filter", "Sink: csv_sink"]

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
            outcome=None,
        )
        edges = [
            FakeEdge(edge_id="edge-src-tfm", from_node_id="src-1", to_node_id="tfm-1", label="default"),
            FakeEdge(edge_id="edge-tfm-sink", from_node_id="tfm-1", to_node_id="sink-1", label="default"),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes, edges=edges))

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
        view = screen.get_lineage_view()
        assert view is not None
        token_items = [item for item in view.items if item.token_id == "tok-123"]
        assert len(token_items) == 1
        labels = [node["label"] for node in screen.state.tree.get_tree_nodes()]
        assert "Token: tok-123 (row: row-456)" in labels

    def test_focused_lineage_outcome_detail_shows_sink_and_artifact(self) -> None:
        """Focused token outcome detail exposes final sink and artifact evidence."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        edges = [FakeEdge(edge_id="edge-sink", from_node_id="src-1", to_node_id="sink-1", label="default")]
        sink_state = SimpleNamespace(node_id="sink-1", state_id="state-sink")
        lineage_result = SimpleNamespace(
            token=SimpleNamespace(token_id="tok-123", row_id="row-456"),
            node_states=(SimpleNamespace(node_id="src-1", state_id="state-src"), sink_state),
            outcome=TokenOutcome(
                outcome_id="outcome-1",
                run_id="run-123",
                token_id="tok-123",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                completed=True,
                sink_name="csv_sink",
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )
        artifact = Artifact(
            artifact_id="artifact-1",
            run_id="run-123",
            produced_by_state_id="state-sink",
            sink_node_id="sink-1",
            artifact_type="csv",
            path_or_uri="/tmp/out.csv",
            content_hash="sha256:abc",
            size_bytes=1536,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        factory = FakeRecorderFactory(
            data_flow=FakeDataFlow(nodes=nodes, edges=edges),
            execution=FakeExecution(artifacts=[artifact]),
        )

        with (
            patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory),
            patch("elspeth.tui.screens.explain_screen.explain_lineage", autospec=True, return_value=lineage_result),
        ):
            screen = ExplainScreen(db=db, run_id="run-123", token_id="tok-123")

        assert isinstance(screen.state, LoadedState)
        outcome_selection = next(node["selection"] for node in screen.state.tree.get_tree_nodes() if node["node_type"] == "outcome")
        screen.on_tree_select(outcome_selection)
        content = screen.detail_panel.render_content()

        assert "Outcome:   success" in content
        assert "Path:      default_flow" in content
        assert "Sink:      csv_sink" in content
        assert "ID:   artifact-1" in content
        assert "Path: /tmp/out.csv" in content
        assert "Hash: sha256:abc" in content

    def test_load_pipeline_structure_tree_preserves_multi_source_graph(self) -> None:
        """Loaded TUI tree uses graph edges instead of first-source linearization."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-a", plugin_name="csv", node_type=NodeType.SOURCE, sequence=0),
            self._make_node(node_id="src-b", plugin_name="json", node_type=NodeType.SOURCE, sequence=1),
            self._make_node(node_id="merge", plugin_name="coalesce", node_type=NodeType.COALESCE, sequence=2),
            self._make_node(node_id="sink", plugin_name="json_sink", node_type=NodeType.SINK, sequence=3),
        ]
        edges = [
            FakeEdge(edge_id="edge-a", from_node_id="src-a", to_node_id="merge", label="a"),
            FakeEdge(edge_id="edge-b", from_node_id="src-b", to_node_id="merge", label="b"),
            FakeEdge(edge_id="edge-sink", from_node_id="merge", to_node_id="sink", label="default"),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes, edges=edges))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-123")

        assert isinstance(screen.state, LoadedState)
        labels = [node["label"] for node in screen.state.tree.get_tree_nodes()]
        assert "Source: csv" in labels
        assert "Source: json" in labels
        assert "Branch: a" in labels
        assert "Branch: b" in labels
        assert "Repeated: Coalesce: coalesce (already shown)" in labels

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
        edges = [
            FakeEdge(edge_id="edge-src-tfm", from_node_id="src-1", to_node_id="tfm-1", label="default"),
            FakeEdge(edge_id="edge-tfm-gate", from_node_id="tfm-1", to_node_id="gate-1", label="default"),
            FakeEdge(edge_id="edge-gate-agg", from_node_id="gate-1", to_node_id="agg-1", label="default"),
            FakeEdge(edge_id="edge-agg-coal", from_node_id="agg-1", to_node_id="coal-1", label="default"),
            FakeEdge(edge_id="edge-coal-sink", from_node_id="coal-1", to_node_id="sink-1", label="default"),
        ]
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes, edges=edges))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-456")

        assert isinstance(screen.state, LoadedState)
        view = screen.get_lineage_view()
        assert view is not None
        labels = [item.label for item in view.items]
        assert labels == [
            "Run: run-456",
            "Source: csv_source",
            "Transform: mapper",
            "Gate: threshold",
            "Aggregation: batch",
            "Coalesce: merge",
            "Sink: output",
        ]

    def test_load_empty_pipeline(self) -> None:
        """Pipeline with no nodes produces LoadedState with empty fields."""
        db = _fake_db()
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=[]))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-empty")

        assert isinstance(screen.state, LoadedState)
        view = screen.get_lineage_view()
        assert view is not None
        assert [item.label for item in view.items] == ["Run: run-empty", "No recorded nodes"]

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
        view = screen.get_lineage_view()  # type: ignore[unreachable]
        assert view is not None
        assert view.run_id == "run-789"

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

    def _completed_state(self, *, state_id: str, token_id: str, node_id: str, offset_seconds: int) -> NodeStateCompleted:
        """Create a completed node state with deterministic timestamps."""
        from datetime import UTC, datetime, timedelta

        started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=offset_seconds)
        return NodeStateCompleted(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=1,
            attempt=0,
            status=NodeStateStatus.COMPLETED,
            input_hash="abc123",
            output_hash="def456",
            started_at=started_at,
            completed_at=started_at + timedelta(seconds=1),
            duration_ms=1000.0,
        )

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
            data_flow=FakeDataFlow(nodes=[node], node_by_id={"tfm-1": node}),
            query=FakeQuery(node_states=[state]),
        )

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-sel")
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

    def test_select_non_node_payloads_show_details_without_node_lookup(self) -> None:
        """Non-node selections show explicit details without node lookup."""
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
        edge_selection: TreeSelection = {
            "kind": "edge",
            "run_id": "run-sel",
            "edge_id": "edge-1",
            "from_node_id": "src-1",
            "to_node_id": "tfm-1",
            "edge_label": "accepted",
        }

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen.on_tree_select(token_selection)
            token_content = screen.detail_panel.render_content()
            screen.on_tree_select(run_selection)
            run_content = screen.detail_panel.render_content()
            screen.on_tree_select(edge_selection)
            edge_content = screen.detail_panel.render_content()

        assert data_flow.get_node_calls == []
        assert "Token ID: token-001" in token_content
        assert "Row ID:   row-001" in token_content
        assert "Run ID: run-sel" in run_content
        assert "Label: accepted" in edge_content
        assert "From:  src-1" in edge_content
        assert "To:    tfm-1" in edge_content

    def test_select_empty_status_row_shows_status_detail(self) -> None:
        """Empty loaded runs expose an explicit status detail row."""
        db = _fake_db()
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=[]))

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-empty")

        assert isinstance(screen.state, LoadedState)
        status_selection = screen.state.tree.get_tree_nodes()[1]["selection"]
        assert status_selection == {
            "kind": "status",
            "run_id": "run-empty",
            "message": "No recorded nodes",
        }

        screen.on_tree_select(status_selection)

        content = screen.detail_panel.render_content()
        assert "Lineage status" in content
        assert "Run ID: run-empty" in content
        assert "Message: No recorded nodes" in content

    def test_select_loaded_node_uses_cached_latest_state_without_per_selection_scan(self) -> None:
        """Loaded screens cache latest states once instead of scanning on each selection."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
        ]
        state = self._completed_state(state_id="state-latest", token_id="token-001", node_id="tfm-1", offset_seconds=1)
        query = FakeQuery(node_states=[state])
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes, node_by_id={"tfm-1": nodes[0]}), query=query)

        with patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory):
            screen = ExplainScreen(db=db, run_id="run-sel")
            screen.on_tree_select("tfm-1")
            screen.on_tree_select("tfm-1")

        assert query.get_all_node_states_calls == 1
        content = screen.detail_panel.render_content()
        assert "state-latest" in content

    def test_focused_token_state_wins_over_latest_state_for_same_node(self) -> None:
        """Focused row/token detail shows that token state, not another token's latest state."""
        db = _fake_db()
        nodes = [
            self._make_node(node_id="src-1", plugin_name="csv_source", node_type=NodeType.SOURCE),
            self._make_node(node_id="tfm-1", plugin_name="filter", node_type=NodeType.TRANSFORM),
            self._make_node(node_id="sink-1", plugin_name="csv_sink", node_type=NodeType.SINK),
        ]
        focused_state = self._completed_state(state_id="state-focused", token_id="token-focused", node_id="tfm-1", offset_seconds=1)
        latest_other_state = self._completed_state(state_id="state-other", token_id="token-other", node_id="tfm-1", offset_seconds=60)
        lineage_result = SimpleNamespace(
            token=SimpleNamespace(token_id="token-focused", row_id="row-focused"),
            node_states=(focused_state,),
            outcome=None,
        )
        query = FakeQuery(node_states=[latest_other_state])
        factory = FakeRecorderFactory(data_flow=FakeDataFlow(nodes=nodes, node_by_id={"tfm-1": nodes[1]}), query=query)

        with (
            patch("elspeth.tui.screens.explain_screen.RecorderFactory", autospec=True, return_value=factory),
            patch("elspeth.tui.screens.explain_screen.explain_lineage", autospec=True, return_value=lineage_result),
        ):
            screen = ExplainScreen(db=db, run_id="run-sel", token_id="token-focused")
            screen.on_tree_select("tfm-1")

        assert query.get_all_node_states_calls == 0
        content = screen.detail_panel.render_content()
        assert "state-focused" in content
        assert "token-focused" in content
        assert "state-other" not in content
        assert "token-other" not in content
