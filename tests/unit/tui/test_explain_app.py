# tests/unit/tui/test_explain_app.py
"""Tests for Explain TUI app.

Migrated from tests/tui/test_explain_app.py.
Tests that require LandscapeDB (TestExplainAppWithData) are deferred to
integration tier.
"""

from typing import cast

import pytest

from elspeth.core.landscape import LandscapeDB
from elspeth.tui.lineage_view import TuiLineageItem, TuiLineageView
from elspeth.tui.screens.explain_screen import LoadedState
from elspeth.tui.types import LineageData, TreeSelection


def _fake_db() -> LandscapeDB:
    return cast(LandscapeDB, object())


def _sample_lineage_data() -> LineageData:
    return {
        "run_id": "run-1",
        "source": {"name": "csv", "node_id": "source-1"},
        "transforms": [{"name": "mapper", "node_id": "transform-1", "node_type": "transform"}],
        "sinks": [{"name": "json", "node_id": "sink-1", "node_type": "sink"}],
        "tokens": [{"token_id": "token-1", "row_id": "row-1", "path": ["source-1", "transform-1", "sink-1"]}],
    }


def _lineage_view_from_data(data: LineageData) -> TuiLineageView:
    from elspeth.tui.widgets.lineage_tree import LineageTree

    nodes = LineageTree(data).get_tree_nodes()
    return TuiLineageView(
        run_id=data["run_id"],
        items=tuple(
            TuiLineageItem(
                label=node["label"],
                selection=node["selection"],
                depth=node["depth"],
                has_children=node["has_children"],
                expanded=node["expanded"],
                node_id=node["node_id"],
                node_type=node["node_type"],
            )
            for node in nodes
        ),
    )


def _loaded_state_from_data(db: LandscapeDB, data: LineageData) -> LoadedState:
    from elspeth.tui.widgets.lineage_tree import LineageTree

    lineage_view = _lineage_view_from_data(data)
    return LoadedState(
        db=db,
        run_id=data["run_id"],
        lineage_view=lineage_view,
        tree=LineageTree(lineage_view),
    )


class TestExplainApp:
    """Tests for ExplainApp."""

    @pytest.mark.asyncio
    async def test_app_starts(self) -> None:
        """App can start and stop."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as _pilot:
            assert app.is_running

    @pytest.mark.asyncio
    async def test_app_has_header(self) -> None:
        """App has a header with title."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as _pilot:
            # Check for header widget
            from textual.widgets import Header

            header = app.query_one(Header)
            assert header is not None

    @pytest.mark.asyncio
    async def test_app_has_footer(self) -> None:
        """App has a footer with keybindings."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as _pilot:
            from textual.widgets import Footer

            footer = app.query_one(Footer)
            assert footer is not None

    @pytest.mark.asyncio
    async def test_quit_keybinding(self) -> None:
        """q key quits the app."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should exit
            assert not app.is_running

    def test_app_passes_lineage_selectors_to_screen(self) -> None:
        """ExplainApp must not drop CLI row/token/sink selectors."""
        from unittest.mock import patch

        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.screens.explain_screen import LoadingFailedState

        db = _fake_db()
        app = ExplainApp(
            db=db,
            run_id="run-123",
            row_id="row-456",
            token_id="tok-123",
            sink="main",
        )

        with patch("elspeth.tui.explain_app.ExplainScreen") as MockScreen:
            MockScreen.return_value.state = LoadingFailedState(db=db, run_id="run-123", error="boom")
            list(app.compose())

        MockScreen.assert_called_once_with(
            db=db,
            run_id="run-123",
            row_id="row-456",
            token_id="tok-123",
            sink="main",
        )

    @pytest.mark.asyncio
    async def test_loaded_app_uses_selectable_tree_to_update_detail_panel(self) -> None:
        """Loaded lineage is a selectable Tree that drives the detail panel."""
        from unittest.mock import patch

        from textual.widgets import Static, Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        class FakeScreen:
            def __init__(self) -> None:
                lineage_data = _sample_lineage_data()
                self.state = _loaded_state_from_data(_fake_db(), lineage_data)
                self.detail_panel = NodeDetailPanel(None)
                self.selected_node_ids: list[str] = []

            def on_tree_select(self, selection: TreeSelection | str | None) -> None:
                node_id = selection["node_id"] if isinstance(selection, dict) and selection["kind"] == "node" else str(selection or "")
                self.selected_node_ids.append(node_id)
                self.detail_panel.update_state(
                    {
                        "node_id": node_id,
                        "plugin_name": "csv",
                        "node_type": "source",
                    }
                )

        fake_screen = FakeScreen()

        with patch("elspeth.tui.explain_app.ExplainScreen", return_value=fake_screen):
            app = ExplainApp(db=_fake_db(), run_id="run-1")
            async with app.run_test() as pilot:
                lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
                detail_panel = app.query_one(f"#{WidgetIDs.DETAIL_PANEL}", Static)

                source_node = lineage_tree.root.children[0]
                lineage_tree.post_message(Tree.NodeSelected(source_node))
                await pilot.pause()

                assert fake_screen.selected_node_ids == ["source-1"]
                assert "=== csv (source) ===" in str(detail_panel.content)
                assert "Node ID:   source-1" in str(detail_panel.content)

    @pytest.mark.asyncio
    async def test_keyboard_navigation_selects_tree_node_to_update_detail_panel(self) -> None:
        """Keyboard navigation selects the focused tree row and updates details."""
        from unittest.mock import patch

        from textual.widgets import Static, Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        class FakeScreen:
            def __init__(self) -> None:
                lineage_data = _sample_lineage_data()
                self.state = _loaded_state_from_data(_fake_db(), lineage_data)
                self.detail_panel = NodeDetailPanel(None)
                self.selected: list[TreeSelection | str | None] = []

            def on_tree_select(self, selection: TreeSelection | str | None) -> None:
                self.selected.append(selection)
                if isinstance(selection, dict) and selection["kind"] == "node":
                    self.detail_panel.update_state(
                        {
                            "node_id": selection["node_id"],
                            "plugin_name": "csv",
                            "node_type": selection["node_type"],
                        }
                    )

        fake_screen = FakeScreen()

        with patch("elspeth.tui.explain_app.ExplainScreen", return_value=fake_screen):
            app = ExplainApp(db=_fake_db(), run_id="run-1")
            async with app.run_test() as pilot:
                lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
                detail_panel = app.query_one(f"#{WidgetIDs.DETAIL_PANEL}", Static)

                lineage_tree.focus()
                await pilot.press("down", "enter")
                await pilot.pause()

                assert fake_screen.selected
                assert fake_screen.selected[-1] == {
                    "kind": "node",
                    "run_id": "run-1",
                    "node_id": "source-1",
                    "node_type": "source",
                }
                assert "Node ID:   source-1" in str(detail_panel.content)

    @pytest.mark.asyncio
    async def test_token_leaf_selection_preserves_token_payload(self) -> None:
        """Selecting a token leaf sends a token payload, not a node lookup id."""
        from unittest.mock import patch

        from textual.widgets import Static, Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        class FakeScreen:
            def __init__(self) -> None:
                lineage_data = _sample_lineage_data()
                self.state = _loaded_state_from_data(_fake_db(), lineage_data)
                self.detail_panel = NodeDetailPanel(None)
                self.selected: list[TreeSelection | str | None] = []

            def on_tree_select(self, selection: TreeSelection | str | None) -> None:
                self.selected.append(selection)
                self.detail_panel.update_state(None)

        fake_screen = FakeScreen()

        with patch("elspeth.tui.explain_app.ExplainScreen", return_value=fake_screen):
            app = ExplainApp(db=_fake_db(), run_id="run-1")
            async with app.run_test() as pilot:
                lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
                detail_panel = app.query_one(f"#{WidgetIDs.DETAIL_PANEL}", Static)

                token_node = lineage_tree.root.children[0].children[0].children[0].children[0]
                lineage_tree.post_message(Tree.NodeSelected(token_node))
                await pilot.pause()

                assert fake_screen.selected == [
                    {
                        "kind": "token",
                        "run_id": "run-1",
                        "token_id": "token-1",
                        "row_id": "row-1",
                    }
                ]
                assert "No node selected" in str(detail_panel.content)

    @pytest.mark.asyncio
    async def test_no_database_state_uses_tree_status_widget(self) -> None:
        """No-data state still mounts the stable lineage tree widget."""
        from textual.widgets import Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as _pilot:
            lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)

            assert "No database connection" in str(lineage_tree.root.label)
            assert lineage_tree.root.data is None

    @pytest.mark.asyncio
    async def test_refresh_loaded_to_failed_replaces_stale_tree(self) -> None:
        """Refresh from loaded to failed replaces stale lineage with failure status."""
        from unittest.mock import patch

        from textual.widgets import Static, Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.screens.explain_screen import LoadingFailedState, ScreenState, UninitializedState
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        class FakeScreen:
            def __init__(self) -> None:
                lineage_data = _sample_lineage_data()
                self.state: ScreenState = _loaded_state_from_data(_fake_db(), lineage_data)
                self.detail_panel = NodeDetailPanel(
                    {
                        "node_id": "source-1",
                        "plugin_name": "csv",
                        "node_type": "source",
                    }
                )

            def clear(self) -> None:
                self.state = UninitializedState()
                self.detail_panel.update_state(None)

            def load(self, db: object, run_id: str, **kwargs: object) -> None:
                self.state = LoadingFailedState(db=_fake_db(), run_id="run-1", error="boom")

        fake_screen = FakeScreen()

        with patch("elspeth.tui.explain_app.ExplainScreen", return_value=fake_screen):
            app = ExplainApp(db=_fake_db(), run_id="run-1")
            async with app.run_test() as pilot:
                app.action_refresh()
                await pilot.pause()

                lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
                detail_panel = app.query_one(f"#{WidgetIDs.DETAIL_PANEL}", Static)
                assert "Loading failed: boom" in str(lineage_tree.root.label)
                assert lineage_tree.root.data == {
                    "kind": "status",
                    "run_id": "run-1",
                    "message": "Loading failed: boom",
                }
                assert "No node selected" in str(detail_panel.content)

    @pytest.mark.asyncio
    async def test_refresh_failed_to_loaded_keeps_tree_widget(self) -> None:
        """Refresh from failed to loaded updates the stable tree widget."""
        from unittest.mock import patch

        from textual.widgets import Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.screens.explain_screen import LoadingFailedState, ScreenState, UninitializedState
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        class FakeScreen:
            def __init__(self) -> None:
                self.state: ScreenState = LoadingFailedState(db=_fake_db(), run_id="run-1", error="boom")
                self.detail_panel = NodeDetailPanel(None)

            def clear(self) -> None:
                self.state = UninitializedState()
                self.detail_panel.update_state(None)

            def load(self, db: object, run_id: str, **kwargs: object) -> None:
                lineage_data = _sample_lineage_data()
                self.state = _loaded_state_from_data(_fake_db(), lineage_data)

        fake_screen = FakeScreen()

        with patch("elspeth.tui.explain_app.ExplainScreen", return_value=fake_screen):
            app = ExplainApp(db=_fake_db(), run_id="run-1")
            async with app.run_test() as pilot:
                lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
                assert "Loading failed: boom" in str(lineage_tree.root.label)

                app.action_refresh()
                await pilot.pause()

                assert "Run: run-1" in str(lineage_tree.root.label)
                assert lineage_tree.root.children

    @pytest.mark.asyncio
    async def test_loaded_empty_state_shows_status_row_not_unknown_source(self) -> None:
        """Loaded runs with no recorded nodes should not invent an unknown source."""
        from unittest.mock import patch

        from textual.widgets import Tree

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        empty_data: LineageData = {
            "run_id": "run-empty",
            "source": {"name": None, "node_id": None},
            "transforms": [],
            "sinks": [],
            "tokens": [],
        }

        class FakeScreen:
            def __init__(self) -> None:
                self.state = _loaded_state_from_data(_fake_db(), empty_data)
                self.detail_panel = NodeDetailPanel(None)

        with patch("elspeth.tui.explain_app.ExplainScreen", return_value=FakeScreen()):
            app = ExplainApp(db=_fake_db(), run_id="run-empty")
            async with app.run_test() as _pilot:
                lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
                labels = [str(lineage_tree.root.label), *(str(child.label) for child in lineage_tree.root.children)]

                assert "Source: (unknown)" not in labels
                assert any("No recorded nodes" in label for label in labels)
                assert lineage_tree.root.children[0].data == {
                    "kind": "status",
                    "run_id": "run-empty",
                    "message": "No recorded nodes",
                }
