"""Explain TUI application for ELSPETH.

Provides interactive lineage exploration using ExplainScreen.
"""

from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static, Tree

from elspeth.core.landscape import LandscapeDB
from elspeth.tui.constants import WidgetIDs
from elspeth.tui.screens.explain_screen import (
    ExplainScreen,
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)
from elspeth.tui.types import TreeNodeDict, TreeSelection


class ExplainApp(App[None]):
    """Interactive TUI for exploring run lineage.

    Wraps ExplainScreen in a Textual application with keybindings
    and lifecycle management.
    """

    TITLE = "ELSPETH Explain"
    CSS = f"""
    Screen {{
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }}

    #{WidgetIDs.LINEAGE_TREE} {{
        height: 100%;
        border: solid green;
    }}

    #{WidgetIDs.DETAIL_PANEL} {{
        height: 100%;
        border: solid blue;
    }}
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        db: LandscapeDB | None = None,
        run_id: str | None = None,
        *,
        token_id: str | None = None,
        row_id: str | None = None,
        sink: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._db = db
        self._run_id = run_id
        self._token_id = token_id
        self._row_id = row_id
        self._sink = sink
        self._screen: ExplainScreen | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        # Create ExplainScreen with database connection if available
        if self._db is not None and self._run_id is not None:
            self._screen = ExplainScreen(
                db=self._db,
                run_id=self._run_id,
                token_id=self._token_id,
                row_id=self._row_id,
                sink=self._sink,
            )

            # Handle state explicitly - no defensive fallback
            screen_state = self._screen.state
            match screen_state:
                case LoadedState():
                    yield self._build_lineage_tree_widget(screen_state.tree.get_tree_nodes())

                    # Render detail panel content
                    detail_content = self._screen.detail_panel.render_content()
                    yield Static(detail_content, id=WidgetIDs.DETAIL_PANEL)

                case LoadingFailedState(error=err):
                    message = f"Loading failed: {err or 'Unknown error'}"
                    yield self._build_status_tree(message, {"kind": "status", "run_id": screen_state.run_id, "message": message})
                    yield Static("", id=WidgetIDs.DETAIL_PANEL)

                case UninitializedState():
                    yield self._build_status_tree("Screen not initialized. This should not happen.")
                    yield Static("", id=WidgetIDs.DETAIL_PANEL)
        else:
            # No data - show placeholder
            yield self._build_status_tree("No database connection. Use --database option.")
            yield Static("", id=WidgetIDs.DETAIL_PANEL)

        yield Footer()

    def _build_status_tree(self, message: str, selection: TreeSelection | None = None) -> Tree[TreeSelection]:
        """Build the stable lineage tree widget for status-only states."""
        return Tree(message, data=selection, id=WidgetIDs.LINEAGE_TREE)

    def _build_lineage_tree_widget(self, nodes: list[TreeNodeDict]) -> Tree[TreeSelection]:
        """Build a selectable Textual tree from flattened lineage nodes."""
        if not nodes:
            return self._build_status_tree("No nodes found")

        root_node = nodes[0]
        tree: Tree[TreeSelection] = Tree(
            root_node["label"],
            data=root_node["selection"],
            id=WidgetIDs.LINEAGE_TREE,
        )
        self._populate_lineage_tree_widget(tree, nodes)
        return tree

    def _populate_lineage_tree_widget(self, tree: Tree[TreeSelection], nodes: list[TreeNodeDict]) -> None:
        """Populate an existing Textual tree from flattened lineage nodes."""
        tree.clear()
        if not nodes:
            self._populate_status_tree(tree, "No nodes found")
            return

        root_node = nodes[0]
        tree.root.set_label(root_node["label"])
        tree.root.data = root_node["selection"]
        if root_node["expanded"]:
            tree.root.expand()

        stack: list[tuple[int, Any]] = [(root_node["depth"], tree.root)]
        for node in nodes[1:]:
            while stack and stack[-1][0] >= node["depth"]:
                stack.pop()
            parent = stack[-1][1] if stack else tree.root
            child = parent.add(
                node["label"],
                data=node["selection"],
                expand=node["expanded"],
                allow_expand=node["has_children"],
            )
            stack.append((node["depth"], child))

    def _populate_status_tree(self, tree: Tree[TreeSelection], message: str, selection: TreeSelection | None = None) -> None:
        """Replace the lineage tree contents with a nonselectable status row."""
        tree.clear()
        tree.root.set_label(message)
        tree.root.data = selection
        tree.root.collapse()

    def on_tree_node_selected(self, event: Tree.NodeSelected[TreeSelection]) -> None:
        """Update detail panel when a lineage tree node is selected."""
        if self._screen is None:
            return

        self._screen.on_tree_select(event.node.data)
        self._update_detail_panel()

    def _update_detail_panel(self) -> None:
        """Render the current screen detail panel into the mounted widget."""
        if self._screen is None:
            return
        detail_panel = self.query_one(f"#{WidgetIDs.DETAIL_PANEL}", Static)
        detail_panel.update(self._screen.detail_panel.render_content())

    def _update_lineage_tree_widget(self) -> None:
        """Render the current screen state into the mounted lineage tree."""
        if self._screen is None:
            return
        tree = self.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Tree)
        match self._screen.state:
            case LoadedState(tree=screen_tree):
                self._populate_lineage_tree_widget(tree, screen_tree.get_tree_nodes())
            case LoadingFailedState(error=err, run_id=run_id):
                message = f"Loading failed: {err or 'Unknown error'}"
                self._populate_status_tree(
                    tree,
                    message,
                    {"kind": "status", "run_id": run_id, "message": message},
                )
            case UninitializedState():
                self._populate_status_tree(tree, "Screen not initialized. This should not happen.")

    def action_refresh(self) -> None:
        """Refresh lineage data.

        Reloads screen state and updates the mounted tree/detail widgets.
        """
        if self._screen is not None:
            # Clear and reload
            self._screen.clear()
            if self._db and self._run_id:
                self._screen.load(
                    self._db,
                    self._run_id,
                    token_id=self._token_id,
                    row_id=self._row_id,
                    sink=self._sink,
                )
                self._update_lineage_tree_widget()
                self._update_detail_panel()
        self.notify("Refreshed")

    def action_help(self) -> None:
        """Show help."""
        self.notify("Use arrow keys to navigate, Enter to select, r to refresh, q to quit")
