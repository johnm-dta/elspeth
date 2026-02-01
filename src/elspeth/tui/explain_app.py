# src/elspeth/tui/explain_app.py
"""Explain TUI application for ELSPETH.

Provides interactive lineage exploration using ExplainScreen.
"""

from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from elspeth.core.landscape import LandscapeDB
from elspeth.tui.constants import WidgetIDs
from elspeth.tui.screens.explain_screen import (
    ExplainScreen,
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)


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
        token_id: str | None = None,
        row_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._db = db
        self._run_id = run_id
        self._token_id = token_id
        self._row_id = row_id
        self._screen: ExplainScreen | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        # Create ExplainScreen with database connection if available
        if self._db is not None and self._run_id is not None:
            self._screen = ExplainScreen(db=self._db, run_id=self._run_id)

            # Handle state explicitly - no defensive fallback
            match self._screen.state:
                case LoadedState(tree=tree):
                    # Render tree nodes as text for Static widget
                    tree_lines = []
                    for node in tree.get_tree_nodes():
                        indent = "  " * node["depth"]
                        tree_lines.append(f"{indent}{node['label']}")
                    tree_content = "\n".join(tree_lines) if tree_lines else "No nodes found"
                    yield Static(tree_content, id=WidgetIDs.LINEAGE_TREE)

                    # Render detail panel content
                    detail_content = self._screen.detail_panel.render_content()
                    yield Static(detail_content, id=WidgetIDs.DETAIL_PANEL)

                case LoadingFailedState(error=err):
                    yield Static(f"Loading failed: {err or 'Unknown error'}", id=WidgetIDs.LINEAGE_TREE)
                    yield Static("", id=WidgetIDs.DETAIL_PANEL)

                case UninitializedState():
                    yield Static("Screen not initialized. This should not happen.", id=WidgetIDs.LINEAGE_TREE)
                    yield Static("", id=WidgetIDs.DETAIL_PANEL)
        else:
            # No data - show placeholder
            yield Static("No database connection. Use --database option.", id=WidgetIDs.LINEAGE_TREE)
            yield Static("", id=WidgetIDs.DETAIL_PANEL)

        yield Footer()

    def action_refresh(self) -> None:
        """Refresh lineage data.

        Note: This clears and reloads the screen state. For the static widgets
        to update, we notify the user but would need widget remounting for
        full visual refresh.
        """
        if self._screen is not None:
            # Clear and reload
            self._screen.clear()
            if self._db and self._run_id:
                self._screen.load(self._db, self._run_id)
        self.notify("Refreshed")

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press q to quit, r to refresh, arrow keys to navigate")
