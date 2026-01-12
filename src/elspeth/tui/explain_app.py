# src/elspeth/tui/explain_app.py
"""Explain TUI application for ELSPETH.

Provides interactive lineage exploration.
"""

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static


class ExplainApp(App[None]):
    """Interactive TUI for exploring run lineage.

    Displays lineage tree and allows drilling into node states,
    routing decisions, and external calls.
    """

    TITLE = "ELSPETH Explain"
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }

    #lineage-tree {
        height: 100%;
        border: solid green;
    }

    #detail-panel {
        height: 100%;
        border: solid blue;
    }
    """

    BINDINGS = [  # noqa: RUF012 - Textual pattern
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        run_id: str | None = None,
        token_id: str | None = None,
        row_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        self.token_id = token_id
        self.row_id = row_id

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Static("Lineage Tree (placeholder)", id="lineage-tree")
        yield Static("Detail Panel (placeholder)", id="detail-panel")
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh lineage data."""
        self.notify("Refreshing...")

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press q to quit, arrow keys to navigate")
