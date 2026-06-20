# tests/unit/tui/test_explain_app.py
"""Tests for Explain TUI app.

Migrated from tests/tui/test_explain_app.py.
Tests that require LandscapeDB (TestExplainAppWithData) are deferred to
integration tier.
"""

import pytest


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
        from unittest.mock import MagicMock, patch

        from elspeth.tui.explain_app import ExplainApp
        from elspeth.tui.screens.explain_screen import LoadingFailedState

        mock_db = MagicMock()
        app = ExplainApp(
            db=mock_db,
            run_id="run-123",
            row_id="row-456",
            token_id="tok-123",
            sink="main",
        )

        with patch("elspeth.tui.explain_app.ExplainScreen") as MockScreen:
            MockScreen.return_value.state = LoadingFailedState(db=mock_db, run_id="run-123", error="boom")
            list(app.compose())

        MockScreen.assert_called_once_with(
            db=mock_db,
            run_id="run-123",
            row_id="row-456",
            token_id="tok-123",
            sink="main",
        )
