"""Tests for Explain TUI app."""

import pytest


class TestExplainApp:
    """Tests for ExplainApp."""

    def test_app_exists(self) -> None:
        """ExplainApp can be imported."""
        from elspeth.tui.explain_app import ExplainApp

        assert ExplainApp is not None

    @pytest.mark.asyncio
    async def test_app_starts(self) -> None:
        """App can start and stop."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            assert app.is_running

    @pytest.mark.asyncio
    async def test_app_has_header(self) -> None:
        """App has a header with title."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            # Check for header widget
            from textual.widgets import Header

            header = app.query_one(Header)
            assert header is not None

    @pytest.mark.asyncio
    async def test_app_has_footer(self) -> None:
        """App has a footer with keybindings."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
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
