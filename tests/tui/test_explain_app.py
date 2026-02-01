"""Tests for Explain TUI app."""

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


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


class TestExplainAppWithData:
    """Tests for ExplainApp with database connection."""

    def test_app_accepts_db_parameter(self) -> None:
        """App can be initialized with database connection."""
        from elspeth.tui.explain_app import ExplainApp

        db = LandscapeDB.in_memory()
        app = ExplainApp(db=db, run_id="test-run")

        assert app._db is db
        assert app._run_id == "test-run"

    def test_app_accepts_all_parameters(self) -> None:
        """App accepts db, run_id, token_id, row_id."""
        from elspeth.tui.explain_app import ExplainApp

        db = LandscapeDB.in_memory()
        app = ExplainApp(
            db=db,
            run_id="run-123",
            token_id="tok-456",
            row_id="row-789",
        )

        assert app._db is db
        assert app._run_id == "run-123"
        assert app._token_id == "tok-456"
        assert app._row_id == "row-789"

    @pytest.mark.asyncio
    async def test_app_loads_lineage_data(self) -> None:
        """App loads and displays real lineage data."""
        from elspeth.tui.explain_app import ExplainApp

        # Create test database with pipeline data
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        app = ExplainApp(db=db, run_id=run.run_id)
        async with app.run_test() as _pilot:
            # App should be running and have loaded the screen
            assert app.is_running
            assert app._screen is not None

    @pytest.mark.asyncio
    async def test_app_displays_source_in_tree(self) -> None:
        """App displays source plugin name in tree."""
        from textual.widgets import Static

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp

        # Create test database with pipeline data
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        app = ExplainApp(db=db, run_id=run.run_id)
        async with app.run_test() as _pilot:
            tree_widget = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Static)
            # Tree content should contain the source name
            # Use render() to get the content as a string
            content = str(tree_widget.render())
            assert "csv_source" in content

    @pytest.mark.asyncio
    async def test_app_handles_empty_run(self) -> None:
        """App handles run with no nodes gracefully."""
        from textual.widgets import Static

        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp

        # Create empty database - no nodes registered for the run
        db = LandscapeDB.in_memory()

        # Load a run that has no nodes (shows "unknown" source)
        app = ExplainApp(db=db, run_id="empty-run")
        async with app.run_test() as _pilot:
            tree_widget = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}", Static)
            # Should show the run with "unknown" source
            content = str(tree_widget.render())
            assert "unknown" in content.lower()
            assert "empty-run" in content
