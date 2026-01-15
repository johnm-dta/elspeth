"""Tests for explain command TUI integration."""

from elspeth.tui.screens.explain_screen import (
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)


class TestExplainScreen:
    """Tests for ExplainScreen component."""

    def test_can_import_screen(self) -> None:
        """ExplainScreen can be imported."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        assert ExplainScreen is not None

    def test_explain_screen_has_tree_widget(self) -> None:
        """Explain screen includes LineageTree widget."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        widgets = screen.get_widget_types()

        assert "LineageTree" in widgets

    def test_explain_screen_has_detail_panel(self) -> None:
        """Explain screen includes NodeDetailPanel widget."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        widgets = screen.get_widget_types()

        assert "NodeDetailPanel" in widgets

    def test_screen_initializes_with_db(self) -> None:
        """Screen can be initialized with database connection."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        # Create test database
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Screen should accept db and run_id
        screen = ExplainScreen(db=db, run_id=run.run_id)
        assert screen is not None

    def test_screen_loads_pipeline_structure(self) -> None:
        """Screen loads pipeline structure from database."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        # Create test data
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Screen should load this data
        screen = ExplainScreen(db=db, run_id=run.run_id)
        lineage = screen.get_lineage_data()

        assert lineage is not None
        assert lineage["run_id"] == run.run_id

    def test_tree_selection_updates_detail_panel(self) -> None:
        """Selecting a node in tree updates detail panel."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()

        # Initially no selection
        assert screen.get_detail_panel_state() is None

        # Simulate selecting a node
        mock_node_id = "node-001"
        screen.on_tree_select(mock_node_id)

        # Selection should be recorded (actual state loading depends on DB)
        assert screen._selected_node_id == mock_node_id

    def test_render_without_data(self) -> None:
        """Screen renders gracefully without data."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        content = screen.render()

        assert "ELSPETH" in content
        assert "No node selected" in content or "Select a node" in content

    def test_render_with_pipeline_data(self) -> None:
        """Screen renders pipeline structure when available."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        # Create test data with source and sink
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        screen = ExplainScreen(db=db, run_id=run.run_id)
        content = screen.render()

        assert "csv_source" in content
        assert "csv_sink" in content


class TestExplainScreenStateModel:
    """Tests for the discriminated union state model."""

    def test_uninitialized_state_without_db(self) -> None:
        """Screen without db/run_id enters UninitializedState."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

    def test_uninitialized_state_with_partial_config(self) -> None:
        """Screen with only db (no run_id) enters UninitializedState."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        screen = ExplainScreen(db=db)  # No run_id
        assert isinstance(screen.state, UninitializedState)

    def test_loaded_state_with_valid_data(self) -> None:
        """Screen with valid db/run_id enters LoadedState."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        screen = ExplainScreen(db=db, run_id=run.run_id)
        assert isinstance(screen.state, LoadedState)
        assert screen.state.run_id == run.run_id
        assert screen.state.db is db

    def test_loaded_state_has_lineage_data(self) -> None:
        """LoadedState contains lineage_data and tree."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        screen = ExplainScreen(db=db, run_id=run.run_id)
        assert isinstance(screen.state, LoadedState)
        assert screen.state.lineage_data is not None
        assert screen.state.tree is not None

    def test_loaded_state_with_empty_data_on_nonexistent_run(self) -> None:
        """Screen with non-existent run_id enters LoadedState with empty data.

        When a run_id doesn't exist in the database, get_nodes() returns an
        empty list (no exception). The screen successfully loads into LoadedState
        but with placeholder data (source name "unknown", empty transforms/sinks).
        """
        from elspeth.core.landscape import LandscapeDB
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        # Use a non-existent run_id
        screen = ExplainScreen(db=db, run_id="non-existent-run-id")
        # Enters LoadedState with empty/placeholder data
        assert isinstance(screen.state, LoadedState)
        assert screen.state.lineage_data["source"]["name"] == "unknown"
        assert screen.state.lineage_data["transforms"] == []
        assert screen.state.lineage_data["sinks"] == []

    def test_state_exhaustive_matching(self) -> None:
        """Demonstrate exhaustive pattern matching on states."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen, ScreenStateType

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        screen = ExplainScreen(db=db, run_id=run.run_id)

        # Pattern matching covers all cases
        match screen.state:
            case UninitializedState():
                result = "uninitialized"
            case LoadingFailedState(run_id=rid):
                result = f"failed:{rid}"
            case LoadedState(run_id=rid, lineage_data=data):
                result = f"loaded:{rid}:{data['run_id']}"

        assert result.startswith("loaded:")
        assert screen.state.state_type == ScreenStateType.LOADED
