"""Tests for explain command TUI integration."""


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
        assert lineage.get("run_id") == run.run_id

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
