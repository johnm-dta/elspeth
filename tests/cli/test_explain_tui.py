"""Tests for explain command TUI integration."""

from sqlalchemy.exc import OperationalError

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.tui.screens.explain_screen import (
    InvalidStateTransitionError,
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


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

    def test_detail_panel_property_accessible(self) -> None:
        """detail_panel property provides public access to NodeDetailPanel."""
        from elspeth.tui.screens.explain_screen import ExplainScreen
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        screen = ExplainScreen()
        panel = screen.detail_panel

        assert isinstance(panel, NodeDetailPanel)

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

        # Create test data with source and sink
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
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Screen should load this data
        screen = ExplainScreen(db=db, run_id=run.run_id)
        lineage = screen.get_lineage_data()

        assert lineage is not None
        assert lineage["run_id"] == run.run_id

        # Verify source is correctly identified - not just run_id
        assert lineage["source"]["name"] == "csv_source", f"Expected source 'csv_source', got '{lineage['source']['name']}'"
        assert lineage["source"]["node_id"] is not None, "Source node_id should be set"

        # Verify sinks list contains our sink
        sink_names = [s["name"] for s in lineage["sinks"]]
        assert "csv_sink" in sink_names, f"Expected 'csv_sink' in sinks, got {sink_names}"

    def test_tree_selection_updates_detail_panel(self) -> None:
        """Selecting a node in tree updates detail panel with node info."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        # Create test data with a node we can select
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        screen = ExplainScreen(db=db, run_id=run.run_id)

        # Initially no selection
        assert screen.get_detail_panel_state() is None

        # Select the node - use node.node_id (string), not node object
        screen.on_tree_select(node.node_id)

        # Detail panel should now show node info via public API
        detail_state = screen.get_detail_panel_state()
        assert detail_state is not None, "Detail panel should have state after selection"
        assert detail_state["node_id"] == node.node_id, f"Expected node_id '{node.node_id}', got '{detail_state.get('node_id')}'"
        assert detail_state["plugin_name"] == "test_transform", (
            f"Expected plugin_name 'test_transform', got '{detail_state.get('plugin_name')}'"
        )
        assert detail_state["node_type"] == "transform", f"Expected node_type 'transform', got '{detail_state.get('node_type')}'"

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
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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

    def test_loading_failed_state_on_db_error(self) -> None:
        """Screen enters LoadingFailedState when database query fails.

        This tests the error path that was previously unexercised.
        """
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen, ScreenStateType

        db = LandscapeDB.in_memory()

        # Monkeypatch get_nodes to raise an exception
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Simulated database error")),
        ):
            screen = ExplainScreen(db=db, run_id="test-run-id")

        # Should enter LoadingFailedState, not crash
        assert isinstance(screen.state, LoadingFailedState), f"Expected LoadingFailedState, got {type(screen.state).__name__}"
        assert screen.state.run_id == "test-run-id"
        assert screen.state.error is not None, "LoadingFailedState should have error message"
        assert "database error" in screen.state.error.lower(), f"Expected error about database, got: {screen.state.error}"
        assert screen.state.state_type == ScreenStateType.LOADING_FAILED

    def test_loading_failed_state_preserves_db_for_retry(self) -> None:
        """LoadingFailedState preserves db and run_id for potential retry."""
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()

        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Connection failed")),
        ):
            screen = ExplainScreen(db=db, run_id="retry-test-run")

        assert isinstance(screen.state, LoadingFailedState)
        # Should preserve db for retry
        assert screen.state.db is db
        assert screen.state.run_id == "retry-test-run"


class TestExplainScreenStateTransitions:
    """Tests for state transition methods: load(), retry(), clear()."""

    # =========================================================================
    # load() transitions
    # =========================================================================

    def test_load_from_uninitialized_succeeds(self) -> None:
        """load() from UninitializedState → LoadedState on success."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        # Start with uninitialized screen
        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

        # Create test data
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # load() should transition to LoadedState
        screen.load(db, run.run_id)

        assert isinstance(screen.state, LoadedState)
        assert screen.state.run_id == run.run_id
        assert screen.state.db is db

    def test_load_from_uninitialized_fails_gracefully(self) -> None:
        """load() from UninitializedState → LoadingFailedState on db error."""
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        db = LandscapeDB.in_memory()

        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Network timeout")),
        ):
            screen.load(db, "test-run-id")

        assert isinstance(screen.state, LoadingFailedState)
        assert screen.state.error is not None
        assert "Network timeout" in screen.state.error

    def test_load_from_loaded_state_raises(self) -> None:
        """load() from LoadedState raises InvalidStateTransitionError."""
        import pytest

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Start in LoadedState
        screen = ExplainScreen(db=db, run_id=run.run_id)
        assert isinstance(screen.state, LoadedState)

        # Attempting load() should raise
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.load(db, "another-run-id")

        assert exc_info.value.method == "load"
        assert exc_info.value.current_state == "LoadedState"
        assert "UninitializedState" in exc_info.value.allowed_states

    def test_load_from_loading_failed_raises(self) -> None:
        """load() from LoadingFailedState raises InvalidStateTransitionError."""
        from unittest.mock import patch

        import pytest

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()

        # Get into LoadingFailedState
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Error")),
        ):
            screen = ExplainScreen(db=db, run_id="test-run")

        assert isinstance(screen.state, LoadingFailedState)

        # Attempting load() should raise (use retry() instead)
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.load(db, "another-run")

        assert exc_info.value.method == "load"
        assert exc_info.value.current_state == "LoadingFailedState"

    # =========================================================================
    # retry() transitions
    # =========================================================================

    def test_retry_from_loading_failed_succeeds(self) -> None:
        """retry() from LoadingFailedState → LoadedState when error is fixed."""
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # First attempt fails
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Temporary error")),
        ):
            screen = ExplainScreen(db=db, run_id=run.run_id)

        assert isinstance(screen.state, LoadingFailedState)

        # retry() should now succeed (patch removed, real get_nodes works)
        screen.retry()

        assert isinstance(screen.state, LoadedState)
        assert screen.state.run_id == run.run_id

    def test_retry_from_loading_failed_still_fails(self) -> None:
        """retry() from LoadingFailedState → LoadingFailedState on persistent error."""
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()

        # Get into LoadingFailedState
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("First error")),
        ):
            screen = ExplainScreen(db=db, run_id="test-run")

        assert isinstance(screen.state, LoadingFailedState)
        first_error = screen.state.error

        # retry() with different error
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Second error")),
        ):
            screen.retry()

        assert isinstance(screen.state, LoadingFailedState)
        assert screen.state.error is not None
        assert "Second error" in screen.state.error
        assert screen.state.error != first_error

    def test_retry_from_uninitialized_raises(self) -> None:
        """retry() from UninitializedState raises InvalidStateTransitionError."""
        import pytest

        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.retry()

        assert exc_info.value.method == "retry"
        assert exc_info.value.current_state == "UninitializedState"
        assert "LoadingFailedState" in exc_info.value.allowed_states

    def test_retry_from_loaded_raises(self) -> None:
        """retry() from LoadedState raises InvalidStateTransitionError."""
        import pytest

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        screen = ExplainScreen(db=db, run_id=run.run_id)
        assert isinstance(screen.state, LoadedState)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            screen.retry()

        assert exc_info.value.method == "retry"
        assert exc_info.value.current_state == "LoadedState"

    # =========================================================================
    # clear() transitions
    # =========================================================================

    def test_clear_from_loaded_state(self) -> None:
        """clear() from LoadedState → UninitializedState."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        screen = ExplainScreen(db=db, run_id=run.run_id)
        assert isinstance(screen.state, LoadedState)

        screen.clear()

        assert isinstance(screen.state, UninitializedState)

    def test_clear_from_loading_failed_state(self) -> None:
        """clear() from LoadingFailedState → UninitializedState."""
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()

        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Error")),
        ):
            screen = ExplainScreen(db=db, run_id="test-run")

        assert isinstance(screen.state, LoadingFailedState)

        screen.clear()

        assert isinstance(screen.state, UninitializedState)

    def test_clear_from_uninitialized_is_idempotent(self) -> None:
        """clear() from UninitializedState → UninitializedState (no-op)."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        assert isinstance(screen.state, UninitializedState)

        screen.clear()

        assert isinstance(screen.state, UninitializedState)

    def test_clear_resets_selection_and_detail_panel(self) -> None:
        """clear() resets selected node and detail panel state."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_node",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        screen = ExplainScreen(db=db, run_id=run.run_id)
        screen.on_tree_select(node.node_id)

        # Verify selection exists
        assert screen.get_detail_panel_state() is not None

        screen.clear()

        # Selection should be cleared
        assert screen.get_detail_panel_state() is None

    # =========================================================================
    # Transition sequences
    # =========================================================================

    def test_clear_then_load_different_data(self) -> None:
        """clear() → load() allows loading different data."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create two runs
        run1 = recorder.begin_run(config={"name": "run1"}, canonical_version="v1")
        run2 = recorder.begin_run(config={"name": "run2"}, canonical_version="v1")

        # Load first run
        screen = ExplainScreen(db=db, run_id=run1.run_id)
        assert isinstance(screen.state, LoadedState)
        assert screen.state.run_id == run1.run_id

        # Clear and load second run
        screen.clear()
        screen.load(db, run2.run_id)

        assert isinstance(screen.state, LoadedState)
        assert screen.state.run_id == run2.run_id

    def test_retry_then_clear_then_load(self) -> None:
        """Complex transition: retry (fail) → clear → load (success)."""
        from unittest.mock import patch

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Start with failure
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Error")),
        ):
            screen = ExplainScreen(db=db, run_id=run.run_id)

        assert isinstance(screen.state, LoadingFailedState)

        # Retry still fails
        with patch.object(
            LandscapeRecorder,
            "get_nodes",
            side_effect=OperationalError("SELECT 1", {}, Exception("Still broken")),
        ):
            screen.retry()

        assert isinstance(screen.state, LoadingFailedState)

        # User gives up and clears
        screen.clear()
        assert isinstance(screen.state, UninitializedState)

        # Try loading fresh (now works)
        screen.load(db, run.run_id)
        assert isinstance(screen.state, LoadedState)
