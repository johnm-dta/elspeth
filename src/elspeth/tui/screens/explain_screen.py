"""Explain screen for lineage visualization.

Uses discriminated union pattern to represent screen states.
Invalid state combinations are prevented at the type level.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import structlog

from elspeth.contracts import NodeType
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.tui.types import LineageData, NodeStateInfo
from elspeth.tui.widgets.lineage_tree import LineageTree
from elspeth.tui.widgets.node_detail import NodeDetailPanel

logger = structlog.get_logger(__name__)


class ScreenStateType(Enum):
    """Discriminator for screen state types."""

    UNINITIALIZED = auto()  # No data source configured
    LOADING_FAILED = auto()  # Data source configured but loading failed
    LOADED = auto()  # Data loaded successfully


@dataclass(frozen=True)
class UninitializedState:
    """Screen has no data source configured.

    This is the default state when created without db/run_id.
    """

    state_type: ScreenStateType = ScreenStateType.UNINITIALIZED


@dataclass(frozen=True)
class LoadingFailedState:
    """Data source configured but loading failed.

    Preserves db and run_id so retry is possible.
    Error message included for user visibility.
    """

    db: LandscapeDB
    run_id: str
    error: str | None = field(default=None)
    state_type: ScreenStateType = field(default=ScreenStateType.LOADING_FAILED)


@dataclass(frozen=True)
class LoadedState:
    """Data loaded successfully.

    All required data is present and validated.
    """

    db: LandscapeDB
    run_id: str
    lineage_data: LineageData
    tree: LineageTree
    state_type: ScreenStateType = ScreenStateType.LOADED


# Discriminated union type - exhaustive pattern matching possible
ScreenState = UninitializedState | LoadingFailedState | LoadedState


class ExplainScreen:
    """Screen for visualizing pipeline lineage.

    Combines LineageTree and NodeDetailPanel widgets to provide
    an interactive exploration of run lineage.

    Layout:
        +------------------+------------------+
        |                  |                  |
        |  Lineage Tree    |   Detail Panel   |
        |                  |                  |
        |                  |                  |
        +------------------+------------------+

    State Model:
        The screen uses a discriminated union to represent its state.
        Invalid state combinations (e.g., lineage_data without db)
        cannot exist - they're unrepresentable in the type system.

        - UninitializedState: No data source
        - LoadingFailedState: Data source exists but loading failed
        - LoadedState: Data loaded successfully
    """

    def __init__(
        self,
        db: LandscapeDB | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize explain screen.

        Args:
            db: Landscape database connection
            run_id: Run ID to explain

        The screen starts in UninitializedState if no db/run_id provided,
        otherwise attempts to load data and enters LoadedState or LoadingFailedState.
        """
        # Selected node is tracked separately - it's a UI concern, not data state
        self._selected_node_id: str | None = None

        # Detail panel always exists, displays None state when nothing selected
        self._detail_panel = NodeDetailPanel(None)

        # Determine initial state based on inputs
        if db is None or run_id is None:
            self._state: ScreenState = UninitializedState()
        else:
            self._state = self._load_pipeline_structure(db, run_id)

    @property
    def state(self) -> ScreenState:
        """Current screen state for pattern matching."""
        return self._state

    def _load_pipeline_structure(self, db: LandscapeDB, run_id: str) -> LoadedState | LoadingFailedState:
        """Load pipeline structure from database.

        Args:
            db: Database connection
            run_id: Run ID to load

        Returns:
            LoadedState on success, LoadingFailedState on failure.
        """
        try:
            recorder = LandscapeRecorder(db)
            nodes = recorder.get_nodes(run_id)

            # Organize nodes by type
            source_nodes = [n for n in nodes if n.node_type == NodeType.SOURCE]
            transform_nodes = [n for n in nodes if n.node_type == NodeType.TRANSFORM]
            sink_nodes = [n for n in nodes if n.node_type == NodeType.SINK]

            lineage_data: LineageData = {
                "run_id": run_id,
                "source": {
                    "name": source_nodes[0].plugin_name if source_nodes else "unknown",
                    "node_id": source_nodes[0].node_id if source_nodes else None,
                }
                if source_nodes
                else {"name": "unknown", "node_id": None},
                "transforms": [{"name": n.plugin_name, "node_id": n.node_id} for n in transform_nodes],
                "sinks": [{"name": n.plugin_name, "node_id": n.node_id} for n in sink_nodes],
                "tokens": [],  # Tokens loaded separately when needed
            }
            tree = LineageTree(lineage_data)
            return LoadedState(
                db=db,
                run_id=run_id,
                lineage_data=lineage_data,
                tree=tree,
            )
        except Exception as e:
            # Log the actual error - silent failures hide bugs
            logger.error(
                "Failed to load lineage data",
                run_id=run_id,
                error=str(e),
                exc_info=True,
            )
            return LoadingFailedState(db=db, run_id=run_id, error=str(e))

    def get_widget_types(self) -> list[str]:
        """Get list of widget types in this screen.

        Returns:
            List of widget type names
        """
        return ["LineageTree", "NodeDetailPanel"]

    def get_lineage_data(self) -> LineageData | None:
        """Get current lineage data.

        Returns:
            Lineage data dict or None if not in LoadedState
        """
        match self._state:
            case LoadedState(lineage_data=data):
                return data
            case _:
                return None

    def _get_db(self) -> LandscapeDB | None:
        """Get database connection if available."""
        match self._state:
            case LoadedState(db=db) | LoadingFailedState(db=db):
                return db
            case _:
                return None

    def _get_run_id(self) -> str | None:
        """Get run ID if available."""
        match self._state:
            case LoadedState(run_id=rid) | LoadingFailedState(run_id=rid):
                return rid
            case _:
                return None

    def on_tree_select(self, node_id: str) -> None:
        """Handle tree node selection.

        Args:
            node_id: Selected node ID
        """
        self._selected_node_id = node_id

        # Load node state from database if in a state with db access
        db = self._get_db()
        run_id = self._get_run_id()
        if db and run_id and node_id:
            node_state = self._load_node_state(db, run_id, node_id)
            self._detail_panel.update_state(node_state)
        else:
            self._detail_panel.update_state(None)

    def _load_node_state(self, db: LandscapeDB, run_id: str, node_id: str) -> NodeStateInfo | None:
        """Load node state from database.

        Returns node information with required fields always populated.
        Optional execution state fields are included when available.

        Args:
            db: Database connection
            run_id: Run ID
            node_id: Node ID to load

        Returns:
            NodeStateInfo with at minimum node_id, plugin_name, node_type,
            or None if node not found
        """
        try:
            recorder = LandscapeRecorder(db)
            node = recorder.get_node(node_id)

            if node is None or node.run_id != run_id:
                return None

            # Build result with required fields - direct access, crash on missing
            # node_type is an enum, convert to string for display
            result: NodeStateInfo = {
                "node_id": node.node_id,
                "plugin_name": node.plugin_name,
                "node_type": node.node_type.value,
            }

            # Note: Full node state requires a token_id to look up execution state.
            # When selecting a node in the tree (not a specific token), we only
            # have the registered node info. Token-specific state (state_id,
            # token_id, status, timing, hashes, errors) would be populated when
            # the user selects a specific token-node combination.

            return result
        except Exception as e:
            logger.error(
                "Failed to load node state",
                run_id=run_id,
                node_id=node_id,
                error=str(e),
                exc_info=True,
            )
            return None

    def get_detail_panel_state(self) -> NodeStateInfo | None:
        """Get current detail panel state.

        Returns:
            Node state being displayed or None
        """
        return self._detail_panel._state

    def render(self) -> str:
        """Render the screen as text.

        Returns:
            Rendered screen content
        """
        lines = []
        lines.append("=" * 60)

        # Get run_id for display based on state
        run_id_display = self._get_run_id() or "(none)"
        lines.append(f"  ELSPETH Lineage Explorer - Run: {run_id_display}")
        lines.append("=" * 60)
        lines.append("")

        # Render tree if in loaded state
        match self._state:
            case LoadedState(tree=tree):
                lines.append("--- Lineage Tree ---")
                for node in tree.get_tree_nodes():
                    indent = "  " * node["depth"]
                    lines.append(f"{indent}{node['label']}")
                lines.append("")
            case _:
                pass  # No tree to render

        lines.append("--- Node Details ---")
        lines.append(self._detail_panel.render_content())

        return "\n".join(lines)
