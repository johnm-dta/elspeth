"""Explain screen for lineage visualization.

Uses discriminated union pattern to represent screen states.
Invalid state combinations are prevented at the type level.
"""

from dataclasses import dataclass, field

import structlog
from sqlalchemy.exc import DatabaseError, OperationalError

from elspeth.contracts import NodeState, NodeStateCompleted, NodeStateFailed, NodeStateOpen, NodeStatePending, NodeType
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape import explain as explain_lineage
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.tui.types import LineageData, NodeStateInfo, TokenDisplayInfo
from elspeth.tui.widgets.lineage_tree import LineageTree
from elspeth.tui.widgets.node_detail import NodeDetailPanel

logger = structlog.get_logger(__name__)


# Database errors that indicate connection/availability issues (recoverable)
# Other exceptions indicate bugs in our code and should crash
_RECOVERABLE_DB_ERRORS = (DatabaseError, OperationalError)


class InvalidStateTransitionError(Exception):
    """Raised when a state transition is not allowed from the current state."""

    def __init__(self, method: str, current_state: str, allowed_states: list[str]) -> None:
        self.method = method
        self.current_state = current_state
        self.allowed_states = allowed_states
        allowed = ", ".join(allowed_states)
        super().__init__(f"{method}() requires state in [{allowed}], but current state is {current_state}")


@dataclass(frozen=True, slots=True)
class UninitializedState:
    """Screen has no data source configured.

    This is the default state when created without db/run_id.
    """


@dataclass(frozen=True, slots=True)
class LoadingFailedState:
    """Data source configured but loading failed.

    Preserves db and run_id so retry is possible.
    Error message included for user visibility.
    """

    db: LandscapeDB
    run_id: str
    error: str | None = field(default=None)


@dataclass(frozen=True, slots=True)
class LoadedState:
    """Data loaded successfully.

    All required data is present and validated.
    """

    db: LandscapeDB
    run_id: str
    lineage_data: LineageData
    tree: LineageTree


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
        *,
        token_id: str | None = None,
        row_id: str | None = None,
        sink: str | None = None,
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
        self._token_id = token_id
        self._row_id = row_id
        self._sink = sink

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

    @property
    def detail_panel(self) -> NodeDetailPanel:
        """Get the detail panel for composition.

        This is a public interface for ExplainApp to access the detail panel
        without directly accessing the private _detail_panel attribute.
        """
        return self._detail_panel

    def _load_pipeline_structure(self, db: LandscapeDB, run_id: str) -> LoadedState | LoadingFailedState:
        """Load pipeline structure from database.

        Args:
            db: Database connection
            run_id: Run ID to load

        Returns:
            LoadedState on success, LoadingFailedState on failure.
        """
        try:
            factory = RecorderFactory(db)
            nodes = factory.data_flow.get_nodes(run_id)
            focused_tokens: list[TokenDisplayInfo] = []
            if self._token_id is not None or self._row_id is not None:
                try:
                    lineage_result = explain_lineage(
                        factory.query,
                        factory.data_flow,
                        run_id=run_id,
                        token_id=self._token_id,
                        row_id=self._row_id,
                        sink=self._sink,
                    )
                except ValueError as e:
                    return LoadingFailedState(db=db, run_id=run_id, error=str(e))
                if lineage_result is None:
                    return LoadingFailedState(
                        db=db,
                        run_id=run_id,
                        error="Token or row not found, or no terminal tokens exist yet",
                    )
                focused_tokens.append(
                    TokenDisplayInfo(
                        token_id=lineage_result.token.token_id,
                        row_id=lineage_result.token.row_id,
                        path=[state.node_id for state in lineage_result.node_states],
                    )
                )

            # Organize nodes by type
            source_nodes = [n for n in nodes if n.node_type == NodeType.SOURCE]
            # Processing nodes: transforms, gates, aggregations, coalesces
            # All appear in the transform chain in pipeline order
            _PROCESSING_TYPES = {NodeType.TRANSFORM, NodeType.GATE, NodeType.AGGREGATION, NodeType.COALESCE}
            transform_nodes = [n for n in nodes if n.node_type in _PROCESSING_TYPES]
            sink_nodes = [n for n in nodes if n.node_type == NodeType.SINK]

            lineage_data: LineageData = {
                "run_id": run_id,
                "source": {
                    "name": source_nodes[0].plugin_name if source_nodes else None,
                    "node_id": source_nodes[0].node_id if source_nodes else None,
                },
                "transforms": [{"name": n.plugin_name, "node_id": n.node_id, "node_type": n.node_type.value} for n in transform_nodes],
                "sinks": [{"name": n.plugin_name, "node_id": n.node_id, "node_type": n.node_type.value} for n in sink_nodes],
                "tokens": focused_tokens,
            }
            tree = LineageTree(lineage_data)
            return LoadedState(
                db=db,
                run_id=run_id,
                lineage_data=lineage_data,
                tree=tree,
            )
        except _RECOVERABLE_DB_ERRORS as e:
            # Database connection/availability errors are recoverable via retry
            # Other exceptions (bugs in our code) should crash - don't hide them
            logger.warning(
                "Database error loading lineage data - recoverable via retry",
                run_id=run_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return LoadingFailedState(db=db, run_id=run_id, error=str(e))

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

        if not node_id:
            self._detail_panel.update_state(None)
            return

        if isinstance(self._state, (LoadedState, LoadingFailedState)):
            node_state = self._load_node_state(self._state.db, self._state.run_id, node_id)
            self._detail_panel.update_state(node_state)
            return

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
        factory = RecorderFactory(db)
        # Query by composite PK (node_id, run_id) - no post-hoc validation needed
        node = factory.data_flow.get_node(node_id, run_id)

        if node is None:
            return None

        # Build result with required fields - direct access, crash on missing
        # node_type is an enum, convert to string for display
        result: NodeStateInfo = {
            "node_id": node.node_id,
            "plugin_name": node.plugin_name,
            "node_type": node.node_type.value,
        }

        node_states = [state for state in factory.query.get_all_node_states_for_run(run_id) if state.node_id == node_id]
        if node_states:
            self._add_execution_state(result, self._select_node_state_for_display(node_states))

        return result

    def _select_node_state_for_display(self, node_states: list[NodeState]) -> NodeState:
        """Choose the most recent execution state for a selected node."""
        return max(
            node_states,
            key=lambda state: (
                state.completed_at if isinstance(state, (NodeStatePending, NodeStateCompleted, NodeStateFailed)) else state.started_at,
                state.token_id,
                state.step_index,
                state.attempt,
            ),
        )

    def _add_execution_state(self, result: NodeStateInfo, state: NodeState) -> None:
        """Add execution-state fields to a node detail result."""
        result.update(
            {
                "state_id": state.state_id,
                "token_id": state.token_id,
                "status": state.status.value,
                "input_hash": state.input_hash,
                "started_at": state.started_at.isoformat(),
            }
        )

        if isinstance(state, NodeStateOpen):
            return

        result.update(
            {
                "completed_at": state.completed_at.isoformat(),
                "duration_ms": state.duration_ms,
            }
        )

        if state.context_after_json is not None:
            result["context_after_json"] = state.context_after_json

        if isinstance(state, NodeStatePending):
            return

        if state.output_hash is not None:
            result["output_hash"] = state.output_hash

        if isinstance(state, NodeStateCompleted):
            if state.success_reason_json is not None:
                result["success_reason_json"] = state.success_reason_json
            return

        if isinstance(state, NodeStateFailed):
            if state.error_json is not None:
                result["error_json"] = state.error_json
            return

        raise TypeError(f"Unsupported node state type: {type(state).__name__}")

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
        if isinstance(self._state, LoadedState):
            lines.append("--- Lineage Tree ---")
            for node in self._state.tree.get_tree_nodes():
                indent = "  " * node["depth"]
                lines.append(f"{indent}{node['label']}")
            lines.append("")

        lines.append("--- Node Details ---")
        lines.append(self._detail_panel.render_content())

        return "\n".join(lines)

    def load(
        self,
        db: LandscapeDB,
        run_id: str,
        *,
        token_id: str | None = None,
        row_id: str | None = None,
        sink: str | None = None,
    ) -> None:
        """Load pipeline data from database.

        Transitions: UninitializedState → LoadedState | LoadingFailedState

        Args:
            db: Landscape database connection
            run_id: Run ID to load

        Raises:
            InvalidStateTransitionError: If not in UninitializedState.
                Call clear() first to load different data.
        """
        if not isinstance(self._state, UninitializedState):
            raise InvalidStateTransitionError(
                method="load",
                current_state=type(self._state).__name__,
                allowed_states=["UninitializedState"],
            )

        self._token_id = token_id
        self._row_id = row_id
        self._sink = sink
        self._state = self._load_pipeline_structure(db, run_id)

    def retry(self) -> None:
        """Retry loading after a failure.

        Transitions: LoadingFailedState → LoadedState | LoadingFailedState

        Uses the db and run_id preserved in LoadingFailedState to attempt
        loading again. Useful for transient errors like network issues.

        Raises:
            InvalidStateTransitionError: If not in LoadingFailedState.
        """
        if not isinstance(self._state, LoadingFailedState):
            raise InvalidStateTransitionError(
                method="retry",
                current_state=type(self._state).__name__,
                allowed_states=["LoadingFailedState"],
            )

        # LoadingFailedState preserves db and run_id for exactly this purpose
        self._state = self._load_pipeline_structure(self._state.db, self._state.run_id)

    def clear(self) -> None:
        """Clear loaded data and return to uninitialized state.

        Transitions: Any → UninitializedState

        Resets the screen to its initial state. Call load() after this
        to load new data.
        """
        self._state = UninitializedState()
        self._selected_node_id = None
        self._detail_panel.update_state(None)
