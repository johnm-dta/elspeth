# src/elspeth/core/landscape/recorder.py
"""LandscapeRecorder: High-level API for audit recording.

This is the main interface for recording audit trail entries during
pipeline execution. It wraps the low-level database operations.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from elspeth.core.landscape.reproducibility import ReproducibilityGrade

from sqlalchemy import select

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import (
    Artifact,
    Batch,
    BatchMember,
    Call,
    Edge,
    Node,
    NodeState,
    RoutingEvent,
    Row,
    RowLineage,
    Run,
    Token,
    TokenParent,
)
from elspeth.core.landscape.schema import (
    artifacts_table,
    batch_members_table,
    batches_table,
    calls_table,
    edges_table,
    node_states_table,
    nodes_table,
    routing_events_table,
    rows_table,
    runs_table,
    token_parents_table,
    tokens_table,
)
from elspeth.plugins.enums import Determinism, NodeType

E = TypeVar("E", bound=Enum)


def _now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(UTC)


def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex


def _coerce_enum(value: str | E, enum_type: type[E]) -> E:
    """Coerce a string or enum value to the target enum type.

    Args:
        value: String value or enum instance
        enum_type: Target enum class

    Returns:
        Enum instance

    Raises:
        ValueError: If string doesn't match any enum value

    Example:
        >>> _coerce_enum("transform", NodeType)
        <NodeType.TRANSFORM: 'transform'>
        >>> _coerce_enum(NodeType.TRANSFORM, NodeType)
        <NodeType.TRANSFORM: 'transform'>
    """
    if isinstance(value, enum_type):
        return value
    # str-based enums use value lookup
    return enum_type(value)


class LandscapeRecorder:
    """High-level API for recording audit trail entries.

    This class provides methods to record:
    - Runs and their configuration
    - Nodes (plugin instances) and edges
    - Rows and tokens (data flow)
    - Node states (processing records)
    - Routing events, batches, artifacts

    Example:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": "data.csv"})
        # ... execute pipeline ...
        recorder.complete_run(run.run_id, status="completed")
    """

    def __init__(self, db: LandscapeDB, *, payload_store: Any | None = None) -> None:
        """Initialize recorder with database connection.

        Args:
            db: LandscapeDB instance for audit storage
            payload_store: Optional payload store for retrieving row data
        """
        self._db = db
        self._payload_store = payload_store

    # === Run Management ===

    def begin_run(
        self,
        config: dict[str, Any],
        canonical_version: str,
        *,
        run_id: str | None = None,
        reproducibility_grade: str | None = None,
    ) -> Run:
        """Begin a new pipeline run.

        Args:
            config: Resolved configuration dictionary
            canonical_version: Version of canonical hash algorithm
            run_id: Optional run ID (generated if not provided)
            reproducibility_grade: Optional grade (FULL_REPRODUCIBLE, etc.)

        Returns:
            Run model with generated run_id
        """
        run_id = run_id or _generate_id()
        settings_json = canonical_json(config)
        config_hash = stable_hash(config)
        now = _now()

        run = Run(
            run_id=run_id,
            started_at=now,
            config_hash=config_hash,
            settings_json=settings_json,
            canonical_version=canonical_version,
            status="running",
            reproducibility_grade=reproducibility_grade,
        )

        with self._db.connection() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run.run_id,
                    started_at=run.started_at,
                    config_hash=run.config_hash,
                    settings_json=run.settings_json,
                    canonical_version=run.canonical_version,
                    status=run.status,
                    reproducibility_grade=run.reproducibility_grade,
                )
            )

        return run

    def complete_run(
        self,
        run_id: str,
        status: str,
        *,
        reproducibility_grade: str | None = None,
    ) -> Run:
        """Complete a pipeline run.

        Args:
            run_id: Run to complete
            status: Final status (completed, failed)
            reproducibility_grade: Optional final grade

        Returns:
            Updated Run model
        """
        now = _now()

        with self._db.connection() as conn:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(
                    status=status,
                    completed_at=now,
                    reproducibility_grade=reproducibility_grade,
                )
            )

        return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID.

        Args:
            run_id: Run ID to retrieve

        Returns:
            Run model or None if not found
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(runs_table).where(runs_table.c.run_id == run_id)
            )
            row = result.fetchone()

        if row is None:
            return None

        return Run(
            run_id=row.run_id,
            started_at=row.started_at,
            completed_at=row.completed_at,
            config_hash=row.config_hash,
            settings_json=row.settings_json,
            canonical_version=row.canonical_version,
            status=row.status,
            reproducibility_grade=row.reproducibility_grade,
            export_status=row.export_status,
            export_error=row.export_error,
            exported_at=row.exported_at,
            export_format=row.export_format,
            export_sink=row.export_sink,
        )

    def list_runs(self, *, status: str | None = None) -> list[Run]:
        """List all runs in the database.

        Args:
            status: Optional filter by status (running, completed, failed)

        Returns:
            List of Run models, ordered by started_at (newest first)
        """
        query = select(runs_table).order_by(runs_table.c.started_at.desc())

        if status:
            query = query.where(runs_table.c.status == status)

        with self._db.connection() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        return [
            Run(
                run_id=row.run_id,
                started_at=row.started_at,
                completed_at=row.completed_at,
                config_hash=row.config_hash,
                settings_json=row.settings_json,
                canonical_version=row.canonical_version,
                status=row.status,
                reproducibility_grade=row.reproducibility_grade,
                export_status=row.export_status,
                export_error=row.export_error,
                exported_at=row.exported_at,
                export_format=row.export_format,
                export_sink=row.export_sink,
            )
            for row in rows
        ]

    def set_export_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        export_format: str | None = None,
        export_sink: str | None = None,
    ) -> None:
        """Set export status for a run.

        This is separate from run status so export failures don't mask
        successful pipeline completion.

        Args:
            run_id: Run to update
            status: Export status (pending, completed, failed)
            error: Error message if status is 'failed'
            export_format: Format used (csv, json)
            export_sink: Sink name used for export
        """
        updates: dict[str, Any] = {"export_status": status}

        if status == "completed":
            updates["exported_at"] = _now()
        if error is not None:
            updates["export_error"] = error
        if export_format is not None:
            updates["export_format"] = export_format
        if export_sink is not None:
            updates["export_sink"] = export_sink

        with self._db.connection() as conn:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(**updates)
            )

    # === Node and Edge Registration ===

    def register_node(
        self,
        run_id: str,
        plugin_name: str,
        node_type: NodeType | str,
        plugin_version: str,
        config: dict[str, Any],
        *,
        node_id: str | None = None,
        sequence: int | None = None,
        schema_hash: str | None = None,
        determinism: Determinism | str = Determinism.DETERMINISTIC,
    ) -> Node:
        """Register a plugin instance (node) in the execution graph.

        Args:
            run_id: Run this node belongs to
            plugin_name: Name of the plugin
            node_type: Type (source, transform, gate, aggregation, coalesce, sink)
                       Accepts NodeType enum or string (will be validated)
            plugin_version: Version of the plugin
            config: Plugin configuration
            node_id: Optional node ID (generated if not provided)
            sequence: Position in pipeline
            schema_hash: Optional input/output schema hash
            determinism: Reproducibility grade (Determinism enum or string)

        Returns:
            Node model

        Raises:
            ValueError: If node_type or determinism string is not a valid enum value
        """
        # Validate and coerce enums early - fail fast on typos
        node_type_enum = _coerce_enum(node_type, NodeType)
        determinism_enum = _coerce_enum(determinism, Determinism)

        node_id = node_id or _generate_id()
        config_json = canonical_json(config)
        config_hash = stable_hash(config)
        now = _now()

        node = Node(
            node_id=node_id,
            run_id=run_id,
            plugin_name=plugin_name,
            node_type=node_type_enum.value,  # Store string in DB
            plugin_version=plugin_version,
            determinism=determinism_enum.value,  # Store string in DB
            config_hash=config_hash,
            config_json=config_json,
            schema_hash=schema_hash,
            sequence_in_pipeline=sequence,
            registered_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                nodes_table.insert().values(
                    node_id=node.node_id,
                    run_id=node.run_id,
                    plugin_name=node.plugin_name,
                    node_type=node.node_type,
                    plugin_version=node.plugin_version,
                    determinism=node.determinism,
                    config_hash=node.config_hash,
                    config_json=node.config_json,
                    schema_hash=node.schema_hash,
                    sequence_in_pipeline=node.sequence_in_pipeline,
                    registered_at=node.registered_at,
                )
            )

        return node

    def register_edge(
        self,
        run_id: str,
        from_node_id: str,
        to_node_id: str,
        label: str,
        mode: str,
        *,
        edge_id: str | None = None,
    ) -> Edge:
        """Register an edge in the execution graph.

        Args:
            run_id: Run this edge belongs to
            from_node_id: Source node
            to_node_id: Destination node
            label: Edge label ("continue", route name, etc.)
            mode: Default routing mode ("move" or "copy")
            edge_id: Optional edge ID (generated if not provided)

        Returns:
            Edge model
        """
        edge_id = edge_id or _generate_id()
        now = _now()

        edge = Edge(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,
            default_mode=mode,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                edges_table.insert().values(
                    edge_id=edge.edge_id,
                    run_id=edge.run_id,
                    from_node_id=edge.from_node_id,
                    to_node_id=edge.to_node_id,
                    label=edge.label,
                    default_mode=edge.default_mode,
                    created_at=edge.created_at,
                )
            )

        return edge

    def get_nodes(self, run_id: str) -> list[Node]:
        """Get all nodes for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Node models, ordered by sequence (NULL sequences last)
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(nodes_table)
                .where(nodes_table.c.run_id == run_id)
                # Use nullslast() for consistent NULL handling across databases
                # Nodes without sequence (e.g., dynamically added) sort last
                .order_by(nodes_table.c.sequence_in_pipeline.nullslast())
            )
            rows = result.fetchall()

        return [
            Node(
                node_id=row.node_id,
                run_id=row.run_id,
                plugin_name=row.plugin_name,
                node_type=row.node_type,
                plugin_version=row.plugin_version,
                determinism=row.determinism,
                config_hash=row.config_hash,
                config_json=row.config_json,
                schema_hash=row.schema_hash,
                sequence_in_pipeline=row.sequence_in_pipeline,
                registered_at=row.registered_at,
            )
            for row in rows
        ]

    def get_edges(self, run_id: str) -> list[Edge]:
        """Get all edges for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Edge models for this run, ordered by created_at then edge_id
            for deterministic export signatures.
        """
        query = (
            select(edges_table)
            .where(edges_table.c.run_id == run_id)
            .order_by(edges_table.c.created_at, edges_table.c.edge_id)
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        return [
            Edge(
                edge_id=r.edge_id,
                run_id=r.run_id,
                from_node_id=r.from_node_id,
                to_node_id=r.to_node_id,
                label=r.label,
                default_mode=r.default_mode,
                created_at=r.created_at,
            )
            for r in rows
        ]

    # === Row and Token Management ===

    def create_row(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        data: dict[str, Any],
        *,
        row_id: str | None = None,
        payload_ref: str | None = None,
    ) -> Row:
        """Create a source row record.

        Args:
            run_id: Run this row belongs to
            source_node_id: Source node that loaded this row
            row_index: Position in source (0-indexed)
            data: Row data for hashing
            row_id: Optional row ID (generated if not provided)
            payload_ref: Optional reference to payload store

        Returns:
            Row model
        """
        row_id = row_id or _generate_id()
        data_hash = stable_hash(data)
        now = _now()

        row = Row(
            row_id=row_id,
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            source_data_hash=data_hash,
            source_data_ref=payload_ref,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                rows_table.insert().values(
                    row_id=row.row_id,
                    run_id=row.run_id,
                    source_node_id=row.source_node_id,
                    row_index=row.row_index,
                    source_data_hash=row.source_data_hash,
                    source_data_ref=row.source_data_ref,
                    created_at=row.created_at,
                )
            )

        return row

    def create_token(
        self,
        row_id: str,
        *,
        token_id: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
    ) -> Token:
        """Create a token (row instance in DAG path).

        Args:
            row_id: Source row this token represents
            token_id: Optional token ID (generated if not provided)
            branch_name: Optional branch name (for forked tokens)
            fork_group_id: Optional fork group (links siblings)
            join_group_id: Optional join group (links merged tokens)

        Returns:
            Token model
        """
        token_id = token_id or _generate_id()
        now = _now()

        token = Token(
            token_id=token_id,
            row_id=row_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            branch_name=branch_name,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                tokens_table.insert().values(
                    token_id=token.token_id,
                    row_id=token.row_id,
                    fork_group_id=token.fork_group_id,
                    join_group_id=token.join_group_id,
                    branch_name=token.branch_name,
                    created_at=token.created_at,
                )
            )

        return token

    def fork_token(
        self,
        parent_token_id: str,
        row_id: str,
        branches: list[str],
        *,
        step_in_pipeline: int | None = None,
    ) -> list[Token]:
        """Fork a token to multiple branches.

        Creates child tokens for each branch, all sharing a fork_group_id.
        Records parent relationships.

        Args:
            parent_token_id: Token being forked
            row_id: Row ID (same for all children)
            branches: List of branch names
            step_in_pipeline: Step in the DAG where the fork occurs

        Returns:
            List of child Token models
        """
        fork_group_id = _generate_id()
        children = []

        with self._db.connection() as conn:
            for ordinal, branch_name in enumerate(branches):
                child_id = _generate_id()
                now = _now()

                # Create child token
                conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        step_in_pipeline=step_in_pipeline,
                        created_at=now,
                    )
                )

                # Record parent relationship
                conn.execute(
                    token_parents_table.insert().values(
                        token_id=child_id,
                        parent_token_id=parent_token_id,
                        ordinal=ordinal,
                    )
                )

                children.append(
                    Token(
                        token_id=child_id,
                        row_id=row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        step_in_pipeline=step_in_pipeline,
                        created_at=now,
                    )
                )

        return children

    def coalesce_tokens(
        self,
        parent_token_ids: list[str],
        row_id: str,
        *,
        step_in_pipeline: int | None = None,
    ) -> Token:
        """Coalesce multiple tokens into one (join operation).

        Creates a new token representing the merged result.
        Records all parent relationships.

        Args:
            parent_token_ids: Tokens being merged
            row_id: Row ID for the merged token
            step_in_pipeline: Step in the DAG where the coalesce occurs

        Returns:
            Merged Token model
        """
        join_group_id = _generate_id()
        token_id = _generate_id()
        now = _now()

        with self._db.connection() as conn:
            # Create merged token
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row_id,
                    join_group_id=join_group_id,
                    step_in_pipeline=step_in_pipeline,
                    created_at=now,
                )
            )

            # Record all parent relationships
            for ordinal, parent_id in enumerate(parent_token_ids):
                conn.execute(
                    token_parents_table.insert().values(
                        token_id=token_id,
                        parent_token_id=parent_id,
                        ordinal=ordinal,
                    )
                )

        return Token(
            token_id=token_id,
            row_id=row_id,
            join_group_id=join_group_id,
            step_in_pipeline=step_in_pipeline,
            created_at=now,
        )

    # === Node State Recording ===

    def begin_node_state(
        self,
        token_id: str,
        node_id: str,
        step_index: int,
        input_data: dict[str, Any],
        *,
        state_id: str | None = None,
        attempt: int = 0,
        context_before: dict[str, Any] | None = None,
    ) -> NodeState:
        """Begin recording a node state (token visiting a node).

        Args:
            token_id: Token being processed
            node_id: Node processing the token
            step_index: Position in token's execution path
            input_data: Input data for hashing
            state_id: Optional state ID (generated if not provided)
            attempt: Attempt number (0 for first attempt)
            context_before: Optional context snapshot before processing

        Returns:
            NodeState model with status="open"
        """
        state_id = state_id or _generate_id()
        input_hash = stable_hash(input_data)
        now = _now()

        context_json = canonical_json(context_before) if context_before else None

        state = NodeState(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=step_index,
            attempt=attempt,
            status="open",
            input_hash=input_hash,
            context_before_json=context_json,
            started_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                node_states_table.insert().values(
                    state_id=state.state_id,
                    token_id=state.token_id,
                    node_id=state.node_id,
                    step_index=state.step_index,
                    attempt=state.attempt,
                    status=state.status,
                    input_hash=state.input_hash,
                    context_before_json=state.context_before_json,
                    started_at=state.started_at,
                )
            )

        return state

    def complete_node_state(
        self,
        state_id: str,
        status: str,
        *,
        output_data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        error: dict[str, Any] | None = None,
        context_after: dict[str, Any] | None = None,
    ) -> NodeState:
        """Complete a node state.

        Args:
            state_id: State to complete
            status: Final status (completed, failed)
            output_data: Output data for hashing (if success)
            duration_ms: Processing duration
            error: Error details (if failed)
            context_after: Optional context snapshot after processing

        Returns:
            Updated NodeState model
        """
        now = _now()
        output_hash = stable_hash(output_data) if output_data else None
        error_json = canonical_json(error) if error else None
        context_json = canonical_json(context_after) if context_after else None

        with self._db.connection() as conn:
            conn.execute(
                node_states_table.update()
                .where(node_states_table.c.state_id == state_id)
                .values(
                    status=status,
                    output_hash=output_hash,
                    duration_ms=duration_ms,
                    error_json=error_json,
                    context_after_json=context_json,
                    completed_at=now,
                )
            )

        return self.get_node_state(state_id)  # type: ignore[return-value]

    def get_node_state(self, state_id: str) -> NodeState | None:
        """Get a node state by ID.

        Args:
            state_id: State ID to retrieve

        Returns:
            NodeState model or None
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.state_id == state_id
                )
            )
            row = result.fetchone()

        if row is None:
            return None

        return NodeState(
            state_id=row.state_id,
            token_id=row.token_id,
            node_id=row.node_id,
            step_index=row.step_index,
            attempt=row.attempt,
            status=row.status,
            input_hash=row.input_hash,
            output_hash=row.output_hash,
            context_before_json=row.context_before_json,
            context_after_json=row.context_after_json,
            duration_ms=row.duration_ms,
            error_json=row.error_json,
            started_at=row.started_at,
            completed_at=row.completed_at,
        )

    # === Routing Event Recording ===

    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: str,
        reason: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
        routing_group_id: str | None = None,
        ordinal: int = 0,
        reason_ref: str | None = None,
    ) -> RoutingEvent:
        """Record a single routing event.

        Args:
            state_id: Node state that made the routing decision
            edge_id: Edge that was taken
            mode: Routing mode (move or copy)
            reason: Reason for this routing decision
            event_id: Optional event ID
            routing_group_id: Group ID (for multi-destination routing)
            ordinal: Position in routing group
            reason_ref: Optional payload store reference

        Returns:
            RoutingEvent model
        """
        event_id = event_id or _generate_id()
        routing_group_id = routing_group_id or _generate_id()
        reason_hash = stable_hash(reason) if reason else None
        now = _now()

        event = RoutingEvent(
            event_id=event_id,
            state_id=state_id,
            edge_id=edge_id,
            routing_group_id=routing_group_id,
            ordinal=ordinal,
            mode=mode,
            reason_hash=reason_hash,
            reason_ref=reason_ref,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                routing_events_table.insert().values(
                    event_id=event.event_id,
                    state_id=event.state_id,
                    edge_id=event.edge_id,
                    routing_group_id=event.routing_group_id,
                    ordinal=event.ordinal,
                    mode=event.mode,
                    reason_hash=event.reason_hash,
                    reason_ref=event.reason_ref,
                    created_at=event.created_at,
                )
            )

        return event

    def record_routing_events(
        self,
        state_id: str,
        routes: list[dict[str, str]],
        reason: dict[str, Any] | None = None,
    ) -> list[RoutingEvent]:
        """Record multiple routing events (fork/multi-destination).

        All events share the same routing_group_id.

        Args:
            state_id: Node state that made the routing decision
            routes: List of {"edge_id": str, "mode": str}
            reason: Shared reason for all routes

        Returns:
            List of RoutingEvent models
        """
        routing_group_id = _generate_id()
        reason_hash = stable_hash(reason) if reason else None
        now = _now()
        events = []

        with self._db.connection() as conn:
            for ordinal, route in enumerate(routes):
                event_id = _generate_id()
                event = RoutingEvent(
                    event_id=event_id,
                    state_id=state_id,
                    edge_id=route["edge_id"],
                    routing_group_id=routing_group_id,
                    ordinal=ordinal,
                    mode=route["mode"],
                    reason_hash=reason_hash,
                    reason_ref=None,
                    created_at=now,
                )

                conn.execute(
                    routing_events_table.insert().values(
                        event_id=event.event_id,
                        state_id=event.state_id,
                        edge_id=event.edge_id,
                        routing_group_id=event.routing_group_id,
                        ordinal=event.ordinal,
                        mode=event.mode,
                        reason_hash=event.reason_hash,
                        created_at=event.created_at,
                    )
                )

                events.append(event)

        return events

    # === Batch Management ===

    def create_batch(
        self,
        run_id: str,
        aggregation_node_id: str,
        *,
        batch_id: str | None = None,
        attempt: int = 0,
    ) -> Batch:
        """Create a new batch for aggregation.

        Args:
            run_id: Run this batch belongs to
            aggregation_node_id: Aggregation node collecting tokens
            batch_id: Optional batch ID (generated if not provided)
            attempt: Attempt number (0 for first attempt)

        Returns:
            Batch model with status="draft"
        """
        batch_id = batch_id or _generate_id()
        now = _now()

        batch = Batch(
            batch_id=batch_id,
            run_id=run_id,
            aggregation_node_id=aggregation_node_id,
            attempt=attempt,
            status="draft",
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                batches_table.insert().values(
                    batch_id=batch.batch_id,
                    run_id=batch.run_id,
                    aggregation_node_id=batch.aggregation_node_id,
                    attempt=batch.attempt,
                    status=batch.status,
                    created_at=batch.created_at,
                )
            )

        return batch

    def add_batch_member(
        self,
        batch_id: str,
        token_id: str,
        ordinal: int,
    ) -> BatchMember:
        """Add a token to a batch.

        Args:
            batch_id: Batch to add to
            token_id: Token to add
            ordinal: Order in batch

        Returns:
            BatchMember model
        """
        member = BatchMember(
            batch_id=batch_id,
            token_id=token_id,
            ordinal=ordinal,
        )

        with self._db.connection() as conn:
            conn.execute(
                batch_members_table.insert().values(
                    batch_id=member.batch_id,
                    token_id=member.token_id,
                    ordinal=member.ordinal,
                )
            )

        return member

    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        *,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> None:
        """Update batch status.

        Args:
            batch_id: Batch to update
            status: New status (executing, completed, failed)
            trigger_reason: Why the batch was triggered
            state_id: Node state for the flush operation
        """
        updates: dict[str, Any] = {"status": status}

        if trigger_reason:
            updates["trigger_reason"] = trigger_reason
        if state_id:
            updates["aggregation_state_id"] = state_id
        if status in ("completed", "failed"):
            updates["completed_at"] = _now()

        with self._db.connection() as conn:
            conn.execute(
                batches_table.update()
                .where(batches_table.c.batch_id == batch_id)
                .values(**updates)
            )

    def complete_batch(
        self,
        batch_id: str,
        status: str,
        *,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> Batch:
        """Complete a batch.

        Args:
            batch_id: Batch to complete
            status: Final status (completed, failed)
            trigger_reason: Why the batch was triggered
            state_id: Optional node state for the aggregation

        Returns:
            Updated Batch model
        """
        now = _now()

        with self._db.connection() as conn:
            conn.execute(
                batches_table.update()
                .where(batches_table.c.batch_id == batch_id)
                .values(
                    status=status,
                    trigger_reason=trigger_reason,
                    aggregation_state_id=state_id,
                    completed_at=now,
                )
            )

        return self.get_batch(batch_id)  # type: ignore[return-value]

    def get_batch(self, batch_id: str) -> Batch | None:
        """Get a batch by ID.

        Args:
            batch_id: Batch ID to retrieve

        Returns:
            Batch model or None
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(batches_table).where(batches_table.c.batch_id == batch_id)
            )
            row = result.fetchone()

        if row is None:
            return None

        return Batch(
            batch_id=row.batch_id,
            run_id=row.run_id,
            aggregation_node_id=row.aggregation_node_id,
            aggregation_state_id=row.aggregation_state_id,
            trigger_reason=row.trigger_reason,
            attempt=row.attempt,
            status=row.status,
            created_at=row.created_at,
            completed_at=row.completed_at,
        )

    def get_batches(
        self,
        run_id: str,
        *,
        status: str | None = None,
        node_id: str | None = None,
    ) -> list[Batch]:
        """Get batches for a run.

        Args:
            run_id: Run ID
            status: Optional status filter
            node_id: Optional aggregation node filter

        Returns:
            List of Batch models, ordered by created_at then batch_id
            for deterministic export signatures.
        """
        query = select(batches_table).where(batches_table.c.run_id == run_id)

        if status:
            query = query.where(batches_table.c.status == status)
        if node_id:
            query = query.where(batches_table.c.aggregation_node_id == node_id)

        # Order for deterministic export signatures
        query = query.order_by(batches_table.c.created_at, batches_table.c.batch_id)

        with self._db.connection() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        return [
            Batch(
                batch_id=row.batch_id,
                run_id=row.run_id,
                aggregation_node_id=row.aggregation_node_id,
                aggregation_state_id=row.aggregation_state_id,
                trigger_reason=row.trigger_reason,
                attempt=row.attempt,
                status=row.status,
                created_at=row.created_at,
                completed_at=row.completed_at,
            )
            for row in rows
        ]

    def get_batch_members(self, batch_id: str) -> list[BatchMember]:
        """Get all members of a batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of BatchMember models (ordered by ordinal)
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(batch_members_table)
                .where(batch_members_table.c.batch_id == batch_id)
                .order_by(batch_members_table.c.ordinal)
            )
            rows = result.fetchall()

        return [
            BatchMember(
                batch_id=row.batch_id,
                token_id=row.token_id,
                ordinal=row.ordinal,
            )
            for row in rows
        ]

    # === Artifact Registration ===

    def register_artifact(
        self,
        run_id: str,
        state_id: str,
        sink_node_id: str,
        artifact_type: str,
        path: str,
        content_hash: str,
        size_bytes: int,
        *,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Register an artifact produced by a sink.

        Args:
            run_id: Run that produced this artifact
            state_id: Node state that produced this artifact
            sink_node_id: Sink node that wrote the artifact
            artifact_type: Type of artifact (csv, json, etc.)
            path: File path or URI
            content_hash: Hash of artifact content
            size_bytes: Size of artifact in bytes
            artifact_id: Optional artifact ID

        Returns:
            Artifact model
        """
        artifact_id = artifact_id or _generate_id()
        now = _now()

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                artifacts_table.insert().values(
                    artifact_id=artifact.artifact_id,
                    run_id=artifact.run_id,
                    produced_by_state_id=artifact.produced_by_state_id,
                    sink_node_id=artifact.sink_node_id,
                    artifact_type=artifact.artifact_type,
                    path_or_uri=artifact.path_or_uri,
                    content_hash=artifact.content_hash,
                    size_bytes=artifact.size_bytes,
                    created_at=artifact.created_at,
                )
            )

        return artifact

    def get_artifacts(
        self,
        run_id: str,
        *,
        sink_node_id: str | None = None,
    ) -> list[Artifact]:
        """Get artifacts for a run.

        Args:
            run_id: Run ID
            sink_node_id: Optional filter by sink

        Returns:
            List of Artifact models
        """
        query = select(artifacts_table).where(artifacts_table.c.run_id == run_id)

        if sink_node_id:
            query = query.where(artifacts_table.c.sink_node_id == sink_node_id)

        with self._db.connection() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        return [
            Artifact(
                artifact_id=row.artifact_id,
                run_id=row.run_id,
                produced_by_state_id=row.produced_by_state_id,
                sink_node_id=row.sink_node_id,
                artifact_type=row.artifact_type,
                path_or_uri=row.path_or_uri,
                content_hash=row.content_hash,
                size_bytes=row.size_bytes,
                created_at=row.created_at,
            )
            for row in rows
        ]

    # === Row and Token Query Methods ===

    def get_rows(self, run_id: str) -> list[Row]:
        """Get all rows for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Row models, ordered by row_index
        """
        query = (
            select(rows_table)
            .where(rows_table.c.run_id == run_id)
            .order_by(rows_table.c.row_index)
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            db_rows = result.fetchall()

        return [
            Row(
                row_id=r.row_id,
                run_id=r.run_id,
                source_node_id=r.source_node_id,
                row_index=r.row_index,
                source_data_hash=r.source_data_hash,
                source_data_ref=r.source_data_ref,
                created_at=r.created_at,
            )
            for r in db_rows
        ]

    def get_tokens(self, row_id: str) -> list[Token]:
        """Get all tokens for a row.

        Args:
            row_id: Row ID

        Returns:
            List of Token models, ordered by created_at then token_id
            for deterministic export signatures.
        """
        query = (
            select(tokens_table)
            .where(tokens_table.c.row_id == row_id)
            .order_by(tokens_table.c.created_at, tokens_table.c.token_id)
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            db_rows = result.fetchall()

        return [
            Token(
                token_id=r.token_id,
                row_id=r.row_id,
                fork_group_id=r.fork_group_id,
                join_group_id=r.join_group_id,
                branch_name=r.branch_name,
                step_in_pipeline=r.step_in_pipeline,
                created_at=r.created_at,
            )
            for r in db_rows
        ]

    def get_node_states_for_token(self, token_id: str) -> list[NodeState]:
        """Get all node states for a token.

        Args:
            token_id: Token ID

        Returns:
            List of NodeState models, ordered by step_index
        """
        query = (
            select(node_states_table)
            .where(node_states_table.c.token_id == token_id)
            .order_by(node_states_table.c.step_index)
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            db_rows = result.fetchall()

        return [
            NodeState(
                state_id=r.state_id,
                token_id=r.token_id,
                node_id=r.node_id,
                step_index=r.step_index,
                attempt=r.attempt,
                status=r.status,
                input_hash=r.input_hash,
                output_hash=r.output_hash,
                started_at=r.started_at,
                completed_at=r.completed_at,
                duration_ms=r.duration_ms,
                context_before_json=r.context_before_json,
                context_after_json=r.context_after_json,
                error_json=r.error_json,
            )
            for r in db_rows
        ]

    def get_row(self, row_id: str) -> Row | None:
        """Get a row by ID.

        Args:
            row_id: Row ID

        Returns:
            Row model or None if not found
        """
        query = select(rows_table).where(rows_table.c.row_id == row_id)

        with self._db.connection() as conn:
            result = conn.execute(query)
            r = result.fetchone()

        if r is None:
            return None

        return Row(
            row_id=r.row_id,
            run_id=r.run_id,
            source_node_id=r.source_node_id,
            row_index=r.row_index,
            source_data_hash=r.source_data_hash,
            source_data_ref=r.source_data_ref,
            created_at=r.created_at,
        )

    def get_row_data(self, row_id: str) -> dict[str, Any] | None:
        """Get the payload data for a row.

        Retrieves the actual row content from payload store if available.

        Args:
            row_id: Row ID

        Returns:
            Row data dict, or None if row not found or payload purged
        """
        row = self.get_row(row_id)
        if row is None:
            return None

        if row.source_data_ref and self._payload_store:
            # Retrieve from payload store
            import json
            from typing import cast

            payload_bytes = self._payload_store.retrieve(row.source_data_ref)
            data: dict[str, Any] = cast(dict[str, Any], json.loads(payload_bytes.decode("utf-8")))
            return data

        # No payload store or no ref - data not available
        return None

    def get_token(self, token_id: str) -> Token | None:
        """Get a token by ID.

        Args:
            token_id: Token ID

        Returns:
            Token model or None if not found
        """
        query = select(tokens_table).where(tokens_table.c.token_id == token_id)

        with self._db.connection() as conn:
            result = conn.execute(query)
            r = result.fetchone()

        if r is None:
            return None

        return Token(
            token_id=r.token_id,
            row_id=r.row_id,
            fork_group_id=r.fork_group_id,
            join_group_id=r.join_group_id,
            branch_name=r.branch_name,
            step_in_pipeline=r.step_in_pipeline,
            created_at=r.created_at,
        )

    def get_token_parents(self, token_id: str) -> list[TokenParent]:
        """Get parent relationships for a token.

        Args:
            token_id: Token ID

        Returns:
            List of TokenParent models (ordered by ordinal)
        """
        query = (
            select(token_parents_table)
            .where(token_parents_table.c.token_id == token_id)
            .order_by(token_parents_table.c.ordinal)
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            db_rows = result.fetchall()

        return [
            TokenParent(
                token_id=r.token_id,
                parent_token_id=r.parent_token_id,
                ordinal=r.ordinal,
            )
            for r in db_rows
        ]

    def get_routing_events(self, state_id: str) -> list[RoutingEvent]:
        """Get routing events for a node state.

        Args:
            state_id: State ID

        Returns:
            List of RoutingEvent models, ordered by ordinal then event_id
            for deterministic export signatures.
        """
        query = (
            select(routing_events_table)
            .where(routing_events_table.c.state_id == state_id)
            .order_by(
                routing_events_table.c.ordinal, routing_events_table.c.event_id
            )
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            db_rows = result.fetchall()

        return [
            RoutingEvent(
                event_id=r.event_id,
                state_id=r.state_id,
                edge_id=r.edge_id,
                routing_group_id=r.routing_group_id,
                ordinal=r.ordinal,
                mode=r.mode,
                reason_hash=r.reason_hash,
                reason_ref=r.reason_ref,
                created_at=r.created_at,
            )
            for r in db_rows
        ]

    def get_calls(self, state_id: str) -> list[Call]:
        """Get external calls for a node state.

        Args:
            state_id: State ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = (
            select(calls_table)
            .where(calls_table.c.state_id == state_id)
            .order_by(calls_table.c.call_index)
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            db_rows = result.fetchall()

        return [
            Call(
                call_id=r.call_id,
                state_id=r.state_id,
                call_index=r.call_index,
                call_type=r.call_type,
                status=r.status,
                request_hash=r.request_hash,
                request_ref=r.request_ref,
                response_hash=r.response_hash,
                response_ref=r.response_ref,
                error_json=r.error_json,
                latency_ms=r.latency_ms,
                created_at=r.created_at,
            )
            for r in db_rows
        ]

    # === Explain Methods (Graceful Degradation) ===

    def explain_row(self, run_id: str, row_id: str) -> RowLineage | None:
        """Get lineage for a row, gracefully handling purged payloads.

        This method returns row lineage information even when the actual
        payload data has been purged by retention policies. The hash is
        always preserved, ensuring audit integrity can be verified.

        Args:
            run_id: Run this row belongs to
            row_id: Row ID to explain

        Returns:
            RowLineage with hash and optionally source data, or None if row not found
            or if row doesn't belong to the specified run
        """
        import json

        row = self.get_row(row_id)
        if row is None:
            return None

        # Validate row belongs to the specified run - audit systems must be strict
        if row.run_id != run_id:
            return None

        # Try to load payload
        source_data: dict[str, Any] | None = None
        payload_available = False

        if row.source_data_ref and self._payload_store:
            try:
                payload_bytes = self._payload_store.retrieve(row.source_data_ref)
                source_data = json.loads(payload_bytes.decode("utf-8"))
                payload_available = True
            except (KeyError, json.JSONDecodeError, OSError):
                # Payload has been purged or is corrupted
                # KeyError: raised by PayloadStore when content not found
                # JSONDecodeError: content corrupted
                # OSError: filesystem issues
                pass

        return RowLineage(
            row_id=row_id,
            run_id=run_id,
            source_hash=row.source_data_hash,
            source_data=source_data,
            payload_available=payload_available,
        )

    # === Reproducibility Grade Management ===

    def compute_reproducibility_grade(self, run_id: str) -> "ReproducibilityGrade":
        """Compute reproducibility grade for a run based on node determinism.

        Logic:
        - If any node has determinism='nondeterministic', returns REPLAY_REPRODUCIBLE
        - Otherwise returns FULL_REPRODUCIBLE
        - 'seeded' counts as reproducible

        Args:
            run_id: Run ID to compute grade for

        Returns:
            ReproducibilityGrade enum value
        """
        from elspeth.core.landscape.reproducibility import compute_grade

        return compute_grade(self._db, run_id)

    def finalize_run(self, run_id: str, status: str) -> Run:
        """Finalize a run by computing grade and completing it.

        Convenience method that:
        1. Computes the reproducibility grade based on node determinism
        2. Completes the run with the specified status and computed grade

        Args:
            run_id: Run to finalize
            status: Final status (completed, failed)

        Returns:
            Updated Run model
        """
        grade = self.compute_reproducibility_grade(run_id)
        return self.complete_run(run_id, status, reproducibility_grade=grade.value)
