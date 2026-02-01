# src/elspeth/core/landscape/recorder.py
"""LandscapeRecorder: High-level API for audit recording.

This is the main interface for recording audit trail entries during
pipeline execution. It wraps the low-level database operations.
"""

from __future__ import annotations

import json
import logging
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    from elspeth.contracts.errors import TransformSuccessReason
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.landscape.reproducibility import ReproducibilityGrade

from sqlalchemy import select

from elspeth.contracts import (
    Artifact,
    Batch,
    BatchMember,
    BatchStatus,
    Call,
    CallStatus,
    CallType,
    CoalesceFailureReason,
    Determinism,
    Edge,
    ExecutionError,
    ExportStatus,
    FrameworkBugError,
    Node,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    NodeType,
    NonCanonicalMetadata,
    Operation,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
    Row,
    RowLineage,
    RowOutcome,
    Run,
    RunStatus,
    Token,
    TokenOutcome,
    TokenParent,
    TransformErrorReason,
    TransformErrorRecord,
    TriggerType,
    ValidationErrorRecord,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json, repr_hash, stable_hash
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.repositories import (
    ArtifactRepository,
    BatchMemberRepository,
    BatchRepository,
    CallRepository,
    EdgeRepository,
    NodeRepository,
    NodeStateRepository,
    RoutingEventRepository,
    RowRepository,
    RunRepository,
    TokenOutcomeRepository,
    TokenParentRepository,
    TokenRepository,
    TransformErrorRepository,
    ValidationErrorRepository,
)
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.landscape.schema import (
    artifacts_table,
    batch_members_table,
    batches_table,
    calls_table,
    edges_table,
    node_states_table,
    nodes_table,
    operations_table,
    routing_events_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
    transform_errors_table,
    validation_errors_table,
)


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
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
    """

    def __init__(self, db: LandscapeDB, *, payload_store: PayloadStore | None = None) -> None:
        """Initialize recorder with database connection.

        Args:
            db: LandscapeDB instance for audit storage
            payload_store: Optional payload store for retrieving row data
        """
        self._db = db
        self._payload_store = payload_store

        # Per-state_id call index allocation
        # Ensures UNIQUE(state_id, call_index) across all client types and retries
        self._call_indices: dict[str, int] = {}  # state_id → next_index
        self._call_index_lock: Lock = Lock()

        # Per-operation_id call index allocation (parallel to state call indices)
        # Operations (source/sink I/O) need their own call numbering
        self._operation_call_indices: dict[str, int] = {}  # operation_id → next_index

        # Database operations helper for reduced boilerplate
        self._ops = DatabaseOps(db)

        # Repository instances for row-to-object conversions
        # Session is passed per-call so we pass None here
        self._run_repo = RunRepository(None)
        self._node_repo = NodeRepository(None)
        self._edge_repo = EdgeRepository(None)
        self._row_repo = RowRepository(None)
        self._token_repo = TokenRepository(None)
        self._token_parent_repo = TokenParentRepository(None)
        self._call_repo = CallRepository(None)
        self._routing_event_repo = RoutingEventRepository(None)
        self._batch_repo = BatchRepository(None)
        self._node_state_repo = NodeStateRepository(None)
        self._validation_error_repo = ValidationErrorRepository(None)
        self._transform_error_repo = TransformErrorRepository(None)
        self._token_outcome_repo = TokenOutcomeRepository(None)
        self._artifact_repo = ArtifactRepository(None)
        self._batch_member_repo = BatchMemberRepository(None)

    # === Run Management ===

    def begin_run(
        self,
        config: dict[str, Any],
        canonical_version: str,
        *,
        run_id: str | None = None,
        reproducibility_grade: str | None = None,
        status: RunStatus = RunStatus.RUNNING,
        source_schema_json: str | None = None,
    ) -> Run:
        """Begin a new pipeline run.

        Args:
            config: Resolved configuration dictionary
            canonical_version: Version of canonical hash algorithm
            run_id: Optional run ID (generated if not provided)
            reproducibility_grade: Optional grade (FULL_REPRODUCIBLE, etc.)
            status: Initial RunStatus (defaults to RUNNING)
            source_schema_json: Optional serialized source schema for resume type restoration.
                Should be Pydantic model_json_schema() output. Required for proper resume
                type fidelity (datetime/Decimal restoration from payload JSON strings).

        Returns:
            Run model with generated run_id
        """
        run_id = run_id or generate_id()
        settings_json = canonical_json(config)
        config_hash = stable_hash(config)
        timestamp = now()

        run = Run(
            run_id=run_id,
            started_at=timestamp,
            config_hash=config_hash,
            settings_json=settings_json,
            canonical_version=canonical_version,
            status=status,
            reproducibility_grade=reproducibility_grade,
        )

        self._ops.execute_insert(
            runs_table.insert().values(
                run_id=run.run_id,
                started_at=run.started_at,
                config_hash=run.config_hash,
                settings_json=run.settings_json,
                canonical_version=run.canonical_version,
                status=run.status,
                reproducibility_grade=run.reproducibility_grade,
                source_schema_json=source_schema_json,
            )
        )

        return run

    def complete_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        reproducibility_grade: str | None = None,
    ) -> Run:
        """Complete a pipeline run.

        Args:
            run_id: Run to complete
            status: Final RunStatus (COMPLETED or FAILED)
            reproducibility_grade: Optional final grade

        Returns:
            Updated Run model
        """
        timestamp = now()

        self._ops.execute_update(
            runs_table.update()
            .where(runs_table.c.run_id == run_id)
            .values(
                status=status.value,
                completed_at=timestamp,
                reproducibility_grade=reproducibility_grade,
            )
        )

        result = self.get_run(run_id)
        if result is None:
            raise AuditIntegrityError(f"Run {run_id} not found after INSERT/UPDATE - database corruption or transaction failure")
        return result

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID.

        Args:
            run_id: Run ID to retrieve

        Returns:
            Run model or None if not found
        """
        query = select(runs_table).where(runs_table.c.run_id == run_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._run_repo.load(row)

    def get_source_schema(self, run_id: str) -> str:
        """Get source schema JSON for a run (for resume/type restoration).

        Args:
            run_id: Run to query

        Returns:
            Source schema JSON string

        Raises:
            ValueError: If run not found or has no source schema

        Note:
            This encapsulates Landscape schema access for Orchestrator resume.
            Schema is required for type fidelity when restoring rows from payloads.
        """
        query = select(runs_table.c.source_schema_json).where(runs_table.c.run_id == run_id)
        run_row = self._ops.execute_fetchone(query)

        if run_row is None:
            raise ValueError(f"Run {run_id} not found in database")

        source_schema_json = run_row.source_schema_json
        if source_schema_json is None:
            raise ValueError(
                f"Run {run_id} has no source schema stored. "
                f"This run was created before source schema storage was implemented. "
                f"Cannot resume without schema - type fidelity would be violated."
            )

        return str(source_schema_json)

    def record_source_field_resolution(
        self,
        run_id: str,
        resolution_mapping: dict[str, str],
        normalization_version: str | None,
    ) -> None:
        """Record field resolution mapping computed during source.load().

        This captures the mapping from original header names (as read from the file)
        to final field names (after normalization and/or field_mapping applied).
        Must be called after source.load() completes but before processing begins.

        Args:
            run_id: Run to update
            resolution_mapping: Dict mapping original header name → final field name
            normalization_version: Algorithm version used for normalization, or None if
                                   no normalization was applied (passthrough or explicit columns)

        Note:
            This is necessary because field resolution depends on actual file headers
            which are only known after load() runs, but node config is registered
            before load(). Without this, audit trail cannot recover original headers.
        """
        resolution_data = {
            "resolution_mapping": resolution_mapping,
            "normalization_version": normalization_version,
        }
        resolution_json = canonical_json(resolution_data)

        self._ops.execute_update(
            runs_table.update().where(runs_table.c.run_id == run_id).values(source_field_resolution_json=resolution_json)
        )

    def get_edge_map(self, run_id: str) -> dict[tuple[str, str], str]:
        """Get edge mapping for a run (from_node_id, label) -> edge_id.

        Args:
            run_id: Run to query

        Returns:
            Dictionary mapping (from_node_id, label) to edge_id

        Raises:
            ValueError: If run has no edges registered (data corruption)

        Note:
            This encapsulates Landscape schema access for Orchestrator resume.
            Edge IDs are required for FK integrity when recording routing events.
        """
        query = select(edges_table).where(edges_table.c.run_id == run_id)
        edges = self._ops.execute_fetchall(query)

        edge_map: dict[tuple[str, str], str] = {}
        for edge in edges:
            edge_map[(edge.from_node_id, edge.label)] = edge.edge_id

        return edge_map

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        """Update run status without setting completed_at.

        Used for intermediate status changes (e.g., paused → running during resume).
        For final completion, use complete_run() instead.

        Args:
            run_id: Run to update
            status: New RunStatus

        Note:
            This encapsulates run status updates for Orchestrator recovery.
            Only updates status field - does not set completed_at or reproducibility_grade.
        """
        self._ops.execute_update(runs_table.update().where(runs_table.c.run_id == run_id).values(status=status.value))

    def list_runs(self, *, status: RunStatus | None = None) -> list[Run]:
        """List all runs in the database.

        Args:
            status: Optional RunStatus filter

        Returns:
            List of Run models, ordered by started_at (newest first)
        """
        query = select(runs_table).order_by(runs_table.c.started_at.desc())

        if status is not None:
            query = query.where(runs_table.c.status == status.value)

        rows = self._ops.execute_fetchall(query)
        return [self._run_repo.load(row) for row in rows]

    def set_export_status(
        self,
        run_id: str,
        status: ExportStatus,
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
            status: ExportStatus (PENDING, COMPLETED, or FAILED)
            error: Error message if status is FAILED
            export_format: Format used (csv, json)
            export_sink: Sink name used for export
        """
        updates: dict[str, Any] = {"export_status": status.value}

        if status == ExportStatus.COMPLETED:
            updates["exported_at"] = now()
            # Clear stale error when transitioning to completed
            updates["export_error"] = None
        elif status == ExportStatus.PENDING:
            # Clear stale error when transitioning to pending
            updates["export_error"] = None

        # Only set error if explicitly provided (for FAILED status)
        if error is not None:
            updates["export_error"] = error

        if export_format is not None:
            updates["export_format"] = export_format
        if export_sink is not None:
            updates["export_sink"] = export_sink

        self._ops.execute_update(runs_table.update().where(runs_table.c.run_id == run_id).values(**updates))

    # === Node and Edge Registration ===

    def register_node(
        self,
        run_id: str,
        plugin_name: str,
        node_type: NodeType,
        plugin_version: str,
        config: dict[str, Any],
        *,
        node_id: str | None = None,
        sequence: int | None = None,
        schema_hash: str | None = None,
        determinism: Determinism = Determinism.DETERMINISTIC,
        schema_config: SchemaConfig,
    ) -> Node:
        """Register a plugin instance (node) in the execution graph.

        Args:
            run_id: Run this node belongs to
            plugin_name: Name of the plugin
            node_type: NodeType enum (SOURCE, TRANSFORM, GATE, AGGREGATION, COALESCE, SINK)
            plugin_version: Version of the plugin
            config: Plugin configuration
            node_id: Optional node ID (generated if not provided)
            sequence: Position in pipeline
            schema_hash: Optional input/output schema hash
            determinism: Determinism enum (defaults to DETERMINISTIC)
            schema_config: Schema configuration for audit trail (WP-11.99)

        Returns:
            Node model
        """
        node_id = node_id or generate_id()
        config_json = canonical_json(config)
        config_hash = stable_hash(config)
        timestamp = now()

        # Extract schema info for audit (WP-11.99)
        schema_fields_json: str | None = None
        schema_fields_list: list[dict[str, object]] | None = None

        if schema_config.is_dynamic:
            schema_mode = "dynamic"
        else:
            # mode is non-None when is_dynamic is False
            schema_mode = schema_config.mode or "free"  # Fallback shouldn't happen
            if schema_config.fields:
                # FieldDefinition.to_dict() returns dict[str, str | bool]
                # Cast each dict to wider type for storage
                field_dicts = [f.to_dict() for f in schema_config.fields]
                schema_fields_list = [dict(d) for d in field_dicts]
                schema_fields_json = canonical_json(field_dicts)

        node = Node(
            node_id=node_id,
            run_id=run_id,
            plugin_name=plugin_name,
            node_type=node_type,
            plugin_version=plugin_version,
            determinism=determinism,
            config_hash=config_hash,
            config_json=config_json,
            schema_hash=schema_hash,
            sequence_in_pipeline=sequence,
            registered_at=timestamp,
            schema_mode=schema_mode,
            schema_fields=schema_fields_list,
        )

        self._ops.execute_insert(
            nodes_table.insert().values(
                node_id=node.node_id,
                run_id=node.run_id,
                plugin_name=node.plugin_name,
                node_type=node.node_type.value,  # Store string in DB
                plugin_version=node.plugin_version,
                determinism=node.determinism.value,  # Store string in DB
                config_hash=node.config_hash,
                config_json=node.config_json,
                schema_hash=node.schema_hash,
                sequence_in_pipeline=node.sequence_in_pipeline,
                registered_at=node.registered_at,
                schema_mode=node.schema_mode,
                schema_fields_json=schema_fields_json,
            )
        )

        return node

    def register_edge(
        self,
        run_id: str,
        from_node_id: str,
        to_node_id: str,
        label: str,
        mode: RoutingMode,
        *,
        edge_id: str | None = None,
    ) -> Edge:
        """Register an edge in the execution graph.

        Args:
            run_id: Run this edge belongs to
            from_node_id: Source node
            to_node_id: Destination node
            label: Edge label ("continue", route name, etc.)
            mode: RoutingMode enum (MOVE or COPY)
            edge_id: Optional edge ID (generated if not provided)

        Returns:
            Edge model
        """
        edge_id = edge_id or generate_id()
        timestamp = now()

        edge = Edge(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,
            default_mode=mode,
            created_at=timestamp,
        )

        self._ops.execute_insert(
            edges_table.insert().values(
                edge_id=edge.edge_id,
                run_id=edge.run_id,
                from_node_id=edge.from_node_id,
                to_node_id=edge.to_node_id,
                label=edge.label,
                default_mode=edge.default_mode.value,  # Store string in DB
                created_at=edge.created_at,
            )
        )

        return edge

    def get_node(self, node_id: str, run_id: str) -> Node | None:
        """Get a node by its composite primary key (node_id, run_id).

        NOTE: The nodes table has a composite PK (node_id, run_id). The same
        node_id can exist in multiple runs, so run_id is required to identify
        the specific node.

        Args:
            node_id: Node ID to retrieve
            run_id: Run ID the node belongs to

        Returns:
            Node model or None if not found
        """
        query = select(nodes_table).where((nodes_table.c.node_id == node_id) & (nodes_table.c.run_id == run_id))
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._node_repo.load(row)

    def get_nodes(self, run_id: str) -> list[Node]:
        """Get all nodes for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Node models, ordered by sequence (NULL sequences last)
        """
        query = (
            select(nodes_table)
            .where(nodes_table.c.run_id == run_id)
            # Use nullslast() for consistent NULL handling across databases
            # Nodes without sequence (e.g., dynamically added) sort last
            .order_by(nodes_table.c.sequence_in_pipeline.nullslast())
        )
        rows = self._ops.execute_fetchall(query)
        return [self._node_repo.load(row) for row in rows]

    def get_edges(self, run_id: str) -> list[Edge]:
        """Get all edges for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Edge models for this run, ordered by created_at then edge_id
            for deterministic export signatures.
        """
        query = select(edges_table).where(edges_table.c.run_id == run_id).order_by(edges_table.c.created_at, edges_table.c.edge_id)
        rows = self._ops.execute_fetchall(query)
        return [self._edge_repo.load(row) for row in rows]

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
            data: Row data for hashing and optional storage
            row_id: Optional row ID (generated if not provided)
            payload_ref: DEPRECATED - payload persistence now handled internally

        Returns:
            Row model

        Note:
            Payload persistence is now handled by LandscapeRecorder, not callers.
            If self._payload_store is configured, the method will:
            1. Serialize data using canonical_json (handles pandas/numpy/datetime/Decimal)
            2. Store in payload store
            3. Record reference in audit trail

            This ensures Landscape owns its audit format end-to-end.
        """
        from elspeth.core.canonical import canonical_json

        row_id = row_id or generate_id()
        data_hash = stable_hash(data)
        timestamp = now()

        # Landscape owns payload persistence - serialize and store if configured
        final_payload_ref = payload_ref  # Legacy path (will be removed)
        if self._payload_store is not None and payload_ref is None:
            # Canonical JSON handles pandas/numpy/Decimal/datetime types
            payload_bytes = canonical_json(data).encode("utf-8")
            final_payload_ref = self._payload_store.store(payload_bytes)

        row = Row(
            row_id=row_id,
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            source_data_hash=data_hash,
            source_data_ref=final_payload_ref,
            created_at=timestamp,
        )

        self._ops.execute_insert(
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
        token_id = token_id or generate_id()
        timestamp = now()

        token = Token(
            token_id=token_id,
            row_id=row_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            branch_name=branch_name,
            created_at=timestamp,
        )

        self._ops.execute_insert(
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
        run_id: str,
        step_in_pipeline: int | None = None,
    ) -> tuple[list[Token], str]:
        """Fork a token to multiple branches.

        ATOMIC: Creates children AND records parent FORKED outcome in single transaction.
        Stores branch contract for recovery validation.

        Args:
            parent_token_id: Token being forked
            row_id: Row ID (same for all children)
            branches: List of branch names (must have at least one)
            run_id: Run ID (required for outcome recording)
            step_in_pipeline: Step in the DAG where the fork occurs

        Returns:
            Tuple of (child Token models, fork_group_id)

        Raises:
            ValueError: If branches is empty (defense-in-depth for audit integrity)
        """
        # Defense-in-depth: validate even though RoutingAction.fork_to_paths()
        # already validates. Per CLAUDE.md "no silent drops" - empty forks
        # would cause tokens to disappear without audit trail.
        if not branches:
            raise ValueError("fork_token requires at least one branch")

        fork_group_id = generate_id()
        children = []

        with self._db.connection() as conn:
            # 1. Create child tokens
            for ordinal, branch_name in enumerate(branches):
                child_id = generate_id()
                timestamp = now()

                # Create child token
                conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
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
                        created_at=timestamp,
                    )
                )

            # 2. Record parent FORKED outcome in SAME transaction (atomic)
            outcome_id = f"out_{generate_id()[:12]}"
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id=outcome_id,
                    run_id=run_id,
                    token_id=parent_token_id,
                    outcome=RowOutcome.FORKED.value,
                    is_terminal=1,
                    recorded_at=now(),
                    fork_group_id=fork_group_id,
                    expected_branches_json=json.dumps(branches),
                )
            )

        return children, fork_group_id

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
        join_group_id = generate_id()
        token_id = generate_id()
        timestamp = now()

        with self._db.connection() as conn:
            # Create merged token
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row_id,
                    join_group_id=join_group_id,
                    step_in_pipeline=step_in_pipeline,
                    created_at=timestamp,
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
            created_at=timestamp,
        )

    def expand_token(
        self,
        parent_token_id: str,
        row_id: str,
        count: int,
        *,
        run_id: str,
        step_in_pipeline: int | None = None,
        record_parent_outcome: bool = True,
    ) -> tuple[list[Token], str]:
        """Expand a token into multiple child tokens (deaggregation).

        ATOMIC: Creates children AND optionally records parent EXPANDED outcome
        in single transaction.

        Creates N child tokens from a single parent for 1->N expansion.
        All children share the same row_id (same source row) and are
        linked to the parent via token_parents table.

        Unlike fork_token (parallel DAG paths with branch names), expand_token
        creates sequential children for deaggregation transforms.

        Args:
            parent_token_id: Token being expanded
            row_id: Row ID (same for all children)
            count: Number of child tokens to create (must be >= 1)
            run_id: Run ID (required for atomic outcome recording)
            step_in_pipeline: Step where expansion occurs (optional)
            record_parent_outcome: If True (default), record EXPANDED outcome for parent.
                Set to False for batch aggregation where parent gets CONSUMED_IN_BATCH.

        Returns:
            Tuple of (child Token list, expand_group_id)

        Raises:
            ValueError: If count < 1
        """
        if count < 1:
            raise ValueError("expand_token requires at least 1 child")

        expand_group_id = generate_id()
        children = []

        with self._db.connection() as conn:
            for ordinal in range(count):
                child_id = generate_id()
                timestamp = now()

                # Create child token with expand_group_id
                conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        expand_group_id=expand_group_id,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
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
                        expand_group_id=expand_group_id,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
                    )
                )

            # Optionally record parent EXPANDED outcome in SAME transaction (atomic)
            # This eliminates the crash window where children exist but parent
            # outcome is not yet recorded.
            #
            # Set record_parent_outcome=False for batch aggregation where the
            # parent token gets CONSUMED_IN_BATCH instead of EXPANDED.
            if record_parent_outcome:
                outcome_id = f"out_{generate_id()[:12]}"
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=outcome_id,
                        run_id=run_id,
                        token_id=parent_token_id,
                        outcome=RowOutcome.EXPANDED.value,
                        is_terminal=1,
                        recorded_at=now(),
                        expand_group_id=expand_group_id,
                        # Store expected count for recovery validation
                        expected_branches_json=json.dumps({"count": count}),
                    )
                )

        return children, expand_group_id

    # === Node State Recording ===

    def begin_node_state(
        self,
        token_id: str,
        node_id: str,
        run_id: str,
        step_index: int,
        input_data: dict[str, Any],
        *,
        state_id: str | None = None,
        attempt: int = 0,
        context_before: dict[str, Any] | None = None,
    ) -> NodeStateOpen:
        """Begin recording a node state (token visiting a node).

        Args:
            token_id: Token being processed
            node_id: Node processing the token
            run_id: Run ID for composite FK to nodes table
            step_index: Position in token's execution path
            input_data: Input data for hashing
            state_id: Optional state ID (generated if not provided)
            attempt: Attempt number (0 for first attempt)
            context_before: Optional context snapshot before processing

        Returns:
            NodeStateOpen model with status=OPEN
        """
        state_id = state_id or generate_id()
        input_hash = stable_hash(input_data)
        timestamp = now()

        context_json = canonical_json(context_before) if context_before is not None else None

        state = NodeStateOpen(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=step_index,
            attempt=attempt,
            status=NodeStateStatus.OPEN,
            input_hash=input_hash,
            context_before_json=context_json,
            started_at=timestamp,
        )

        self._ops.execute_insert(
            node_states_table.insert().values(
                state_id=state.state_id,
                token_id=state.token_id,
                node_id=state.node_id,
                run_id=run_id,  # Added for composite FK to nodes
                step_index=state.step_index,
                attempt=state.attempt,
                status=state.status.value,
                input_hash=state.input_hash,
                context_before_json=state.context_before_json,
                started_at=state.started_at,
            )
        )

        return state

    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.PENDING],
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        context_after: dict[str, Any] | None = None,
    ) -> NodeStatePending: ...

    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.COMPLETED],
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,
        context_after: dict[str, Any] | None = None,
    ) -> NodeStateCompleted: ...

    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.FAILED],
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        context_after: dict[str, Any] | None = None,
    ) -> NodeStateFailed: ...

    def complete_node_state(
        self,
        state_id: str,
        status: NodeStateStatus,
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,
        context_after: dict[str, Any] | None = None,
    ) -> NodeStatePending | NodeStateCompleted | NodeStateFailed:
        """Complete a node state.

        Args:
            state_id: State to complete
            status: NodeStateStatus (PENDING, COMPLETED, or FAILED)
            output_data: Output data for hashing (if success)
            duration_ms: Processing duration (required)
            error: Error details (if failed)
            context_after: Optional context snapshot after processing

        Returns:
            NodeStatePending if status is pending, NodeStateCompleted if completed, NodeStateFailed if failed

        Raises:
            ValueError: If status is OPEN (not a valid terminal status)
            ValueError: If duration_ms is not provided
        """
        if status == NodeStateStatus.OPEN:
            raise ValueError("Cannot complete a node state with status OPEN")

        if duration_ms is None:
            raise ValueError("duration_ms is required when completing a node state")

        timestamp = now()
        output_hash = stable_hash(output_data) if output_data is not None else None
        error_json = canonical_json(error) if error is not None else None
        context_json = canonical_json(context_after) if context_after is not None else None
        # Serialize success reason if provided (use canonical_json for audit consistency)
        success_reason_json = canonical_json(success_reason) if success_reason is not None else None

        self._ops.execute_update(
            node_states_table.update()
            .where(node_states_table.c.state_id == state_id)
            .values(
                status=status.value,
                output_hash=output_hash,
                duration_ms=duration_ms,
                error_json=error_json,
                success_reason_json=success_reason_json,
                context_after_json=context_json,
                completed_at=timestamp,
            )
        )

        result = self.get_node_state(state_id)
        if result is None:
            raise AuditIntegrityError(f"NodeState {state_id} not found after update - database corruption or transaction failure")
        # Type narrowing: result is guaranteed to be terminal (PENDING/COMPLETED/FAILED)
        if isinstance(result, NodeStateOpen):
            raise AuditIntegrityError(f"NodeState {state_id} should be terminal after completion but has status OPEN")
        return result

    def get_node_state(self, state_id: str) -> NodeState | None:
        """Get a node state by ID.

        Args:
            state_id: State ID to retrieve

        Returns:
            NodeState (union of Open, Completed, or Failed) or None
        """
        query = select(node_states_table).where(node_states_table.c.state_id == state_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._node_state_repo.load(row)

    # === Routing Event Recording ===

    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: RoutingMode,
        reason: RoutingReason | None = None,
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
            mode: RoutingMode enum (MOVE or COPY)
            reason: Reason for this routing decision
            event_id: Optional event ID
            routing_group_id: Group ID (for multi-destination routing)
            ordinal: Position in routing group
            reason_ref: Optional payload store reference

        Returns:
            RoutingEvent model
        """
        event_id = event_id or generate_id()
        routing_group_id = routing_group_id or generate_id()
        reason_hash = stable_hash(reason) if reason else None
        timestamp = now()

        event = RoutingEvent(
            event_id=event_id,
            state_id=state_id,
            edge_id=edge_id,
            routing_group_id=routing_group_id,
            ordinal=ordinal,
            mode=mode,
            reason_hash=reason_hash,
            reason_ref=reason_ref,
            created_at=timestamp,
        )

        self._ops.execute_insert(
            routing_events_table.insert().values(
                event_id=event.event_id,
                state_id=event.state_id,
                edge_id=event.edge_id,
                routing_group_id=event.routing_group_id,
                ordinal=event.ordinal,
                mode=event.mode.value,  # Store string in DB
                reason_hash=event.reason_hash,
                reason_ref=event.reason_ref,
                created_at=event.created_at,
            )
        )

        return event

    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
    ) -> list[RoutingEvent]:
        """Record multiple routing events (fork/multi-destination).

        All events share the same routing_group_id.

        Args:
            state_id: Node state that made the routing decision
            routes: List of RoutingSpec objects specifying edge_id and mode
            reason: Shared reason for all routes

        Returns:
            List of RoutingEvent models
        """
        routing_group_id = generate_id()
        reason_hash = stable_hash(reason) if reason else None
        timestamp = now()
        events = []

        with self._db.connection() as conn:
            for ordinal, route in enumerate(routes):
                event_id = generate_id()
                event = RoutingEvent(
                    event_id=event_id,
                    state_id=state_id,
                    edge_id=route.edge_id,
                    routing_group_id=routing_group_id,
                    ordinal=ordinal,
                    mode=route.mode,  # Already RoutingMode enum from RoutingSpec
                    reason_hash=reason_hash,
                    reason_ref=None,
                    created_at=timestamp,
                )

                conn.execute(
                    routing_events_table.insert().values(
                        event_id=event.event_id,
                        state_id=event.state_id,
                        edge_id=event.edge_id,
                        routing_group_id=event.routing_group_id,
                        ordinal=event.ordinal,
                        mode=event.mode.value,  # Store string in DB
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
        batch_id = batch_id or generate_id()
        timestamp = now()

        batch = Batch(
            batch_id=batch_id,
            run_id=run_id,
            aggregation_node_id=aggregation_node_id,
            attempt=attempt,
            status=BatchStatus.DRAFT,  # Strict: enum type
            created_at=timestamp,
        )

        self._ops.execute_insert(
            batches_table.insert().values(
                batch_id=batch.batch_id,
                run_id=batch.run_id,
                aggregation_node_id=batch.aggregation_node_id,
                attempt=batch.attempt,
                status=batch.status.value,  # Store string in DB
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

        self._ops.execute_insert(
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
        status: BatchStatus,
        *,
        trigger_type: TriggerType | None = None,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> None:
        """Update batch status.

        Args:
            batch_id: Batch to update
            status: New BatchStatus
            trigger_type: TriggerType enum value
            trigger_reason: Human-readable reason for the trigger
            state_id: Node state for the flush operation
        """
        updates: dict[str, Any] = {"status": status.value}

        if trigger_type:
            updates["trigger_type"] = trigger_type.value
        if trigger_reason:
            updates["trigger_reason"] = trigger_reason
        if state_id:
            updates["aggregation_state_id"] = state_id
        if status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            updates["completed_at"] = now()

        self._ops.execute_update(batches_table.update().where(batches_table.c.batch_id == batch_id).values(**updates))

    def complete_batch(
        self,
        batch_id: str,
        status: BatchStatus,
        *,
        trigger_type: TriggerType | None = None,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> Batch:
        """Complete a batch.

        Args:
            batch_id: Batch to complete
            status: Final BatchStatus (COMPLETED or FAILED)
            trigger_type: TriggerType enum value
            trigger_reason: Human-readable reason for the trigger
            state_id: Optional node state for the aggregation

        Returns:
            Updated Batch model
        """
        timestamp = now()

        self._ops.execute_update(
            batches_table.update()
            .where(batches_table.c.batch_id == batch_id)
            .values(
                status=status.value,
                trigger_type=trigger_type.value if trigger_type else None,
                trigger_reason=trigger_reason,
                aggregation_state_id=state_id,
                completed_at=timestamp,
            )
        )

        result = self.get_batch(batch_id)
        if result is None:
            raise AuditIntegrityError(f"Batch {batch_id} not found after update - database corruption or transaction failure")
        return result

    def get_batch(self, batch_id: str) -> Batch | None:
        """Get a batch by ID.

        Args:
            batch_id: Batch ID to retrieve

        Returns:
            Batch model or None
        """
        query = select(batches_table).where(batches_table.c.batch_id == batch_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._batch_repo.load(row)

    def get_batches(
        self,
        run_id: str,
        *,
        status: BatchStatus | None = None,
        node_id: str | None = None,
    ) -> list[Batch]:
        """Get batches for a run.

        Args:
            run_id: Run ID
            status: Optional BatchStatus filter
            node_id: Optional aggregation node filter

        Returns:
            List of Batch models, ordered by created_at then batch_id
            for deterministic export signatures.
        """
        query = select(batches_table).where(batches_table.c.run_id == run_id)

        if status:
            query = query.where(batches_table.c.status == status.value)
        if node_id:
            query = query.where(batches_table.c.aggregation_node_id == node_id)

        # Order for deterministic export signatures
        query = query.order_by(batches_table.c.created_at, batches_table.c.batch_id)
        rows = self._ops.execute_fetchall(query)
        return [self._batch_repo.load(row) for row in rows]

    def get_incomplete_batches(self, run_id: str) -> list[Batch]:
        """Get batches that need recovery (draft, executing, or failed).

        Used during crash recovery to find batches that were:
        - draft: Still collecting rows when crash occurred
        - executing: Mid-flush when crash occurred
        - failed: Flush failed and needs retry

        Args:
            run_id: The run to query

        Returns:
            List of Batch objects with status in (draft, executing, failed),
            ordered by created_at ascending (oldest first for deterministic recovery)
        """
        query = (
            select(batches_table)
            .where(batches_table.c.run_id == run_id)
            .where(batches_table.c.status.in_([BatchStatus.DRAFT.value, BatchStatus.EXECUTING.value, BatchStatus.FAILED.value]))
            .order_by(batches_table.c.created_at.asc())
        )
        result = self._ops.execute_fetchall(query)
        return [self._batch_repo.load(row) for row in result]

    def get_batch_members(self, batch_id: str) -> list[BatchMember]:
        """Get all members of a batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of BatchMember models (ordered by ordinal)
        """
        query = select(batch_members_table).where(batch_members_table.c.batch_id == batch_id).order_by(batch_members_table.c.ordinal)
        rows = self._ops.execute_fetchall(query)
        return [self._batch_member_repo.load(row) for row in rows]

    def retry_batch(self, batch_id: str) -> Batch:
        """Create a new batch attempt from a failed batch.

        Copies batch metadata and members to a new batch with
        incremented attempt counter and draft status.

        Args:
            batch_id: The failed batch to retry

        Returns:
            New Batch with attempt = original.attempt + 1

        Raises:
            ValueError: If original batch not found or not in failed status
        """
        original = self.get_batch(batch_id)
        if original is None:
            raise ValueError(f"Batch not found: {batch_id}")
        if original.status != BatchStatus.FAILED:
            raise ValueError(f"Can only retry failed batches, got status: {original.status}")

        # Create new batch with incremented attempt
        new_batch = self.create_batch(
            run_id=original.run_id,
            aggregation_node_id=original.aggregation_node_id,
            attempt=original.attempt + 1,
        )

        # Copy members to new batch
        original_members = self.get_batch_members(batch_id)
        for member in original_members:
            self.add_batch_member(
                batch_id=new_batch.batch_id,
                token_id=member.token_id,
                ordinal=member.ordinal,
            )

        return new_batch

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
        idempotency_key: str | None = None,
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
            idempotency_key: Optional key for retry deduplication

        Returns:
            Artifact model
        """
        artifact_id = artifact_id or generate_id()
        timestamp = now()

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=timestamp,
            idempotency_key=idempotency_key,
        )

        self._ops.execute_insert(
            artifacts_table.insert().values(
                artifact_id=artifact.artifact_id,
                run_id=artifact.run_id,
                produced_by_state_id=artifact.produced_by_state_id,
                sink_node_id=artifact.sink_node_id,
                artifact_type=artifact.artifact_type,
                path_or_uri=artifact.path_or_uri,
                content_hash=artifact.content_hash,
                size_bytes=artifact.size_bytes,
                idempotency_key=artifact.idempotency_key,
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

        rows = self._ops.execute_fetchall(query)
        return [self._artifact_repo.load(row) for row in rows]

    # === Row and Token Query Methods ===

    def get_rows(self, run_id: str) -> list[Row]:
        """Get all rows for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Row models, ordered by row_index
        """
        query = select(rows_table).where(rows_table.c.run_id == run_id).order_by(rows_table.c.row_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._row_repo.load(r) for r in db_rows]

    def get_tokens(self, row_id: str) -> list[Token]:
        """Get all tokens for a row.

        Args:
            row_id: Row ID

        Returns:
            List of Token models, ordered by created_at then token_id
            for deterministic export signatures.
        """
        query = select(tokens_table).where(tokens_table.c.row_id == row_id).order_by(tokens_table.c.created_at, tokens_table.c.token_id)
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_repo.load(r) for r in db_rows]

    def get_node_states_for_token(self, token_id: str) -> list[NodeState]:
        """Get all node states for a token.

        Args:
            token_id: Token ID

        Returns:
            List of NodeState models (discriminated union), ordered by (step_index, attempt)
        """
        # Order by (step_index, attempt) for deterministic ordering across retries
        # Bug fix: P2-2026-01-19-node-state-ordering-missing-attempt
        query = (
            select(node_states_table)
            .where(node_states_table.c.token_id == token_id)
            .order_by(node_states_table.c.step_index, node_states_table.c.attempt)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._node_state_repo.load(r) for r in db_rows]

    def get_row(self, row_id: str) -> Row | None:
        """Get a row by ID.

        Args:
            row_id: Row ID

        Returns:
            Row model or None if not found
        """
        query = select(rows_table).where(rows_table.c.row_id == row_id)
        r = self._ops.execute_fetchone(query)
        if r is None:
            return None
        return self._row_repo.load(r)

    def get_row_data(self, row_id: str) -> RowDataResult:
        """Get the payload data for a row with explicit state.

        Returns a RowDataResult with explicit state indicating why data
        may be unavailable. This replaces the previous ambiguous None return.

        Args:
            row_id: Row ID

        Returns:
            RowDataResult with state and data (if available)
        """
        row = self.get_row(row_id)
        if row is None:
            return RowDataResult(state=RowDataState.ROW_NOT_FOUND, data=None)

        if row.source_data_ref is None:
            return RowDataResult(state=RowDataState.NEVER_STORED, data=None)

        if self._payload_store is None:
            return RowDataResult(state=RowDataState.STORE_NOT_CONFIGURED, data=None)

        try:
            import json

            payload_bytes = self._payload_store.retrieve(row.source_data_ref)
            data = json.loads(payload_bytes.decode("utf-8"))
            return RowDataResult(state=RowDataState.AVAILABLE, data=data)
        except KeyError:
            return RowDataResult(state=RowDataState.PURGED, data=None)

    def get_token(self, token_id: str) -> Token | None:
        """Get a token by ID.

        Args:
            token_id: Token ID

        Returns:
            Token model or None if not found
        """
        query = select(tokens_table).where(tokens_table.c.token_id == token_id)
        r = self._ops.execute_fetchone(query)
        if r is None:
            return None
        return self._token_repo.load(r)

    def get_token_parents(self, token_id: str) -> list[TokenParent]:
        """Get parent relationships for a token.

        Args:
            token_id: Token ID

        Returns:
            List of TokenParent models (ordered by ordinal)
        """
        query = select(token_parents_table).where(token_parents_table.c.token_id == token_id).order_by(token_parents_table.c.ordinal)
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_parent_repo.load(r) for r in db_rows]

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
            .order_by(routing_events_table.c.ordinal, routing_events_table.c.event_id)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._routing_event_repo.load(r) for r in db_rows]

    def get_calls(self, state_id: str) -> list[Call]:
        """Get external calls for a node state.

        Args:
            state_id: State ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = select(calls_table).where(calls_table.c.state_id == state_id).order_by(calls_table.c.call_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

    def allocate_call_index(self, state_id: str) -> int:
        """Allocate next call index for a state_id (thread-safe).

        Provides centralized call index allocation ensuring UNIQUE(state_id, call_index)
        across all client types (HTTP, LLM) and retry attempts.

        This is the single source of truth for call numbering. All AuditedClient
        instances MUST delegate to this method rather than maintaining their own counters.

        Thread Safety:
            Uses a lock to prevent race conditions when multiple threads allocate
            indices concurrently. Safe for pooled execution scenarios.

        Persistence:
            Counter persists across retries within the same run. The LandscapeRecorder
            lifecycle matches the run lifecycle, not the execution attempt.

        Args:
            state_id: Node state ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this state_id

        Example:
            # Two different client types, same state_id
            http_client = AuditedHTTPClient(recorder, state_id="state-001")
            llm_client = AuditedLLMClient(recorder, state_id="state-001")

            # Both delegate to same recorder - indices coordinate correctly
            http_client.post(...)  # allocates index 0
            llm_client.query(...)   # allocates index 1 (not 0!)
        """
        with self._call_index_lock:
            if state_id not in self._call_indices:
                self._call_indices[state_id] = 0
            idx = self._call_indices[state_id]
            self._call_indices[state_id] += 1
            return idx

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call:
        """Record an external call for a node state.

        Args:
            state_id: The node_state this call belongs to
            call_index: 0-based index of this call within the state
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (will be hashed)
            response_data: Response payload (will be hashed, optional for errors)
            error: Error details if status is ERROR (stored as JSON)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response

        Returns:
            The recorded Call model

        Note:
            Duplicate (state_id, call_index) will raise IntegrityError from SQLAlchemy.
            Invalid state_id will raise IntegrityError due to foreign key constraint.
            Call indices should be allocated via allocate_call_index() for coordination.
        """
        call_id = generate_id()
        timestamp = now()

        # Hash request (always required)
        request_hash = stable_hash(request_data)

        # Hash response (optional - None for errors without response)
        response_hash = stable_hash(response_data) if response_data is not None else None

        # Auto-persist request to payload store if available and ref not provided
        # This enables replay/verify modes to retrieve the original request
        if request_ref is None and self._payload_store is not None:
            request_bytes = canonical_json(request_data).encode("utf-8")
            request_ref = self._payload_store.store(request_bytes)

        # Auto-persist response to payload store if available and ref not provided
        # This enables replay/verify modes to retrieve the original response
        if response_data is not None and response_ref is None and self._payload_store is not None:
            response_bytes = canonical_json(response_data).encode("utf-8")
            response_ref = self._payload_store.store(response_bytes)

        # Serialize error if present
        error_json = canonical_json(error) if error is not None else None

        values = {
            "call_id": call_id,
            "state_id": state_id,
            "operation_id": None,  # State call, not operation call
            "call_index": call_index,
            "call_type": call_type.value,  # Store enum value
            "status": status.value,  # Store enum value
            "request_hash": request_hash,
            "request_ref": request_ref,
            "response_hash": response_hash,
            "response_ref": response_ref,
            "error_json": error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        self._ops.execute_insert(calls_table.insert().values(**values))

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,  # Pass enum directly per Call contract
            status=status,  # Pass enum directly per Call contract
            request_hash=request_hash,
            created_at=timestamp,
            state_id=state_id,  # Parent is node_state
            operation_id=None,  # Not an operation call
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            latency_ms=latency_ms,
        )

    # === Operations (Source/Sink I/O) ===

    def begin_operation(
        self,
        run_id: str,
        node_id: str,
        operation_type: Literal["source_load", "sink_write"],
        *,
        input_data: dict[str, Any] | None = None,
    ) -> Operation:
        """Begin an operation for source/sink I/O.

        Operations are the source/sink equivalent of node_states - they provide
        a parent context for external calls made during load() or write().

        Args:
            run_id: Run this operation belongs to
            node_id: Source or sink node performing the operation
            operation_type: Type of operation
            input_data: Optional input context (stored via payload store)

        Returns:
            Operation with operation_id for call attribution
        """
        from uuid import uuid4

        # Use pure UUID for operation_id - run_id + node_id can exceed 64 chars
        # (run_id=36 + node_id=45 + prefixes would be 94+ chars)
        operation_id = f"op_{uuid4().hex}"  # "op_" + 32 hex = 35 chars, well under 64

        input_ref = None
        if input_data and self._payload_store is not None:
            input_bytes = canonical_json(input_data).encode("utf-8")
            input_ref = self._payload_store.store(input_bytes)

        timestamp = now()
        operation = Operation(
            operation_id=operation_id,
            run_id=run_id,
            node_id=node_id,
            operation_type=operation_type,
            started_at=timestamp,
            status="open",
            input_data_ref=input_ref,
        )

        self._ops.execute_insert(operations_table.insert().values(**operation.to_dict()))
        return operation

    def complete_operation(
        self,
        operation_id: str,
        status: Literal["completed", "failed", "pending"],
        *,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Complete an operation.

        Args:
            operation_id: Operation to complete
            status: Final status ('completed', 'failed', or 'pending' for BatchPendingError)
            output_data: Optional output context
            error: Error message if failed
            duration_ms: Operation duration

        Raises:
            FrameworkBugError: If operation doesn't exist or is already completed
        """
        # Validate operation exists and is open (prevent double-complete)
        query = select(operations_table.c.status).where(operations_table.c.operation_id == operation_id)
        current = self._ops.execute_fetchone(query)

        if current is None:
            raise FrameworkBugError(f"Completing non-existent operation: {operation_id}")

        if current.status != "open":
            raise FrameworkBugError(
                f"Completing already-completed operation {operation_id}: current status={current.status}, new status={status}"
            )

        output_ref = None
        if output_data and self._payload_store is not None:
            output_bytes = canonical_json(output_data).encode("utf-8")
            output_ref = self._payload_store.store(output_bytes)

        timestamp = now()
        self._ops.execute_update(
            operations_table.update()
            .where(operations_table.c.operation_id == operation_id)
            .values(
                completed_at=timestamp,
                status=status,
                output_data_ref=output_ref,
                error_message=error,
                duration_ms=duration_ms,
            )
        )

    def allocate_operation_call_index(self, operation_id: str) -> int:
        """Allocate next call index for an operation_id (thread-safe).

        Provides centralized call index allocation ensuring unique call numbering
        within each operation. Parallel to allocate_call_index() for node_states.

        Args:
            operation_id: Operation ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this operation_id
        """
        with self._call_index_lock:  # Reuse existing lock
            if operation_id not in self._operation_call_indices:
                self._operation_call_indices[operation_id] = 0
            idx = self._operation_call_indices[operation_id]
            self._operation_call_indices[operation_id] += 1
            return idx

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        provider: str | None = None,
    ) -> Call:
        """Record an external call made during an operation.

        This is the operation equivalent of record_call() - attributes calls
        to operations instead of node_states.

        Args:
            operation_id: The operation this call belongs to
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (will be hashed)
            response_data: Response payload (will be hashed, optional for errors)
            error: Error details if status is ERROR (stored as JSON)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response
            provider: Optional provider name for telemetry

        Returns:
            The recorded Call model
        """
        call_index = self.allocate_operation_call_index(operation_id)
        call_id = f"call_{operation_id}_{call_index}"
        timestamp = now()

        # Hash request (always required)
        request_hash = stable_hash(request_data)

        # Hash response (optional - None for errors without response)
        response_hash = stable_hash(response_data) if response_data is not None else None

        # Auto-persist request to payload store if available and ref not provided
        if request_ref is None and self._payload_store is not None:
            request_bytes = canonical_json(request_data).encode("utf-8")
            request_ref = self._payload_store.store(request_bytes)

        # Auto-persist response to payload store if available and ref not provided
        if response_data is not None and response_ref is None and self._payload_store is not None:
            response_bytes = canonical_json(response_data).encode("utf-8")
            response_ref = self._payload_store.store(response_bytes)

        # Serialize error if present
        error_json = canonical_json(error) if error is not None else None

        values = {
            "call_id": call_id,
            "state_id": None,  # NOT a node_state call
            "operation_id": operation_id,  # Operation call
            "call_index": call_index,
            "call_type": call_type.value,
            "status": status.value,
            "request_hash": request_hash,
            "request_ref": request_ref,
            "response_hash": response_hash,
            "response_ref": response_ref,
            "error_json": error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        self._ops.execute_insert(calls_table.insert().values(**values))

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=request_hash,
            created_at=timestamp,
            state_id=None,  # Not a node_state call
            operation_id=operation_id,  # Parent is operation
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            latency_ms=latency_ms,
        )

    def get_operation(self, operation_id: str) -> Operation | None:
        """Get an operation by ID.

        Args:
            operation_id: Operation ID to retrieve

        Returns:
            Operation model or None if not found
        """
        query = select(operations_table).where(operations_table.c.operation_id == operation_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None

        return Operation(
            operation_id=row.operation_id,
            run_id=row.run_id,
            node_id=row.node_id,
            operation_type=row.operation_type,
            started_at=row.started_at,
            completed_at=row.completed_at,
            status=row.status,
            input_data_ref=row.input_data_ref,
            output_data_ref=row.output_data_ref,
            error_message=row.error_message,
            duration_ms=row.duration_ms,
        )

    def get_operation_calls(self, operation_id: str) -> list[Call]:
        """Get external calls for an operation.

        Args:
            operation_id: Operation ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = select(calls_table).where(calls_table.c.operation_id == operation_id).order_by(calls_table.c.call_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

    def get_operations_for_run(self, run_id: str) -> list[Operation]:
        """Get all operations for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Operation models, ordered by started_at
        """
        query = select(operations_table).where(operations_table.c.run_id == run_id).order_by(operations_table.c.started_at)
        db_rows = self._ops.execute_fetchall(query)
        return [
            Operation(
                operation_id=row.operation_id,
                run_id=row.run_id,
                node_id=row.node_id,
                operation_type=row.operation_type,
                started_at=row.started_at,
                completed_at=row.completed_at,
                status=row.status,
                input_data_ref=row.input_data_ref,
                output_data_ref=row.output_data_ref,
                error_message=row.error_message,
                duration_ms=row.duration_ms,
            )
            for row in db_rows
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
            row_id=row.row_id,
            run_id=row.run_id,
            source_node_id=row.source_node_id,
            row_index=row.row_index,
            source_data_hash=row.source_data_hash,
            created_at=row.created_at,
            source_data=source_data,
            payload_available=payload_available,
        )

    # === Reproducibility Grade Management ===

    def compute_reproducibility_grade(self, run_id: str) -> ReproducibilityGrade:
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

    def finalize_run(self, run_id: str, status: RunStatus) -> Run:
        """Finalize a run by computing grade and completing it.

        Convenience method that:
        1. Computes the reproducibility grade based on node determinism
        2. Completes the run with the specified status and computed grade

        Args:
            run_id: Run to finalize
            status: Final RunStatus (COMPLETED or FAILED)

        Returns:
            Updated Run model
        """
        grade = self.compute_reproducibility_grade(run_id)
        return self.complete_run(run_id, status, reproducibility_grade=grade.value)

    # === Token Outcome Recording (AUD-001) ===

    def _validate_outcome_fields(
        self,
        outcome: RowOutcome,
        *,
        sink_name: str | None,
        batch_id: str | None,
        fork_group_id: str | None,
        join_group_id: str | None,
        expand_group_id: str | None,
        error_hash: str | None,
    ) -> None:
        """Validate required fields are present for each outcome type.

        Enforces the token outcome contract from docs/audit/tokens/00-token-outcome-contract.md.
        This is defense-in-depth: callers SHOULD pass correct fields, but this catches bugs.

        Raises:
            ValueError: If a required field is missing for the outcome type
        """
        # Map outcome to required field(s)
        # Contract: Each outcome type has specific required fields
        if outcome == RowOutcome.COMPLETED:
            if sink_name is None:
                raise ValueError(
                    "COMPLETED outcome requires sink_name but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.ROUTED:
            if sink_name is None:
                raise ValueError(
                    "ROUTED outcome requires sink_name but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.FORKED:
            if fork_group_id is None:
                raise ValueError(
                    "FORKED outcome requires fork_group_id but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.FAILED:
            if error_hash is None:
                raise ValueError(
                    "FAILED outcome requires error_hash but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.QUARANTINED:
            if error_hash is None:
                raise ValueError(
                    "QUARANTINED outcome requires error_hash but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.CONSUMED_IN_BATCH:
            if batch_id is None:
                raise ValueError(
                    "CONSUMED_IN_BATCH outcome requires batch_id but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.COALESCED:
            if join_group_id is None:
                raise ValueError(
                    "COALESCED outcome requires join_group_id but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.EXPANDED:
            if expand_group_id is None:
                raise ValueError(
                    "EXPANDED outcome requires expand_group_id but got None. "
                    "Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
                )
        elif outcome == RowOutcome.BUFFERED and batch_id is None:
            raise ValueError(
                "BUFFERED outcome requires batch_id but got None. Contract violation - see docs/audit/tokens/00-token-outcome-contract.md"
            )
        # No else needed - exhaustive enum handling above

    def record_token_outcome(
        self,
        run_id: str,
        token_id: str,
        outcome: RowOutcome,
        *,
        sink_name: str | None = None,
        batch_id: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
        expand_group_id: str | None = None,
        error_hash: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Record a token's outcome in the audit trail.

        Called at the moment the outcome is determined in processor.py.
        For BUFFERED tokens, a second call records the terminal outcome
        when the batch flushes.

        Args:
            run_id: Current run ID
            token_id: Token that reached this outcome
            outcome: The RowOutcome enum value
            sink_name: For ROUTED/COMPLETED - which sink (REQUIRED)
            batch_id: For CONSUMED_IN_BATCH/BUFFERED - which batch (REQUIRED)
            fork_group_id: For FORKED - the fork group (REQUIRED)
            join_group_id: For COALESCED - the join group (REQUIRED)
            expand_group_id: For EXPANDED - the expand group (REQUIRED)
            error_hash: For FAILED/QUARANTINED - hash of error details (REQUIRED)
            context: Optional additional context (stored as JSON)

        Returns:
            outcome_id for tracking

        Raises:
            ValueError: If required fields for outcome type are missing
            IntegrityError: If terminal outcome already exists for token
        """
        # Validate required fields per outcome type (contract enforcement)
        # See docs/audit/tokens/00-token-outcome-contract.md
        self._validate_outcome_fields(
            outcome=outcome,
            sink_name=sink_name,
            batch_id=batch_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            error_hash=error_hash,
        )

        outcome_id = f"out_{generate_id()[:12]}"
        is_terminal = outcome.is_terminal
        context_json = canonical_json(context) if context is not None else None

        self._ops.execute_insert(
            token_outcomes_table.insert().values(
                outcome_id=outcome_id,
                run_id=run_id,
                token_id=token_id,
                outcome=outcome.value,
                is_terminal=1 if is_terminal else 0,
                recorded_at=now(),
                sink_name=sink_name,
                batch_id=batch_id,
                fork_group_id=fork_group_id,
                join_group_id=join_group_id,
                expand_group_id=expand_group_id,
                error_hash=error_hash,
                context_json=context_json,
            )
        )

        return outcome_id

    def get_token_outcome(self, token_id: str) -> TokenOutcome | None:
        """Get the terminal outcome for a token.

        Returns the terminal outcome if one exists, otherwise the most
        recent non-terminal outcome (BUFFERED).

        Args:
            token_id: Token to look up

        Returns:
            TokenOutcome dataclass or None if no outcome recorded
        """
        # Get most recent outcome (terminal preferred)
        query = (
            select(token_outcomes_table)
            .where(token_outcomes_table.c.token_id == token_id)
            .order_by(
                token_outcomes_table.c.is_terminal.desc(),  # Terminal first
                token_outcomes_table.c.recorded_at.desc(),  # Then by time
            )
            .limit(1)
        )
        result = self._ops.execute_fetchone(query)
        if result is None:
            return None
        return self._token_outcome_repo.load(result)

    def get_token_outcomes_for_row(self, run_id: str, row_id: str) -> list[TokenOutcome]:
        """Get all token outcomes for a row in a single query.

        Uses JOIN to avoid N+1 query pattern when resolving row_id to tokens.
        Critical for explain() disambiguation with forks/expands.

        Args:
            run_id: Run ID to filter by (prevents cross-run contamination)
            row_id: Row ID

        Returns:
            List of TokenOutcome objects, empty if no outcomes recorded.
            Ordered by recorded_at for deterministic behavior.
        """
        # Single JOIN query: tokens + outcomes
        query = (
            select(
                token_outcomes_table.c.outcome_id,
                token_outcomes_table.c.run_id,
                token_outcomes_table.c.token_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.is_terminal,
                token_outcomes_table.c.recorded_at,
                token_outcomes_table.c.sink_name,
                token_outcomes_table.c.batch_id,
                token_outcomes_table.c.fork_group_id,
                token_outcomes_table.c.join_group_id,
                token_outcomes_table.c.expand_group_id,
                token_outcomes_table.c.error_hash,
                token_outcomes_table.c.context_json,
                token_outcomes_table.c.expected_branches_json,
            )
            .join(
                tokens_table,
                token_outcomes_table.c.token_id == tokens_table.c.token_id,
            )
            .where(tokens_table.c.row_id == row_id)
            .where(token_outcomes_table.c.run_id == run_id)
            .order_by(token_outcomes_table.c.recorded_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._token_outcome_repo.load(r) for r in rows]

    # === Validation Error Recording (WP-11.99) ===

    def record_validation_error(
        self,
        run_id: str,
        node_id: str | None,
        row_data: Any,
        error: str,
        schema_mode: str,
        destination: str,
    ) -> str:
        """Record a validation error in the audit trail.

        Called when a source row fails schema validation. The row is
        quarantined (not processed further) but we record what we saw
        for complete audit coverage.

        Args:
            run_id: Current run ID
            node_id: Node where validation failed
            row_data: The row that failed validation (may be non-dict or contain non-finite values)
            error: Error description
            schema_mode: Schema mode that caught the error ("strict", "free", "dynamic")
            destination: Where row was routed ("discard" or sink name)

        Returns:
            error_id for tracking
        """
        logger = logging.getLogger(__name__)
        error_id = f"verr_{generate_id()[:12]}"

        # Tier-3 (external data) trust boundary: row_data may be non-canonical
        # Try canonical hash/JSON first, fall back to safe representations
        try:
            row_hash = stable_hash(row_data)
            row_data_json = canonical_json(row_data)
        except (ValueError, TypeError) as e:
            # Non-canonical data (NaN, Infinity, non-dict, etc.)
            # Use repr() fallback to preserve audit trail
            row_preview = repr(row_data)[:200] + "..." if len(repr(row_data)) > 200 else repr(row_data)
            logger.warning(
                "Validation error row not canonically serializable (using repr fallback): %s | Row preview: %s",
                str(e),
                row_preview,
            )
            row_hash = repr_hash(row_data)
            # Store non-canonical representation with type metadata
            metadata = NonCanonicalMetadata.from_error(row_data, e)
            row_data_json = json.dumps(metadata.to_dict())

        self._ops.execute_insert(
            validation_errors_table.insert().values(
                error_id=error_id,
                run_id=run_id,
                node_id=node_id,
                row_hash=row_hash,
                row_data_json=row_data_json,
                error=error,
                schema_mode=schema_mode,
                destination=destination,
                created_at=now(),
            )
        )

        return error_id

    # === Transform Error Recording (WP-11.99b) ===

    def record_transform_error(
        self,
        run_id: str,
        token_id: str,
        transform_id: str,
        row_data: dict[str, Any],
        error_details: TransformErrorReason,
        destination: str,
    ) -> str:
        """Record a transform processing error in the audit trail.

        Called when a transform returns TransformResult.error().
        This is for legitimate errors, NOT transform bugs.

        Args:
            run_id: Current run ID
            token_id: Token ID for the row
            transform_id: Transform that returned the error
            row_data: The row that could not be processed
            error_details: Error details from TransformResult (TransformErrorReason TypedDict)
            destination: Where row was routed ("discard" or sink name)

        Returns:
            error_id for tracking
        """
        error_id = f"terr_{generate_id()[:12]}"

        self._ops.execute_insert(
            transform_errors_table.insert().values(
                error_id=error_id,
                run_id=run_id,
                token_id=token_id,
                transform_id=transform_id,
                row_hash=stable_hash(row_data),
                row_data_json=canonical_json(row_data),
                error_details_json=canonical_json(error_details),
                destination=destination,
                created_at=now(),
            )
        )

        return error_id

    # === Error Query Methods ===

    def get_validation_errors_for_row(self, run_id: str, row_hash: str) -> list[ValidationErrorRecord]:
        """Get validation errors for a row by its hash.

        Validation errors are keyed by row_hash since quarantined rows
        never get row_ids (they're rejected before entering the pipeline).

        Args:
            run_id: Run ID to query
            row_hash: Hash of the row data

        Returns:
            List of ValidationErrorRecord models
        """
        query = select(validation_errors_table).where(
            validation_errors_table.c.run_id == run_id,
            validation_errors_table.c.row_hash == row_hash,
        )
        rows = self._ops.execute_fetchall(query)
        return [self._validation_error_repo.load(r) for r in rows]

    def get_validation_errors_for_run(self, run_id: str) -> list[ValidationErrorRecord]:
        """Get all validation errors for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of ValidationErrorRecord models, ordered by created_at
        """
        query = (
            select(validation_errors_table).where(validation_errors_table.c.run_id == run_id).order_by(validation_errors_table.c.created_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._validation_error_repo.load(r) for r in rows]

    def get_transform_errors_for_token(self, token_id: str) -> list[TransformErrorRecord]:
        """Get transform errors for a specific token.

        Args:
            token_id: Token ID to query

        Returns:
            List of TransformErrorRecord models
        """
        query = select(transform_errors_table).where(
            transform_errors_table.c.token_id == token_id,
        )
        rows = self._ops.execute_fetchall(query)
        return [self._transform_error_repo.load(r) for r in rows]

    def get_transform_errors_for_run(self, run_id: str) -> list[TransformErrorRecord]:
        """Get all transform errors for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of TransformErrorRecord models, ordered by created_at
        """
        query = (
            select(transform_errors_table).where(transform_errors_table.c.run_id == run_id).order_by(transform_errors_table.c.created_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._transform_error_repo.load(r) for r in rows]

    # === Call Lookup Methods (for Replay Mode) ===

    def find_call_by_request_hash(
        self,
        run_id: str,
        call_type: str,
        request_hash: str,
        *,
        sequence_index: int = 0,
    ) -> Call | None:
        """Find a call by its request hash within a run.

        Used for replay mode to look up previously recorded calls by
        the hash of their request data.

        Args:
            run_id: Run ID to search within
            call_type: Type of call (llm, http, etc.)
            request_hash: SHA-256 hash of the canonical request data
            sequence_index: 0-based index for duplicate request hashes.
                When the same request is made multiple times in a run
                (e.g., retries, loops), use sequence_index to get the
                Nth occurrence (0=first, 1=second, etc.).

        Returns:
            Call model if found, None otherwise

        Note:
            Calls are ordered chronologically by created_at. The sequence_index
            parameter allows disambiguation when the same request was made
            multiple times (each returning a different response).
        """
        # Join to node_states to filter by run_id
        # NOTE: Use node_states.run_id directly (denormalized column) instead of
        # joining through nodes table. The nodes table has composite PK (node_id, run_id),
        # so joining on node_id alone would be ambiguous when node_id is reused across runs.
        query = (
            select(calls_table)
            .join(
                node_states_table,
                calls_table.c.state_id == node_states_table.c.state_id,
            )
            .where(node_states_table.c.run_id == run_id)
            .where(calls_table.c.call_type == call_type)
            .where(calls_table.c.request_hash == request_hash)
            .order_by(calls_table.c.created_at)
            .limit(1)
            .offset(sequence_index)
        )
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._call_repo.load(row)

    def get_call_response_data(self, call_id: str) -> dict[str, Any] | None:
        """Retrieve the response data for a call.

        Fetches response data from the payload store if response_ref is set,
        otherwise returns None.

        Args:
            call_id: The call ID to get response data for

        Returns:
            Response data dict if available, None if no response was recorded
            or if payload store is not configured

        Note:
            Returns None if:
            - Call not found
            - No response_ref set on the call (error calls may not have response)
            - Payload store not configured
            - Response data has been purged from payload store
        """
        # Get the call record first
        query = select(calls_table).where(calls_table.c.call_id == call_id)
        row = self._ops.execute_fetchone(query)

        if row is None:
            return None

        if row.response_ref is None:
            return None

        if self._payload_store is None:
            return None

        try:
            payload_bytes = self._payload_store.retrieve(row.response_ref)
            data: dict[str, Any] = json.loads(payload_bytes.decode("utf-8"))
            return data
        except KeyError:
            # Payload has been purged
            return None
