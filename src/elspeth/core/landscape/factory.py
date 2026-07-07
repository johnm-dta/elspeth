"""RecorderFactory: construction point for Landscape repositories.

Single place that wires up loaders, database operations, and repository instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.core.landscape._database_ops import DatabaseOps, ReadOnlyDatabaseOps
from elspeth.core.landscape.auth_audit_repository import AuthAuditRepository
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    EdgeLoader,
    NodeLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
    RowLoader,
    RunLoader,
    SchedulerEventLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.plugin_audit_writer import PluginAuditWriterAdapter as _PluginAuditWriterAdapterImpl
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.core.landscape.scheduler import BarrierRestoreReadModel
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore


class RecorderFactory:
    """Construction point for Landscape repositories.

    Creates the Landscape repository graph from a LandscapeDB, sharing loader
    instances to ensure consistent object construction across repositories.
    """

    def __init__(self, db: LandscapeDB, *, payload_store: PayloadStore | None = None) -> None:
        self._db = db
        self._payload_store = payload_store

        # Database operations helper for reduced boilerplate
        ops = DatabaseOps(db)
        read_ops = ReadOnlyDatabaseOps(db)

        # Loader instances for row-to-object conversions
        run_loader = RunLoader()
        node_loader = NodeLoader()
        edge_loader = EdgeLoader()
        row_loader = RowLoader()
        token_loader = TokenLoader()
        token_parent_loader = TokenParentLoader()
        call_loader = CallLoader()
        operation_loader = OperationLoader()
        routing_event_loader = RoutingEventLoader()
        batch_loader = BatchLoader()
        node_state_loader = NodeStateLoader()
        validation_error_loader = ValidationErrorLoader()
        transform_error_loader = TransformErrorLoader()
        token_outcome_loader = TokenOutcomeLoader()
        scheduler_event_loader = SchedulerEventLoader()
        artifact_loader = ArtifactLoader()
        batch_member_loader = BatchMemberLoader()

        # Composed repository for run lifecycle
        self._run_lifecycle = RunLifecycleRepository(db, ops, run_loader)
        self._auth_audit = AuthAuditRepository(ops)

        # Composed repository for execution recording
        self._execution = ExecutionRepository(
            db,
            ops,
            node_state_loader=node_state_loader,
            routing_event_loader=routing_event_loader,
            call_loader=call_loader,
            operation_loader=operation_loader,
            batch_loader=batch_loader,
            batch_member_loader=batch_member_loader,
            artifact_loader=artifact_loader,
            payload_store=payload_store,
        )

        # Composed repository for data flow recording
        self._data_flow = DataFlowRepository(
            db,
            ops,
            token_outcome_loader=token_outcome_loader,
            node_loader=node_loader,
            edge_loader=edge_loader,
            validation_error_loader=validation_error_loader,
            transform_error_loader=transform_error_loader,
            payload_store=payload_store,
        )
        self._barrier_restore = BarrierRestoreReadModel(
            ops,
            token_outcome_loader=token_outcome_loader,
        )

        # Composed repository for read-only queries
        self._query = QueryRepository(
            read_ops,
            row_loader=row_loader,
            token_loader=token_loader,
            token_parent_loader=token_parent_loader,
            node_state_loader=node_state_loader,
            routing_event_loader=routing_event_loader,
            call_loader=call_loader,
            token_outcome_loader=token_outcome_loader,
            scheduler_event_loader=scheduler_event_loader,
            payload_store=payload_store,
        )
        # The scheduler repository is a pure write surface (its constructor
        # runs a SQLite Tier-1 PRAGMA probe when applicable).  On a
        # read-only handle — MCP analyzer, web read surfaces, immutable
        # snapshot opens whose journal_mode legitimately reads ``delete`` —
        # there is nothing it could ever do, so skip construction entirely
        # and fail loudly on access instead.
        self._scheduler: TokenSchedulerRepository | None = None if db.is_read_only else TokenSchedulerRepository(db.engine)
        # Same posture for the coordination substrate (epoch 21, ADR-030): a
        # pure write/arbitration surface whose constructor runs the same
        # SQLite Tier-1 PRAGMA probe when applicable — nothing it could do on
        # a read-only handle.
        self._run_coordination: RunCoordinationRepository | None = None if db.is_read_only else RunCoordinationRepository(db.engine)

    @property
    def run_lifecycle(self) -> RunLifecycleRepository:
        return self._run_lifecycle

    @property
    def auth_audit(self) -> AuthAuditRepository:
        return self._auth_audit

    @property
    def execution(self) -> ExecutionRepository:
        return self._execution

    @property
    def data_flow(self) -> DataFlowRepository:
        return self._data_flow

    @property
    def barrier_restore(self) -> BarrierRestoreReadModel:
        return self._barrier_restore

    @property
    def query(self) -> QueryRepository:
        return self._query

    @property
    def scheduler(self) -> TokenSchedulerRepository:
        if self._scheduler is None:
            raise RuntimeError("scheduler repository is not available on a read-only LandscapeDB handle")
        return self._scheduler

    @property
    def run_coordination(self) -> RunCoordinationRepository:
        if self._run_coordination is None:
            raise RuntimeError("run coordination repository is not available on a read-only LandscapeDB handle")
        return self._run_coordination

    @property
    def payload_store(self) -> PayloadStore | None:
        return self._payload_store

    def plugin_audit_writer(self) -> PluginAuditWriter:
        """Create a PluginAuditWriter adapter composing the three repositories."""
        return _PluginAuditWriterAdapterImpl(self._execution, self._data_flow, self._run_lifecycle)
