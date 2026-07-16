"""RecorderFactory: construction point for Landscape repositories.

Single place that wires up loaders, database operations, and repository instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.core.landscape._database_ops import DatabaseOps, ReadOnlyDatabaseOps
from elspeth.core.landscape.auth_audit_repository import AuthAuditRepository
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.audit_export_snapshots import AuditExportSnapshotRepository
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
    SinkEffectLoader,
    SinkEffectMemberLoader,
    SinkEffectStreamLoader,
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
from elspeth.core.landscape.run_status_projection import AuditRunStatusProjection
from elspeth.core.landscape.scheduler import BarrierRestoreReadModel
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

if TYPE_CHECKING:
    from elspeth.contracts import Run, SecretResolution
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.landscape.run_lifecycle_repository import (
        RunSourceFieldResolutionRecord,
        RunSourceLifecycleRecord,
        RunSourceResumeRecord,
    )


class RunLifecycleReadRepository:
    """Read-only run lifecycle port exposed by ``LandscapeReadRepositories``."""

    __slots__ = ("_repo",)

    def __init__(self, repo: RunLifecycleRepository) -> None:
        self._repo = repo

    def get_run(self, run_id: str) -> Run | None:
        return self._repo.get_run(run_id)

    def get_run_attribution(self, run_id: str) -> tuple[str, str] | None:
        return self._repo.get_run_attribution(run_id)

    def get_source_schema(self, run_id: str) -> str:
        return self._repo.get_source_schema(run_id)

    def get_runtime_val_manifest(self, run_id: str) -> str:
        return self._repo.get_runtime_val_manifest(run_id)

    def get_run_source_resume_records(self, run_id: str) -> dict[str, RunSourceResumeRecord]:
        return self._repo.get_run_source_resume_records(run_id)

    def get_run_source_lifecycle_records(self, run_id: str) -> dict[str, RunSourceLifecycleRecord]:
        return self._repo.get_run_source_lifecycle_records(run_id)

    def get_source_field_resolution(self, run_id: str) -> dict[str, str] | None:
        return self._repo.get_source_field_resolution(run_id)

    def get_source_field_resolutions(self, run_id: str) -> dict[str, RunSourceFieldResolutionRecord]:
        return self._repo.get_source_field_resolutions(run_id)

    def get_resume_field_resolution(self, run_id: str) -> dict[str, str] | None:
        return self._repo.get_resume_field_resolution(run_id)

    def get_secret_resolutions_for_run(self, run_id: str) -> list[SecretResolution]:
        return self._repo.get_secret_resolutions_for_run(run_id)

    def list_runs(self, *, status: Any | None = None) -> list[Run]:
        return self._repo.list_runs(status=status)


class DataFlowReadRepository:
    """Read-only data-flow port exposed by ``LandscapeReadRepositories``."""

    __slots__ = ("_repo",)

    def __init__(self, repo: DataFlowRepository) -> None:
        self._repo = repo

    def get_token_outcome(self, token_id: str) -> Any | None:
        return self._repo.get_token_outcome(token_id)

    def get_token_outcomes_for_row(self, run_id: str, row_id: str) -> list[Any]:
        return self._repo.get_token_outcomes_for_row(run_id, row_id)

    def get_node(self, node_id: str, run_id: str) -> Any | None:
        return self._repo.get_node(node_id, run_id)

    def get_nodes(self, run_id: str) -> list[Any]:
        return self._repo.get_nodes(run_id)

    def get_node_contracts(self, *args: Any, **kwargs: Any) -> Any:
        return self._repo.get_node_contracts(*args, **kwargs)

    def get_edges(self, run_id: str) -> list[Any]:
        return self._repo.get_edges(run_id)

    def get_edge(self, edge_id: str) -> Any:
        return self._repo.get_edge(edge_id)

    def get_edge_map(self, run_id: str) -> dict[tuple[str, str], str]:
        return self._repo.get_edge_map(run_id)

    def get_validation_errors_for_row(
        self,
        run_id: str,
        row_hash: str | None = None,
        *,
        row_id: str | None = None,
    ) -> list[Any]:
        return self._repo.get_validation_errors_for_row(run_id, row_hash, row_id=row_id)

    def get_validation_errors_for_run(self, run_id: str) -> list[Any]:
        return self._repo.get_validation_errors_for_run(run_id)

    def get_transform_errors_for_token(self, token_id: str) -> list[Any]:
        return self._repo.get_transform_errors_for_token(token_id)

    def get_transform_errors_for_run(self, run_id: str) -> list[Any]:
        return self._repo.get_transform_errors_for_run(run_id)


class ExecutionReadRepository:
    """Read-only execution port exposed by ``LandscapeReadRepositories``."""

    __slots__ = ("_repo",)

    def __init__(self, repo: ExecutionRepository) -> None:
        self._repo = repo

    def get_node_state(self, state_id: str) -> Any | None:
        return self._repo.get_node_state(state_id)

    def get_max_node_state_attempts(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        return self._repo.get_max_node_state_attempts(*args, **kwargs)

    def get_open_node_state_ids(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        return self._repo.get_open_node_state_ids(*args, **kwargs)

    def get_completed_row_ids_for_nodes(self, *args: Any, **kwargs: Any) -> set[tuple[str, str]]:
        return self._repo.get_completed_row_ids_for_nodes(*args, **kwargs)

    def get_operation(self, operation_id: str) -> Any | None:
        return self._repo.get_operation(operation_id)

    def get_operation_calls(self, operation_id: str) -> list[Any]:
        return self._repo.get_operation_calls(operation_id)

    def get_operations_for_run(self, run_id: str) -> list[Any]:
        return self._repo.get_operations_for_run(run_id)

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Any]:
        return self._repo.get_all_operation_calls_for_run(run_id)

    def get_call_response_data(self, call_id: str) -> Any:
        return self._repo.get_call_response_data(call_id)

    def get_batch(self, batch_id: str) -> Any | None:
        return self._repo.get_batch(batch_id)

    def get_batches(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self._repo.get_batches(*args, **kwargs)

    def get_incomplete_batches(self, run_id: str) -> list[Any]:
        return self._repo.get_incomplete_batches(run_id)

    def get_batch_members(self, batch_id: str) -> list[Any]:
        return self._repo.get_batch_members(batch_id)

    def get_all_batch_members_for_run(self, run_id: str) -> list[Any]:
        return self._repo.get_all_batch_members_for_run(run_id)

    def get_artifacts(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self._repo.get_artifacts(*args, **kwargs)


class LandscapeReadRepositories:
    """Typed read repository surface for inspection-only Landscape callers."""

    __slots__ = (
        "barrier_restore",
        "data_flow",
        "execution",
        "payload_store",
        "query",
        "run_lifecycle",
        "run_status_projection",
    )

    def __init__(
        self,
        *,
        run_lifecycle: RunLifecycleReadRepository,
        data_flow: DataFlowReadRepository,
        execution: ExecutionReadRepository,
        query: QueryRepository,
        run_status_projection: AuditRunStatusProjection,
        barrier_restore: BarrierRestoreReadModel,
        payload_store: PayloadStore | None,
    ) -> None:
        self.run_lifecycle = run_lifecycle
        self.data_flow = data_flow
        self.execution = execution
        self.query = query
        self.run_status_projection = run_status_projection
        self.barrier_restore = barrier_restore
        self.payload_store = payload_store


class LandscapeWriteRepositories:
    """Typed writable repository surface for audit-recording callers."""

    __slots__ = (
        "audit_export_snapshots",
        "auth_audit",
        "barrier_restore",
        "data_flow",
        "execution",
        "payload_store",
        "query",
        "read",
        "run_coordination",
        "run_lifecycle",
        "run_status_projection",
        "scheduler",
    )

    def __init__(
        self,
        *,
        read: LandscapeReadRepositories,
        run_lifecycle: RunLifecycleRepository,
        auth_audit: AuthAuditRepository,
        execution: ExecutionRepository,
        data_flow: DataFlowRepository,
        scheduler: TokenSchedulerRepository,
        run_coordination: RunCoordinationRepository,
        audit_export_snapshots: AuditExportSnapshotRepository,
    ) -> None:
        self.read = read
        self.run_lifecycle = run_lifecycle
        self.auth_audit = auth_audit
        self.execution = execution
        self.data_flow = data_flow
        self.query = read.query
        self.run_status_projection = read.run_status_projection
        self.barrier_restore = read.barrier_restore
        self.payload_store = read.payload_store
        self.scheduler = scheduler
        self.run_coordination = run_coordination
        self.audit_export_snapshots = audit_export_snapshots

    def plugin_audit_writer(self) -> PluginAuditWriter:
        """Create a PluginAuditWriter adapter composing the writable repositories."""
        return _PluginAuditWriterAdapterImpl(self.execution, self.data_flow, self.run_lifecycle)


class RecorderFactory:
    """Construction point for Landscape repositories.

    Creates the Landscape repository graph from a LandscapeDB, sharing loader
    instances to ensure consistent object construction across repositories.
    """

    @classmethod
    def read_only(cls, db: LandscapeDB, *, payload_store: PayloadStore | None = None) -> LandscapeReadRepositories:
        """Construct an explicit read-only repository surface."""
        return cls._build_read_repositories(db, payload_store=payload_store)

    @classmethod
    def writable(cls, db: LandscapeDB, *, payload_store: PayloadStore | None = None) -> LandscapeWriteRepositories:
        """Construct an explicit writable repository surface.

        Raises:
            RuntimeError: If ``db`` is an inspection-only handle.
        """
        if db.is_read_only:
            raise RuntimeError("writable repositories are not available on a read-only LandscapeDB handle")
        return cls(db, payload_store=payload_store).write_repositories()

    @staticmethod
    def _build_read_repositories(db: LandscapeDB, *, payload_store: PayloadStore | None) -> LandscapeReadRepositories:
        """Build read repository ports without constructing write-only helpers."""
        read_ops = ReadOnlyDatabaseOps(db)
        read_ops_as_common = cast(DatabaseOps, read_ops)

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
        sink_effect_loader = SinkEffectLoader()
        sink_effect_member_loader = SinkEffectMemberLoader()
        sink_effect_stream_loader = SinkEffectStreamLoader()

        run_lifecycle = RunLifecycleRepository(db, read_ops_as_common, run_loader)
        data_flow = DataFlowRepository(
            db,
            read_ops_as_common,
            token_outcome_loader=token_outcome_loader,
            node_loader=node_loader,
            edge_loader=edge_loader,
            validation_error_loader=validation_error_loader,
            transform_error_loader=transform_error_loader,
            payload_store=payload_store,
        )
        execution = ExecutionRepository(
            db,
            read_ops_as_common,
            node_state_loader=node_state_loader,
            routing_event_loader=routing_event_loader,
            call_loader=call_loader,
            operation_loader=operation_loader,
            batch_loader=batch_loader,
            batch_member_loader=batch_member_loader,
            artifact_loader=artifact_loader,
            sink_effect_loader=sink_effect_loader,
            sink_effect_member_loader=sink_effect_member_loader,
            sink_effect_stream_loader=sink_effect_stream_loader,
            payload_store=payload_store,
        )
        barrier_restore = BarrierRestoreReadModel(
            read_ops_as_common,
            token_outcome_loader=token_outcome_loader,
        )
        query = QueryRepository(
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
        run_status_projection = AuditRunStatusProjection(read_ops)

        return LandscapeReadRepositories(
            run_lifecycle=RunLifecycleReadRepository(run_lifecycle),
            data_flow=DataFlowReadRepository(data_flow),
            execution=ExecutionReadRepository(execution),
            query=query,
            run_status_projection=run_status_projection,
            barrier_restore=barrier_restore,
            payload_store=payload_store,
        )

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
        sink_effect_loader = SinkEffectLoader()
        sink_effect_member_loader = SinkEffectMemberLoader()
        sink_effect_stream_loader = SinkEffectStreamLoader()

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
            sink_effect_loader=sink_effect_loader,
            sink_effect_member_loader=sink_effect_member_loader,
            sink_effect_stream_loader=sink_effect_stream_loader,
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
        self._run_status_projection = AuditRunStatusProjection(read_ops)
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
        self._audit_export_snapshots = AuditExportSnapshotRepository()

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
    def run_status_projection(self) -> AuditRunStatusProjection:
        return self._run_status_projection

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
    def audit_export_snapshots(self) -> AuditExportSnapshotRepository:
        """Return the transaction-bound immutable snapshot registry capability."""
        if self._db.is_read_only:
            raise RuntimeError("audit export snapshot registry is not available on a read-only LandscapeDB handle")
        return self._audit_export_snapshots

    @property
    def payload_store(self) -> PayloadStore | None:
        return self._payload_store

    def plugin_audit_writer(self) -> PluginAuditWriter:
        """Create a PluginAuditWriter adapter composing the three repositories."""
        return _PluginAuditWriterAdapterImpl(self._execution, self._data_flow, self._run_lifecycle)

    def read_repositories(self) -> LandscapeReadRepositories:
        """Return a read-only capability view over this factory."""
        return LandscapeReadRepositories(
            run_lifecycle=RunLifecycleReadRepository(self._run_lifecycle),
            data_flow=DataFlowReadRepository(self._data_flow),
            execution=ExecutionReadRepository(self._execution),
            query=self._query,
            run_status_projection=self._run_status_projection,
            barrier_restore=self._barrier_restore,
            payload_store=self._payload_store,
        )

    def write_repositories(self) -> LandscapeWriteRepositories:
        """Return a writable capability view over this factory."""
        if self._db.is_read_only:
            raise RuntimeError("writable repositories are not available on a read-only LandscapeDB handle")
        return LandscapeWriteRepositories(
            read=self.read_repositories(),
            run_lifecycle=self._run_lifecycle,
            auth_audit=self._auth_audit,
            execution=self._execution,
            data_flow=self._data_flow,
            scheduler=self.scheduler,
            run_coordination=self.run_coordination,
            audit_export_snapshots=self.audit_export_snapshots,
        )
