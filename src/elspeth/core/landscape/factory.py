"""RecorderFactory: construction point for Landscape repositories.

Single place that wires up loaders, database operations, and repository instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from elspeth.contracts import (
    Call,
    CallStatus,
    CallType,
    NodeState,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import ContractViolation, TransformErrorReason
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape._database_ops import DatabaseOps, ReadOnlyDatabaseOps
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
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.schema_contract import PipelineRow


class _PluginAuditWriterAdapter:
    """Composes three repositories into the PluginAuditWriter interface.

    Each method delegates to the correct repository. This is a thin adapter
    with no logic of its own.

    Method-count budget: Do not exceed 20 methods.
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        data_flow: DataFlowRepository,
        run_lifecycle: RunLifecycleRepository,
    ) -> None:
        self._execution = execution
        self._data_flow = data_flow
        self._run_lifecycle = run_lifecycle

    # ── ExecutionRepository delegation ───────────────────────────────────

    def allocate_call_index(self, state_id: str) -> int:
        return self._execution.allocate_call_index(state_id)

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call:
        return self._execution.record_call(
            state_id,
            call_index,
            call_type,
            status,
            request_data,
            response_data,
            error,
            latency_ms,
            request_ref=request_ref,
            response_ref=response_ref,
        )

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call:
        return self._execution.record_operation_call(
            operation_id,
            call_type,
            status,
            request_data,
            response_data,
            error,
            latency_ms,
            request_ref=request_ref,
            response_ref=response_ref,
        )

    def get_node_state(self, state_id: str) -> NodeState | None:
        return self._execution.get_node_state(state_id)

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
        return self._execution.record_routing_event(
            state_id,
            edge_id,
            mode,
            reason,
            event_id=event_id,
            routing_group_id=routing_group_id,
            ordinal=ordinal,
            reason_ref=reason_ref,
        )

    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
    ) -> list[RoutingEvent]:
        return self._execution.record_routing_events(state_id, routes, reason)

    # ── DataFlowRepository delegation ────────────────────────────────────

    def record_validation_error(
        self,
        run_id: str,
        node_id: str | None,
        row_data: Any,
        error: str,
        schema_mode: str,
        destination: str,
        *,
        contract_violation: ContractViolation | None = None,
    ) -> str:
        return self._data_flow.record_validation_error(
            run_id,
            node_id,
            row_data,
            error,
            schema_mode,
            destination,
            contract_violation=contract_violation,
        )

    def record_transform_error(
        self,
        ref: TokenRef,
        transform_id: str,
        row_data: Mapping[str, object] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> str:
        return self._data_flow.record_transform_error(ref, transform_id, row_data, error_details, destination)

    def update_node_output_contract(
        self,
        run_id: str,
        node_id: str,
        contract: SchemaContract,
    ) -> None:
        self._data_flow.update_node_output_contract(run_id, node_id, contract)

    def get_node_contracts(
        self,
        run_id: str,
        node_id: str,
        *,
        allow_missing: bool = False,
    ) -> tuple[SchemaContract | None, SchemaContract | None]:
        return self._data_flow.get_node_contracts(run_id, node_id, allow_missing=allow_missing)

    # ── RunLifecycleRepository delegation ────────────────────────────────

    def get_source_field_resolution(self, run_id: str) -> dict[str, str] | None:
        return self._run_lifecycle.get_source_field_resolution(run_id)

    def record_readiness_check(
        self,
        run_id: str,
        *,
        name: str,
        collection: str,
        reachable: bool,
        count: int | None,
        message: str,
    ) -> None:
        self._run_lifecycle.record_readiness_check(
            run_id,
            name=name,
            collection=collection,
            reachable=reachable,
            count=count,
            message=message,
        )


class RecorderFactory:
    """Construction point for Landscape repositories.

    Creates all 4 repositories from a LandscapeDB, sharing loader instances
    to ensure consistent object construction across repositories.
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
        artifact_loader = ArtifactLoader()
        batch_member_loader = BatchMemberLoader()

        # Composed repository for run lifecycle
        self._run_lifecycle = RunLifecycleRepository(db, ops, run_loader)

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
            payload_store=payload_store,
        )

    @property
    def run_lifecycle(self) -> RunLifecycleRepository:
        return self._run_lifecycle

    @property
    def execution(self) -> ExecutionRepository:
        return self._execution

    @property
    def data_flow(self) -> DataFlowRepository:
        return self._data_flow

    @property
    def query(self) -> QueryRepository:
        return self._query

    @property
    def payload_store(self) -> PayloadStore | None:
        return self._payload_store

    def plugin_audit_writer(self) -> PluginAuditWriter:
        """Create a PluginAuditWriter adapter composing the three repositories."""
        return _PluginAuditWriterAdapter(self._execution, self._data_flow, self._run_lifecycle)
