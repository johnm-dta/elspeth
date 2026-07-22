"""Plugin-facing audit writer adapter for Landscape repositories."""

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
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import ContractViolation, TransformErrorReason
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import PipelineRow


class PluginAuditWriterAdapter:
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

    def allocate_operation_call_index(self, operation_id: str) -> int:
        return self._execution.allocate_operation_call_index(operation_id)

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
        resolved_prompt_template_hash: str | None = None,
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
            resolved_prompt_template_hash=resolved_prompt_template_hash,
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
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        return self._execution.record_operation_call(
            operation_id,
            call_type,
            status,
            request_data,
            response_data,
            error,
            latency_ms,
            call_index=call_index,
            request_ref=request_ref,
            response_ref=response_ref,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
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
        """Record a complete one-route decision; use the plural API for forks."""
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
        """Atomically record every route in one complete fork decision."""
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
