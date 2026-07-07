"""Tests for PluginAuditWriter protocol and adapter delegation routing."""

from __future__ import annotations

from unittest.mock import MagicMock, sentinel

import pytest

from elspeth.contracts import CallStatus, CallType, RoutingMode, RoutingSpec, SchemaContract, create_contract_from_config
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.audit_protocols import CallRecorder, PluginAuditWriter
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.errors import MissingFieldViolation, SinkDiversionReason, TransformErrorReason
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.plugin_audit_writer import PluginAuditWriterAdapter
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository


def _public_method_names(cls: type[object]) -> set[str]:
    return {name for name, value in vars(cls).items() if not name.startswith("_") and callable(value)}


def _sample_payload(name: str) -> RawCallPayload:
    return RawCallPayload({"payload": name})


def _sample_contract() -> SchemaContract:
    schema_config = SchemaConfig.from_dict({"mode": "fixed", "fields": ["id: int"]})
    return create_contract_from_config(schema_config)


@pytest.fixture()
def repos() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create mock repositories with spec matching the real classes."""
    execution = MagicMock(spec=ExecutionRepository)
    data_flow = MagicMock(spec=DataFlowRepository)
    run_lifecycle = MagicMock(spec=RunLifecycleRepository)
    return execution, data_flow, run_lifecycle


@pytest.fixture()
def writer(
    repos: tuple[MagicMock, MagicMock, MagicMock],
) -> PluginAuditWriterAdapter:
    execution, data_flow, run_lifecycle = repos
    return PluginAuditWriterAdapter(execution, data_flow, run_lifecycle)


class TestAdapterConstruction:
    """Verify the adapter constructs without error and has the right type."""

    def test_adapter_constructs_successfully(self, writer: PluginAuditWriterAdapter) -> None:
        assert isinstance(writer, PluginAuditWriterAdapter)

    def test_adapter_satisfies_plugin_audit_writer_protocol(self, writer: PluginAuditWriterAdapter) -> None:
        assert isinstance(writer, PluginAuditWriter)

    def test_adapter_surface_matches_protocol_methods(self) -> None:
        protocol_methods = _public_method_names(PluginAuditWriter)
        adapter_methods = _public_method_names(PluginAuditWriterAdapter)

        assert len(protocol_methods) <= 20
        assert adapter_methods == protocol_methods

    def test_call_recorder_remains_narrow_subset(self) -> None:
        call_recorder_methods = _public_method_names(CallRecorder)

        assert call_recorder_methods == {
            "allocate_call_index",
            "allocate_operation_call_index",
            "record_call",
            "record_operation_call",
        }
        assert call_recorder_methods < _public_method_names(PluginAuditWriter)


class TestCallRecordingRoutesToExecution:
    """Verify call-related methods delegate to ExecutionRepository."""

    def test_allocate_call_index_routes_to_execution(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        execution, _data_flow, _run_lifecycle = repos
        execution.allocate_call_index.return_value = 42

        result = writer.allocate_call_index("state-1")

        assert result == 42
        execution.allocate_call_index.assert_called_once_with("state-1")
        # DataFlowRepository does not have allocate_call_index — if it
        # routed there, the adapter would have crashed.

    def test_record_call_routes_to_execution(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        execution, _data_flow, _run_lifecycle = repos
        execution.record_call.return_value = sentinel.call

        request = _sample_payload("request")
        response = _sample_payload("response")
        result = writer.record_call(
            "state-1",
            7,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request,
            response,
            None,
            12.5,
            request_ref="blob-request",
            response_ref="blob-response",
        )

        assert result is sentinel.call
        execution.record_call.assert_called_once_with(
            "state-1",
            7,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request,
            response,
            None,
            12.5,
            request_ref="blob-request",
            response_ref="blob-response",
            # Phase 5b Task 9: cross-DB hash anchor; None unless the caller
            # is an LLM transform downstream of a resolved interpretation
            # event. The PluginAuditWriter adapter unconditionally forwards
            # the kwarg so the LLM transform plugin path can pass-through
            # without per-call-site adapter changes.
            resolved_prompt_template_hash=None,
        )


class TestErrorRecordingRoutesToDataFlow:
    """Verify error recording methods delegate to DataFlowRepository."""

    def test_record_validation_error_routes_to_data_flow(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        _execution, data_flow, _run_lifecycle = repos
        data_flow.record_validation_error.return_value = "err-1"
        violation = MissingFieldViolation(normalized_name="field", original_name="Field")

        result = writer.record_validation_error(
            "run-1",
            "node-1",
            {"field": "value"},
            "bad data",
            "strict",
            "sink-1",
            contract_violation=violation,
        )

        assert result == "err-1"
        data_flow.record_validation_error.assert_called_once_with(
            "run-1",
            "node-1",
            {"field": "value"},
            "bad data",
            "strict",
            "sink-1",
            contract_violation=violation,
        )


class TestNodeStateRoutesToExecution:
    """Verify get_node_state delegates to ExecutionRepository."""

    def test_get_node_state_routes_to_execution(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        execution, _data_flow, _run_lifecycle = repos
        execution.get_node_state.return_value = sentinel.state

        result = writer.get_node_state("state-1")

        assert result is sentinel.state
        execution.get_node_state.assert_called_once_with("state-1")


class TestOperationCallRoutesToExecution:
    """Verify record_operation_call delegates to ExecutionRepository."""

    def test_record_operation_call_routes_to_execution(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        execution, _data_flow, _run_lifecycle = repos
        execution.record_operation_call.return_value = sentinel.call

        request = _sample_payload("operation-request")
        error = _sample_payload("operation-error")
        result = writer.record_operation_call(
            "op-1",
            CallType.HTTP,
            CallStatus.ERROR,
            request,
            None,
            error,
            3.25,
            request_ref="op-request-ref",
            response_ref=None,
        )

        assert result is sentinel.call
        execution.record_operation_call.assert_called_once_with(
            "op-1",
            CallType.HTTP,
            CallStatus.ERROR,
            request,
            None,
            error,
            3.25,
            request_ref="op-request-ref",
            response_ref=None,
            # Phase 5b Task 9: cross-DB hash anchor; None unless the caller
            # is an LLM operation downstream of a resolved interpretation.
            resolved_prompt_template_hash=None,
        )


class TestRoutingEventRoutesToExecution:
    """Verify routing event methods delegate to ExecutionRepository."""

    def test_record_routing_event_routes_to_execution(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        execution, _data_flow, _run_lifecycle = repos
        execution.record_routing_event.return_value = sentinel.event
        reason: SinkDiversionReason = {"diversion_reason": "rejected by sink"}

        result = writer.record_routing_event(
            "state-1",
            "edge-1",
            RoutingMode.MOVE,
            reason,
            event_id="event-1",
            routing_group_id="group-1",
            ordinal=2,
            reason_ref="reason-blob",
        )

        assert result is sentinel.event
        execution.record_routing_event.assert_called_once_with(
            "state-1",
            "edge-1",
            RoutingMode.MOVE,
            reason,
            event_id="event-1",
            routing_group_id="group-1",
            ordinal=2,
            reason_ref="reason-blob",
        )

    def test_record_routing_events_routes_to_execution(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        execution, _data_flow, _run_lifecycle = repos
        execution.record_routing_events.return_value = [sentinel.event]
        routes = [RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)]
        reason: SinkDiversionReason = {"diversion_reason": "bulk route"}

        result = writer.record_routing_events("state-1", routes, reason)

        assert result == [sentinel.event]
        execution.record_routing_events.assert_called_once_with("state-1", routes, reason)


class TestTransformErrorRoutesToDataFlow:
    """Verify record_transform_error delegates to DataFlowRepository."""

    def test_record_transform_error_routes_to_data_flow(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        _execution, data_flow, _run_lifecycle = repos
        data_flow.record_transform_error.return_value = "err-1"

        ref = TokenRef(token_id="tok-1", run_id="run-1")
        reason = TransformErrorReason(reason="api_error", error_type="ValueError", message="bad")
        result = writer.record_transform_error(ref, "xform-1", {"field": "val"}, reason, "sink-1")

        assert result == "err-1"
        data_flow.record_transform_error.assert_called_once_with(ref, "xform-1", {"field": "val"}, reason, "sink-1")


class TestContractMethodsRouteToDataFlow:
    """Verify contract-related methods delegate to DataFlowRepository."""

    def test_update_node_output_contract_routes_to_data_flow(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        _execution, data_flow, _run_lifecycle = repos

        contract = _sample_contract()
        writer.update_node_output_contract("run-1", "node-1", contract)

        data_flow.update_node_output_contract.assert_called_once_with("run-1", "node-1", contract)

    def test_get_node_contracts_routes_to_data_flow(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        _execution, data_flow, _run_lifecycle = repos
        data_flow.get_node_contracts.return_value = (sentinel.input, sentinel.output)

        result = writer.get_node_contracts("run-1", "node-1", allow_missing=True)

        assert result == (sentinel.input, sentinel.output)
        data_flow.get_node_contracts.assert_called_once_with("run-1", "node-1", allow_missing=True)


class TestReadinessCheckRoutesToRunLifecycle:
    """Verify record_readiness_check delegates to RunLifecycleRepository."""

    def test_record_readiness_check_routes_to_run_lifecycle(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        _execution, _data_flow, run_lifecycle = repos

        writer.record_readiness_check(
            "run-1",
            name="chroma",
            collection="docs",
            reachable=True,
            count=42,
            message="OK",
        )

        run_lifecycle.record_readiness_check.assert_called_once_with(
            "run-1",
            name="chroma",
            collection="docs",
            reachable=True,
            count=42,
            message="OK",
        )


class TestRunLifecycleRoutesToRunLifecycle:
    """Verify run lifecycle methods delegate to RunLifecycleRepository."""

    def test_get_source_field_resolution_routes_to_run_lifecycle(
        self,
        writer: PluginAuditWriterAdapter,
        repos: tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        _execution, _data_flow, run_lifecycle = repos
        run_lifecycle.get_source_field_resolution.return_value = {"a": "b"}

        result = writer.get_source_field_resolution("run-1")

        assert result == {"a": "b"}
        run_lifecycle.get_source_field_resolution.assert_called_once_with("run-1")
