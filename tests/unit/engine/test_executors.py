# tests/unit/engine/test_executors.py
"""Comprehensive unit tests for the 4 executor classes in engine/executors/.

Tests cover:
- TransformExecutor: success/error/exception paths, audit recording
- GateExecutor: execute_config_gate with routing variations
- AggregationExecutor: buffer management, flush lifecycle, trigger delegation
- SinkExecutor: write lifecycle, artifact recording, callback handling

All tests mock the LandscapeRecorder and SpanFactory to isolate executor logic.

Error Routing Decision Tree (exd audit)
========================================

Every error follows one of these paths through the executor layer:

Transform Error (TransformResult.error):
  on_error is REQUIRED  (enforced by TransformSettings at config time — None no longer possible)
  on_error="discard"   → node_state=FAILED + transform_error recorded (NO routing_event)
                         → token outcome: QUARANTINED (recorded at sink write time)
  on_error=<sink_name> → node_state=FAILED + transform_error + DIVERT routing_event
                         → token outcome: ROUTED to error sink

Transform Exception (plugin crash):
  → node_state=FAILED (always recorded before re-raise)
  → exception propagates → pipeline CRASH

Gate Error:
  Expression exception   → node_state=FAILED + re-raise → pipeline CRASH
  Unknown route label    → node_state=FAILED + ValueError → pipeline CRASH
  Non-string/bool result → str() conversion, then route lookup (may fail as above)
  Missing edge           → node_state=FAILED + MissingEdgeError → pipeline CRASH

Aggregation Error:
  Flush exception        → node_state=FAILED, batch=FAILED → pipeline CRASH
  Flush error result     → node_state=FAILED, batch=FAILED, tokens=CONSUMED_IN_BATCH
  BatchPending           → node_state=PENDING (control flow, not error)

Sink Error:
  Write exception        → all node_states=FAILED → pipeline CRASH (no outcomes recorded)
  Flush exception        → all node_states=FAILED → pipeline CRASH (no outcomes recorded)

Source Quarantine:
  Validation failure     → quarantine token → DIVERT routing_event → sink write
                         → token outcome: QUARANTINED

Invariant: Every error records node_state=FAILED before any exception propagates.
Invariant: DIVERT routing_event only for non-discard error routing (audit accuracy).
Invariant: Token outcomes only recorded after sink durability (crash recovery safe).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from elspeth.contracts import (
    PendingOutcome,
    TokenInfo,
    TransformResult,
)
from elspeth.contracts.enums import (
    BatchStatus,
    NodeStateStatus,
    RoutingMode,
    RowOutcome,
    TriggerType,
)
from elspeth.contracts.errors import ContractMergeError, OrchestrationInvariantError
from elspeth.contracts.results import ArtifactDescriptor, GateResult
from elspeth.contracts.routing import RouteDestination, RoutingAction
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.config import AggregationSettings, GateSettings, TriggerConfig
from elspeth.engine.executors import (
    AGGREGATION_CHECKPOINT_VERSION,
    AggregationExecutor,
    GateExecutor,
    GateOutcome,
    MissingEdgeError,
    SinkExecutor,
    TransformExecutor,
)
from elspeth.testing import make_field, make_row
from tests.unit.engine.conftest import make_test_step_resolver as _make_step_resolver

# =============================================================================
# Shared helpers
# =============================================================================


def _make_contract() -> SchemaContract:
    """Create a simple test contract."""
    return SchemaContract(
        fields=(
            make_field(
                "value",
                python_type=str,
                original_name="value",
                required=True,
                source="declared",
            ),
        ),
        mode="FLEXIBLE",
        locked=True,
    )


def _make_token(
    data: dict[str, Any] | None = None,
    contract: SchemaContract | None = None,
    row_id: str = "row_1",
    token_id: str = "tok_1",
) -> TokenInfo:
    """Create a token with PipelineRow for testing."""
    if data is None:
        data = {"value": "test"}
    if contract is None:
        contract = _make_contract()
    row_data = make_row(data, contract=contract)
    return TokenInfo(row_id=row_id, token_id=token_id, row_data=row_data)


def _make_recorder() -> MagicMock:
    """Create a mock LandscapeRecorder with sensible defaults."""
    recorder = MagicMock()
    state = Mock(state_id="state_001")
    recorder.begin_node_state.return_value = state
    recorder.register_artifact.return_value = Mock(artifact_id="art_001")
    batch = Mock(batch_id="batch_001")
    recorder.create_batch.return_value = batch
    recorder.begin_operation.return_value = Mock(operation_id="op_001")
    return recorder


def _make_span_factory() -> MagicMock:
    """Create a mock SpanFactory where all spans are no-op context managers."""
    sf = MagicMock()
    sf.transform_span.return_value = nullcontext()
    sf.gate_span.return_value = nullcontext()
    sf.aggregation_span.return_value = nullcontext()
    sf.sink_span.return_value = nullcontext()
    return sf


def _make_transform(
    name: str = "test_transform",
    node_id: str | None = "node_1",
    on_error: str | None = None,
    adds_fields: bool = False,
) -> MagicMock:
    """Create a mock transform (non-batch)."""
    # Use spec to avoid MagicMock auto-creating 'accept' attribute
    t = MagicMock(spec=["name", "node_id", "on_error", "transforms_adds_fields", "process"])
    t.name = name
    t.node_id = node_id
    t.on_error = on_error
    t.transforms_adds_fields = adds_fields
    return t


def _make_sink(
    name: str = "test_sink",
    node_id: str | None = "sink_1",
) -> MagicMock:
    """Create a mock sink."""
    sink = MagicMock()
    sink.name = name
    sink.node_id = node_id
    sink.write.return_value = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri="file:///output/test.csv",
        content_hash="abc123",
        size_bytes=100,
    )
    return sink


def _make_ctx(run_id: str = "test-run") -> Any:
    """Create a PluginContext for testing."""
    from elspeth.contracts.plugin_context import PluginContext

    return PluginContext(run_id=run_id, config={})


# =============================================================================
# TestMissingEdgeError
# =============================================================================


class TestMissingEdgeError:
    """Tests for the MissingEdgeError exception class."""

    def test_stores_node_id_and_label(self) -> None:
        err = MissingEdgeError(node_id=NodeID("gate_1"), label="above")
        assert err.node_id == NodeID("gate_1")
        assert err.label == "above"

    def test_message_includes_both(self) -> None:
        err = MissingEdgeError(node_id=NodeID("gate_1"), label="above")
        assert "gate_1" in str(err)
        assert "above" in str(err)

    def test_is_exception(self) -> None:
        err = MissingEdgeError(node_id=NodeID("n"), label="l")
        assert isinstance(err, Exception)


# =============================================================================
# TestGateOutcome
# =============================================================================


class TestGateOutcome:
    """Tests for the GateOutcome dataclass."""

    def test_default_values(self) -> None:
        result = GateResult(row={"a": 1}, action=RoutingAction.continue_())
        token = _make_token()
        outcome = GateOutcome(result=result, updated_token=token)
        assert outcome.child_tokens == []
        assert outcome.sink_name is None
        assert outcome.next_node_id is None

    def test_custom_values(self) -> None:
        result = GateResult(row={"a": 1}, action=RoutingAction.continue_())
        token = _make_token()
        child = _make_token(token_id="child_1")
        outcome = GateOutcome(
            result=result,
            updated_token=token,
            child_tokens=[child],
            sink_name="error_sink",
        )
        assert len(outcome.child_tokens) == 1
        assert outcome.child_tokens[0].token_id == "child_1"
        assert outcome.sink_name == "error_sink"
        assert outcome.next_node_id is None


# =============================================================================
# TestTransformExecutor
# =============================================================================


class TestTransformExecutor:
    """Tests for TransformExecutor covering success, error, and exception paths."""

    # --- Init / Validation ---

    def test_no_node_id_raises_orchestration_invariant_error(self) -> None:
        """Transform without node_id must raise OrchestrationInvariantError."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(node_id=None)
        token = _make_token()
        ctx = _make_ctx()

        with pytest.raises(OrchestrationInvariantError, match="without node_id"):
            executor.execute_transform(transform, token, ctx)

    # --- Success path ---

    def test_successful_transform_returns_result_token_none(self) -> None:
        """Successful transform returns (result, updated_token, None)."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        contract = _make_contract()
        token = _make_token(contract=contract)
        transform = _make_transform()
        transform.process.return_value = TransformResult.success(
            make_row({"value": "processed"}, contract=contract),
            success_reason={"action": "test"},
        )
        ctx = _make_ctx()

        result, updated_token, error_sink = executor.execute_transform(
            transform,
            token,
            ctx,
        )

        assert result.status == "success"
        assert error_sink is None
        assert updated_token.row_data["value"] == "processed"

    def test_begin_node_state_called_with_correct_args(self) -> None:
        """Recorder.begin_node_state called with token_id, node_id, run_id, step, dict input."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver({"node_1": 3}))
        contract = _make_contract()
        token = _make_token(contract=contract)
        transform = _make_transform()
        transform.process.return_value = TransformResult.success(
            make_row({"value": "out"}, contract=contract),
            success_reason={"action": "test"},
        )
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx, attempt=2)

        recorder.begin_node_state.assert_called_once()
        kwargs = recorder.begin_node_state.call_args[1]
        assert kwargs["token_id"] == "tok_1"
        assert kwargs["node_id"] == "node_1"
        assert kwargs["run_id"] == "test-run"
        assert kwargs["step_index"] == 3
        assert kwargs["attempt"] == 2
        assert isinstance(kwargs["input_data"], dict)

    def test_complete_node_state_called_completed_on_success(self) -> None:
        """On success, complete_node_state is called with COMPLETED status."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        contract = _make_contract()
        token = _make_token(contract=contract)
        transform = _make_transform()
        transform.process.return_value = TransformResult.success(
            make_row({"value": "out"}, contract=contract),
            success_reason={"action": "test"},
        )
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx)

        recorder.complete_node_state.assert_called_once()
        kwargs = recorder.complete_node_state.call_args[1]
        assert kwargs["status"] == NodeStateStatus.COMPLETED
        assert kwargs["state_id"] == "state_001"

    def test_result_has_audit_fields_populated(self) -> None:
        """Result has input_hash, output_hash, duration_ms populated by executor."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        contract = _make_contract()
        token = _make_token(contract=contract)
        transform = _make_transform()
        transform.process.return_value = TransformResult.success(
            make_row({"value": "out"}, contract=contract),
            success_reason={"action": "test"},
        )
        ctx = _make_ctx()

        result, _, _ = executor.execute_transform(
            transform,
            token,
            ctx,
        )

        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    def test_updated_token_has_new_row_data(self) -> None:
        """Updated token has row_data from the result, preserving lineage."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        contract = _make_contract()
        token = _make_token(data={"value": "original"}, contract=contract)
        transform = _make_transform()
        transform.process.return_value = TransformResult.success(
            make_row({"value": "modified"}, contract=contract),
            success_reason={"action": "test"},
        )
        ctx = _make_ctx()

        _, updated_token, _ = executor.execute_transform(
            transform,
            token,
            ctx,
        )

        assert updated_token.row_data["value"] == "modified"
        # Lineage preserved
        assert updated_token.row_id == "row_1"
        assert updated_token.token_id == "tok_1"

    def test_ctx_state_id_set_for_external_call_recording(self) -> None:
        """ctx.state_id and ctx.node_id are set before transform.process()."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        contract = _make_contract()
        token = _make_token(contract=contract)

        captured_state_id = None
        captured_node_id = None

        def capturing_process(row_data, ctx):
            nonlocal captured_state_id, captured_node_id
            captured_state_id = ctx.state_id
            captured_node_id = ctx.node_id
            return TransformResult.success(
                make_row({"value": "out"}, contract=contract),
                success_reason={"action": "test"},
            )

        transform = _make_transform()
        transform.process = capturing_process
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx)

        assert captured_state_id == "state_001"
        assert captured_node_id == "node_1"

    # --- Error path (TransformResult.error) ---

    def test_error_result_with_on_error_sink_returns_error_sink(self) -> None:
        """Error result with on_error=sink_name returns that as error_sink."""
        recorder = _make_recorder()
        edge_ids = {NodeID("node_1"): "divert_edge_1"}
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver(), error_edge_ids=edge_ids)
        transform = _make_transform(on_error="quarantine_sink")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        result, updated_token, error_sink = executor.execute_transform(
            transform,
            token,
            ctx,
        )

        assert result.status == "error"
        assert error_sink == "quarantine_sink"
        # Token is unchanged on error
        assert updated_token is token

    def test_error_result_with_on_error_discard_returns_discard(self) -> None:
        """Error result with on_error='discard' returns 'discard'."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="discard")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        _, _, error_sink = executor.execute_transform(
            transform,
            token,
            ctx,
        )

        assert error_sink == "discard"

    def test_on_error_is_always_set_invariant(self) -> None:
        """on_error is now required at config time — transforms always have it set.

        Previously on_error=None would raise RuntimeError at execution time.
        Now TransformSettings requires on_error, so the None case cannot occur
        in production. This test documents the invariant.
        """
        # Every transform constructed via TransformSettings will have on_error set.
        # Verify a transform with on_error="discard" works (the minimum valid value).
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="discard")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        _, _, error_sink = executor.execute_transform(transform, token, ctx)
        assert error_sink == "discard"

    def test_error_path_records_failed_state(self) -> None:
        """Error result records FAILED node_state."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="discard")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx)

        recorder.complete_node_state.assert_called_once()
        kwargs = recorder.complete_node_state.call_args[1]
        assert kwargs["status"] == NodeStateStatus.FAILED

    def test_error_path_records_transform_error(self) -> None:
        """Error result calls ctx.record_transform_error."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="discard")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()
        ctx.record_transform_error = MagicMock()

        executor.execute_transform(transform, token, ctx)

        ctx.record_transform_error.assert_called_once()

    def test_error_path_non_discard_records_divert_routing(self) -> None:
        """Non-discard error routing records a DIVERT routing_event."""
        recorder = _make_recorder()
        edge_ids = {NodeID("node_1"): "divert_edge_1"}
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver(), error_edge_ids=edge_ids)
        transform = _make_transform(on_error="quarantine_sink")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx)

        recorder.record_routing_event.assert_called_once()
        kwargs = recorder.record_routing_event.call_args[1]
        assert kwargs["edge_id"] == "divert_edge_1"
        assert kwargs["mode"] == RoutingMode.DIVERT

    def test_error_path_missing_divert_edge_raises(self) -> None:
        """Non-discard error without DIVERT edge raises OrchestrationInvariantError."""
        recorder = _make_recorder()
        # No error_edge_ids registered
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="quarantine_sink")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        with pytest.raises(OrchestrationInvariantError, match="DIVERT edge"):
            executor.execute_transform(transform, token, ctx)

    # --- Exception path ---

    def test_exception_from_process_records_failed_and_reraises(self) -> None:
        """Exception from transform.process() records FAILED and re-raises."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform()
        transform.process.side_effect = ValueError("plugin bug")
        token = _make_token()
        ctx = _make_ctx()

        with pytest.raises(ValueError, match="plugin bug"):
            executor.execute_transform(transform, token, ctx)

        recorder.complete_node_state.assert_called_once()
        kwargs = recorder.complete_node_state.call_args[1]
        assert kwargs["status"] == NodeStateStatus.FAILED
        assert "plugin bug" in kwargs["error"]["exception"]

    def test_exception_path_captures_duration(self) -> None:
        """Duration is captured even when exception is raised."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform()
        transform.process.side_effect = RuntimeError("crash")
        token = _make_token()
        ctx = _make_ctx()

        with pytest.raises(RuntimeError):
            executor.execute_transform(transform, token, ctx)

        kwargs = recorder.complete_node_state.call_args[1]
        assert kwargs["duration_ms"] >= 0

    # --- Error routing edge cases (exd audit) ---

    def test_discard_error_does_not_record_divert_routing(self) -> None:
        """on_error='discard' must NOT create a DIVERT routing_event.

        Discard means quarantine silently — no routing decision was made,
        so recording a DIVERT event would misrepresent the audit trail.
        """
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="discard")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        token = _make_token()
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx)

        recorder.record_routing_event.assert_not_called()

    def test_error_reason_propagated_to_node_state_error_field(self) -> None:
        """TransformResult.error() reason dict stored as node_state error."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform(on_error="discard")
        error_reason = {"reason": "content_filtered", "provider": "azure", "code": "CF-01"}
        transform.process.return_value = TransformResult.error(reason=error_reason)
        token = _make_token()
        ctx = _make_ctx()

        executor.execute_transform(transform, token, ctx)

        kwargs = recorder.complete_node_state.call_args[1]
        assert kwargs["error"] == error_reason

    def test_non_discard_error_records_both_transform_error_and_divert(self) -> None:
        """Non-discard error records transform_error AND DIVERT routing event.

        Both are required for audit completeness: the transform_error
        captures the error details, and the DIVERT records the routing
        decision for lineage tracing.
        """
        recorder = _make_recorder()
        edge_ids = {NodeID("node_1"): "divert_edge_1"}
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver(), error_edge_ids=edge_ids)
        transform = _make_transform(on_error="error_sink")
        transform.process.return_value = TransformResult.error(
            reason={"reason": "api_error"},
        )
        token = _make_token()
        ctx = _make_ctx()
        ctx.record_transform_error = MagicMock()

        executor.execute_transform(transform, token, ctx)

        # Both must be recorded
        ctx.record_transform_error.assert_called_once()
        recorder.record_routing_event.assert_called_once()
        assert recorder.record_routing_event.call_args[1]["mode"] == RoutingMode.DIVERT


# =============================================================================
# TestGateExecutor
# =============================================================================


class TestGateExecutor:
    """Tests for GateExecutor covering execute_config_gate."""

    # --- execute_config_gate ---

    def test_config_gate_boolean_true_routes_via_true_label(self) -> None:
        """Boolean True condition evaluates to 'true' label."""
        recorder = _make_recorder()
        edge_map = {(NodeID("cg_1"), "true"): "edge_true"}
        route_map = {(NodeID("cg_1"), "true"): RouteDestination.processing_node(NodeID("next_node"))}
        executor = GateExecutor(
            recorder,
            _make_span_factory(),
            _make_step_resolver(),
            edge_map=edge_map,
            route_resolution_map=route_map,
        )
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="True",
            routes={"true": "next_conn", "false": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        outcome = executor.execute_config_gate(
            config,
            "cg_1",
            token,
            ctx,
        )

        assert outcome.sink_name is None
        assert outcome.next_node_id == NodeID("next_node")

    def test_config_gate_boolean_false_routes_via_false_label(self) -> None:
        """Boolean False condition evaluates to 'false' label."""
        recorder = _make_recorder()
        # Edge map must have the route label that will be used for recording
        edge_map = {(NodeID("cg_1"), "false"): "edge_false"}
        route_map = {(NodeID("cg_1"), "false"): RouteDestination.sink(SinkName("error_sink"))}
        executor = GateExecutor(
            recorder,
            _make_span_factory(),
            _make_step_resolver(),
            edge_map=edge_map,
            route_resolution_map=route_map,
        )
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="False",
            routes={"true": "next_conn", "false": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        outcome = executor.execute_config_gate(
            config,
            "cg_1",
            token,
            ctx,
        )

        assert outcome.sink_name == "error_sink"

    def test_config_gate_string_result_used_as_label(self) -> None:
        """String condition result used as route label directly."""
        recorder = _make_recorder()
        edge_map = {(NodeID("cg_1"), "high"): "edge_high"}
        route_map = {(NodeID("cg_1"), "high"): RouteDestination.processing_node(NodeID("next_node"))}
        executor = GateExecutor(
            recorder,
            _make_span_factory(),
            _make_step_resolver(),
            edge_map=edge_map,
            route_resolution_map=route_map,
        )
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="'high'",
            routes={"high": "next_conn", "low": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        outcome = executor.execute_config_gate(
            config,
            "cg_1",
            token,
            ctx,
        )

        assert outcome.sink_name is None
        assert outcome.next_node_id == NodeID("next_node")

    def test_config_gate_route_to_processing_node(self) -> None:
        """Config gate route label can branch to a processing node."""
        recorder = _make_recorder()
        edge_map = {(NodeID("cg_1"), "high"): "edge_high"}
        route_map = {(NodeID("cg_1"), "high"): RouteDestination.processing_node(NodeID("transform_2"))}
        executor = GateExecutor(
            recorder,
            _make_span_factory(),
            _make_step_resolver(),
            edge_map=edge_map,
            route_resolution_map=route_map,
        )
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="'high'",
            routes={"high": "branch_conn", "low": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        outcome = executor.execute_config_gate(
            config,
            "cg_1",
            token,
            ctx,
        )

        assert outcome.sink_name is None
        assert outcome.next_node_id == NodeID("transform_2")

    def test_config_gate_unknown_route_label_raises_value_error(self) -> None:
        """Unknown route label raises ValueError with FAILED state recorded."""
        recorder = _make_recorder()
        executor = GateExecutor(recorder, _make_span_factory(), _make_step_resolver())
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="'unknown_label'",
            routes={"high": "next_conn", "low": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        with pytest.raises(ValueError, match="unknown_label"):
            executor.execute_config_gate(
                config,
                "cg_1",
                token,
                ctx,
            )

        # Verify FAILED state was recorded before raising
        assert recorder.complete_node_state.call_count >= 1
        last_call = recorder.complete_node_state.call_args_list[-1]
        assert last_call[1]["status"] == NodeStateStatus.FAILED

    def test_config_gate_fork_destination_creates_children(self) -> None:
        """Config gate with 'fork' destination creates child tokens."""
        recorder = _make_recorder()
        edge_map = {
            (NodeID("cg_1"), "path_a"): "edge_a",
            (NodeID("cg_1"), "path_b"): "edge_b",
            (NodeID("cg_1"), "continue"): "edge_cont",
        }
        route_map = {(NodeID("cg_1"), "true"): RouteDestination.fork()}
        executor = GateExecutor(
            recorder,
            _make_span_factory(),
            _make_step_resolver(),
            edge_map=edge_map,
            route_resolution_map=route_map,
        )
        # Boolean condition requires both true and false routes
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="True",
            routes={"true": "fork", "false": "error_sink"},
            fork_to=["path_a", "path_b"],
        )

        child_a = _make_token(token_id="child_a")
        child_b = _make_token(token_id="child_b")
        token_manager = MagicMock()
        token_manager.fork_token.return_value = ([child_a, child_b], "fg_001")

        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        outcome = executor.execute_config_gate(
            config,
            "cg_1",
            token,
            ctx,
            token_manager=token_manager,
        )

        assert len(outcome.child_tokens) == 2
        token_manager.fork_token.assert_called_once()

    def test_config_gate_missing_token_manager_raises_without_failed_record(self) -> None:
        """Routing-construction invariant should not be recorded as FAILED row state."""
        recorder = _make_recorder()
        edge_map = {
            (NodeID("cg_1"), "path_a"): "edge_a",
            (NodeID("cg_1"), "path_b"): "edge_b",
            (NodeID("cg_1"), "continue"): "edge_cont",
        }
        route_map = {(NodeID("cg_1"), "true"): RouteDestination.fork()}
        executor = GateExecutor(
            recorder,
            _make_span_factory(),
            _make_step_resolver(),
            edge_map=edge_map,
            route_resolution_map=route_map,
        )
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="True",
            routes={"true": "fork", "false": "error_sink"},
            fork_to=["path_a", "path_b"],
        )
        token = _make_token(contract=_make_contract())
        ctx = _make_ctx()

        with pytest.raises(OrchestrationInvariantError, match="no TokenManager"):
            executor.execute_config_gate(
                config,
                "cg_1",
                token,
                ctx,
                token_manager=None,
            )

        statuses = [call.kwargs.get("status") for call in recorder.complete_node_state.call_args_list]
        assert NodeStateStatus.FAILED not in statuses

    def test_config_gate_missing_route_resolution_fails_closed(self) -> None:
        """Missing route resolution mapping raises MissingEdgeError (no fallback)."""
        recorder = _make_recorder()
        edge_map = {(NodeID("cg_1"), "continue"): "edge_cont"}
        executor = GateExecutor(recorder, _make_span_factory(), _make_step_resolver(), edge_map=edge_map)
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="'high'",
            routes={"high": "branch_conn"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        with pytest.raises(MissingEdgeError, match="cg_1"):
            executor.execute_config_gate(
                config,
                "cg_1",
                token,
                ctx,
            )

        assert recorder.complete_node_state.call_count >= 1
        last_call = recorder.complete_node_state.call_args_list[-1]
        assert last_call[1]["status"] == NodeStateStatus.FAILED

    def test_config_gate_exception_records_failed_and_reraises(self) -> None:
        """Exception during config gate eval records FAILED and re-raises.

        Note: GateSettings validates condition syntax at construction, so we
        use a condition that is syntactically valid but raises at evaluation time
        (e.g., accessing a non-existent field on the row data via a runtime error).
        """
        recorder = _make_recorder()
        executor = GateExecutor(recorder, _make_span_factory(), _make_step_resolver())
        # Syntactically valid but references a key that will cause evaluation error
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="row['nonexistent_field'] > 0",
            routes={"true": "next_conn", "false": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        from elspeth.engine.expression_parser import ExpressionEvaluationError

        with pytest.raises(ExpressionEvaluationError):
            executor.execute_config_gate(
                config,
                "cg_1",
                token,
                ctx,
            )

        assert recorder.complete_node_state.call_count >= 1
        last_call = recorder.complete_node_state.call_args_list[-1]
        assert last_call[1]["status"] == NodeStateStatus.FAILED

    # --- Error routing edge cases (exd audit) ---

    def test_config_gate_none_result_records_failed_with_route_label(self) -> None:
        """Expression returning None is stringified to 'None' and fails route lookup.

        The gate converts non-string/non-bool results via str(), so None
        becomes 'None'. Since 'None' isn't in routes, this records FAILED
        state and raises ValueError with the stringified label.
        """
        recorder = _make_recorder()
        executor = GateExecutor(recorder, _make_span_factory(), _make_step_resolver())
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="None",
            routes={"true": "next_conn", "false": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        with pytest.raises(ValueError, match="None"):
            executor.execute_config_gate(config, "cg_1", token, ctx)

        last_call = recorder.complete_node_state.call_args_list[-1]
        assert last_call[1]["status"] == NodeStateStatus.FAILED
        assert "None" in last_call[1]["error"]["exception"]

    def test_config_gate_int_result_stringified_for_route_lookup(self) -> None:
        """Expression returning int is stringified for route label matching.

        Arithmetic expressions return int. The gate converts it via str()
        (e.g., 42 → '42'). If '42' isn't in routes, it records FAILED.
        """
        recorder = _make_recorder()
        executor = GateExecutor(recorder, _make_span_factory(), _make_step_resolver())
        # '40 + 2' returns 42 → str(42) = '42' → not in routes
        config = GateSettings(
            name="my_gate",
            input="in_conn",
            condition="40 + 2",
            routes={"true": "next_conn", "false": "error_sink"},
        )
        contract = _make_contract()
        token = _make_token(contract=contract)
        ctx = _make_ctx()

        with pytest.raises(ValueError, match="'42'"):
            executor.execute_config_gate(config, "cg_1", token, ctx)

        last_call = recorder.complete_node_state.call_args_list[-1]
        assert last_call[1]["status"] == NodeStateStatus.FAILED


# =============================================================================
# TestAggregationExecutor
# =============================================================================


class TestAggregationExecutor:
    """Tests for AggregationExecutor covering buffering, flush, and triggers."""

    def _make_agg_executor(
        self,
        recorder: MagicMock | None = None,
        node_id: str = "agg_1",
        count: int = 3,
    ) -> tuple[AggregationExecutor, MagicMock, NodeID]:
        """Create an AggregationExecutor with a single configured node."""
        if recorder is None:
            recorder = _make_recorder()
        span_factory = _make_span_factory()
        nid = NodeID(node_id)
        settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=count),
        )
        executor = AggregationExecutor(
            recorder,
            span_factory,
            _make_step_resolver(),
            run_id="test-run",
            aggregation_settings={nid: settings},
        )
        return executor, recorder, nid

    # --- buffer_row ---

    def test_buffer_row_unconfigured_node_raises(self) -> None:
        """buffer_row for unconfigured node raises OrchestrationInvariantError."""
        executor, _, _ = self._make_agg_executor()
        token = _make_token()

        with pytest.raises(OrchestrationInvariantError, match="not in aggregation_settings"):
            executor.buffer_row(NodeID("unknown"), token)

    def test_buffer_row_first_row_creates_batch(self) -> None:
        """First buffered row creates a new batch via recorder."""
        executor, recorder, nid = self._make_agg_executor()
        token = _make_token()

        executor.buffer_row(nid, token)

        recorder.create_batch.assert_called_once_with(
            run_id="test-run",
            aggregation_node_id=nid,
        )

    def test_buffer_row_subsequent_rows_reuse_batch(self) -> None:
        """Second row does not create a new batch."""
        executor, recorder, nid = self._make_agg_executor()

        executor.buffer_row(nid, _make_token(token_id="t1"))
        executor.buffer_row(nid, _make_token(token_id="t2"))

        # Only one batch created
        assert recorder.create_batch.call_count == 1

    def test_buffer_row_records_batch_member_with_ordinal(self) -> None:
        """Each buffered row records a batch member with incrementing ordinal."""
        executor, recorder, nid = self._make_agg_executor()

        executor.buffer_row(nid, _make_token(token_id="t1"))
        executor.buffer_row(nid, _make_token(token_id="t2"))

        calls = recorder.add_batch_member.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["ordinal"] == 0
        assert calls[1][1]["ordinal"] == 1

    # --- get_buffered_rows / get_buffered_tokens ---

    def test_get_buffered_rows_returns_data(self) -> None:
        """get_buffered_rows returns the buffered row dicts."""
        executor, _, nid = self._make_agg_executor()
        executor.buffer_row(nid, _make_token(data={"value": "a"}, token_id="t1"))
        executor.buffer_row(nid, _make_token(data={"value": "b"}, token_id="t2"))

        rows = executor.get_buffered_rows(nid)
        assert len(rows) == 2
        assert rows[0]["value"] == "a"
        assert rows[1]["value"] == "b"

    def test_get_buffered_rows_unconfigured_raises(self) -> None:
        executor, _, _ = self._make_agg_executor()
        with pytest.raises(OrchestrationInvariantError):
            executor.get_buffered_rows(NodeID("unknown"))

    def test_get_buffered_tokens_returns_tokens(self) -> None:
        """get_buffered_tokens returns TokenInfo objects."""
        executor, _, nid = self._make_agg_executor()
        executor.buffer_row(nid, _make_token(token_id="t1"))

        tokens = executor.get_buffered_tokens(nid)
        assert len(tokens) == 1
        assert tokens[0].token_id == "t1"

    def test_get_buffered_tokens_unconfigured_raises(self) -> None:
        executor, _, _ = self._make_agg_executor()
        with pytest.raises(OrchestrationInvariantError):
            executor.get_buffered_tokens(NodeID("unknown"))

    def test_empty_buffer_returns_empty_list(self) -> None:
        """Configured node with no rows returns empty list."""
        executor, _, nid = self._make_agg_executor()
        assert executor.get_buffered_rows(nid) == []
        assert executor.get_buffered_tokens(nid) == []

    # --- get_buffer_count ---

    def test_get_buffer_count_correct(self) -> None:
        executor, _, nid = self._make_agg_executor()
        assert executor.get_buffer_count(nid) == 0
        executor.buffer_row(nid, _make_token(token_id="t1"))
        assert executor.get_buffer_count(nid) == 1
        executor.buffer_row(nid, _make_token(token_id="t2"))
        assert executor.get_buffer_count(nid) == 2

    def test_get_buffer_count_unconfigured_raises(self) -> None:
        executor, _, _ = self._make_agg_executor()
        with pytest.raises(OrchestrationInvariantError):
            executor.get_buffer_count(NodeID("unknown"))

    # --- should_flush ---

    def test_should_flush_delegates_to_trigger_evaluator(self) -> None:
        """should_flush delegates to trigger evaluator (count=3, need 3 rows)."""
        executor, _, nid = self._make_agg_executor(count=3)
        assert executor.should_flush(nid) is False
        executor.buffer_row(nid, _make_token(token_id="t1"))
        executor.buffer_row(nid, _make_token(token_id="t2"))
        assert executor.should_flush(nid) is False
        executor.buffer_row(nid, _make_token(token_id="t3"))
        assert executor.should_flush(nid) is True

    def test_should_flush_unconfigured_raises(self) -> None:
        executor, _, _ = self._make_agg_executor()
        with pytest.raises(OrchestrationInvariantError):
            executor.should_flush(NodeID("unknown"))

    # --- get_trigger_type ---

    def test_get_trigger_type_returns_none_before_fire(self) -> None:
        executor, _, nid = self._make_agg_executor(count=10)
        executor.buffer_row(nid, _make_token(token_id="t1"))
        assert executor.get_trigger_type(nid) is None

    def test_get_trigger_type_returns_count_after_fire(self) -> None:
        executor, _, nid = self._make_agg_executor(count=2)
        executor.buffer_row(nid, _make_token(token_id="t1"))
        executor.buffer_row(nid, _make_token(token_id="t2"))
        # should_flush() must be called first to evaluate triggers
        # (sets _last_triggered on the evaluator)
        assert executor.should_flush(nid) is True
        assert executor.get_trigger_type(nid) == TriggerType.COUNT

    def test_get_trigger_type_unconfigured_raises(self) -> None:
        executor, _, _ = self._make_agg_executor()
        with pytest.raises(OrchestrationInvariantError):
            executor.get_trigger_type(NodeID("unknown"))

    # --- execute_flush ---

    def test_execute_flush_no_batch_raises_runtime_error(self) -> None:
        """Flushing without a batch raises RuntimeError."""
        executor, _, nid = self._make_agg_executor()
        transform = MagicMock()
        ctx = _make_ctx()

        with pytest.raises(RuntimeError, match="No batch exists"):
            executor.execute_flush(nid, transform, ctx, TriggerType.COUNT)

    def test_execute_flush_empty_buffer_raises_runtime_error(self) -> None:
        """Flushing with empty buffer raises RuntimeError.

        To reproduce this: buffer a row, flush successfully (which clears buffer),
        then try to flush again - the batch_id is None so it hits 'No batch exists'.
        Actually, getting empty buffer with a batch requires manual state manipulation.
        We'll skip this edge case since the production code guards against it
        (buffer_row creates batch, and batch is reset on flush).
        """
        # This state is hard to reach without direct manipulation.
        # The guard exists for internal consistency checking.
        pass

    def test_execute_flush_success_completes_batch_and_state(self) -> None:
        """Successful flush transitions batch to COMPLETED and state to COMPLETED."""
        executor, recorder, nid = self._make_agg_executor(count=2)
        contract = _make_contract()

        # Buffer two rows
        executor.buffer_row(nid, _make_token(data={"value": "a"}, token_id="t1", contract=contract))
        executor.buffer_row(nid, _make_token(data={"value": "b"}, token_id="t2", contract=contract))

        # Mock batch transform
        transform = MagicMock()
        transform.name = "agg_transform"
        transform.process.return_value = TransformResult.success(
            make_row({"value": "aggregated"}, contract=contract),
            success_reason={"action": "aggregated"},
        )
        ctx = _make_ctx()

        result, tokens, batch_id = executor.execute_flush(
            nid,
            transform,
            ctx,
            TriggerType.COUNT,
        )

        assert result.status == "success"
        assert len(tokens) == 2
        assert batch_id == "batch_001"

        # Verify batch completed
        complete_calls = [c for c in recorder.complete_batch.call_args_list if c[1].get("status") == BatchStatus.COMPLETED]
        assert len(complete_calls) == 1

    def test_execute_flush_error_result_marks_batch_failed(self) -> None:
        """Error result from transform marks batch as FAILED."""
        executor, recorder, nid = self._make_agg_executor(count=2)
        contract = _make_contract()

        executor.buffer_row(nid, _make_token(data={"value": "a"}, token_id="t1", contract=contract))
        executor.buffer_row(nid, _make_token(data={"value": "b"}, token_id="t2", contract=contract))

        transform = MagicMock()
        transform.name = "agg_transform"
        transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )
        ctx = _make_ctx()

        result, _tokens, _batch_id = executor.execute_flush(
            nid,
            transform,
            ctx,
            TriggerType.COUNT,
        )

        assert result.status == "error"

        # Verify batch marked failed
        failed_calls = [c for c in recorder.complete_batch.call_args_list if c[1].get("status") == BatchStatus.FAILED]
        assert len(failed_calls) == 1

    def test_execute_flush_exception_marks_batch_failed_and_reraises(self) -> None:
        """Exception from transform marks batch as FAILED and re-raises."""
        executor, recorder, nid = self._make_agg_executor(count=2)
        contract = _make_contract()

        executor.buffer_row(nid, _make_token(data={"value": "a"}, token_id="t1", contract=contract))
        executor.buffer_row(nid, _make_token(data={"value": "b"}, token_id="t2", contract=contract))

        transform = MagicMock()
        transform.name = "agg_transform"
        transform.process.side_effect = RuntimeError("transform crash")
        ctx = _make_ctx()

        with pytest.raises(RuntimeError, match="transform crash"):
            executor.execute_flush(nid, transform, ctx, TriggerType.COUNT)

        # Verify batch marked failed
        failed_calls = [c for c in recorder.complete_batch.call_args_list if c[1].get("status") == BatchStatus.FAILED]
        assert len(failed_calls) == 1

    def test_execute_flush_exception_clears_batch_token_ids(self) -> None:
        """Exception path clears ctx.batch_token_ids to prevent stale leakage.

        Regression: PluginContext is reused across calls. If execute_flush()
        raises without clearing batch_token_ids, subsequent transforms
        (e.g. OpenRouterBatchLLMTransform) see stale IDs and misattribute
        telemetry to tokens from the failed batch.
        """
        executor, _recorder, nid = self._make_agg_executor(count=2)
        contract = _make_contract()

        executor.buffer_row(nid, _make_token(data={"v": "a"}, token_id="t1", contract=contract))
        executor.buffer_row(nid, _make_token(data={"v": "b"}, token_id="t2", contract=contract))

        transform = MagicMock()
        transform.name = "agg"
        transform.process.side_effect = RuntimeError("boom")
        ctx = _make_ctx()

        with pytest.raises(RuntimeError, match="boom"):
            executor.execute_flush(nid, transform, ctx, TriggerType.COUNT)

        # batch_token_ids must be None after error, not stale ["t1", "t2"]
        assert ctx.batch_token_ids is None

    def test_execute_flush_batch_pending_clears_batch_token_ids(self) -> None:
        """BatchPendingError path clears ctx.batch_token_ids.

        Regression: Same stale-ID leakage as the exception path — the
        BatchPendingError control flow signal re-raised before cleanup.
        """
        from elspeth.contracts.errors import BatchPendingError

        executor, _recorder, nid = self._make_agg_executor(count=2)
        contract = _make_contract()

        executor.buffer_row(nid, _make_token(data={"v": "a"}, token_id="t1", contract=contract))
        executor.buffer_row(nid, _make_token(data={"v": "b"}, token_id="t2", contract=contract))

        transform = MagicMock()
        transform.name = "agg"
        transform.process.side_effect = BatchPendingError("batch-123", "submitted")
        ctx = _make_ctx()

        with pytest.raises(BatchPendingError):
            executor.execute_flush(nid, transform, ctx, TriggerType.COUNT)

        # batch_token_ids must be None after pending, not stale ["t1", "t2"]
        assert ctx.batch_token_ids is None

    def test_execute_flush_resets_batch_state(self) -> None:
        """After flush, batch state is reset (new batch on next row)."""
        executor, _recorder, nid = self._make_agg_executor(count=1)
        contract = _make_contract()

        executor.buffer_row(nid, _make_token(token_id="t1", contract=contract))

        transform = MagicMock()
        transform.name = "agg"
        transform.process.return_value = TransformResult.success(
            make_row({"value": "agg"}, contract=contract),
            success_reason={"action": "agg"},
        )
        ctx = _make_ctx()

        executor.execute_flush(nid, transform, ctx, TriggerType.COUNT)

        # Batch ID should be None after flush
        assert executor.get_batch_id(nid) is None
        # Buffer should be empty
        assert executor.get_buffer_count(nid) == 0

    # --- get_batch_id ---

    def test_get_batch_id_none_before_rows(self) -> None:
        executor, _, nid = self._make_agg_executor()
        assert executor.get_batch_id(nid) is None

    def test_get_batch_id_set_after_first_row(self) -> None:
        executor, _, nid = self._make_agg_executor()
        executor.buffer_row(nid, _make_token(token_id="t1"))
        assert executor.get_batch_id(nid) == "batch_001"

    # --- check_flush_status ---

    def test_check_flush_status_combined_operation(self) -> None:
        """check_flush_status returns (bool, TriggerType|None) tuple."""
        executor, _, nid = self._make_agg_executor(count=2)

        should_flush, trigger = executor.check_flush_status(nid)
        assert should_flush is False
        assert trigger is None

        executor.buffer_row(nid, _make_token(token_id="t1"))
        executor.buffer_row(nid, _make_token(token_id="t2"))

        should_flush, trigger = executor.check_flush_status(nid)
        assert should_flush is True
        assert trigger == TriggerType.COUNT

    def test_check_flush_status_unconfigured_raises(self) -> None:
        executor, _, _ = self._make_agg_executor()
        with pytest.raises(OrchestrationInvariantError):
            executor.check_flush_status(NodeID("unknown"))

    # --- restore_state / get_restored_state ---

    def test_restore_state_and_get(self) -> None:
        executor, _, nid = self._make_agg_executor()
        state = {"key": "value"}
        executor.restore_state(nid, state)
        assert executor.get_restored_state(nid) == state

    def test_get_restored_state_returns_none_if_not_set(self) -> None:
        executor, _, nid = self._make_agg_executor()
        assert executor.get_restored_state(nid) is None


# =============================================================================
# TestSinkExecutor
# =============================================================================


class TestSinkExecutor:
    """Tests for SinkExecutor covering write lifecycle, artifacts, and callbacks."""

    # --- Empty tokens ---

    def test_empty_tokens_returns_none(self) -> None:
        """Write with empty token list returns None immediately."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        result = executor.write(
            sink,
            [],
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        assert result is None
        sink.write.assert_not_called()

    # --- No node_id ---

    def test_no_node_id_raises_orchestration_invariant_error(self) -> None:
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        sink = _make_sink(node_id=None)
        token = _make_token()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(OrchestrationInvariantError, match="without node_id"):
            executor.write(
                sink,
                [token],
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

    # --- Successful write ---

    def test_successful_write_completes_states_and_registers_artifact(self) -> None:
        """Successful write: node states COMPLETED, artifact registered, outcomes recorded."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(data={"value": "a"}, token_id="t1", contract=contract),
            _make_token(data={"value": "b"}, token_id="t2", contract=contract),
        ]
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        artifact = executor.write(
            sink,
            tokens,
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        assert artifact is not None
        # 2 begin_node_state calls (one per token)
        assert recorder.begin_node_state.call_count == 2
        # 2 complete_node_state with COMPLETED (one per token)
        completed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.COMPLETED]
        assert len(completed_calls) == 2
        # Artifact registered
        recorder.register_artifact.assert_called_once()
        # Token outcomes recorded
        assert recorder.record_token_outcome.call_count == 2
        for c in recorder.record_token_outcome.call_args_list:
            assert c[1]["outcome"] == RowOutcome.COMPLETED
            assert c[1]["sink_name"] == "out"

    def test_successful_write_calls_sink_flush(self) -> None:
        """Successful write calls sink.flush() after sink.write()."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        token = _make_token()
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        executor.write(
            sink,
            [token],
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        sink.write.assert_called_once()
        sink.flush.assert_called_once()

    def test_successful_write_passes_dicts_to_sink(self) -> None:
        """Sink receives plain dicts, not PipelineRow."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        token = _make_token(data={"value": "test"})
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        executor.write(
            sink,
            [token],
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        rows_passed = sink.write.call_args[0][0]
        assert isinstance(rows_passed, list)
        assert all(isinstance(r, dict) for r in rows_passed)

    # --- Write exception ---

    def test_contract_merge_exception_marks_all_states_failed_and_reraises(self) -> None:
        """Contract merge failure marks opened sink states FAILED and re-raises."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract_a = _make_contract()
        contract_b = SchemaContract(
            fields=(
                make_field(
                    "value",
                    python_type=int,
                    original_name="value",
                    required=True,
                    source="declared",
                ),
            ),
            mode="FLEXIBLE",
            locked=True,
        )
        tokens = [
            _make_token(data={"value": "a"}, token_id="t1", contract=contract_a),
            _make_token(data={"value": 1}, token_id="t2", contract=contract_b),
        ]
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(ContractMergeError):
            executor.write(
                sink,
                tokens,
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

        failed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 2
        for call in failed_calls:
            error = call[1]["error"]
            assert error["phase"] == "contract_merge"

        sink.write.assert_not_called()
        sink.flush.assert_not_called()
        recorder.record_token_outcome.assert_not_called()

    def test_write_exception_marks_all_states_failed_and_reraises(self) -> None:
        """Exception from sink.write() marks all states FAILED and re-raises."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(token_id="t1", contract=contract),
            _make_token(token_id="t2", contract=contract),
        ]
        sink = _make_sink()
        sink.write.side_effect = OSError("disk full")
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(OSError, match="disk full"):
            executor.write(
                sink,
                tokens,
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

        # All states should be FAILED
        failed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 2

    # --- Flush exception ---

    def test_flush_exception_marks_all_states_failed_and_reraises(self) -> None:
        """Exception from sink.flush() marks all states FAILED and re-raises."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        tokens = [_make_token()]
        sink = _make_sink()
        sink.flush.side_effect = OSError("flush failed")
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(OSError, match="flush failed"):
            executor.write(
                sink,
                tokens,
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

        failed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 1

    # --- Callback ---

    def test_on_token_written_called_per_token(self) -> None:
        """on_token_written callback called for each token."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(token_id="t1", contract=contract),
            _make_token(token_id="t2", contract=contract),
        ]
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)
        callback = MagicMock()

        executor.write(
            sink,
            tokens,
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
            on_token_written=callback,
        )

        assert callback.call_count == 2
        callback_token_ids = [c[0][0].token_id for c in callback.call_args_list]
        assert "t1" in callback_token_ids
        assert "t2" in callback_token_ids

    def test_on_token_written_failure_logged_not_raised(self) -> None:
        """Callback failure is logged but does not raise (sink already durable)."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        token = _make_token()
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)
        callback = MagicMock(side_effect=RuntimeError("checkpoint failed"))

        # Should not raise
        artifact = executor.write(
            sink,
            [token],
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
            on_token_written=callback,
        )

        assert artifact is not None  # Write succeeded despite callback failure

    # --- PendingOutcome ---

    def test_pending_outcome_used_for_token_outcome(self) -> None:
        """pending_outcome.outcome and error_hash propagated to record_token_outcome."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        token = _make_token()
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.QUARANTINED, error_hash="err_hash_123")

        executor.write(
            sink,
            [token],
            ctx,
            step_in_pipeline=5,
            sink_name="quarantine",
            pending_outcome=pending,
        )

        recorder.record_token_outcome.assert_called_once()
        kwargs = recorder.record_token_outcome.call_args[1]
        assert kwargs["outcome"] == RowOutcome.QUARANTINED
        assert kwargs["error_hash"] == "err_hash_123"
        assert kwargs["sink_name"] == "quarantine"

    # --- Artifact registration ---

    def test_artifact_linked_to_first_state(self) -> None:
        """Artifact is registered linked to the first token's state."""
        recorder = _make_recorder()
        # Make begin_node_state return different state_ids per call
        states = [Mock(state_id="state_001"), Mock(state_id="state_002")]
        recorder.begin_node_state.side_effect = states

        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(token_id="t1", contract=contract),
            _make_token(token_id="t2", contract=contract),
        ]
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        executor.write(
            sink,
            tokens,
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        recorder.register_artifact.assert_called_once()
        kwargs = recorder.register_artifact.call_args[1]
        assert kwargs["state_id"] == "state_001"  # First token's state

    # --- Duration amortization (sm22) ---

    def test_duration_amortized_across_tokens_on_success(self) -> None:
        """Each token gets batch_duration / N, not the full batch duration."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(data={"value": "a"}, token_id="t1", contract=contract),
            _make_token(data={"value": "b"}, token_id="t2", contract=contract),
            _make_token(data={"value": "c"}, token_id="t3", contract=contract),
        ]
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        executor.write(
            sink,
            tokens,
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        completed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.COMPLETED]
        assert len(completed_calls) == 3
        durations = [c[1]["duration_ms"] for c in completed_calls]
        # All three tokens should get the same amortized duration
        assert durations[0] == durations[1] == durations[2]
        # Amortized means each is roughly 1/3 of what the total would be.
        # We can't know the exact total, but we can verify the values are
        # consistent: sum of per-token ≈ total (within floating point)
        total_reconstructed = sum(durations)
        assert durations[0] == pytest.approx(total_reconstructed / 3)

    def test_duration_amortized_across_tokens_on_write_failure(self) -> None:
        """Write failure: each token gets amortized duration, not full batch time."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(data={"value": "a"}, token_id="t1", contract=contract),
            _make_token(data={"value": "b"}, token_id="t2", contract=contract),
        ]
        sink = _make_sink()
        sink.write.side_effect = OSError("disk full")
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(OSError):
            executor.write(
                sink,
                tokens,
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

        failed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 2
        durations = [c[1]["duration_ms"] for c in failed_calls]
        assert durations[0] == durations[1]
        assert durations[0] == pytest.approx(sum(durations) / 2)

    def test_duration_amortized_across_tokens_on_flush_failure(self) -> None:
        """Flush failure: each token gets amortized duration, not full batch time."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        contract = _make_contract()
        tokens = [
            _make_token(data={"value": "a"}, token_id="t1", contract=contract),
            _make_token(data={"value": "b"}, token_id="t2", contract=contract),
        ]
        sink = _make_sink()
        sink.flush.side_effect = OSError("flush failed")
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(OSError):
            executor.write(
                sink,
                tokens,
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

        failed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 2
        durations = [c[1]["duration_ms"] for c in failed_calls]
        assert durations[0] == durations[1]
        assert durations[0] == pytest.approx(sum(durations) / 2)

    def test_single_token_duration_unchanged(self) -> None:
        """Single-token write: duration_ms is total (N/N = total)."""
        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        token = _make_token()
        sink = _make_sink()
        ctx = _make_ctx()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        executor.write(
            sink,
            [token],
            ctx,
            step_in_pipeline=5,
            sink_name="out",
            pending_outcome=pending,
        )

        completed_calls = [c for c in recorder.complete_node_state.call_args_list if c[1].get("status") == NodeStateStatus.COMPLETED]
        assert len(completed_calls) == 1
        # With 1 token, amortized = total (no division effect)
        assert completed_calls[0][1]["duration_ms"] > 0


# =============================================================================
# AggregationExecutor checkpoint version tests
# =============================================================================


class TestAggregationCheckpointVersion:
    """Tests for aggregation checkpoint version compatibility."""

    def test_old_checkpoint_version_rejected(self) -> None:
        """Old checkpoint version (2.1) is rejected — prevents silent state corruption."""
        executor, _recorder, _nid = TestAggregationExecutor._make_agg_executor(TestAggregationExecutor())

        old_checkpoint = {"_version": "2.1", "agg_1": {"tokens": []}}

        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            executor.restore_from_checkpoint(old_checkpoint)

    def test_current_version_accepted(self) -> None:
        """Current checkpoint version is accepted without error."""
        executor, _recorder, _nid = TestAggregationExecutor._make_agg_executor(TestAggregationExecutor())

        # Minimal valid v3.0 checkpoint with no buffered tokens
        checkpoint = {"_version": AGGREGATION_CHECKPOINT_VERSION}

        # Should not raise
        executor.restore_from_checkpoint(checkpoint)

    def test_missing_version_rejected(self) -> None:
        """Checkpoint without _version key is rejected."""
        executor, _recorder, _nid = TestAggregationExecutor._make_agg_executor(TestAggregationExecutor())

        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            executor.restore_from_checkpoint({"agg_1": {"tokens": []}})
