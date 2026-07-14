"""Tests for telemetry.filtering -- event granularity filtering."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.enums import (
    CallStatus,
    CallType,
    NodeStateStatus,
    RoutingMode,
    RunStatus,
    TelemetryGranularity,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.contracts.events import (
    ExternalCallCompleted,
    FieldResolutionApplied,
    GateEvaluated,
    PhaseAction,
    PhaseChanged,
    PipelinePhase,
    RowCreated,
    RunFinished,
    RunStarted,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.telemetry.filtering import should_emit
from elspeth.telemetry.serialization import serialize_event_attributes

# =============================================================================
# Constants and Factories
# =============================================================================

_NOW = datetime(2026, 1, 15, tzinfo=UTC)
_RUN_ID = "run-test"


def _run_started() -> RunStarted:
    return RunStarted(
        timestamp=_NOW,
        run_id=_RUN_ID,
        config_hash="h1",
        source_plugin="csv",
    )


def _run_finished() -> RunFinished:
    return RunFinished(
        timestamp=_NOW,
        run_id=_RUN_ID,
        status=RunStatus.COMPLETED,
        row_count=10,
        duration_ms=100.0,
    )


def _phase_changed() -> PhaseChanged:
    return PhaseChanged(
        timestamp=_NOW,
        run_id=_RUN_ID,
        phase=PipelinePhase.PROCESS,
        action=PhaseAction.PROCESSING,
    )


def _row_created() -> RowCreated:
    return RowCreated(
        timestamp=_NOW,
        run_id=_RUN_ID,
        row_id="row-1",
        token_id="tok-1",
        content_hash="abc123",
    )


def _transform_completed() -> TransformCompleted:
    return TransformCompleted(
        timestamp=_NOW,
        run_id=_RUN_ID,
        row_id="row-1",
        token_id="tok-1",
        node_id="node-1",
        plugin_name="passthrough",
        status=NodeStateStatus.COMPLETED,
        duration_ms=5.0,
        input_hash="in-hash",
        output_hash="out-hash",
    )


def _gate_evaluated() -> GateEvaluated:
    return GateEvaluated(
        timestamp=_NOW,
        run_id=_RUN_ID,
        row_id="row-1",
        token_id="tok-1",
        node_id="gate-1",
        plugin_name="threshold_gate",
        routing_mode=RoutingMode.MOVE,
        destinations=("sink-a",),
    )


def _token_completed() -> TokenCompleted:
    return TokenCompleted(
        timestamp=_NOW,
        run_id=_RUN_ID,
        row_id="row-1",
        token_id="tok-1",
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.DEFAULT_FLOW,
        sink_name="output",
    )


def _external_call() -> ExternalCallCompleted:
    return ExternalCallCompleted(
        timestamp=_NOW,
        run_id=_RUN_ID,
        call_type=CallType.LLM,
        provider="test-provider",
        status=CallStatus.SUCCESS,
        latency_ms=50.0,
        state_id="state-1",
        request_payload=RawCallPayload({"operation": "apply_guardrail"}),
        response_payload=RawCallPayload({"status": "blocked", "matched_filters": ("PROMPT_ATTACK",)}),
    )


def _field_resolution_applied() -> FieldResolutionApplied:
    return FieldResolutionApplied(
        timestamp=_NOW,
        run_id=_RUN_ID,
        source_plugin="csv",
        field_count=2,
        normalization_version="v1",
        resolution_mapping={"Customer ID": "customer_id", "Order Amount": "order_amount"},
    )


# =============================================================================
# Lifecycle Events: Always Emit at Any Granularity
# =============================================================================

LIFECYCLE = TelemetryGranularity.LIFECYCLE
ROWS = TelemetryGranularity.ROWS
FULL = TelemetryGranularity.FULL


class TestLifecycleEventsAlwaysEmit:
    @pytest.mark.parametrize("granularity", [LIFECYCLE, ROWS, FULL])
    def test_run_started_emits_at_all_granularities(self, granularity: TelemetryGranularity) -> None:
        assert should_emit(_run_started(), granularity) is True

    @pytest.mark.parametrize("granularity", [LIFECYCLE, ROWS, FULL])
    def test_run_finished_emits_at_all_granularities(self, granularity: TelemetryGranularity) -> None:
        assert should_emit(_run_finished(), granularity) is True

    @pytest.mark.parametrize("granularity", [LIFECYCLE, ROWS, FULL])
    def test_phase_changed_emits_at_all_granularities(self, granularity: TelemetryGranularity) -> None:
        assert should_emit(_phase_changed(), granularity) is True


# =============================================================================
# Row Events: Emit at ROWS or FULL Only
# =============================================================================


class TestRowEventsFilteredByGranularity:
    def test_row_created_suppressed_at_lifecycle(self) -> None:
        assert should_emit(_row_created(), LIFECYCLE) is False

    def test_row_created_emits_at_rows(self) -> None:
        assert should_emit(_row_created(), ROWS) is True

    def test_row_created_emits_at_full(self) -> None:
        assert should_emit(_row_created(), FULL) is True

    def test_transform_completed_suppressed_at_lifecycle(self) -> None:
        assert should_emit(_transform_completed(), LIFECYCLE) is False

    def test_transform_completed_emits_at_rows(self) -> None:
        assert should_emit(_transform_completed(), ROWS) is True

    def test_transform_completed_emits_at_full(self) -> None:
        assert should_emit(_transform_completed(), FULL) is True

    def test_gate_evaluated_suppressed_at_lifecycle(self) -> None:
        assert should_emit(_gate_evaluated(), LIFECYCLE) is False

    def test_gate_evaluated_emits_at_rows(self) -> None:
        assert should_emit(_gate_evaluated(), ROWS) is True

    def test_gate_evaluated_emits_at_full(self) -> None:
        assert should_emit(_gate_evaluated(), FULL) is True

    def test_token_completed_suppressed_at_lifecycle(self) -> None:
        assert should_emit(_token_completed(), LIFECYCLE) is False

    def test_token_completed_emits_at_rows(self) -> None:
        assert should_emit(_token_completed(), ROWS) is True

    def test_token_completed_emits_at_full(self) -> None:
        assert should_emit(_token_completed(), FULL) is True

    def test_token_completed_serializes_outcome_and_path(self) -> None:
        attrs = serialize_event_attributes(_token_completed())

        assert attrs["event_type"] == "TokenCompleted"
        assert attrs["outcome"] == "success"
        assert attrs["path"] == "default_flow"
        assert attrs["sink_name"] == "output"

    def test_field_resolution_applied_suppressed_at_lifecycle(self) -> None:
        assert should_emit(_field_resolution_applied(), LIFECYCLE) is False

    def test_field_resolution_applied_emits_at_rows(self) -> None:
        assert should_emit(_field_resolution_applied(), ROWS) is True

    def test_field_resolution_applied_emits_at_full(self) -> None:
        assert should_emit(_field_resolution_applied(), FULL) is True


# =============================================================================
# External Call Events: Emit Only at FULL
# =============================================================================


class TestExternalCallEventsFullOnly:
    def test_external_call_suppressed_at_lifecycle(self) -> None:
        assert should_emit(_external_call(), LIFECYCLE) is False

    def test_external_call_suppressed_at_rows(self) -> None:
        assert should_emit(_external_call(), ROWS) is False

    def test_external_call_emits_at_full(self) -> None:
        assert should_emit(_external_call(), FULL) is True

    def test_bedrock_external_call_payload_serializes_without_provider_text(self) -> None:
        attributes = serialize_event_attributes(_external_call())
        rendered = repr(attributes)

        assert "apply_guardrail" in rendered
        assert "PROMPT_ATTACK" in rendered
        assert "guardrail_identifier" not in rendered
        assert "provider_output" not in rendered
