"""Tests for telemetry.filtering -- event granularity filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from elspeth.contracts.enums import (
    CallStatus,
    CallType,
    NodeStateStatus,
    RoutingMode,
    RowOutcome,
    RunStatus,
    TelemetryGranularity,
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
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.telemetry.filtering import should_emit

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
        outcome=RowOutcome.COMPLETED,
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


@dataclass(frozen=True, slots=True)
class _UnknownEvent(TelemetryEvent):
    """Custom event subclass to test forward-compatible filtering."""

    custom_field: str = "unknown"


def _unknown_event() -> _UnknownEvent:
    return _UnknownEvent(
        timestamp=_NOW,
        run_id=_RUN_ID,
        custom_field="test",
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


# =============================================================================
# Unknown Events: Pass Through (Fail-Open for Forward Compatibility)
# =============================================================================


class TestUnknownEventsPassThrough:
    def test_unknown_event_passes_at_lifecycle(self) -> None:
        assert should_emit(_unknown_event(), LIFECYCLE) is True

    def test_unknown_event_passes_at_rows(self) -> None:
        assert should_emit(_unknown_event(), ROWS) is True

    def test_unknown_event_passes_at_full(self) -> None:
        assert should_emit(_unknown_event(), FULL) is True


# =============================================================================
# Parametrized Full Matrix
# =============================================================================


_LIFECYCLE_FACTORIES = [
    pytest.param(_run_started, id="RunStarted"),
    pytest.param(_run_finished, id="RunFinished"),
    pytest.param(_phase_changed, id="PhaseChanged"),
]

_ROW_FACTORIES = [
    pytest.param(_row_created, id="RowCreated"),
    pytest.param(_transform_completed, id="TransformCompleted"),
    pytest.param(_gate_evaluated, id="GateEvaluated"),
    pytest.param(_token_completed, id="TokenCompleted"),
    pytest.param(_field_resolution_applied, id="FieldResolutionApplied"),
]


class TestFullMatrix:
    @pytest.mark.parametrize("factory", _LIFECYCLE_FACTORIES)
    @pytest.mark.parametrize("granularity", [LIFECYCLE, ROWS, FULL])
    def test_lifecycle_events_always_true(self, factory, granularity) -> None:
        assert should_emit(factory(), granularity) is True

    @pytest.mark.parametrize("factory", _ROW_FACTORIES)
    def test_row_events_false_at_lifecycle(self, factory) -> None:
        assert should_emit(factory(), LIFECYCLE) is False

    @pytest.mark.parametrize("factory", _ROW_FACTORIES)
    @pytest.mark.parametrize("granularity", [ROWS, FULL])
    def test_row_events_true_at_rows_or_full(self, factory, granularity) -> None:
        assert should_emit(factory(), granularity) is True

    def test_external_call_only_at_full(self) -> None:
        event = _external_call()
        assert should_emit(event, LIFECYCLE) is False
        assert should_emit(event, ROWS) is False
        assert should_emit(event, FULL) is True
