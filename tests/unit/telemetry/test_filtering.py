# tests/unit/telemetry/test_filtering.py
"""Unit tests for telemetry event filtering based on granularity.

Tests cover:
- Lifecycle events always pass (at any granularity)
- Row-level events pass at ROWS and FULL granularity
- External call events pass only at FULL granularity
- Unknown event types pass through (forward compatibility)
"""

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
    GateEvaluated,
    PhaseAction,
    PipelinePhase,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.telemetry import (
    ExternalCallCompleted,
    PhaseChanged,
    RowCreated,
    RunFinished,
    RunStarted,
    should_emit,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed timestamp for deterministic tests."""
    return datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def base_run_id() -> str:
    """Fixed run ID for tests."""
    return "run-filter-test"


# =============================================================================
# Lifecycle Event Tests
# =============================================================================


class TestLifecycleEventsAlwaysPass:
    """Lifecycle events should always be emitted at any granularity."""

    @pytest.mark.parametrize(
        "granularity",
        [
            TelemetryGranularity.LIFECYCLE,
            TelemetryGranularity.ROWS,
            TelemetryGranularity.FULL,
        ],
    )
    def test_run_started_always_passes(self, base_timestamp: datetime, base_run_id: str, granularity: TelemetryGranularity) -> None:
        """RunStarted passes at any granularity."""
        event = RunStarted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            config_hash="abc123",
            source_plugin="csv",
        )
        assert should_emit(event, granularity) is True

    @pytest.mark.parametrize(
        "granularity",
        [
            TelemetryGranularity.LIFECYCLE,
            TelemetryGranularity.ROWS,
            TelemetryGranularity.FULL,
        ],
    )
    def test_run_finished_always_passes(self, base_timestamp: datetime, base_run_id: str, granularity: TelemetryGranularity) -> None:
        """RunFinished passes at any granularity."""
        event = RunFinished(
            timestamp=base_timestamp,
            run_id=base_run_id,
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=5000.0,
        )
        assert should_emit(event, granularity) is True

    @pytest.mark.parametrize(
        "granularity",
        [
            TelemetryGranularity.LIFECYCLE,
            TelemetryGranularity.ROWS,
            TelemetryGranularity.FULL,
        ],
    )
    def test_phase_changed_always_passes(self, base_timestamp: datetime, base_run_id: str, granularity: TelemetryGranularity) -> None:
        """PhaseChanged passes at any granularity."""
        event = PhaseChanged(
            timestamp=base_timestamp,
            run_id=base_run_id,
            phase=PipelinePhase.PROCESS,
            action=PhaseAction.PROCESSING,
        )
        assert should_emit(event, granularity) is True


# =============================================================================
# Row-Level Event Tests
# =============================================================================


class TestRowLevelEvents:
    """Row-level events should pass at ROWS and FULL, but not LIFECYCLE."""

    @pytest.mark.parametrize(
        "granularity,expected",
        [
            (TelemetryGranularity.LIFECYCLE, False),
            (TelemetryGranularity.ROWS, True),
            (TelemetryGranularity.FULL, True),
        ],
    )
    def test_row_created_filtering(
        self,
        base_timestamp: datetime,
        base_run_id: str,
        granularity: TelemetryGranularity,
        expected: bool,
    ) -> None:
        """RowCreated filtering based on granularity."""
        event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash123",
        )
        assert should_emit(event, granularity) is expected

    @pytest.mark.parametrize(
        "granularity,expected",
        [
            (TelemetryGranularity.LIFECYCLE, False),
            (TelemetryGranularity.ROWS, True),
            (TelemetryGranularity.FULL, True),
        ],
    )
    def test_transform_completed_filtering(
        self,
        base_timestamp: datetime,
        base_run_id: str,
        granularity: TelemetryGranularity,
        expected: bool,
    ) -> None:
        """TransformCompleted filtering based on granularity."""
        event = TransformCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="field_mapper",
            status=NodeStateStatus.COMPLETED,
            duration_ms=10.0,
            input_hash="in-hash",
            output_hash="out-hash",
        )
        assert should_emit(event, granularity) is expected

    @pytest.mark.parametrize(
        "granularity,expected",
        [
            (TelemetryGranularity.LIFECYCLE, False),
            (TelemetryGranularity.ROWS, True),
            (TelemetryGranularity.FULL, True),
        ],
    )
    def test_gate_evaluated_filtering(
        self,
        base_timestamp: datetime,
        base_run_id: str,
        granularity: TelemetryGranularity,
        expected: bool,
    ) -> None:
        """GateEvaluated filtering based on granularity."""
        event = GateEvaluated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("sink_a",),
        )
        assert should_emit(event, granularity) is expected

    @pytest.mark.parametrize(
        "granularity,expected",
        [
            (TelemetryGranularity.LIFECYCLE, False),
            (TelemetryGranularity.ROWS, True),
            (TelemetryGranularity.FULL, True),
        ],
    )
    def test_token_completed_filtering(
        self,
        base_timestamp: datetime,
        base_run_id: str,
        granularity: TelemetryGranularity,
        expected: bool,
    ) -> None:
        """TokenCompleted filtering based on granularity."""
        event = TokenCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        assert should_emit(event, granularity) is expected


# =============================================================================
# External Call Event Tests
# =============================================================================


class TestExternalCallEvents:
    """External call events should pass only at FULL granularity."""

    @pytest.mark.parametrize(
        "granularity,expected",
        [
            (TelemetryGranularity.LIFECYCLE, False),
            (TelemetryGranularity.ROWS, False),
            (TelemetryGranularity.FULL, True),
        ],
    )
    def test_external_call_completed_filtering(
        self,
        base_timestamp: datetime,
        base_run_id: str,
        granularity: TelemetryGranularity,
        expected: bool,
    ) -> None:
        """ExternalCallCompleted filtering based on granularity."""
        event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.LLM,
            provider="azure-openai",
            status=CallStatus.SUCCESS,
            latency_ms=1500.0,
        )
        assert should_emit(event, granularity) is expected


# =============================================================================
# Unknown Event Type Tests (Forward Compatibility)
# =============================================================================


class TestUnknownEventTypes:
    """Unknown event types should pass through for forward compatibility."""

    @pytest.mark.parametrize(
        "granularity",
        [
            TelemetryGranularity.LIFECYCLE,
            TelemetryGranularity.ROWS,
            TelemetryGranularity.FULL,
        ],
    )
    def test_base_telemetry_event_passes_through(
        self, base_timestamp: datetime, base_run_id: str, granularity: TelemetryGranularity
    ) -> None:
        """Base TelemetryEvent (unknown type) passes through."""
        # Using the base class simulates an unknown event type
        event = TelemetryEvent(
            timestamp=base_timestamp,
            run_id=base_run_id,
        )
        assert should_emit(event, granularity) is True


# =============================================================================
# Granularity Level Ordering Tests
# =============================================================================


class TestGranularityOrdering:
    """Verify granularity levels are correctly ordered: LIFECYCLE < ROWS < FULL."""

    def test_lifecycle_is_minimal(self, base_timestamp: datetime, base_run_id: str) -> None:
        """LIFECYCLE granularity filters out the most events."""
        lifecycle_events = [
            RunStarted(
                timestamp=base_timestamp,
                run_id=base_run_id,
                config_hash="abc",
                source_plugin="csv",
            ),
            RunFinished(
                timestamp=base_timestamp,
                run_id=base_run_id,
                status=RunStatus.COMPLETED,
                row_count=100,
                duration_ms=5000.0,
            ),
            PhaseChanged(
                timestamp=base_timestamp,
                run_id=base_run_id,
                phase=PipelinePhase.PROCESS,
                action=PhaseAction.PROCESSING,
            ),
        ]

        row_event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash",
        )

        call_event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.HTTP,
            provider="api",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
        )

        # All lifecycle events pass at LIFECYCLE
        for event in lifecycle_events:
            assert should_emit(event, TelemetryGranularity.LIFECYCLE) is True

        # Row and call events are filtered at LIFECYCLE
        assert should_emit(row_event, TelemetryGranularity.LIFECYCLE) is False
        assert should_emit(call_event, TelemetryGranularity.LIFECYCLE) is False

    def test_rows_includes_lifecycle_and_rows(self, base_timestamp: datetime, base_run_id: str) -> None:
        """ROWS granularity includes lifecycle and row events, filters external calls."""
        lifecycle_event = RunStarted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            config_hash="abc",
            source_plugin="csv",
        )

        row_event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash",
        )

        call_event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.HTTP,
            provider="api",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
        )

        # Lifecycle and row events pass at ROWS
        assert should_emit(lifecycle_event, TelemetryGranularity.ROWS) is True
        assert should_emit(row_event, TelemetryGranularity.ROWS) is True

        # External calls are filtered at ROWS
        assert should_emit(call_event, TelemetryGranularity.ROWS) is False

    def test_full_includes_everything(self, base_timestamp: datetime, base_run_id: str) -> None:
        """FULL granularity includes all event types."""
        lifecycle_event = RunStarted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            config_hash="abc",
            source_plugin="csv",
        )

        row_event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash",
        )

        call_event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.HTTP,
            provider="api",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
        )

        # All event types pass at FULL
        assert should_emit(lifecycle_event, TelemetryGranularity.FULL) is True
        assert should_emit(row_event, TelemetryGranularity.FULL) is True
        assert should_emit(call_event, TelemetryGranularity.FULL) is True
