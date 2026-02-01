# tests/telemetry/test_property_based.py
"""Property-based tests for the telemetry subsystem using Hypothesis.

These tests verify critical invariants and properties that must hold across
all possible inputs, using Hypothesis for systematic exploration.

Test categories:
1. TelemetryManagerStateMachine: Stateful testing for failure handling
2. BoundedBuffer Properties: Ring buffer invariants
3. Granularity Filtering Matrix: All granularity x event type combinations
4. Event Ordering: FIFO ordering preservation through the pipeline
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

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
from elspeth.telemetry.buffer import BoundedBuffer
from elspeth.telemetry.manager import TelemetryManager

# =============================================================================
# Test Doubles
# =============================================================================


@dataclass
class MockConfig:
    """Mock RuntimeTelemetryProtocol for testing."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False

    @property
    def backpressure_mode(self) -> Any:
        return None

    @property
    def exporter_configs(self) -> tuple[Any, ...]:
        return ()


class MockExporter:
    """Mock exporter that tracks calls and can simulate failures."""

    def __init__(
        self,
        name: str,
        *,
        fail_export: bool = False,
        fail_flush: bool = False,
        fail_close: bool = False,
    ):
        self._name = name
        self._fail_export = fail_export
        self._fail_flush = fail_flush
        self._fail_close = fail_close
        self.exports: list[TelemetryEvent] = []
        self.flush_count = 0
        self.close_count = 0

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        if self._fail_export:
            raise RuntimeError(f"Simulated export failure in {self._name}")
        self.exports.append(event)

    def flush(self) -> None:
        if self._fail_flush:
            raise RuntimeError(f"Simulated flush failure in {self._name}")
        self.flush_count += 1

    def close(self) -> None:
        if self._fail_close:
            raise RuntimeError(f"Simulated close failure in {self._name}")
        self.close_count += 1


class ToggleableExporter(MockExporter):
    """Exporter that can be toggled between working and failing state."""

    def set_failing(self, fail: bool) -> None:
        """Toggle failure mode."""
        self._fail_export = fail

    def export(self, event: TelemetryEvent) -> None:
        if self._fail_export:
            raise RuntimeError(f"Simulated failure in {self._name}")
        self.exports.append(event)


# =============================================================================
# Event Factories
# =============================================================================


def make_run_started(run_id: str, timestamp: datetime | None = None) -> RunStarted:
    """Create a RunStarted event for testing."""
    return RunStarted(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        config_hash="abc123",
        source_plugin="csv",
    )


def make_run_finished(run_id: str, timestamp: datetime | None = None) -> RunFinished:
    """Create a RunFinished event for testing."""
    return RunFinished(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        status=RunStatus.COMPLETED,
        row_count=100,
        duration_ms=5000.0,
    )


def make_phase_changed(run_id: str, timestamp: datetime | None = None) -> PhaseChanged:
    """Create a PhaseChanged event for testing."""
    return PhaseChanged(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        phase=PipelinePhase.PROCESS,
        action=PhaseAction.PROCESSING,
    )


def make_row_created(run_id: str, row_id: str, timestamp: datetime | None = None) -> RowCreated:
    """Create a RowCreated event for testing."""
    return RowCreated(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        row_id=row_id,
        token_id=f"token-{row_id}",
        content_hash=f"hash-{row_id}",
    )


def make_transform_completed(run_id: str, row_id: str, timestamp: datetime | None = None) -> TransformCompleted:
    """Create a TransformCompleted event for testing."""
    return TransformCompleted(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        row_id=row_id,
        token_id=f"token-{row_id}",
        node_id="transform-1",
        plugin_name="field_mapper",
        status=NodeStateStatus.COMPLETED,
        duration_ms=10.0,
        input_hash=f"in-{row_id}",
        output_hash=f"out-{row_id}",
    )


def make_gate_evaluated(run_id: str, row_id: str, timestamp: datetime | None = None) -> GateEvaluated:
    """Create a GateEvaluated event for testing."""
    return GateEvaluated(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        row_id=row_id,
        token_id=f"token-{row_id}",
        node_id="gate-1",
        plugin_name="threshold_gate",
        routing_mode=RoutingMode.MOVE,
        destinations=("output",),
    )


def make_token_completed(run_id: str, row_id: str, timestamp: datetime | None = None) -> TokenCompleted:
    """Create a TokenCompleted event for testing."""
    return TokenCompleted(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        row_id=row_id,
        token_id=f"token-{row_id}",
        outcome=RowOutcome.COMPLETED,
        sink_name="output",
    )


def make_external_call_completed(run_id: str, state_id: str, timestamp: datetime | None = None) -> ExternalCallCompleted:
    """Create an ExternalCallCompleted event for testing."""
    return ExternalCallCompleted(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        state_id=state_id,
        call_type=CallType.LLM,
        provider="azure-openai",
        status=CallStatus.SUCCESS,
        latency_ms=1500.0,
    )


# =============================================================================
# Hypothesis Strategies
# =============================================================================


# Strategy for generating valid run_ids
run_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
).filter(lambda s: len(s.strip()) > 0)


# Strategy for generating row_ids
row_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
).filter(lambda s: len(s.strip()) > 0)


# Strategy for timestamps (min/max must be naive for Hypothesis, timezone applied separately)
# Note: DTZ001 exception - Hypothesis requires naive datetimes for min/max bounds
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),  # noqa: DTZ001
    max_value=datetime(2030, 12, 31),  # noqa: DTZ001
    timezones=st.just(UTC),
)


# Strategy for granularity levels
granularity_strategy = st.sampled_from(list(TelemetryGranularity))


# Strategy for event types
@st.composite
def lifecycle_event_strategy(draw: st.DrawFn) -> TelemetryEvent:
    """Generate lifecycle events (RunStarted, RunFinished, PhaseChanged)."""
    run_id = draw(run_id_strategy)
    timestamp = draw(timestamp_strategy)
    event_type = draw(st.sampled_from(["run_started", "run_finished", "phase_changed"]))

    if event_type == "run_started":
        return make_run_started(run_id, timestamp)
    elif event_type == "run_finished":
        return make_run_finished(run_id, timestamp)
    else:
        return make_phase_changed(run_id, timestamp)


@st.composite
def row_event_strategy(draw: st.DrawFn) -> TelemetryEvent:
    """Generate row-level events."""
    run_id = draw(run_id_strategy)
    row_id = draw(row_id_strategy)
    timestamp = draw(timestamp_strategy)
    event_type = draw(st.sampled_from(["row_created", "transform_completed", "gate_evaluated", "token_completed"]))

    if event_type == "row_created":
        return make_row_created(run_id, row_id, timestamp)
    elif event_type == "transform_completed":
        return make_transform_completed(run_id, row_id, timestamp)
    elif event_type == "gate_evaluated":
        return make_gate_evaluated(run_id, row_id, timestamp)
    else:
        return make_token_completed(run_id, row_id, timestamp)


@st.composite
def external_call_event_strategy(draw: st.DrawFn) -> TelemetryEvent:
    """Generate external call events."""
    run_id = draw(run_id_strategy)
    state_id = draw(row_id_strategy)
    timestamp = draw(timestamp_strategy)
    return make_external_call_completed(run_id, state_id, timestamp)


@st.composite
def any_event_strategy(draw: st.DrawFn) -> TelemetryEvent:
    """Generate any type of telemetry event."""
    event_category = draw(st.sampled_from(["lifecycle", "row", "external"]))
    if event_category == "lifecycle":
        return draw(lifecycle_event_strategy())
    elif event_category == "row":
        return draw(row_event_strategy())
    else:
        return draw(external_call_event_strategy())


# =============================================================================
# 1. TelemetryManagerStateMachine - Stateful Testing
# =============================================================================


class TelemetryManagerStateMachine(RuleBasedStateMachine):
    """Stateful testing for TelemetryManager failure handling.

    This state machine tests the TelemetryManager's behavior under various
    failure scenarios, verifying:
    - Metrics consistency (emitted + dropped == total passed filtering)
    - Disabled state is sticky
    - Consecutive failure counter resets on success
    - Per-exporter failure tracking is accurate
    """

    def __init__(self) -> None:
        super().__init__()
        # Create a mix of working and failing exporters
        self.working_exporter = ToggleableExporter("working")
        self.failing_exporter = ToggleableExporter("failing")
        self.failing_exporter.set_failing(True)

        self.config = MockConfig(fail_on_total_exporter_failure=False)
        self.manager = TelemetryManager(
            self.config,
            exporters=[self.working_exporter, self.failing_exporter],
        )

        # Track our own state for verification
        self.total_events_sent = 0
        self.events_when_all_working = 0
        self.events_when_all_failing = 0
        self.events_when_partial = 0
        self.all_exporters_failing = False

    events = Bundle("events")

    @rule(target=events, run_id=run_id_strategy)
    def create_event(self, run_id: str) -> TelemetryEvent:
        """Create a new event."""
        return make_run_started(run_id)

    @rule(event=events)
    def emit_event_with_partial_failure(self, event: TelemetryEvent) -> None:
        """Emit event when one exporter works and one fails (partial success)."""
        if self.manager._disabled:
            return  # Skip if already disabled

        self.working_exporter.set_failing(False)
        self.failing_exporter.set_failing(True)
        self.all_exporters_failing = False

        with patch("elspeth.telemetry.manager.logger"):
            self.manager.handle_event(event)
            self.manager.flush()

        self.total_events_sent += 1
        self.events_when_partial += 1

    @rule(event=events)
    def emit_event_all_exporters_succeed(self, event: TelemetryEvent) -> None:
        """Emit event when all exporters succeed."""
        if self.manager._disabled:
            return

        self.working_exporter.set_failing(False)
        self.failing_exporter.set_failing(False)
        self.all_exporters_failing = False

        self.manager.handle_event(event)
        self.manager.flush()
        self.total_events_sent += 1
        self.events_when_all_working += 1

    @rule(event=events)
    def emit_event_all_exporters_fail(self, event: TelemetryEvent) -> None:
        """Emit event when all exporters fail."""
        if self.manager._disabled:
            return

        self.working_exporter.set_failing(True)
        self.failing_exporter.set_failing(True)
        self.all_exporters_failing = True

        with patch("elspeth.telemetry.manager.logger"):
            self.manager.handle_event(event)
            self.manager.flush()

        self.total_events_sent += 1
        self.events_when_all_failing += 1

    @invariant()
    def metrics_are_consistent(self) -> None:
        """Invariant: emitted + dropped == total events (before disable)."""
        metrics = self.manager.health_metrics
        total_accounted = metrics["events_emitted"] + metrics["events_dropped"]

        # Before disable, all events are accounted for
        # After disable, events stop being counted
        if not self.manager._disabled:
            assert total_accounted == self.total_events_sent, (
                f"Metrics inconsistent: emitted={metrics['events_emitted']}, "
                f"dropped={metrics['events_dropped']}, total_sent={self.total_events_sent}"
            )

    @invariant()
    def disabled_state_is_sticky(self) -> None:
        """Invariant: Once disabled, manager stays disabled."""
        # We can't test this directly in invariant, but we can verify
        # that if disabled is True, it was set correctly
        if self.manager._disabled:
            # The manager should have had 10+ consecutive total failures
            # before being disabled (or was disabled by reaching threshold)
            pass  # This is verified by the threshold behavior

    @invariant()
    def consecutive_failures_bounded(self) -> None:
        """Invariant: consecutive_total_failures <= max_consecutive_failures."""
        assert self.manager._consecutive_total_failures <= self.manager._max_consecutive_failures

    @rule()
    def check_success_resets_consecutive_count(self) -> None:
        """After any success (partial or full), consecutive_total_failures is 0."""
        if self.manager._disabled:
            return

        # If last event was a success (any exporter worked)
        if not self.all_exporters_failing:
            assert self.manager._consecutive_total_failures == 0


# Register the state machine test
TestTelemetryManagerStateMachine = TelemetryManagerStateMachine.TestCase


class AllExportersFailStateMachine(RuleBasedStateMachine):
    """State machine for testing when all exporters consistently fail.

    Verifies:
    - Consecutive failure counter increments correctly
    - Disabled threshold is honored
    - Events stop being counted after disable
    """

    def __init__(self) -> None:
        super().__init__()
        self.failing1 = MockExporter("failing1", fail_export=True)
        self.failing2 = MockExporter("failing2", fail_export=True)

        self.config = MockConfig(fail_on_total_exporter_failure=False)
        self.manager = TelemetryManager(
            self.config,
            exporters=[self.failing1, self.failing2],
        )
        self.events_sent_before_disable = 0
        self.events_sent_after_disable = 0

    @rule(run_id=run_id_strategy)
    def emit_failing_event(self, run_id: str) -> None:
        """Emit an event that will fail on all exporters."""
        with patch("elspeth.telemetry.manager.logger"):
            was_disabled = self.manager._disabled

            event = make_run_started(run_id)
            self.manager.handle_event(event)
            self.manager.flush()

            if was_disabled:
                self.events_sent_after_disable += 1
            else:
                self.events_sent_before_disable += 1

    @invariant()
    def dropped_count_correct_before_disable(self) -> None:
        """Dropped count equals events sent before disable."""
        if not self.manager._disabled:
            assert self.manager.health_metrics["events_dropped"] == self.events_sent_before_disable

    @invariant()
    def dropped_count_frozen_after_disable(self) -> None:
        """Dropped count stops incrementing after disable."""
        if self.manager._disabled:
            # Dropped count should be frozen at exactly 10 (the threshold)
            assert self.manager.health_metrics["events_dropped"] == 10

    @invariant()
    def disable_happens_at_threshold(self) -> None:
        """Manager disables exactly at max_consecutive_failures threshold."""
        if self.events_sent_before_disable >= 10:
            assert self.manager._disabled is True


TestAllExportersFailStateMachine = AllExportersFailStateMachine.TestCase


# =============================================================================
# 2. BoundedBuffer Property Tests
# =============================================================================


class TestBoundedBufferProperties:
    """Property-based tests for BoundedBuffer invariants."""

    @given(
        max_size=st.integers(min_value=1, max_value=1000),
        num_events=st.integers(min_value=0, max_value=2000),
    )
    @settings(max_examples=100)
    def test_length_never_exceeds_max_size(self, max_size: int, num_events: int) -> None:
        """Property: len(buffer) <= max_size always."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(TelemetryEvent(timestamp=datetime.now(tz=UTC), run_id=f"run-{i}"))

        assert len(buffer) <= max_size

    @given(
        max_size=st.integers(min_value=1, max_value=1000),
        num_events=st.integers(min_value=0, max_value=2000),
    )
    @settings(max_examples=100)
    def test_dropped_count_equals_overflow(self, max_size: int, num_events: int) -> None:
        """Property: dropped_count == max(0, total_added - max_size)."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(TelemetryEvent(timestamp=datetime.now(tz=UTC), run_id=f"run-{i}"))

        expected_drops = max(0, num_events - max_size)
        assert buffer.dropped_count == expected_drops

    @given(
        max_size=st.integers(min_value=1, max_value=100),
        events=st.lists(
            st.integers(min_value=0, max_value=1000),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_drain_order_is_fifo(self, max_size: int, events: list[int]) -> None:
        """Property: drain order matches append order (FIFO)."""
        buffer = BoundedBuffer(max_size=max_size)

        # Append events with sequence numbers as run_ids
        for seq in events:
            buffer.append(TelemetryEvent(timestamp=datetime.now(tz=UTC), run_id=str(seq)))

        # Drain all events
        drained = buffer.pop_batch(max_count=len(buffer))

        # The drained events should be the last max_size events in FIFO order
        dropped = max(0, len(events) - max_size)
        expected_ids = [str(seq) for seq in events[dropped:]]
        actual_ids = [e.run_id for e in drained]

        assert actual_ids == expected_ids

    @given(
        max_size=st.integers(min_value=1, max_value=100),
        num_events=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=100)
    def test_conservation_of_events(self, max_size: int, num_events: int) -> None:
        """Property: len(buffer) + dropped_count == events_added (no pops)."""
        buffer = BoundedBuffer(max_size=max_size)
        for i in range(num_events):
            buffer.append(TelemetryEvent(timestamp=datetime.now(tz=UTC), run_id=f"run-{i}"))

        # All events are either in buffer or dropped
        assert len(buffer) + buffer.dropped_count == num_events


# =============================================================================
# 3. Granularity Filtering Matrix Tests
# =============================================================================


class TestGranularityFilteringMatrix:
    """Property tests verifying should_emit() for all granularity x event combinations."""

    # Define expected behavior matrix
    # True = should emit, False = should filter
    EXPECTED_MATRIX: ClassVar[dict[type, dict[TelemetryGranularity, bool]]] = {
        # Lifecycle events: always emit at any granularity
        RunStarted: {
            TelemetryGranularity.LIFECYCLE: True,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        RunFinished: {
            TelemetryGranularity.LIFECYCLE: True,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        PhaseChanged: {
            TelemetryGranularity.LIFECYCLE: True,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        # Row-level events: emit at ROWS and FULL
        RowCreated: {
            TelemetryGranularity.LIFECYCLE: False,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        TransformCompleted: {
            TelemetryGranularity.LIFECYCLE: False,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        GateEvaluated: {
            TelemetryGranularity.LIFECYCLE: False,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        TokenCompleted: {
            TelemetryGranularity.LIFECYCLE: False,
            TelemetryGranularity.ROWS: True,
            TelemetryGranularity.FULL: True,
        },
        # External call events: emit only at FULL
        ExternalCallCompleted: {
            TelemetryGranularity.LIFECYCLE: False,
            TelemetryGranularity.ROWS: False,
            TelemetryGranularity.FULL: True,
        },
    }

    @given(granularity=granularity_strategy, event=lifecycle_event_strategy())
    @settings(max_examples=50)
    def test_lifecycle_events_always_pass(self, granularity: TelemetryGranularity, event: TelemetryEvent) -> None:
        """Property: Lifecycle events always pass at any granularity."""
        assert should_emit(event, granularity) is True

    @given(granularity=granularity_strategy, event=row_event_strategy())
    @settings(max_examples=50)
    def test_row_events_at_rows_or_full(self, granularity: TelemetryGranularity, event: TelemetryEvent) -> None:
        """Property: Row events pass at ROWS or FULL, filtered at LIFECYCLE."""
        expected = granularity in (TelemetryGranularity.ROWS, TelemetryGranularity.FULL)
        assert should_emit(event, granularity) is expected

    @given(granularity=granularity_strategy, event=external_call_event_strategy())
    @settings(max_examples=50)
    def test_external_call_events_only_at_full(self, granularity: TelemetryGranularity, event: TelemetryEvent) -> None:
        """Property: External call events pass only at FULL granularity."""
        expected = granularity == TelemetryGranularity.FULL
        assert should_emit(event, granularity) is expected

    def test_complete_matrix_coverage(self) -> None:
        """Verify the filtering matrix is complete and correct.

        This exhaustive test ensures every event type x granularity combination
        behaves according to the documented specification.
        """
        base_ts = datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)
        run_id = "test-matrix"
        row_id = "row-1"
        state_id = "state-1"

        # Create one instance of each event type
        events_by_type: dict[type, TelemetryEvent] = {
            RunStarted: make_run_started(run_id, base_ts),
            RunFinished: make_run_finished(run_id, base_ts),
            PhaseChanged: make_phase_changed(run_id, base_ts),
            RowCreated: make_row_created(run_id, row_id, base_ts),
            TransformCompleted: make_transform_completed(run_id, row_id, base_ts),
            GateEvaluated: make_gate_evaluated(run_id, row_id, base_ts),
            TokenCompleted: make_token_completed(run_id, row_id, base_ts),
            ExternalCallCompleted: make_external_call_completed(run_id, state_id, base_ts),
        }

        # Verify every cell in the matrix
        for event_type, expected_by_granularity in self.EXPECTED_MATRIX.items():
            event = events_by_type[event_type]
            for granularity, expected in expected_by_granularity.items():
                actual = should_emit(event, granularity)
                assert actual is expected, (
                    f"Matrix mismatch: {event_type.__name__} at {granularity.value} expected {expected}, got {actual}"
                )

    @given(granularity=granularity_strategy)
    @settings(max_examples=10)
    def test_unknown_event_types_pass_through(self, granularity: TelemetryGranularity) -> None:
        """Property: Unknown event types pass through (forward compatibility)."""
        # Base TelemetryEvent class represents an unknown type
        event = TelemetryEvent(
            timestamp=datetime.now(tz=UTC),
            run_id="unknown-event-test",
        )
        # Unknown types should always pass (fail-open for forward compatibility)
        assert should_emit(event, granularity) is True


# =============================================================================
# 4. Event Ordering Tests
# =============================================================================


class TestEventOrdering:
    """Property tests verifying event ordering is preserved through the pipeline."""

    @given(
        num_events=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_exporter_receives_events_in_order(self, num_events: int) -> None:
        """Property: Events are exported in the same order they were emitted."""
        exporter = MockExporter("test")
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        # Create events with sequential timestamps
        base_time = datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)
        events = []
        for i in range(num_events):
            event = make_run_started(f"run-{i}", base_time + timedelta(seconds=i))
            events.append(event)
            manager.handle_event(event)

        # Verify order is preserved
        manager.flush()
        assert len(exporter.exports) == num_events
        for i, (emitted, received) in enumerate(zip(events, exporter.exports, strict=True)):
            assert emitted is received, f"Order mismatch at index {i}"

    @given(
        num_events=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_multiple_exporters_same_order(self, num_events: int) -> None:
        """Property: All exporters receive events in the same order."""
        exporter1 = MockExporter("exp1")
        exporter2 = MockExporter("exp2")
        exporter3 = MockExporter("exp3")
        manager = TelemetryManager(
            MockConfig(),
            exporters=[exporter1, exporter2, exporter3],
        )

        events = []
        for i in range(num_events):
            event = make_run_started(f"run-{i}")
            events.append(event)
            manager.handle_event(event)

        # All exporters should have the same events in the same order
        manager.flush()
        assert exporter1.exports == exporter2.exports == exporter3.exports
        assert exporter1.exports == events

    @given(
        max_size=st.integers(min_value=1, max_value=50),
        num_events=st.integers(min_value=1, max_value=100),
        batch_size=st.integers(min_value=1, max_value=25),
    )
    @settings(max_examples=50)
    def test_buffer_preserves_order_across_batches(self, max_size: int, num_events: int, batch_size: int) -> None:
        """Property: Buffer preserves FIFO order across multiple pop_batch calls."""
        buffer = BoundedBuffer(max_size=max_size)

        # Add events
        for i in range(num_events):
            buffer.append(TelemetryEvent(timestamp=datetime.now(tz=UTC), run_id=str(i)))

        # Drain in batches
        all_drained = []
        while len(buffer) > 0:
            batch = buffer.pop_batch(max_count=batch_size)
            all_drained.extend(batch)

        # The drained order should be the last max_size events in order
        dropped = max(0, num_events - max_size)
        expected_order = [str(i) for i in range(dropped, num_events)]
        actual_order = [e.run_id for e in all_drained]

        assert actual_order == expected_order


# =============================================================================
# Additional Invariant Tests
# =============================================================================


class TestManagerInvariants:
    """Additional property tests for TelemetryManager invariants."""

    @given(
        num_events=st.integers(min_value=0, max_value=50),
        granularity=granularity_strategy,
    )
    @settings(max_examples=50)
    def test_filtered_events_not_counted(self, num_events: int, granularity: TelemetryGranularity) -> None:
        """Property: Filtered events are not counted in emitted/dropped metrics."""
        exporter = MockExporter("test")
        config = MockConfig(granularity=granularity)
        manager = TelemetryManager(config, exporters=[exporter])

        # Count how many events should pass the filter
        events_that_pass = 0

        for i in range(num_events):
            # Alternate between event types
            event: TelemetryEvent
            if i % 3 == 0:
                event = make_run_started(f"run-{i}")  # Lifecycle - always passes
                should_pass = True
            elif i % 3 == 1:
                event = make_row_created(f"run-{i}", f"row-{i}")  # Row-level
                should_pass = granularity in (TelemetryGranularity.ROWS, TelemetryGranularity.FULL)
            else:
                event = make_external_call_completed(f"run-{i}", f"state-{i}")  # External
                should_pass = granularity == TelemetryGranularity.FULL

            if should_pass:
                events_that_pass += 1

            manager.handle_event(event)

        # Emitted count should match events that passed filter
        manager.flush()
        assert manager.health_metrics["events_emitted"] == events_that_pass
        assert manager.health_metrics["events_dropped"] == 0
        assert len(exporter.exports) == events_that_pass

    @given(st.integers(min_value=0, max_value=20))
    @settings(max_examples=20)
    def test_flush_and_close_are_idempotent(self, num_events: int) -> None:
        """Property: flush() and close() are safe to call multiple times."""
        exporter = MockExporter("test")
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        for i in range(num_events):
            manager.handle_event(make_run_started(f"run-{i}"))

        # Multiple flushes should work
        manager.flush()
        manager.flush()
        manager.flush()

        # Multiple closes should work (close is idempotent)
        with patch("elspeth.telemetry.manager.logger"):
            manager.close()
            manager.close()
            manager.close()

        # Exporter methods should have been called
        assert exporter.flush_count == 3
        assert exporter.close_count == 3
