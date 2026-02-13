# tests/property/telemetry/test_emit_completeness.py
"""Property-based tests for telemetry emit completeness.

Per CLAUDE.md: "Any telemetry emission point MUST either send what it has
OR explicitly acknowledge 'I have nothing' (with failure reason if applicable).
Never silently swallow events or exceptions."

These tests verify:
1. Granularity filtering is monotonic (FULL >= ROWS >= LIFECYCLE)
2. Every event type has a defined filtering behavior (no unknown gaps)
3. TelemetryManager never silently drops events - every event is either
   emitted, filtered by granularity, or accounted for in health_metrics
4. Health metrics are consistent: emitted + dropped accounts for all dispatched events
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.enums import (
    BackpressureMode,
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
from elspeth.telemetry.manager import TelemetryManager

# =============================================================================
# Strategies for generating telemetry events
# =============================================================================

# Fixed timestamp for non-Hypothesis tests (e.g., FutureEvent in forward compat)
_REFERENCE_TIMESTAMP = datetime(2025, 1, 1, tzinfo=UTC)

# Reusable strategies — vary all fields to defend against future regressions
# where filtering might accidentally become value-dependent.
run_ids = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))
timestamps = st.datetimes(
    min_value=datetime(2020, 1, 1),  # noqa: DTZ001 — Hypothesis requires naive bounds
    max_value=datetime(2030, 1, 1),  # noqa: DTZ001
    timezones=st.just(UTC),
)
_short_ids = st.text(min_size=1, max_size=12, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_")
_hash_strings = st.text(min_size=8, max_size=32, alphabet="0123456789abcdef")
_plugin_names = st.sampled_from(["csv", "json", "database", "api", "llm_classifier", "passthrough"])
_sink_names = st.sampled_from(["output", "quarantine", "archive", "default", "errors"])

# Lifecycle events (always emitted at any granularity)
lifecycle_events = st.one_of(
    st.builds(
        RunStarted,
        timestamp=timestamps,
        run_id=run_ids,
        config_hash=_hash_strings,
        source_plugin=_plugin_names,
    ),
    st.builds(
        RunFinished,
        timestamp=timestamps,
        run_id=run_ids,
        status=st.sampled_from(list(RunStatus)),
        row_count=st.integers(min_value=0, max_value=1000),
        duration_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ),
    st.builds(
        PhaseChanged,
        timestamp=timestamps,
        run_id=run_ids,
        phase=st.sampled_from(list(PipelinePhase)),
        action=st.sampled_from(list(PhaseAction)),
    ),
)

# Row-level events (emitted at ROWS and FULL)
row_events = st.one_of(
    st.builds(
        RowCreated,
        timestamp=timestamps,
        run_id=run_ids,
        row_id=_short_ids,
        token_id=_short_ids,
        content_hash=_hash_strings,
    ),
    st.builds(
        TransformCompleted,
        timestamp=timestamps,
        run_id=run_ids,
        row_id=_short_ids,
        token_id=_short_ids,
        node_id=_short_ids,
        plugin_name=_plugin_names,
        status=st.sampled_from(list(NodeStateStatus)),
        duration_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        input_hash=_hash_strings,
        output_hash=_hash_strings,
    ),
    st.builds(
        GateEvaluated,
        timestamp=timestamps,
        run_id=run_ids,
        row_id=_short_ids,
        token_id=_short_ids,
        node_id=_short_ids,
        plugin_name=_plugin_names,
        routing_mode=st.sampled_from(list(RoutingMode)),
        destinations=st.tuples(_sink_names),
    ),
    st.builds(
        TokenCompleted,
        timestamp=timestamps,
        run_id=run_ids,
        row_id=_short_ids,
        token_id=_short_ids,
        outcome=st.sampled_from(list(RowOutcome)),
        sink_name=_sink_names,
    ),
    st.builds(
        FieldResolutionApplied,
        timestamp=timestamps,
        run_id=run_ids,
        source_plugin=_plugin_names,
        field_count=st.integers(min_value=1, max_value=3),
        normalization_version=st.one_of(st.none(), st.sampled_from(["v1", "v2"])),
        resolution_mapping=st.sampled_from(
            [
                {"Customer ID": "customer_id"},
                {"Order Amount": "order_amount", "Order Date": "order_date"},
                {"E-mail": "e_mail", "ZIP": "zip"},
            ]
        ),
    ),
)

# External call events (emitted only at FULL)
# ExternalCallCompleted has XOR constraint: exactly one of state_id or operation_id.
external_call_events = st.one_of(
    # Transform context (has state_id, no operation_id)
    st.builds(
        ExternalCallCompleted,
        timestamp=timestamps,
        run_id=run_ids,
        call_type=st.sampled_from(list(CallType)),
        provider=st.sampled_from(["azure-openai", "openrouter", "litellm", "anthropic"]),
        status=st.sampled_from(list(CallStatus)),
        latency_ms=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        state_id=_short_ids,
        operation_id=st.none(),
    ),
    # Operation context (has operation_id, no state_id)
    st.builds(
        ExternalCallCompleted,
        timestamp=timestamps,
        run_id=run_ids,
        call_type=st.sampled_from(list(CallType)),
        provider=st.sampled_from(["azure-openai", "openrouter", "litellm", "anthropic"]),
        status=st.sampled_from(list(CallStatus)),
        latency_ms=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        state_id=st.none(),
        operation_id=_short_ids,
    ),
)

# All events strategy
all_events = st.one_of(lifecycle_events, row_events, external_call_events)

# Granularity levels
granularity_levels = st.sampled_from(list(TelemetryGranularity))


# =============================================================================
# Helpers
# =============================================================================


@dataclass(frozen=True)
class _FakeTelemetryConfig:
    """Minimal config that satisfies RuntimeTelemetryProtocol."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    backpressure_mode: BackpressureMode = BackpressureMode.DROP
    fail_on_total_exporter_failure: bool = False
    max_consecutive_failures: int = 10
    exporter_configs: tuple[Any, ...] = ()


class _TrackingExporter:
    """Exporter that tracks what was exported."""

    def __init__(self, should_fail: bool = False) -> None:
        self.exported: list[TelemetryEvent] = []
        self.should_fail = should_fail
        self._name = "tracking"

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        if self.should_fail:
            raise RuntimeError("Exporter failure")
        self.exported.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# Granularity Filtering Properties
# =============================================================================


class TestGranularityFilteringProperties:
    """Property tests for granularity-based event filtering."""

    @given(event=all_events, granularity=granularity_levels)
    @settings(max_examples=200)
    def test_filtering_is_deterministic(self, event: TelemetryEvent, granularity: TelemetryGranularity) -> None:
        """Property: Same event + same granularity = same filtering decision, always."""
        result1 = should_emit(event, granularity)
        result2 = should_emit(event, granularity)
        result3 = should_emit(event, granularity)

        assert result1 == result2 == result3, "Filtering decision is non-deterministic!"

    @given(event=lifecycle_events, granularity=granularity_levels)
    @settings(max_examples=100)
    def test_lifecycle_events_always_emitted(self, event: TelemetryEvent, granularity: TelemetryGranularity) -> None:
        """Property: Lifecycle events (RunStarted, RunFinished, PhaseChanged) are
        emitted at ALL granularity levels.

        This is the baseline guarantee - you always know when a run starts/ends.
        """
        assert should_emit(event, granularity) is True, f"Lifecycle event {type(event).__name__} was filtered at granularity={granularity}"

    @given(event=row_events)
    @settings(max_examples=100)
    def test_row_events_emitted_at_rows_and_full(self, event: TelemetryEvent) -> None:
        """Property: Row-level events are emitted at ROWS and FULL, not LIFECYCLE."""
        assert should_emit(event, TelemetryGranularity.LIFECYCLE) is False
        assert should_emit(event, TelemetryGranularity.ROWS) is True
        assert should_emit(event, TelemetryGranularity.FULL) is True

    @given(event=external_call_events)
    @settings(max_examples=50)
    def test_external_call_events_emitted_only_at_full(self, event: TelemetryEvent) -> None:
        """Property: External call events are emitted only at FULL granularity."""
        assert should_emit(event, TelemetryGranularity.LIFECYCLE) is False
        assert should_emit(event, TelemetryGranularity.ROWS) is False
        assert should_emit(event, TelemetryGranularity.FULL) is True

    @given(event=all_events)
    @settings(max_examples=200)
    def test_granularity_is_monotonic(self, event: TelemetryEvent) -> None:
        """Property: If an event is emitted at a lower granularity, it is also
        emitted at all higher granularities.

        LIFECYCLE <= ROWS <= FULL (monotonically increasing visibility)
        """
        emit_lifecycle = should_emit(event, TelemetryGranularity.LIFECYCLE)
        emit_rows = should_emit(event, TelemetryGranularity.ROWS)
        emit_full = should_emit(event, TelemetryGranularity.FULL)

        # If emitted at LIFECYCLE, must be emitted at ROWS and FULL
        if emit_lifecycle:
            assert emit_rows, f"Event {type(event).__name__} emitted at LIFECYCLE but not ROWS"
            assert emit_full, f"Event {type(event).__name__} emitted at LIFECYCLE but not FULL"

        # If emitted at ROWS, must be emitted at FULL
        if emit_rows:
            assert emit_full, f"Event {type(event).__name__} emitted at ROWS but not FULL"

    @given(event=all_events)
    @settings(max_examples=100)
    def test_full_granularity_emits_everything(self, event: TelemetryEvent) -> None:
        """Property: FULL granularity never filters any known event type."""
        assert should_emit(event, TelemetryGranularity.FULL) is True, f"FULL granularity filtered event {type(event).__name__}"


# =============================================================================
# TelemetryManager Emit Completeness Properties
# =============================================================================


class TestManagerEmitCompletenessProperties:
    """Property tests for TelemetryManager event accounting.

    These verify the "no silent drops" guarantee: every event that passes
    filtering is either emitted to exporters or accounted for in health_metrics.
    """

    @given(events=st.lists(all_events, min_size=1, max_size=20))
    @settings(max_examples=50, deadline=10000)
    def test_successful_events_counted(self, events: list[TelemetryEvent]) -> None:
        """Property: Events sent to a healthy exporter are counted as emitted."""
        exporter = _TrackingExporter()
        config = _FakeTelemetryConfig(granularity=TelemetryGranularity.FULL)
        manager = TelemetryManager(config, exporters=[exporter])

        try:
            for event in events:
                manager.handle_event(event)

            manager.flush()
            metrics = manager.health_metrics

            # All events should be emitted (FULL granularity, healthy exporter)
            assert metrics["events_emitted"] == len(events), f"Expected {len(events)} emitted, got {metrics['events_emitted']}"
            assert metrics["events_dropped"] == 0
            assert len(exporter.exported) == len(events)
        finally:
            manager.close()

    @given(events=st.lists(all_events, min_size=1, max_size=20))
    @settings(max_examples=50, deadline=10000)
    def test_failed_events_counted_as_dropped(self, events: list[TelemetryEvent]) -> None:
        """Property: Events that fail all exporters are counted as dropped."""
        exporter = _TrackingExporter(should_fail=True)
        config = _FakeTelemetryConfig(granularity=TelemetryGranularity.FULL)
        manager = TelemetryManager(config, exporters=[exporter])

        try:
            for event in events:
                manager.handle_event(event)

            manager.flush()
            metrics = manager.health_metrics

            # All events failed - they should be dropped
            assert metrics["events_dropped"] == len(events), f"Expected {len(events)} dropped, got {metrics['events_dropped']}"
            assert metrics["events_emitted"] == 0
        finally:
            manager.close()

    @given(events=st.lists(all_events, min_size=1, max_size=10))
    @settings(max_examples=30, deadline=10000)
    def test_no_exporters_means_no_emission(self, events: list[TelemetryEvent]) -> None:
        """Property: With no exporters, events are silently skipped (by design).

        This is the one valid case of "no action" - when no exporters are
        configured, telemetry is effectively disabled. The manager short-circuits
        in handle_event() before queuing.
        """
        config = _FakeTelemetryConfig(granularity=TelemetryGranularity.FULL)
        manager = TelemetryManager(config, exporters=[])

        try:
            for event in events:
                manager.handle_event(event)

            manager.flush()
            metrics = manager.health_metrics

            # No exporters = no emission, no drops (short-circuited)
            assert metrics["events_emitted"] == 0
            assert metrics["events_dropped"] == 0
        finally:
            manager.close()

    @given(
        events=st.lists(all_events, min_size=1, max_size=10),
        granularity=granularity_levels,
    )
    @settings(max_examples=50, deadline=10000)
    def test_filtered_events_not_counted(self, events: list[TelemetryEvent], granularity: TelemetryGranularity) -> None:
        """Property: Events filtered by granularity don't appear in health metrics.

        Filtered events are not queued, so they don't count as emitted or dropped.
        Only events that pass the granularity filter are tracked.
        """
        exporter = _TrackingExporter()
        config = _FakeTelemetryConfig(granularity=granularity)
        manager = TelemetryManager(config, exporters=[exporter])

        try:
            for event in events:
                manager.handle_event(event)

            manager.flush()
            metrics = manager.health_metrics

            # Count how many events should pass the filter
            expected_passed = sum(1 for e in events if should_emit(e, granularity))

            # Emitted should match passed (healthy exporter)
            assert metrics["events_emitted"] == expected_passed, (
                f"Granularity {granularity}: expected {expected_passed} emitted, "
                f"got {metrics['events_emitted']} (from {len(events)} total events)"
            )
            assert metrics["events_dropped"] == 0
            assert len(exporter.exported) == expected_passed
        finally:
            manager.close()


# =============================================================================
# Unknown Event Type Properties
# =============================================================================


class TestUnknownEventForwardCompatibility:
    """Property tests for unknown event type handling."""

    @given(granularity=granularity_levels)
    @settings(max_examples=20)
    def test_unknown_event_types_pass_through(self, granularity: TelemetryGranularity) -> None:
        """Property: Unknown event subclasses pass through filtering (fail-open).

        Per filtering.py's default case: unknown event types are emitted at
        all granularity levels. This ensures forward compatibility when new
        event types are added.
        """

        @dataclass(frozen=True)
        class FutureEvent(TelemetryEvent):
            """Hypothetical future event type."""

            custom_field: str = "test"

        event = FutureEvent(
            timestamp=_REFERENCE_TIMESTAMP,
            run_id="test_run",
            custom_field="value",
        )

        assert should_emit(event, granularity) is True, f"Unknown event type FutureEvent was filtered at granularity={granularity}"
