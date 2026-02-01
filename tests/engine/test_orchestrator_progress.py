# tests/engine/test_orchestrator_progress.py
"""Tests for Orchestrator progress callback functionality.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from elspeth.contracts import SourceRow
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    pass


class TestOrchestratorProgress:
    """Tests for progress callback functionality."""

    def test_progress_callback_called_every_100_rows(self, payload_store) -> None:
        """Verify progress callback is called at 100, 200, and 250 row marks."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, ProgressEvent, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class MultiRowSource(_TestSourceBase):
            """Source that yields N rows for progress testing."""

            name = "multi_row_source"
            output_schema = ValueSchema

            def __init__(self, count: int) -> None:
                self._count = count

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for i in range(self._count):
                    yield SourceRow.valid({"value": i})

        class CollectSink(_TestSinkBase):
            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

        # Create 250-row source
        source = MultiRowSource(count=250)
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        # Track progress events using EventBus
        from elspeth.core import EventBus

        progress_events: list[ProgressEvent] = []

        def track_progress(event: ProgressEvent) -> None:
            progress_events.append(event)

        event_bus = EventBus()
        event_bus.subscribe(ProgressEvent, track_progress)

        orchestrator = Orchestrator(db, event_bus=event_bus)
        orchestrator.run(
            config,
            graph=build_production_graph(config),
            payload_store=payload_store,
        )

        # P1 Fix: Relax exact count assertion - orchestrator also emits on 5-second intervals
        # which can cause extra events on slow machines. Assert required checkpoints exist.
        # Required checkpoints: row 1 (first), 100, 200, and 250 (final)
        assert len(progress_events) >= 4, f"Expected at least 4 progress events (1, 100, 200, 250), got {len(progress_events)}"

        # Extract rows_processed values for verification
        rows_at_events = [e.rows_processed for e in progress_events]

        # Verify required checkpoints exist (may have extra time-based events)
        assert 1 in rows_at_events, "Missing first row progress event"
        assert 100 in rows_at_events, "Missing 100-row checkpoint"
        assert 200 in rows_at_events, "Missing 200-row checkpoint"
        assert 250 in rows_at_events, "Missing final row (250) progress event"

        # Verify ordering: rows_processed should be monotonically increasing
        for i in range(1, len(progress_events)):
            assert progress_events[i].rows_processed >= progress_events[i - 1].rows_processed, (
                f"Progress events not monotonically increasing: "
                f"{progress_events[i - 1].rows_processed} -> {progress_events[i].rows_processed}"
            )

        # Verify timing is recorded
        assert all(e.elapsed_seconds >= 0 for e in progress_events)
        # Elapsed should be monotonically increasing
        for i in range(1, len(progress_events)):
            assert progress_events[i].elapsed_seconds >= progress_events[i - 1].elapsed_seconds, "Elapsed time not monotonically increasing"

    def test_progress_callback_not_called_when_none(self, payload_store) -> None:
        """Verify no crash when on_progress is None."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class SmallSource(_TestSourceBase):
            name = "small_source"
            output_schema = ValueSchema

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for i in range(50):
                    yield SourceRow.valid({"value": i})

        class CollectSink(_TestSinkBase):
            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

        source = SmallSource()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        # Run without progress callback - should not crash
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.rows_processed == 50

    def test_progress_callback_fires_for_quarantined_rows(self, payload_store) -> None:
        """Verify progress callback fires even when rows are quarantined.

        Regression test: progress emission was placed after the quarantine
        continue, so quarantined rows at 100-row boundaries never triggered
        progress updates.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, ProgressEvent, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class QuarantineAtBoundarySource(_TestSourceBase):
            """Source that quarantines specifically at 100-row boundary."""

            name = "quarantine_boundary_source"
            output_schema = ValueSchema

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for i in range(150):
                    if i == 99:  # Row 100 (0-indexed 99) is quarantined
                        yield SourceRow.quarantined(
                            row={"value": i},
                            error="test_quarantine_at_boundary",
                            destination="quarantine",
                        )
                    else:
                        yield SourceRow.valid({"value": i})

        class CollectSink(_TestSinkBase):
            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

        source = QuarantineAtBoundarySource()
        default_sink = CollectSink()
        quarantine_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        # Track progress events using EventBus
        from elspeth.core import EventBus

        progress_events: list[ProgressEvent] = []

        def track_progress(event: ProgressEvent) -> None:
            progress_events.append(event)

        event_bus = EventBus()
        event_bus.subscribe(ProgressEvent, track_progress)

        orchestrator = Orchestrator(db, event_bus=event_bus)
        orchestrator.run(
            config,
            graph=build_production_graph(config),
            payload_store=payload_store,
        )

        # P1 Fix: Relax exact count assertion - orchestrator also emits on 5-second intervals
        # Required checkpoints: row 1 (first), 100, and final 150
        assert len(progress_events) >= 3, f"Expected at least 3 progress events (1, 100, 150), got {len(progress_events)}"

        # Extract rows_processed values for verification
        rows_at_events = [e.rows_processed for e in progress_events]

        # Verify required checkpoints exist
        assert 1 in rows_at_events, "Missing first row progress event"
        assert 100 in rows_at_events, "Missing 100-row checkpoint"
        assert 150 in rows_at_events, "Missing final row (150) progress event"

        # Find specific events by rows_processed for quarantine verification
        first_event = next(e for e in progress_events if e.rows_processed == 1)
        row_100_event = next(e for e in progress_events if e.rows_processed == 100)
        final_event = next(e for e in progress_events if e.rows_processed == 150)

        # Verify quarantine counts at checkpoints
        assert first_event.rows_quarantined == 0  # First row not quarantined yet
        assert row_100_event.rows_quarantined == 1  # Row 100 (0-indexed 99) was quarantined
        assert final_event.rows_quarantined == 1  # Still 1 quarantined at final

    def test_progress_callback_includes_routed_rows_in_success(self, payload_store) -> None:
        """Verify routed rows are counted as successes in progress events.

        Regression test: progress was showing ✓0 for pipelines with gates
        because routed rows weren't included in rows_succeeded.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, ProgressEvent, SourceRow
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class RoutingTestSource(_TestSourceBase):
            """Source that yields 150 rows for routing tests."""

            name = "routing_test_source"
            output_schema = ValueSchema

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for i in range(150):
                    yield SourceRow.valid({"value": i})

        class TrackingSink(_TestSinkBase):
            """Sink that tracks whether it received writes."""

            name = "tracking_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []
                self.write_called = False

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.write_called = True
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

        # Config-driven gate: always routes to "routed_sink"
        routing_gate = GateSettings(
            name="routing_gate",
            condition="True",  # Always routes
            routes={"true": "routed_sink", "false": "continue"},
        )

        source = RoutingTestSource()
        default_sink = TrackingSink()
        routed_sink = TrackingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink), "routed_sink": as_sink(routed_sink)},
            gates=[routing_gate],
        )

        # Track progress events using EventBus
        from elspeth.core import EventBus

        progress_events: list[ProgressEvent] = []

        def track_progress(event: ProgressEvent) -> None:
            progress_events.append(event)

        event_bus = EventBus()
        event_bus.subscribe(ProgressEvent, track_progress)

        orchestrator = Orchestrator(db, event_bus=event_bus)
        orchestrator.run(
            config,
            graph=build_production_graph(config),
            payload_store=payload_store,
        )

        # P1 Fix: Relax exact count assertion - orchestrator also emits on 5-second intervals
        # Required checkpoints: row 1 (first), 100, and final 150
        assert len(progress_events) >= 3, f"Expected at least 3 progress events (1, 100, 150), got {len(progress_events)}"

        # Extract rows_processed values for verification
        rows_at_events = [e.rows_processed for e in progress_events]

        # Verify required checkpoints exist
        assert 1 in rows_at_events, "Missing first row progress event"
        assert 100 in rows_at_events, "Missing 100-row checkpoint"
        assert 150 in rows_at_events, "Missing final row (150) progress event"

        # Find specific events by rows_processed for success verification
        first_event = next(e for e in progress_events if e.rows_processed == 1)
        row_100_event = next(e for e in progress_events if e.rows_processed == 100)
        final_event = next(e for e in progress_events if e.rows_processed == 150)

        # All rows were routed - they should count as succeeded, not zero
        # Bug: without fix, this shows rows_succeeded=0 because routed rows weren't counted
        assert first_event.rows_succeeded == 1  # First row
        assert row_100_event.rows_succeeded == 100
        assert final_event.rows_succeeded == 150

        # Verify routed sink received rows, default did not
        assert routed_sink.write_called
        assert not default_sink.write_called
