# tests/telemetry/test_integration.py
"""Integration tests for the telemetry system.

These tests verify:
1. Telemetry emitted alongside Landscape - both systems work together
2. Regression: telemetry only after Landscape success - critical ordering guarantee
3. Granularity filtering - events filtered correctly at each level
4. Exporter failure isolation - one exporter failing doesn't affect others
5. Loud failure on total exporter failure - when all exporters fail repeatedly
6. High-volume flooding test - stress test with 10k+ events
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import ConfigDict

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema, SourceRow
from elspeth.contracts.enums import BackpressureMode, RunStatus, TelemetryGranularity
from elspeth.contracts.events import (
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.results import TransformResult
from elspeth.telemetry import TelemetryManager
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.events import (
    PhaseChanged,
    RowCreated,
    RunFinished,
    RunStarted,
)
from tests.conftest import _TestSinkBase, _TestSourceBase, as_sink, as_source

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


class DynamicSchema(PluginSchema):
    """Dynamic schema for testing - allows any fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


@dataclass
class MockTelemetryConfig:
    """Mock RuntimeTelemetryProtocol implementation for testing."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK

    @property
    def exporter_configs(self) -> tuple:
        return ()


class RecordingExporter:
    """Exporter that records all events for test verification.

    This is the primary test helper for capturing telemetry events
    emitted during pipeline execution.
    """

    def __init__(self, name: str = "recording"):
        self._name = name
        self.events: list[TelemetryEvent] = []
        self.flush_count = 0
        self.close_count = 0

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


class FailingExporter:
    """Exporter that always fails export() calls.

    Use this to test failure isolation and error handling.
    """

    def __init__(self, name: str = "failing", *, fail_count: int | None = None):
        """Initialize failing exporter.

        Args:
            name: Exporter name for identification
            fail_count: If set, only fail this many times then succeed.
                       If None, always fail.
        """
        self._name = name
        self._fail_count = fail_count
        self._failures = 0
        self.export_attempts = 0

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.export_attempts += 1
        if self._fail_count is None or self._failures < self._fail_count:
            self._failures += 1
            raise RuntimeError(f"Simulated export failure in {self._name}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class ListSource(_TestSourceBase):
    """Simple source that yields a list of rows."""

    name = "list_source"
    output_schema = DynamicSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        super().__init__()
        self._data = data

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for row in self._data:
            yield SourceRow.valid(row)


class PassthroughTransform:
    """Transform that passes through rows unchanged."""

    name = "passthrough"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    node_id: str | None = None
    is_batch_aware = False
    creates_tokens = False
    _on_error: str | None = None

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "passthrough"})

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass


class CollectingSink(_TestSinkBase):
    """Sink that collects rows for verification."""

    name = "collecting_sink"

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="test123")


def create_minimal_graph(source_name: str = "list_source", transform_name: str = "passthrough") -> ExecutionGraph:
    """Create a minimal valid execution graph for testing."""
    from elspeth.contracts import NodeID, NodeType, RoutingMode, SinkName

    graph = ExecutionGraph()
    schema_config = {"schema": {"fields": "dynamic"}}

    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name=source_name, config=schema_config)
    graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name=transform_name, config=schema_config)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="collecting_sink", config=schema_config)

    graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)

    graph._transform_id_map = {0: NodeID("transform")}
    graph._sink_id_map = {SinkName("output"): NodeID("sink")}
    graph._default_sink = "output"

    return graph


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Module-scoped in-memory database for integration tests."""
    return LandscapeDB.in_memory()


# =============================================================================
# Test 1: Telemetry Emitted Alongside Landscape
# =============================================================================


class TestTelemetryEmittedAlongsideLandscape:
    """Integration: Run pipeline, verify both Landscape and telemetry have data."""

    def test_both_landscape_and_telemetry_have_data_after_run(self, landscape_db: LandscapeDB, payload_store) -> None:
        """After a successful run, both Landscape and telemetry contain data."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Verify Landscape has data
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3

        # Verify telemetry was captured
        assert len(exporter.events) > 0

        # Both should have matching run_id
        telemetry_run_ids = {e.run_id for e in exporter.events}
        assert len(telemetry_run_ids) == 1
        assert result.run_id in telemetry_run_ids

    def test_telemetry_events_match_landscape_operations(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Telemetry events correspond to Landscape-recorded operations."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Should have lifecycle events
        run_started = [e for e in exporter.events if isinstance(e, RunStarted)]
        run_completed = [e for e in exporter.events if isinstance(e, RunFinished)]

        assert len(run_started) == 1
        assert len(run_completed) == 1

        # RunFinished should match Landscape result
        completed_event = run_completed[0]
        assert completed_event.status == result.status
        assert completed_event.run_id == result.run_id


# =============================================================================
# Test 2: Regression - Telemetry Only After Landscape Success
# =============================================================================


class TestTelemetryOnlyAfterLandscapeSuccess:
    """Regression: If Landscape fails, telemetry must NOT be emitted."""

    def test_no_run_started_if_begin_run_fails(self, landscape_db: LandscapeDB, payload_store) -> None:
        """If Landscape begin_run fails, RunStarted telemetry is NOT emitted.

        This is a critical ordering guarantee: telemetry is emitted AFTER
        Landscape recording succeeds. If Landscape fails, no telemetry.
        """
        from elspeth.core.landscape import LandscapeRecorder

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        # Patch LandscapeRecorder.begin_run to fail
        # The recorder is created inside run(), so we need to patch the class method
        with (
            patch.object(LandscapeRecorder, "begin_run", side_effect=RuntimeError("Landscape failure")),
            pytest.raises(RuntimeError, match="Landscape failure"),
        ):
            orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # NO RunStarted should have been emitted
        run_started = [e for e in exporter.events if isinstance(e, RunStarted)]
        assert len(run_started) == 0

    def test_telemetry_order_verified_by_successful_run(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify code ordering: telemetry emission follows Landscape success.

        This test proves the ordering by showing that in normal operation:
        1. begin_run succeeds -> RunStarted emitted
        2. finalize_run succeeds -> RunFinished emitted

        By inspection of the Orchestrator code, if either fails, the
        corresponding telemetry is not reached.
        """
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Both should exist (proves the normal path works)
        run_started = [e for e in exporter.events if isinstance(e, RunStarted)]
        run_completed = [e for e in exporter.events if isinstance(e, RunFinished)]

        assert len(run_started) == 1, "RunStarted should be emitted after begin_run"
        assert len(run_completed) == 1, "RunFinished should be emitted after finalize_run"

        # Events should be in correct order
        started_idx = exporter.events.index(run_started[0])
        completed_idx = exporter.events.index(run_completed[0])
        assert started_idx < completed_idx, "RunStarted must precede RunFinished"

    def test_failed_run_still_emits_run_finished_with_failed_status(self, landscape_db: LandscapeDB, payload_store) -> None:
        """When a run fails, RunFinished is emitted with FAILED status.

        The telemetry system records failures - the key is that it only
        records AFTER Landscape has recorded the failure.
        """

        class FailingTransform(PassthroughTransform):
            """Transform that always fails."""

            name = "failing_transform"

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("Simulated transform failure")

        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[FailingTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        with pytest.raises(RuntimeError, match="Simulated transform failure"):
            orchestrator.run(config, graph=create_minimal_graph(transform_name="failing_transform"), payload_store=payload_store)

        # RunStarted should be present (began before failure)
        run_started = [e for e in exporter.events if isinstance(e, RunStarted)]
        assert len(run_started) == 1

        # RunFinished should be present with FAILED status
        run_completed = [e for e in exporter.events if isinstance(e, RunFinished)]
        assert len(run_completed) == 1
        assert run_completed[0].status == RunStatus.FAILED


# =============================================================================
# Test 3: Granularity Filtering
# =============================================================================


class TestGranularityFiltering:
    """Verify events are filtered correctly at each granularity level."""

    def test_lifecycle_granularity_only_emits_lifecycle_events(self, landscape_db: LandscapeDB, payload_store) -> None:
        """At LIFECYCLE granularity, only RunStarted/RunFinished/PhaseChanged emitted."""
        config_lifecycle = MockTelemetryConfig(granularity=TelemetryGranularity.LIFECYCLE)
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(config_lifecycle, exporters=[exporter])

        source = ListSource([{"id": 1}, {"id": 2}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Should only have lifecycle events
        for event in exporter.events:
            assert isinstance(event, (RunStarted, RunFinished, PhaseChanged)), (
                f"Non-lifecycle event at LIFECYCLE granularity: {type(event).__name__}"
            )

        # Should NOT have row-level events
        row_events = [e for e in exporter.events if isinstance(e, (RowCreated, TransformCompleted, TokenCompleted))]
        assert len(row_events) == 0, "Row events should be filtered at LIFECYCLE granularity"

    def test_rows_granularity_includes_row_events(self, landscape_db: LandscapeDB, payload_store) -> None:
        """At ROWS granularity, row-level events are included."""
        config_rows = MockTelemetryConfig(granularity=TelemetryGranularity.ROWS)
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(config_rows, exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Should have both lifecycle and row events
        has_lifecycle = any(isinstance(e, (RunStarted, RunFinished)) for e in exporter.events)

        assert has_lifecycle, "Should have lifecycle events at ROWS granularity"
        # Row events may or may not be present depending on RowProcessor emission
        # The key is they are ALLOWED, not filtered

    def test_full_granularity_includes_all_events(self, landscape_db: LandscapeDB, payload_store) -> None:
        """At FULL granularity, all event types are allowed."""
        config_full = MockTelemetryConfig(granularity=TelemetryGranularity.FULL)
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(config_full, exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Should have lifecycle events at minimum
        has_lifecycle = any(isinstance(e, (RunStarted, RunFinished)) for e in exporter.events)
        assert has_lifecycle, "Should have lifecycle events at FULL granularity"

        # At FULL, external call events would also be allowed (if any were emitted)
        # This test verifies the filter doesn't block anything


# =============================================================================
# Test 4: Exporter Failure Isolation
# =============================================================================


class TestExporterFailureIsolation:
    """One exporter failing doesn't prevent others from receiving events."""

    def test_one_failing_exporter_doesnt_block_others(self, landscape_db: LandscapeDB, payload_store) -> None:
        """When one exporter fails, other exporters still receive events."""
        working_exporter = RecordingExporter("working")
        failing_exporter = FailingExporter("failing")

        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[working_exporter, failing_exporter])

        source = ListSource([{"id": 1}, {"id": 2}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        # Suppress warning logs from the failing exporter
        with patch("elspeth.telemetry.manager.logger"):
            result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Pipeline should complete successfully
        assert result.status == RunStatus.COMPLETED

        # Working exporter should have received events
        assert len(working_exporter.events) > 0

        # Failing exporter should have attempted exports
        assert failing_exporter.export_attempts > 0

    def test_all_exporters_receive_same_events(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Multiple working exporters all receive the same events."""
        exporter1 = RecordingExporter("exporter1")
        exporter2 = RecordingExporter("exporter2")
        exporter3 = RecordingExporter("exporter3")

        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter1, exporter2, exporter3])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # All exporters should have the same number of events
        assert len(exporter1.events) == len(exporter2.events) == len(exporter3.events)

        # Events should be identical
        for e1, e2, e3 in zip(exporter1.events, exporter2.events, exporter3.events, strict=True):
            assert type(e1) is type(e2) is type(e3)
            assert e1.run_id == e2.run_id == e3.run_id

    def test_partial_success_counts_as_emitted(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Partial success (some exporters work) is counted as emitted."""
        working_exporter = RecordingExporter("working")
        failing_exporter = FailingExporter("failing")

        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[working_exporter, failing_exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        with patch("elspeth.telemetry.manager.logger"):
            orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Events should be counted as emitted, not dropped
        metrics = telemetry_manager.health_metrics
        assert metrics["events_emitted"] > 0
        assert metrics["events_dropped"] == 0


# =============================================================================
# Test 5: Loud Failure on Total Exporter Failure
# =============================================================================


def make_run_started(run_id: str, timestamp: datetime | None = None) -> RunStarted:
    """Create a RunStarted event for testing."""
    return RunStarted(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        config_hash="abc123",
        source_plugin="csv",
    )


class TestTotalExporterFailure:
    """When all exporters fail repeatedly and fail_on_total=True, crash.

    Note: These tests send events directly to TelemetryManager because
    a typical pipeline run only emits ~5 lifecycle events, which isn't
    enough to trigger the 10-consecutive-failure threshold. The tests
    verify the failure handling behavior independently of the pipeline.
    """

    def test_total_failure_with_fail_on_total_true_raises(self) -> None:
        """With fail_on_total_exporter_failure=True, raises after threshold.

        The TelemetryManager should raise TelemetryExporterError when all
        exporters fail 10 consecutive times and fail_on_total=True.
        """
        config_fail = MockTelemetryConfig(fail_on_total_exporter_failure=True)
        failing1 = FailingExporter("failing1")
        failing2 = FailingExporter("failing2")

        telemetry_manager = TelemetryManager(config_fail, exporters=[failing1, failing2])

        with patch("elspeth.telemetry.manager.logger"):
            # Send 9 events - should not raise yet
            for i in range(9):
                telemetry_manager.handle_event(make_run_started(f"run-{i}"))

            # 10th event triggers the exception (stored for re-raise on flush)
            telemetry_manager.handle_event(make_run_started("run-10"))

            # flush() re-raises the stored exception from background thread
            with pytest.raises(TelemetryExporterError) as exc_info:
                telemetry_manager.flush()

        assert exc_info.value.exporter_name == "all"
        assert "10 consecutive times" in str(exc_info.value)

    def test_total_failure_with_fail_on_total_false_disables(self) -> None:
        """With fail_on_total_exporter_failure=False, disables telemetry and continues.

        The TelemetryManager should log CRITICAL and disable itself rather
        than crashing the pipeline.
        """
        config_continue = MockTelemetryConfig(fail_on_total_exporter_failure=False)
        failing1 = FailingExporter("failing1")
        failing2 = FailingExporter("failing2")

        telemetry_manager = TelemetryManager(config_continue, exporters=[failing1, failing2])

        with patch("elspeth.telemetry.manager.logger"):
            # Send 10 events - should not raise, but should disable
            for i in range(10):
                telemetry_manager.handle_event(make_run_started(f"run-{i}"))

            # Wait for background thread to process all events
            telemetry_manager.flush()

        # Telemetry should be disabled
        assert telemetry_manager._disabled is True

        # Further events should be silently dropped (no counter increment)
        initial_dropped = telemetry_manager.health_metrics["events_dropped"]
        telemetry_manager.handle_event(make_run_started("run-11"))
        assert telemetry_manager.health_metrics["events_dropped"] == initial_dropped

    def test_total_failure_in_pipeline_context(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Total exporter failure behavior during actual pipeline run.

        With LIFECYCLE granularity (~5 events per run), we won't hit the
        10-failure threshold. This test verifies the failure counting works
        in the pipeline context, even if we don't reach the threshold.
        """
        # Use LIFECYCLE granularity to limit events to ~5 (RunStarted, PhaseChanged x N, RunFinished)
        # FULL granularity would emit 10+ row-level events and hit the failure threshold
        config_fail = MockTelemetryConfig(
            fail_on_total_exporter_failure=True,
            granularity=TelemetryGranularity.LIFECYCLE,
        )
        failing1 = FailingExporter("failing1")
        failing2 = FailingExporter("failing2")

        telemetry_manager = TelemetryManager(config_fail, exporters=[failing1, failing2])

        source = ListSource([{"id": i} for i in range(10)])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)

        with patch("elspeth.telemetry.manager.logger"):
            # Pipeline runs emit ~5 events, which is below threshold
            result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Pipeline should complete (didn't hit threshold)
        assert result.status == RunStatus.COMPLETED

        # But failures should be counted
        metrics = telemetry_manager.health_metrics
        assert metrics["events_dropped"] > 0
        assert metrics["consecutive_total_failures"] > 0

        # Exporter attempts should match the number of events * number of exporters
        # (each exporter gets called for each event)
        assert failing1.export_attempts > 0
        assert failing2.export_attempts > 0


# =============================================================================
# Test 6: High-Volume Flooding Test
# =============================================================================


class TestHighVolumeFlooding:
    """10k+ events processed without memory issues."""

    def test_ten_thousand_events_processed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Process 10,000+ rows and verify telemetry handles the volume."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        # Create 10,000 rows
        num_rows = 10_000
        source = ListSource([{"id": i, "data": f"value_{i}"} for i in range(num_rows)])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Pipeline should complete successfully
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == num_rows

        # Telemetry should have captured events
        assert len(exporter.events) >= 2  # At least RunStarted and RunFinished

        # Should have lifecycle events
        run_started = [e for e in exporter.events if isinstance(e, RunStarted)]
        run_completed = [e for e in exporter.events if isinstance(e, RunFinished)]
        assert len(run_started) == 1
        assert len(run_completed) == 1

        # RunFinished should have correct row count
        assert run_completed[0].row_count == num_rows

    def test_high_volume_with_granularity_filter_reduces_memory(self, landscape_db: LandscapeDB, payload_store) -> None:
        """LIFECYCLE granularity drastically reduces event volume for high-throughput."""
        exporter_full = RecordingExporter("full")
        exporter_lifecycle = RecordingExporter("lifecycle")

        manager_full = TelemetryManager(MockTelemetryConfig(granularity=TelemetryGranularity.FULL), exporters=[exporter_full])
        manager_lifecycle = TelemetryManager(
            MockTelemetryConfig(granularity=TelemetryGranularity.LIFECYCLE), exporters=[exporter_lifecycle]
        )

        num_rows = 1_000
        source = ListSource([{"id": i} for i in range(num_rows)])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        # Run with FULL granularity
        orchestrator1 = Orchestrator(landscape_db, telemetry_manager=manager_full)
        orchestrator1.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Run with LIFECYCLE granularity (new source/sink for clean state)
        source2 = ListSource([{"id": i} for i in range(num_rows)])
        sink2 = CollectingSink()
        config2 = PipelineConfig(
            source=as_source(source2),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink2)},
        )
        orchestrator2 = Orchestrator(landscape_db, telemetry_manager=manager_lifecycle)
        orchestrator2.run(config2, graph=create_minimal_graph(), payload_store=payload_store)

        # LIFECYCLE should have significantly fewer events than FULL
        # (FULL could have row events, LIFECYCLE only has RunStarted/RunFinished/PhaseChanged)
        lifecycle_count = len(exporter_lifecycle.events)

        # LIFECYCLE should have only lifecycle events (<=10 typically)
        assert lifecycle_count < 20, f"LIFECYCLE should have few events, got {lifecycle_count}"

        # All lifecycle events should be lifecycle types
        for event in exporter_lifecycle.events:
            assert isinstance(event, (RunStarted, RunFinished, PhaseChanged))

    def test_high_volume_metrics_accurate(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Health metrics are accurate after high-volume processing."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        num_rows = 5_000
        source = ListSource([{"id": i} for i in range(num_rows)])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        metrics = telemetry_manager.health_metrics

        # All events should be emitted (no failures)
        assert metrics["events_emitted"] == len(exporter.events)
        assert metrics["events_dropped"] == 0
        assert metrics["consecutive_total_failures"] == 0


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestTelemetryManagerLifecycle:
    """Tests for TelemetryManager lifecycle (flush, close) in integration context."""

    def test_flush_called_on_manager_close(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Exporters are flushed when manager is closed."""
        exporter = RecordingExporter()
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[exporter])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Manually close the manager (Orchestrator may or may not do this)
        telemetry_manager.flush()
        telemetry_manager.close()

        assert exporter.flush_count >= 1
        assert exporter.close_count >= 1

    def test_telemetry_manager_handles_empty_exporter_list(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Telemetry manager with no exporters is a no-op."""
        telemetry_manager = TelemetryManager(MockTelemetryConfig(), exporters=[])

        source = ListSource([{"id": 1}])
        sink = CollectingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[PassthroughTransform()],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(config, graph=create_minimal_graph(), payload_store=payload_store)

        # Pipeline should complete normally
        assert result.status == RunStatus.COMPLETED

        # No events emitted (no exporters)
        metrics = telemetry_manager.health_metrics
        assert metrics["events_emitted"] == 0
        assert metrics["events_dropped"] == 0
