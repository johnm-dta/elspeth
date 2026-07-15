# tests/integration/pipeline/orchestrator/test_execution_loop.py
"""Integration tests for the orchestrator's main execution loop.

Covers phase events, row processing, database initialization guards,
RunSummary emission, and graceful shutdown — the highest-risk untested
paths in orchestrator/core.py.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from elspeth.contracts import Determinism, PipelineRow, RunStatus, SourceRow
from elspeth.contracts.enums import OutputMode
from elspeth.contracts.errors import GracefulShutdownError, OrchestrationInvariantError
from elspeth.contracts.events import (
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompletionStatus,
    RunFinished,
    RunSummary,
)
from elspeth.contracts.types import AggregationName
from elspeth.core.config import AggregationSettings, SourceSettings, TriggerConfig
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema, _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline, build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore

# ---------------------------------------------------------------------------
# Capturing event bus
# ---------------------------------------------------------------------------


class CapturingEventBus:
    """Event bus that records all emitted events for test assertions."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def subscribe(self, event_type: type, handler: Any) -> None:
        pass  # Not needed — we capture via emit

    def emit(self, event: Any) -> None:
        self.events.append(event)


class CapturingTelemetryManager:
    """Minimal telemetry manager double for orchestrator lifecycle tests."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def handle_event(self, event: Any) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_pipeline_with_event_capture(
    source_data: list[dict[str, Any]],
    transforms: list[Any] | None = None,
    *,
    shutdown_event: threading.Event | None = None,
) -> tuple[Any, list[Any], CollectSink]:
    """Run a linear pipeline and return (RunResult, events, sink).

    Creates an in-memory LandscapeDB and InMemoryPayloadStore, builds a
    linear pipeline via build_linear_pipeline(), and runs via Orchestrator
    with a CapturingEventBus.
    """
    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()
    event_bus = CapturingEventBus()

    if transforms is None:
        transforms = []

    source, tx_list, sinks, graph = build_linear_pipeline(source_data, transforms=transforms)
    sink = sinks["default"]

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(t) for t in tx_list],
        sinks={"default": as_sink(sink)},
    )

    orchestrator = Orchestrator(db, event_bus=event_bus)
    result = orchestrator.run(
        config,
        graph=graph,
        payload_store=payload_store,
        shutdown_event=shutdown_event,
    )
    return result, event_bus.events, sink


# ---------------------------------------------------------------------------
# Shutdown transform
# ---------------------------------------------------------------------------


class ShutdownAfterNTransform(BaseTransform):
    """Transform that sets a shutdown event after processing N rows."""

    name = "shutdown_trigger"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self, shutdown_event: threading.Event, trigger_after: int = 2) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._shutdown_event = shutdown_event
        self._trigger_after = trigger_after
        self._count = 0

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        self._count += 1
        if self._count >= self._trigger_after:
            self._shutdown_event.set()
        return TransformResult.success(
            make_pipeline_row(dict(row)),
            success_reason={"action": "passthrough"},
        )


class PreYieldFailureSource(_TestSourceBase):
    """Generator source that fails before yielding its first row."""

    name = "pre_yield_fail_source"
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__()
        self.on_success = "default"

    def load(self, ctx: Any) -> Any:
        raise FileNotFoundError("missing-source-file.csv")
        yield  # pragma: no cover - keeps this a generator function


class FailOnSecondRowTransform(BaseTransform):
    """Transform that succeeds once, then crashes on the second row."""

    name = "fail_on_second_row"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._count = 0

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        self._count += 1
        if self._count == 2:
            raise RuntimeError("boom on second row")
        return TransformResult.success(
            make_pipeline_row(dict(row)),
            success_reason={"action": "passthrough"},
        )


class IdleBlockingSource(_TestSourceBase):
    """Source that requires an aggregation timeout while waiting for row two."""

    name = "idle_blocking_source"
    output_schema = _TestSchema
    idle_flush_wait_seconds = 2.0

    def __init__(self, flush_seen: threading.Event) -> None:
        super().__init__()
        self.on_success = "agg_in"
        self._flush_seen = flush_seen

    def load(self, ctx: Any) -> Any:
        rows = list(self.wrap_rows([{"value": 1}]))
        yield rows[0]
        if not self._flush_seen.wait(timeout=self.idle_flush_wait_seconds):
            raise RuntimeError(f"aggregation timeout did not fire while source was idle within {self.idle_flush_wait_seconds:.1f}s")
        yield SourceRow.valid(
            {"value": 2},
            contract=rows[0].contract,
            source_row_index=1,
        )


class ThreadAffinitySource(_TestSourceBase):
    """Source that proves timeout polling does not move iterator execution threads."""

    name = "thread_affinity_source"
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__()
        self.on_success = "agg_in"
        self.load_thread_id: int | None = None
        self.next_thread_ids: list[int] = []

    def load(self, ctx: Any) -> Any:
        self.load_thread_id = threading.get_ident()
        rows = list(self.wrap_rows([{"value": 1}, {"value": 2}]))

        self.next_thread_ids.append(threading.get_ident())
        yield rows[0]

        self.next_thread_ids.append(threading.get_ident())
        if self.next_thread_ids[-1] != self.load_thread_id:
            raise RuntimeError("source iterator advanced on a different thread from load()")
        yield rows[1]


class MultiGapIdleBlockingSource(_TestSourceBase):
    """Source with two idle gaps, each requiring a timeout flush to proceed."""

    name = "multi_gap_idle_blocking_source"
    output_schema = _TestSchema
    idle_flush_wait_seconds = 5.0

    def __init__(self, gap_events: list[threading.Event]) -> None:
        super().__init__()
        self.on_success = "agg_in"
        self._gap_events = gap_events

    def load(self, ctx: Any) -> Any:
        rows = list(self.wrap_rows([{"value": 1}]))
        contract = rows[0].contract
        yield rows[0]
        for gap_index, gap in enumerate(self._gap_events, start=1):
            if not gap.wait(timeout=self.idle_flush_wait_seconds):
                raise RuntimeError(
                    f"aggregation timeout did not fire during idle gap {gap_index} within {self.idle_flush_wait_seconds:.1f}s"
                )
            yield SourceRow.valid(
                {"value": gap_index + 1},
                contract=contract,
                source_row_index=gap_index,
            )


class SequentialFlushSignalingBatchTransform(BaseTransform):
    """Release source gaps only from idle-worker flushes.

    The engine may also flush an aggregation on the orchestrator thread. Those
    callbacks are valid but are not evidence for idle-source polling, so they
    must not advance this helper's gap sequence.
    """

    name = "sequential_idle_flush_signaler"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success = "output"
    on_error = "discard"

    def __init__(
        self,
        gap_events: list[threading.Event],
        idle_flush_threads: list[threading.Thread],
        orchestrator_thread: threading.Thread,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._gap_events = gap_events
        self._idle_flush_threads = idle_flush_threads
        self._orchestrator_thread = orchestrator_thread
        self._flush_index = 0

    def process(  # type: ignore[override]
        self, rows: list[PipelineRow], ctx: Any
    ) -> TransformResult:
        flush_thread = threading.current_thread()
        if flush_thread is not self._orchestrator_thread:
            self._idle_flush_threads.append(flush_thread)
            if self._flush_index < len(self._gap_events):
                self._gap_events[self._flush_index].set()
            self._flush_index += 1
        return TransformResult.success(
            make_pipeline_row({"flushed_count": len(rows)}),
            success_reason={"action": "idle_timeout_flushed"},
        )


class IdleFlushSignalingBatchTransform(BaseTransform):
    """Batch transform that records when timeout flushing occurs."""

    name = "idle_flush_signaler"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success = "output"
    on_error = "discard"

    def __init__(self, flush_seen: threading.Event) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._flush_seen = flush_seen

    def process(  # type: ignore[override]
        self, rows: list[PipelineRow], ctx: Any
    ) -> TransformResult:
        self._flush_seen.set()
        return TransformResult.success(
            make_pipeline_row({"flushed_count": len(rows)}),
            success_reason={"action": "idle_timeout_flushed"},
        )


# ===========================================================================
# Test classes
# ===========================================================================


class TestExecutionLoopPhaseEvents:
    """Phase event lifecycle ordering."""

    def test_database_and_process_phases_emitted(self) -> None:
        """Both DATABASE and PROCESS phases emit Started and Completed events."""
        result, events, _sink = _run_pipeline_with_event_capture(
            [{"value": 1}, {"value": 2}],
        )
        assert result.status == RunStatus.COMPLETED

        db_started = [e for e in events if isinstance(e, PhaseStarted) and e.phase == PipelinePhase.DATABASE]
        db_completed = [e for e in events if isinstance(e, PhaseCompleted) and e.phase == PipelinePhase.DATABASE]
        proc_started = [e for e in events if isinstance(e, PhaseStarted) and e.phase == PipelinePhase.PROCESS]
        proc_completed = [e for e in events if isinstance(e, PhaseCompleted) and e.phase == PipelinePhase.PROCESS]

        assert len(db_started) == 1, f"Expected 1 DATABASE PhaseStarted, got {len(db_started)}"
        assert len(db_completed) == 1, f"Expected 1 DATABASE PhaseCompleted, got {len(db_completed)}"
        assert len(proc_started) == 1, f"Expected 1 PROCESS PhaseStarted, got {len(proc_started)}"
        assert len(proc_completed) == 1, f"Expected 1 PROCESS PhaseCompleted, got {len(proc_completed)}"

    def test_database_phase_completes_before_process_starts(self) -> None:
        """DATABASE PhaseCompleted must precede PROCESS PhaseStarted in event order."""
        _result, events, _sink = _run_pipeline_with_event_capture(
            [{"value": 1}, {"value": 2}],
        )

        db_completed_idx = next(i for i, e in enumerate(events) if isinstance(e, PhaseCompleted) and e.phase == PipelinePhase.DATABASE)
        proc_started_idx = next(i for i, e in enumerate(events) if isinstance(e, PhaseStarted) and e.phase == PipelinePhase.PROCESS)
        assert db_completed_idx < proc_started_idx, (
            f"DATABASE PhaseCompleted (index {db_completed_idx}) must come before PROCESS PhaseStarted (index {proc_started_idx})"
        )

    def test_pre_yield_generator_source_failure_stays_in_source_phase(self) -> None:
        """A generator startup failure must not be misreported as PROCESS."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        event_bus = CapturingEventBus()

        source = PreYieldFailureSource()
        sink = CollectSink(name="default")
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db, event_bus=event_bus)

        with pytest.raises(FileNotFoundError, match=r"missing-source-file\.csv"):
            orchestrator.run(
                config,
                graph=build_production_graph(config),
                payload_store=payload_store,
            )

        source_completed = [e for e in event_bus.events if isinstance(e, PhaseCompleted) and e.phase == PipelinePhase.SOURCE]
        process_started = [e for e in event_bus.events if isinstance(e, PhaseStarted) and e.phase == PipelinePhase.PROCESS]
        source_errors = [e for e in event_bus.events if isinstance(e, PhaseError) and e.phase == PipelinePhase.SOURCE]

        assert source_completed == []
        assert process_started == []
        assert len(source_errors) == 1
        assert source_errors[0].target == source.name


class TestExecutionLoopRowProcessing:
    """Row-level processing and sink delivery."""

    def test_all_rows_reach_sink(self) -> None:
        """All source rows are processed and delivered to the sink."""
        rows = [{"value": i} for i in range(10)]
        result, _events, sink = _run_pipeline_with_event_capture(rows)

        assert result.rows_processed == 10
        assert result.rows_succeeded == 10
        assert len(sink.results) == 10

    def test_empty_source_completes_successfully(self) -> None:
        """An empty source completes with COMPLETED status and 0 rows."""
        result, _events, sink = _run_pipeline_with_event_capture([])

        # Phase 2.2: status taxonomy now distinguishes this shape as EMPTY.
        assert result.status == RunStatus.EMPTY
        assert result.rows_processed == 0
        assert result.rows_succeeded == 0
        assert len(sink.results) == 0

    def test_run_result_has_valid_run_id(self) -> None:
        """RunResult.run_id is a non-empty string."""
        result, _events, _sink = _run_pipeline_with_event_capture(
            [{"value": 1}],
        )
        assert isinstance(result.run_id, str)
        assert len(result.run_id) > 0

    def test_timeout_aggregation_flushes_while_source_is_idle(self, payload_store) -> None:
        """Aggregation timeout must fire without waiting for another source row."""
        flush_seen = threading.Event()
        source = IdleBlockingSource(flush_seen)
        transform = IdleFlushSignalingBatchTransform(flush_seen)
        sink = CollectSink("output")

        agg_settings = AggregationSettings(
            name="idle_flush",
            plugin=transform.name,
            input="agg_in",
            on_success="output",
            on_error="discard",
            trigger=TriggerConfig(timeout_seconds=0.05),
            output_mode=OutputMode.TRANSFORM,
        )
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": as_source(source)},
            source_settings_map={
                "primary": SourceSettings(
                    plugin=source.name,
                    on_success="agg_in",
                    options={},
                ),
            },
            transforms=[],
            sinks={"output": as_sink(sink)},
            aggregations={"idle_flush": (as_transform(transform), agg_settings)},
            gates=[],
        )
        agg_node_id = graph.get_aggregation_id_map()[AggregationName("idle_flush")]
        transform.node_id = agg_node_id

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink)},
            aggregation_settings={agg_node_id: agg_settings},
        )

        result = Orchestrator(LandscapeDB.in_memory()).run(
            config,
            graph=graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert flush_seen.is_set()
        assert sink.results[0]["flushed_count"] == 1

    def test_timeout_aggregation_polls_source_iterator_on_load_thread(self, payload_store) -> None:
        """Timeout-enabled aggregation must not advance source iterators on worker threads."""
        source = ThreadAffinitySource()
        flush_seen = threading.Event()
        transform = IdleFlushSignalingBatchTransform(flush_seen)
        sink = CollectSink("output")

        agg_settings = AggregationSettings(
            name="thread_affinity_agg",
            plugin=transform.name,
            input="agg_in",
            on_success="output",
            on_error="discard",
            trigger=TriggerConfig(count=2, timeout_seconds=30),
            output_mode=OutputMode.TRANSFORM,
        )
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": as_source(source)},
            source_settings_map={
                "primary": SourceSettings(
                    plugin=source.name,
                    on_success="agg_in",
                    options={},
                ),
            },
            transforms=[],
            sinks={"output": as_sink(sink)},
            aggregations={"thread_affinity_agg": (as_transform(transform), agg_settings)},
            gates=[],
        )
        agg_node_id = graph.get_aggregation_id_map()[AggregationName("thread_affinity_agg")]
        transform.node_id = agg_node_id

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink)},
            aggregation_settings={agg_node_id: agg_settings},
        )

        result = Orchestrator(LandscapeDB.in_memory()).run(
            config,
            graph=graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert source.load_thread_id is not None
        assert source.next_thread_ids == [source.load_thread_id, source.load_thread_id]
        assert flush_seen.is_set()
        assert sink.results[0]["flushed_count"] == 2

    def test_idle_timeout_polling_uses_one_pump_for_the_whole_run(self, payload_store) -> None:
        """One persistent IdleTimeoutPump per run — no per-fetch thread churn (elspeth-735df9576d).

        The run performs four source ``next()`` calls (prefetch, two gap rows,
        exhaustion); the historical implementation spawned a poller thread per
        call. This pins: exactly one pump start per run, both idle-gap flushes
        executed on the same single persistent worker thread (never the
        orchestrator thread), and flushes still firing while the source is
        blocked inside ``next()``.
        """
        gap_events = [threading.Event(), threading.Event()]
        orchestrator_thread = threading.current_thread()
        idle_flush_threads: list[threading.Thread] = []
        source = MultiGapIdleBlockingSource(gap_events)
        transform = SequentialFlushSignalingBatchTransform(gap_events, idle_flush_threads, orchestrator_thread)
        sink = CollectSink("output")

        agg_settings = AggregationSettings(
            name="multi_gap_idle_flush",
            plugin=transform.name,
            input="agg_in",
            on_success="output",
            on_error="discard",
            trigger=TriggerConfig(timeout_seconds=0.05),
            output_mode=OutputMode.TRANSFORM,
        )
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": as_source(source)},
            source_settings_map={
                "primary": SourceSettings(
                    plugin=source.name,
                    on_success="agg_in",
                    options={},
                ),
            },
            transforms=[],
            sinks={"output": as_sink(sink)},
            aggregations={"multi_gap_idle_flush": (as_transform(transform), agg_settings)},
            gates=[],
        )
        agg_node_id = graph.get_aggregation_id_map()[AggregationName("multi_gap_idle_flush")]
        transform.node_id = agg_node_id

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink)},
            aggregation_settings={agg_node_id: agg_settings},
        )

        result = Orchestrator(LandscapeDB.in_memory()).run(
            config,
            graph=graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert all(gap.is_set() for gap in gap_events), "idle flushes must fire while the source is blocked in next()"
        # The first two flushes are the idle-gap flushes (the source cannot
        # advance until each fires); both ran on the same persistent worker
        # thread, never on the orchestrator thread. (A third, end-of-input
        # flush runs on the orchestrator thread during finalization.)
        assert len(idle_flush_threads) >= 2
        assert idle_flush_threads[0] is idle_flush_threads[1]
        assert all(thread is not orchestrator_thread for thread in idle_flush_threads)


class TestDatabaseInitialization:
    """Guards on required orchestrator inputs."""

    def test_run_requires_execution_graph(self) -> None:
        """Passing graph=None raises OrchestrationInvariantError."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(ListSource([]))},
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(OrchestrationInvariantError, match="ExecutionGraph is required"):
            orchestrator.run(config, graph=None, payload_store=payload_store)

    def test_run_requires_payload_store(self) -> None:
        """Passing payload_store=None raises OrchestrationInvariantError."""
        db = LandscapeDB.in_memory()
        source_data: list[dict[str, Any]] = []
        source, _tx, sinks, graph = build_linear_pipeline(source_data)
        sink = sinks["default"]

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(OrchestrationInvariantError, match="PayloadStore is required"):
            orchestrator.run(config, graph=graph, payload_store=None)  # type: ignore[arg-type]

    def test_successful_run_records_in_landscape(self) -> None:
        """After a successful run, the Landscape records the run as COMPLETED."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()

        source, tx_list, sinks, graph = build_linear_pipeline([{"value": 42}])
        sink = sinks["default"]

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(t) for t in tx_list],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        factory = RecorderFactory(db)
        run = factory.run_lifecycle.get_run(result.run_id)
        assert run is not None, f"Run {result.run_id} not found in Landscape"
        assert run.status == RunStatus.COMPLETED


class TestRunSummaryEmission:
    """RunSummary event emission with correct metrics."""

    def test_run_summary_emitted_with_correct_counts(self) -> None:
        """Exactly one RunSummary is emitted with matching row counts."""
        rows = [{"value": i} for i in range(5)]
        result, events, _sink = _run_pipeline_with_event_capture(rows)

        summaries = [e for e in events if isinstance(e, RunSummary)]
        assert len(summaries) == 1, f"Expected 1 RunSummary, got {len(summaries)}"

        summary = summaries[0]
        assert summary.run_id == result.run_id
        assert summary.status == RunCompletionStatus.COMPLETED
        assert summary.total_rows == 5
        assert summary.succeeded == 5
        assert summary.failed == 0
        assert summary.exit_code == 0

    def test_run_summary_has_positive_duration(self) -> None:
        """RunSummary.duration_seconds must be > 0."""
        _result, events, _sink = _run_pipeline_with_event_capture(
            [{"value": 1}],
        )

        summaries = [e for e in events if isinstance(e, RunSummary)]
        assert len(summaries) == 1
        assert summaries[0].duration_seconds > 0

    def test_failed_run_summary_and_runfinished_preserve_partial_counts(self) -> None:
        """Mid-run failure must report real partial counters, not zeros."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        event_bus = CapturingEventBus()
        telemetry = CapturingTelemetryManager()

        transform = FailOnSecondRowTransform()
        source, _tx, sinks, graph = build_linear_pipeline([{"value": 1}, {"value": 2}], transforms=[transform])
        sink = sinks["default"]
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db, event_bus=event_bus, telemetry_manager=telemetry)

        with pytest.raises(RuntimeError, match="boom on second row"):
            orchestrator.run(
                config,
                graph=graph,
                payload_store=payload_store,
            )

        summaries = [e for e in event_bus.events if isinstance(e, RunSummary)]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.status == RunCompletionStatus.FAILED
        assert summary.total_rows == 2
        assert summary.succeeded == 1
        assert summary.failed == 0
        assert summary.quarantined == 0
        assert summary.routed_success == 0
        assert summary.routed_failure == 0

        finished = [e for e in telemetry.events if isinstance(e, RunFinished)]
        assert len(finished) == 1
        assert finished[0].status == RunStatus.FAILED
        assert finished[0].row_count == 2


class TestGracefulShutdownIntegration:
    """Shutdown mid-processing raises GracefulShutdownError with counters."""

    def test_shutdown_mid_processing_raises_with_counters(self) -> None:
        """Setting shutdown_event mid-run raises GracefulShutdownError with rows_processed >= trigger_after."""
        shutdown_event = threading.Event()
        trigger_after = 3
        transform = ShutdownAfterNTransform(shutdown_event, trigger_after=trigger_after)

        rows = [{"value": i} for i in range(10)]
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()

        source, _tx, sinks, graph = build_linear_pipeline(rows, transforms=[transform])
        sink = sinks["default"]

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(t) for t in [transform]],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(
                config,
                graph=graph,
                payload_store=payload_store,
                shutdown_event=shutdown_event,
            )

        assert exc_info.value.rows_processed >= trigger_after
