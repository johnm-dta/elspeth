# tests/integration/pipeline/orchestrator/test_export_partial_semantics.py
"""Integration tests for export-failure partial-run semantics in Orchestrator.

Covers bead scug.2:
- export status transitions PENDING -> FAILED
- run summary emits PARTIAL (exit_code=1) when export fails post-completion
- run remains COMPLETED in Landscape when only export fails
- non-export execution failures emit FAILED semantics (exit_code=2)
- event bus ordering for export failure vs execution failure
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

from elspeth.contracts import ExportStatus, RunStatus
from elspeth.contracts.events import (
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompletionStatus,
    RunSummary,
)
from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
from elspeth.core.events import EventBus
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, FailingSource, ListSource


def _make_export_enabled_settings() -> ElspethSettings:
    """Create minimal settings with export enabled to the default sink."""
    return ElspethSettings(
        source=SourceSettings(plugin="list_source", on_success="default", options={}),
        sinks={"default": SinkSettings(plugin="collect", on_write_failure="discard", options={})},
        landscape={"export": {"enabled": True, "sink": "default", "format": "json"}},
    )


def _capture_orchestrator_events() -> tuple[EventBus, list[object]]:
    """Attach handlers for phase and summary events and return the event list."""
    event_bus = EventBus()
    captured: list[object] = []
    for event_type in (PhaseStarted, PhaseCompleted, PhaseError, RunSummary):
        event_bus.subscribe(event_type, captured.append)
    return event_bus, captured


def _event_index(events: list[object], predicate: Callable[[object], bool]) -> int:
    """Return first event index matching predicate, raising if not found."""
    for idx, event in enumerate(events):
        if predicate(event):
            return idx
    raise AssertionError("Expected event was not emitted")


class TestExportFailurePartialRunSemantics:
    """Regression coverage for run/export split-status behavior."""

    def test_export_failure_emits_partial_summary_and_keeps_run_completed(self, payload_store) -> None:
        """If export fails after run completion, run is COMPLETED but summary is PARTIAL."""
        db = make_landscape_db()
        event_bus, events = _capture_orchestrator_events()

        source = ListSource([{"value": 1}], on_success="default")
        sink = CollectSink("default")
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )
        graph = build_production_graph(config)
        settings = _make_export_enabled_settings()
        orchestrator = Orchestrator(db, event_bus=event_bus)

        export_status_calls: list[tuple[str, ExportStatus]] = []
        original_set_export_status = RunLifecycleRepository.set_export_status

        def record_export_status(
            self: RunLifecycleRepository,
            run_id: str,
            status: ExportStatus,
            *,
            error: str | None = None,
            export_format: str | None = None,
            export_sink: str | None = None,
        ) -> None:
            export_status_calls.append((run_id, status))
            original_set_export_status(
                self,
                run_id,
                status,
                error=error,
                export_format=export_format,
                export_sink=export_sink,
            )

        with (
            patch.object(RunLifecycleRepository, "set_export_status", new=record_export_status),
            patch(
                "elspeth.engine.orchestrator.core.export_landscape",
                side_effect=RuntimeError("export sink failed"),
            ),
            pytest.raises(RuntimeError, match="export sink failed"),
        ):
            orchestrator.run(
                config=config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
                sink_factory=lambda name: sink,
            )

        assert [status for _, status in export_status_calls] == [
            ExportStatus.PENDING,
            ExportStatus.FAILED,
        ]
        run_id = export_status_calls[0][0]
        assert all(call_run_id == run_id for call_run_id, _ in export_status_calls)

        factory = make_factory(db)
        run = factory.run_lifecycle.get_run(run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED
        assert run.export_status == ExportStatus.FAILED
        assert run.export_error is not None
        assert "export sink failed" in run.export_error

        summaries = [event for event in events if isinstance(event, RunSummary)]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.run_id == run_id
        assert summary.status == RunCompletionStatus.PARTIAL
        assert summary.exit_code == 1
        assert summary.total_rows == 1

        export_started_idx = _event_index(
            events,
            lambda event: isinstance(event, PhaseStarted) and event.phase == PipelinePhase.EXPORT,
        )
        export_error_idx = _event_index(
            events,
            lambda event: isinstance(event, PhaseError) and event.phase == PipelinePhase.EXPORT,
        )
        summary_idx = _event_index(
            events,
            lambda event: isinstance(event, RunSummary) and event.status == RunCompletionStatus.PARTIAL,
        )
        assert export_started_idx < export_error_idx < summary_idx
        assert not any(isinstance(event, PhaseCompleted) and event.phase == PipelinePhase.EXPORT for event in events)

    def test_precompletion_execution_error_emits_failed_summary_exit_code_2(self, payload_store) -> None:
        """If execution fails before completion, summary is FAILED with exit_code=2."""
        db = make_landscape_db()
        event_bus, events = _capture_orchestrator_events()

        source = FailingSource(error_message="source load failure")
        sink = CollectSink("default")
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )
        graph = build_production_graph(config)
        settings = _make_export_enabled_settings()
        orchestrator = Orchestrator(db, event_bus=event_bus)

        export_status_calls: list[tuple[str, ExportStatus]] = []
        original_set_export_status = RunLifecycleRepository.set_export_status

        def record_export_status(
            self: RunLifecycleRepository,
            run_id: str,
            status: ExportStatus,
            *,
            error: str | None = None,
            export_format: str | None = None,
            export_sink: str | None = None,
        ) -> None:
            export_status_calls.append((run_id, status))
            original_set_export_status(
                self,
                run_id,
                status,
                error=error,
                export_format=export_format,
                export_sink=export_sink,
            )

        with (
            patch.object(RunLifecycleRepository, "set_export_status", new=record_export_status),
            patch("elspeth.engine.orchestrator.core.export_landscape") as mock_export,
            pytest.raises(RuntimeError, match="source load failure"),
        ):
            orchestrator.run(
                config=config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
                sink_factory=lambda name: sink,
            )

        mock_export.assert_not_called()
        assert export_status_calls == []

        summaries = [event for event in events if isinstance(event, RunSummary)]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.status == RunCompletionStatus.FAILED
        assert summary.exit_code == 2
        assert summary.total_rows == 0
        assert summary.run_id is not None

        factory = make_factory(db)
        run = factory.run_lifecycle.get_run(summary.run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED
        assert run.export_status is None

        source_error_idx = _event_index(
            events,
            lambda event: isinstance(event, PhaseError) and event.phase == PipelinePhase.SOURCE,
        )
        summary_idx = _event_index(
            events,
            lambda event: isinstance(event, RunSummary) and event.status == RunCompletionStatus.FAILED,
        )
        assert source_error_idx < summary_idx
        assert not any(
            isinstance(event, (PhaseStarted, PhaseError, PhaseCompleted)) and event.phase == PipelinePhase.EXPORT for event in events
        )
