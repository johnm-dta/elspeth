"""Terminal run event ownership tests."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from elspeth.contracts import RunStatus
from elspeth.contracts.events import RunCompletionStatus, RunFinished, RunSummary
from elspeth.contracts.run_result import RunResult
from elspeth.engine.orchestrator.ceremony import RunCeremony


class RecordingEvents:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def subscribe(self, event_type: type, handler: Any) -> None:
        pass

    def emit(self, event: Any) -> None:
        self.events.append(event)


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def handle_event(self, event: Any) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass


def test_run_ceremony_emits_terminal_run_events() -> None:
    event_bus = RecordingEvents()
    telemetry = RecordingTelemetry()
    ceremony = RunCeremony(events=event_bus, telemetry=telemetry)
    result = RunResult(
        run_id="run-1",
        status=RunStatus.COMPLETED,
        rows_processed=2,
        rows_succeeded=2,
        rows_failed=0,
        rows_routed_success=1,
        rows_routed_failure=0,
        routed_destinations={"success_sink": 1},
    )

    ceremony.emit_run_finished(
        run_id=result.run_id,
        status=result.status,
        row_count=result.rows_processed,
        duration_seconds=1.25,
    )
    ceremony.emit_run_summary(
        run_id=result.run_id,
        status=RunCompletionStatus.COMPLETED,
        rows_processed=result.rows_processed,
        rows_succeeded=result.rows_succeeded,
        rows_failed=result.rows_failed,
        rows_quarantined=result.rows_quarantined,
        duration_seconds=1.25,
        exit_code=0,
        rows_routed_success=result.rows_routed_success,
        rows_routed_failure=result.rows_routed_failure,
        routed_destinations=result.routed_destinations,
    )

    assert isinstance(telemetry.events[-1], RunFinished)
    assert telemetry.events[-1].run_id == "run-1"
    assert telemetry.events[-1].duration_ms == 1250

    assert isinstance(event_bus.events[-1], RunSummary)
    assert event_bus.events[-1].run_id == "run-1"
    assert event_bus.events[-1].status == RunCompletionStatus.COMPLETED
    assert event_bus.events[-1].routed_destinations == (("success_sink", 1),)


def test_run_and_resume_coordinators_do_not_construct_terminal_events_directly() -> None:
    repo_root = Path(__file__).parents[4]
    checked_paths = (
        repo_root / "src/elspeth/engine/orchestrator/run_lifecycle.py",
        repo_root / "src/elspeth/engine/orchestrator/resume.py",
    )
    offenders: list[str] = []
    for path in checked_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"RunFinished", "RunSummary"}:
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}:{node.func.id}")

    assert offenders == []
