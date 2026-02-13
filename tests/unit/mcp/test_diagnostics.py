# tests/unit/mcp/test_diagnostics.py
"""Tests for MCP diagnostics analyzer behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from elspeth.contracts import RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import runs_table
from elspeth.mcp.analyzers.diagnostics import diagnose


def _set_run_started_at(db: LandscapeDB, run_id: str, started_at: datetime) -> None:
    with db.connection() as conn:
        conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(started_at=started_at))


def _create_running_run(recorder: LandscapeRecorder) -> str:
    run = recorder.begin_run(
        config={"source": {"plugin": "csv"}},
        canonical_version="v1",
        status=RunStatus.RUNNING,
    )
    return run.run_id


def _get_problem(report: dict[str, object], problem_type: str) -> dict[str, object] | None:
    problems = report.get("problems")
    if not isinstance(problems, list):
        return None
    for problem in problems:
        if isinstance(problem, dict) and problem.get("type") == problem_type:
            return problem
    return None


def test_diagnose_does_not_flag_recent_running_run_as_stuck() -> None:
    """A newly started run should not appear in stuck_runs."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    _create_running_run(recorder)

    report = diagnose(db, recorder)

    assert _get_problem(report, "stuck_runs") is None


def test_diagnose_flags_old_running_run_as_stuck() -> None:
    """A running run older than one hour should appear in stuck_runs."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    run_id = _create_running_run(recorder)

    _set_run_started_at(db, run_id, datetime.now(UTC) - timedelta(hours=2))
    report = diagnose(db, recorder)

    stuck_runs = _get_problem(report, "stuck_runs")
    assert stuck_runs is not None
    assert stuck_runs["count"] == 1
    assert stuck_runs["run_ids"] == [run_id]


def test_diagnose_reports_only_old_runs_in_mixed_running_set() -> None:
    """Only runs beyond the stuck threshold should be reported."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    recent_run_id = _create_running_run(recorder)
    old_run_id = _create_running_run(recorder)

    _set_run_started_at(db, old_run_id, datetime.now(UTC) - timedelta(hours=2))
    report = diagnose(db, recorder)

    stuck_runs = _get_problem(report, "stuck_runs")
    assert stuck_runs is not None
    assert stuck_runs["count"] == 1
    assert stuck_runs["run_ids"] == [old_run_id]
    assert recent_run_id not in stuck_runs["run_ids"]
