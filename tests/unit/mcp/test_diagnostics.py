# tests/unit/mcp/test_diagnostics.py
"""Tests for MCP diagnostics analyzer behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from elspeth.contracts import NodeType, RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import runs_table
from elspeth.mcp.analyzers.diagnostics import diagnose
from elspeth.mcp.types import DiagnosticProblem, DiagnosticReport
from tests.fixtures.landscape import make_factory, make_landscape_db

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _set_run_started_at(db: LandscapeDB, run_id: str, started_at: datetime) -> None:
    with db.write_connection() as conn:
        conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(started_at=started_at))


def _create_running_run(factory: RecorderFactory) -> str:
    run = factory.run_lifecycle.begin_run(
        config={"source": {"plugin": "csv"}},
        canonical_version="v1",
        status=RunStatus.RUNNING,
    )
    return run.run_id


def _create_completed_run_with_quarantine(
    db: LandscapeDB,
    factory: RecorderFactory,
    *,
    run_id: str,
    started_at: datetime | None = None,
) -> str:
    """Create a completed run with one quarantined token."""
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=f"source-{run_id}",
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=f"source-{run_id}",
        row_index=0,
        data={"col": "bad-value"},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id)
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.QUARANTINED_AT_SOURCE,
        error_hash="deadbeef" * 8,
    )
    factory.run_lifecycle.complete_run(run_id, RunStatus.COMPLETED)
    if started_at is not None:
        _set_run_started_at(db, run_id, started_at)
    return run_id


def _get_problem(report: DiagnosticReport, problem_type: str) -> DiagnosticProblem | None:
    for problem in report["problems"]:
        if problem.get("type") == problem_type:
            return problem
    return None


def test_diagnose_does_not_flag_recent_running_run_as_stuck() -> None:
    """A newly started run should not appear in stuck_runs."""
    db = make_landscape_db()
    factory = make_factory(db)
    _create_running_run(factory)

    report = diagnose(db, factory)

    assert _get_problem(report, "stuck_runs") is None


def test_diagnose_flags_old_running_run_as_stuck() -> None:
    """A running run older than one hour should appear in stuck_runs."""
    db = make_landscape_db()
    factory = make_factory(db)
    run_id = _create_running_run(factory)

    _set_run_started_at(db, run_id, datetime.now(UTC) - timedelta(hours=2))
    report = diagnose(db, factory)

    stuck_runs = _get_problem(report, "stuck_runs")
    assert stuck_runs is not None
    assert stuck_runs["count"] == 1
    assert stuck_runs["run_ids"] == [run_id]


def test_diagnose_reports_only_old_runs_in_mixed_running_set() -> None:
    """Only runs beyond the stuck threshold should be reported."""
    db = make_landscape_db()
    factory = make_factory(db)
    recent_run_id = _create_running_run(factory)
    old_run_id = _create_running_run(factory)

    _set_run_started_at(db, old_run_id, datetime.now(UTC) - timedelta(hours=2))
    report = diagnose(db, factory)

    stuck_runs = _get_problem(report, "stuck_runs")
    assert stuck_runs is not None
    assert stuck_runs["count"] == 1
    assert stuck_runs["run_ids"] == [old_run_id]
    assert recent_run_id not in stuck_runs["run_ids"]


# --- T5: Quarantine count scoping ---


def test_diagnose_quarantine_count_excludes_old_runs() -> None:
    """Quarantine count should only reflect recent runs, not all history."""
    db = make_landscape_db()
    factory = make_factory(db)

    # Old run (30 days ago) with quarantined row
    _create_completed_run_with_quarantine(
        db,
        factory,
        run_id="old-run",
        started_at=datetime.now(UTC) - timedelta(days=30),
    )
    # Recent run (1 hour ago) — no quarantines
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="recent-run")
    factory.run_lifecycle.complete_run("recent-run", RunStatus.COMPLETED)

    report = diagnose(db, factory)

    # Old quarantines should NOT appear in the report
    quarantined = _get_problem(report, "quarantined_rows")
    assert quarantined is None, (
        f"Old quarantined rows from 30 days ago should not appear in diagnose(), but got count={quarantined['count']}"
    )


def test_diagnose_quarantine_count_includes_recent_runs() -> None:
    """Quarantine count should include quarantines from recent runs."""
    db = make_landscape_db()
    factory = make_factory(db)

    # Recent run with quarantined row
    _create_completed_run_with_quarantine(
        db,
        factory,
        run_id="recent-run",
    )

    report = diagnose(db, factory)

    quarantined = _get_problem(report, "quarantined_rows")
    assert quarantined is not None
    assert quarantined["count"] == 1


def test_diagnose_quarantine_count_scoped_to_recent_runs_only() -> None:
    """With both old and recent quarantines, only recent ones counted."""
    db = make_landscape_db()
    factory = make_factory(db)

    # Old run with 1 quarantine
    _create_completed_run_with_quarantine(
        db,
        factory,
        run_id="old-run",
        started_at=datetime.now(UTC) - timedelta(days=30),
    )
    # Recent run with 1 quarantine
    _create_completed_run_with_quarantine(
        db,
        factory,
        run_id="recent-run",
    )

    report = diagnose(db, factory)

    quarantined = _get_problem(report, "quarantined_rows")
    assert quarantined is not None
    # Should only count the recent run's quarantine, not the old one
    assert quarantined["count"] == 1, f"Expected 1 quarantined row (recent only), got {quarantined['count']}"


def test_diagnose_reports_high_error_completed_with_failures_runs() -> None:
    """High-error completed_with_failures runs should be diagnostic warnings."""
    db = make_landscape_db()
    factory = make_factory(db)
    run_id = "high-error-mixed-run"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source",
        schema_config=_DYNAMIC_SCHEMA,
    )
    for index in range(11):
        factory.data_flow.record_validation_error(
            run_id,
            "source",
            {"bad": f"value-{index}"},
            "missing field",
            "observed",
            "quarantine",
        )
    factory.run_lifecycle.complete_run(run_id, RunStatus.COMPLETED_WITH_FAILURES)

    report = diagnose(db, factory)

    high_errors = _get_problem(report, "high_error_rates")
    assert high_errors is not None
    assert {"run_id": run_id, "errors": 11} in high_errors["runs"]
