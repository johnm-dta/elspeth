# src/elspeth/mcp/analyzers/diagnostics.py
"""Emergency diagnostic functions for the Landscape audit database.

Functions: diagnose, get_failure_context, get_recent_activity.

All functions accept (db, recorder) as their first two parameters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.mcp.types import (
    DiagnosticReport,
    ErrorResult,
    FailureContextReport,
    RecentActivityReport,
)


def diagnose(db: LandscapeDB, recorder: LandscapeRecorder) -> DiagnosticReport:
    """Emergency diagnostic: What's broken right now?

    Scans for failed runs, high error rates, stuck runs, and recent problems.
    This is the first tool to use when something is wrong.

    Args:
        db: Database connection
        recorder: Landscape recorder

    Returns:
        Diagnostic summary with problems, recent failures, and recommendations
    """
    from sqlalchemy import func, select

    from elspeth.core.landscape.schema import (
        operations_table,
        runs_table,
        token_outcomes_table,
        validation_errors_table,
    )

    problems: list[dict[str, Any]] = []
    recommendations: list[str] = []
    stuck_cutoff = datetime.now(UTC) - timedelta(hours=1)

    with db.connection() as conn:
        # Find failed runs (most critical)
        failed_runs = conn.execute(
            select(runs_table.c.run_id, runs_table.c.started_at, runs_table.c.completed_at)
            .where(runs_table.c.status == "failed")
            .order_by(runs_table.c.started_at.desc())
            .limit(5)
        ).fetchall()

        if failed_runs:
            problems.append(
                {
                    "severity": "CRITICAL",
                    "type": "failed_runs",
                    "count": len(failed_runs),
                    "run_ids": [r.run_id for r in failed_runs],
                    "message": f"{len(failed_runs)} failed run(s) found",
                }
            )
            recommendations.append("Use get_run_summary(run_id) on failed runs to see error counts")

        # Find stuck runs (running for > 1 hour with no completion)
        stuck_runs = conn.execute(
            select(runs_table.c.run_id, runs_table.c.started_at)
            .where(runs_table.c.status == "running")
            .where(runs_table.c.completed_at.is_(None))
            .where(runs_table.c.started_at < stuck_cutoff)
            .order_by(runs_table.c.started_at)
            .limit(5)
        ).fetchall()

        if stuck_runs:
            problems.append(
                {
                    "severity": "WARNING",
                    "type": "stuck_runs",
                    "count": len(stuck_runs),
                    "run_ids": [r.run_id for r in stuck_runs],
                    "message": f"{len(stuck_runs)} run(s) still in 'running' status",
                }
            )
            recommendations.append("Check if process is still running; may need manual intervention")

        # Find stuck operations (open for > 1 hour)
        # These indicate source/sink I/O that never completed
        stuck_ops = conn.execute(
            select(
                operations_table.c.operation_id,
                operations_table.c.run_id,
                operations_table.c.operation_type,
                operations_table.c.started_at,
            )
            .where(operations_table.c.status == "open")
            .where(operations_table.c.started_at < stuck_cutoff)
            .order_by(operations_table.c.started_at)
            .limit(5)
        ).fetchall()

        if stuck_ops:
            problems.append(
                {
                    "severity": "WARNING",
                    "type": "stuck_operations",
                    "count": len(stuck_ops),
                    "operations": [
                        {
                            "operation_id": op.operation_id[:12] + "...",
                            "run_id": op.run_id[:12] + "...",
                            "type": op.operation_type,
                            "started": op.started_at.isoformat() if op.started_at else None,
                        }
                        for op in stuck_ops
                    ],
                    "message": f"{len(stuck_ops)} operation(s) stuck in 'open' status for > 1 hour",
                }
            )
            recommendations.append(
                "Stuck operations may indicate: (1) streaming source still active, "
                "(2) process crashed during I/O, (3) database connection issue"
            )
            recommendations.append("Use list_operations(run_id, status='open') to see all open operations")

        # Find runs with high error rates
        error_rates = conn.execute(
            select(
                runs_table.c.run_id,
                func.count(validation_errors_table.c.error_id).label("val_errors"),
            )
            .outerjoin(validation_errors_table, runs_table.c.run_id == validation_errors_table.c.run_id)
            .where(runs_table.c.status == "completed")
            .group_by(runs_table.c.run_id)
            .having(func.count(validation_errors_table.c.error_id) > 10)
            .order_by(func.count(validation_errors_table.c.error_id).desc())
            .limit(5)
        ).fetchall()

        if error_rates:
            problems.append(
                {
                    "severity": "WARNING",
                    "type": "high_error_rates",
                    "runs": [{"run_id": r.run_id, "errors": r.val_errors} for r in error_rates],
                    "message": f"{len(error_rates)} run(s) with high error counts",
                }
            )
            recommendations.append("Use get_error_analysis(run_id) to identify error patterns")

        # Get recent run summary
        recent_runs = conn.execute(
            select(
                runs_table.c.run_id,
                runs_table.c.status,
                runs_table.c.started_at,
                runs_table.c.completed_at,
            )
            .order_by(runs_table.c.started_at.desc())
            .limit(10)
        ).fetchall()

        # Count QUARANTINED outcomes across recent runs
        quarantined_count = (
            conn.execute(
                select(func.count(token_outcomes_table.c.outcome_id)).where(token_outcomes_table.c.outcome == "quarantined")
            ).scalar()
            or 0
        )

        if quarantined_count > 0:
            problems.append(
                {
                    "severity": "INFO",
                    "type": "quarantined_rows",
                    "count": quarantined_count,
                    "message": f"{quarantined_count} row(s) have been quarantined across all runs",
                }
            )
            recommendations.append("Quarantined rows indicate data quality issues at source")

    recent_summary = [
        {
            "run_id": r.run_id[:12] + "...",
            "status": r.status,
            "started": r.started_at.isoformat() if r.started_at else None,
        }
        for r in recent_runs
    ]

    return {
        "status": "CRITICAL" if any(p["severity"] == "CRITICAL" for p in problems) else "OK" if not problems else "WARNING",
        "problems": problems,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "recent_runs": recent_summary,  # type: ignore[typeddict-item]
        "recommendations": recommendations,
        "next_steps": [
            "Use list_runs() to see all runs with their status",
            "Use get_run_summary(run_id) on a specific run for details",
            "Use get_error_analysis(run_id) to understand error patterns",
            "Use explain_token(run_id, token_id=...) to trace a specific failure",
        ],
    }


def get_failure_context(db: LandscapeDB, recorder: LandscapeRecorder, run_id: str, limit: int = 10) -> FailureContextReport | ErrorResult:
    """Get comprehensive context about failures in a run.

    Use this when investigating why a run failed. Returns failed node states,
    error samples, and the surrounding context.

    Args:
        db: Database connection
        recorder: Landscape recorder
        run_id: Run ID to investigate
        limit: Max failures to return

    Returns:
        Failure context with node states, errors, and recommendations
    """
    from sqlalchemy import select

    from elspeth.core.landscape.schema import (
        node_states_table,
        nodes_table,
        transform_errors_table,
        validation_errors_table,
    )

    run = recorder.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
        # Get failed node states with node info using composite key
        failed_states = conn.execute(
            select(
                node_states_table.c.state_id,
                node_states_table.c.token_id,
                node_states_table.c.node_id,
                node_states_table.c.step_index,
                node_states_table.c.attempt,
                node_states_table.c.started_at,
                node_states_table.c.completed_at,
                nodes_table.c.plugin_name,
                nodes_table.c.node_type,
            )
            .join(
                nodes_table,
                (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
            )
            .where(node_states_table.c.run_id == run_id)
            .where(node_states_table.c.status == "failed")
            .order_by(node_states_table.c.started_at.desc())
            .limit(limit)
        ).fetchall()

        # Get transform errors with details
        transform_errors = conn.execute(
            select(
                transform_errors_table.c.token_id,
                transform_errors_table.c.transform_id,
                transform_errors_table.c.error_details_json,
                nodes_table.c.plugin_name,
            )
            .outerjoin(
                nodes_table,
                (transform_errors_table.c.transform_id == nodes_table.c.node_id)
                & (transform_errors_table.c.run_id == nodes_table.c.run_id),
            )
            .where(transform_errors_table.c.run_id == run_id)
            .order_by(transform_errors_table.c.created_at.desc())
            .limit(limit)
        ).fetchall()

        # Get validation errors
        validation_errors = conn.execute(
            select(
                validation_errors_table.c.node_id,
                validation_errors_table.c.row_hash,
                validation_errors_table.c.row_data_json,
                nodes_table.c.plugin_name,
            )
            .outerjoin(
                nodes_table,
                (validation_errors_table.c.node_id == nodes_table.c.node_id) & (validation_errors_table.c.run_id == nodes_table.c.run_id),
            )
            .where(validation_errors_table.c.run_id == run_id)
            .limit(limit)
        ).fetchall()

    failed_state_list = [
        {
            "state_id": s.state_id,
            "token_id": s.token_id,
            "plugin": s.plugin_name,
            "type": s.node_type,
            "step": s.step_index,
            "attempt": s.attempt,
            "started": s.started_at.isoformat() if s.started_at else None,
        }
        for s in failed_states
    ]

    transform_error_list = [
        {
            "token_id": e.token_id,
            "plugin": e.plugin_name,
            "details": json.loads(e.error_details_json) if e.error_details_json else None,
        }
        for e in transform_errors
    ]

    validation_error_list = [
        {
            "plugin": e.plugin_name,
            "row_hash": e.row_hash[:12] + "..." if e.row_hash else None,
            "sample_data": json.loads(e.row_data_json) if e.row_data_json else None,
        }
        for e in validation_errors
    ]

    # Identify patterns
    plugins_with_failures = list({s["plugin"] for s in failed_state_list if s["plugin"]})
    has_retries = any(s["attempt"] > 1 for s in failed_state_list)

    return {
        "run_id": run_id,
        "run_status": run.status.value,
        "failed_node_states": failed_state_list,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "transform_errors": transform_error_list,  # type: ignore[typeddict-item]
        "validation_errors": validation_error_list,  # type: ignore[typeddict-item]
        "patterns": {
            "plugins_failing": plugins_with_failures,
            "has_retries": has_retries,
            "failure_count": len(failed_state_list),
            "transform_error_count": len(transform_error_list),
            "validation_error_count": len(validation_error_list),
        },
        "next_steps": [
            f"Use explain_token(run_id='{run_id}', token_id='...') to trace a specific failure",
            "Check transform error details for exception messages",
            "Look at validation errors for data quality issues at source",
        ],
    }


def get_recent_activity(db: LandscapeDB, recorder: LandscapeRecorder, minutes: int = 60) -> RecentActivityReport:
    """Get recent pipeline activity timeline.

    Use this to understand what happened recently when investigating issues.

    Args:
        db: Database connection
        recorder: Landscape recorder
        minutes: Look back this many minutes (default 60)

    Returns:
        Timeline of recent runs and their status
    """
    from datetime import timedelta

    from sqlalchemy import func, select

    from elspeth.core.landscape.schema import (
        node_states_table,
        rows_table,
        runs_table,
    )

    cutoff = datetime.now(tz=UTC) - timedelta(minutes=minutes)

    with db.connection() as conn:
        # Get runs started in the time window
        recent_runs = conn.execute(
            select(
                runs_table.c.run_id,
                runs_table.c.status,
                runs_table.c.started_at,
                runs_table.c.completed_at,
            )
            .where(runs_table.c.started_at >= cutoff)
            .order_by(runs_table.c.started_at.desc())
        ).fetchall()

        if not recent_runs:
            run_stats: list[dict[str, Any]] = []
        else:
            # Collect run IDs for batch queries
            run_ids = [run.run_id for run in recent_runs]

            # Batch query: Get row counts grouped by run_id (N+1 fix)
            row_counts_result = conn.execute(
                select(rows_table.c.run_id, func.count().label("cnt")).where(rows_table.c.run_id.in_(run_ids)).group_by(rows_table.c.run_id)
            ).fetchall()
            row_counts: dict[str, int] = {r.run_id: r.cnt for r in row_counts_result}

            # Batch query: Get state counts grouped by run_id (N+1 fix)
            state_counts_result = conn.execute(
                select(node_states_table.c.run_id, func.count().label("cnt"))
                .where(node_states_table.c.run_id.in_(run_ids))
                .group_by(node_states_table.c.run_id)
            ).fetchall()
            state_counts: dict[str, int] = {r.run_id: r.cnt for r in state_counts_result}

            run_stats = []
            for run in recent_runs:
                duration = None
                if run.started_at and run.completed_at:
                    duration = (run.completed_at - run.started_at).total_seconds()

                run_stats.append(
                    {
                        "run_id": run.run_id[:12] + "...",
                        "full_run_id": run.run_id,
                        "status": run.status,
                        "started": run.started_at.isoformat() if run.started_at else None,
                        "duration_seconds": duration,
                        "rows_processed": row_counts.get(run.run_id, 0),
                        "node_executions": state_counts.get(run.run_id, 0),
                    }
                )

    status_counts: dict[str, int] = {}
    for r in run_stats:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    return {
        "time_window_minutes": minutes,
        "total_runs": len(run_stats),
        "status_summary": status_counts,
        "runs": run_stats,  # type: ignore[typeddict-item]  # structurally correct dict literals
    }
