"""ADR-019 resume aggregation parity for phase-3 predicate counters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from elspeth.contracts.run_result import RunResult
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator.run_status import derive_resume_terminal_status_from_audit
from tests.integration._helpers import (
    build_test_pipeline_with_discard_sink,
    build_test_pipeline_with_failsink_diversion,
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
    build_test_pipeline_with_source_quarantine,
    run_pipeline,
)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    result: RunResult
    db: LandscapeDB
    run_id: str


def _counter_snapshot(result: RunResult) -> dict[str, object]:
    return {
        "status": result.status,
        "rows_processed": result.rows_processed,
        "rows_succeeded": result.rows_succeeded,
        "rows_failed": result.rows_failed,
        "rows_routed_success": result.rows_routed_success,
        "rows_routed_failure": result.rows_routed_failure,
        "rows_quarantined": result.rows_quarantined,
        "rows_forked": result.rows_forked,
        "rows_coalesced": result.rows_coalesced,
        "rows_expanded": result.rows_expanded,
        "rows_buffered": result.rows_buffered,
        "rows_diverted": result.rows_diverted,
        "rows_coalesce_failed": result.rows_coalesce_failed,
        "routed_destinations": dict(result.routed_destinations),
    }


def _resume_counter_snapshot_from_audit(
    db: LandscapeDB,
    run_id: str,
) -> dict[str, object]:
    factory = RecorderFactory(db)
    status, counters = derive_resume_terminal_status_from_audit(factory, run_id)
    return {
        "status": status,
        "rows_processed": counters.rows_processed,
        "rows_succeeded": counters.rows_succeeded,
        "rows_failed": counters.rows_failed,
        "rows_routed_success": counters.rows_routed_success,
        "rows_routed_failure": counters.rows_routed_failure,
        "rows_quarantined": counters.rows_quarantined,
        "rows_forked": counters.rows_forked,
        "rows_coalesced": counters.rows_coalesced,
        "rows_expanded": counters.rows_expanded,
        "rows_buffered": counters.rows_buffered,
        "rows_diverted": counters.rows_diverted,
        "rows_coalesce_failed": counters.rows_coalesce_failed,
        "routed_destinations": dict(counters.routed_destinations),
    }


def run_live_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> ScenarioResult:
    if scenario == "gate_routed_success":
        config, graph, db, store = build_test_pipeline_with_gate_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            routed_row_count=3,
            default_flow_row_count=1,
        )
    elif scenario == "on_error_routed_failure":
        config, graph, db, store = build_test_pipeline_with_on_error_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            on_error_routed_count=2,
            success_count=1,
        )
    elif scenario == "quarantine_failure":
        config, graph, db, store = build_test_pipeline_with_source_quarantine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            quarantine_row_count=2,
        )
    elif scenario == "failsink_mode_diversion":
        config, graph, db, store = build_test_pipeline_with_failsink_diversion(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            diverted_row_count=2,
            success_row_count=1,
        )
    elif scenario == "discard_mode_diversion":
        config, graph, db, store = build_test_pipeline_with_discard_sink(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            success_row_count=1,
            discard_row_count=2,
        )
    else:
        raise AssertionError(f"unknown scenario {scenario!r}")

    result = run_pipeline(config, graph, db, store)
    return ScenarioResult(result=result, db=db, run_id=result.run_id)


@pytest.mark.parametrize(
    "scenario",
    [
        "gate_routed_success",
        "on_error_routed_failure",
        "quarantine_failure",
        "failsink_mode_diversion",
        "discard_mode_diversion",
    ],
)
def test_resume_counter_shape_matches_live_predicate_counters(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit-derived resume counters match live status and RunResult counters."""
    live = run_live_scenario(tmp_path / scenario, monkeypatch, scenario)

    assert _resume_counter_snapshot_from_audit(live.db, live.run_id) == _counter_snapshot(live.result)


@pytest.mark.parametrize("scenario", ["failsink_mode_diversion", "discard_mode_diversion"])
def test_resume_counter_derivation_replays_diversion_structural_count(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural ``rows_diverted`` is replayed from audit in the all-terminal branch."""
    live = run_live_scenario(tmp_path / scenario, monkeypatch, scenario)
    factory = RecorderFactory(live.db)

    _status, counters = derive_resume_terminal_status_from_audit(factory, live.run_id)

    assert counters.rows_diverted == live.result.rows_diverted
