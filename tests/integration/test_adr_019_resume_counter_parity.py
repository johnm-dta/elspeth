"""ADR-019 resume aggregation parity for phase-3 predicate counters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.run_result import RunResult
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import Orchestrator
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


def _predicate_counter_tuple(result: RunResult) -> tuple[RunStatus, int, int, int, int, int, int]:
    return (
        result.status,
        result.rows_processed,
        result.rows_succeeded,
        result.rows_failed,
        result.rows_routed_success,
        result.rows_routed_failure,
        result.rows_quarantined,
    )


def _resume_counter_tuple_from_audit(
    db: LandscapeDB,
    run_id: str,
) -> tuple[RunStatus, int, int, int, int, int, int]:
    factory = RecorderFactory(db)
    status, processed, succeeded, failed, routed_success, routed_failure, quarantined = (
        Orchestrator._derive_resume_terminal_status_from_audit(  # pyright: ignore[reportPrivateUsage]
            Orchestrator(db),
            factory,
            run_id,
        )
    )
    return (status, processed, succeeded, failed, routed_success, routed_failure, quarantined)


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
    """Audit-derived resume counters match live predicate/status counters."""
    live = run_live_scenario(tmp_path / scenario, monkeypatch, scenario)

    assert _resume_counter_tuple_from_audit(live.db, live.run_id) == _predicate_counter_tuple(live.result)


@pytest.mark.parametrize("scenario", ["failsink_mode_diversion", "discard_mode_diversion"])
def test_resume_counter_derivation_stays_scoped_to_predicate_tuple(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural ``rows_diverted`` is intentionally not replayed from audit here."""
    live = run_live_scenario(tmp_path / scenario, monkeypatch, scenario)
    factory = RecorderFactory(live.db)

    derived = Orchestrator._derive_resume_terminal_status_from_audit(  # pyright: ignore[reportPrivateUsage]
        Orchestrator(live.db),
        factory,
        live.run_id,
    )

    assert len(derived) == 7
