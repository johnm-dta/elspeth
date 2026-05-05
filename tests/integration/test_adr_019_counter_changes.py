"""ADR-019 routed counter behaviour-change integration tests."""

from __future__ import annotations

from collections import Counter

from elspeth.contracts.enums import RunStatus, TerminalOutcome, TerminalPath
from elspeth.core.landscape.factory import RecorderFactory
from tests.integration._helpers import (
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
    run_pipeline,
)


class TestGateRoutedCounterDoubling:
    def test_gate_routed_bumps_both_succeeded_and_routed_success(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        config, graph, db, store = build_test_pipeline_with_gate_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            routed_row_count=4,
            default_flow_row_count=0,
        )
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_succeeded == 4
        assert result.rows_routed_success == 4

        outcomes = RecorderFactory(db).query.get_all_token_outcomes_for_run(result.run_id)
        assert Counter((outcome.outcome, outcome.path, outcome.completed) for outcome in outcomes) == Counter(
            {
                (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, True): 4,
            }
        )
        assert {outcome.sink_name for outcome in outcomes} == {"routed"}

    def test_on_error_routed_bumps_both_failed_and_routed_failure(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        config, graph, db, store = build_test_pipeline_with_on_error_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            on_error_routed_count=3,
            success_count=2,
        )
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_failed == 3
        assert result.rows_routed_failure == 3
        assert result.rows_succeeded == 2

        outcomes = RecorderFactory(db).query.get_all_token_outcomes_for_run(result.run_id)
        counts = Counter((outcome.outcome, outcome.path, outcome.completed) for outcome in outcomes)
        assert counts[(TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED, True)] == 3
        assert counts[(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, True)] == 2
        on_error_rows = [outcome for outcome in outcomes if outcome.path == TerminalPath.ON_ERROR_ROUTED]
        assert all(outcome.sink_name == "error_sink" for outcome in on_error_rows)
        assert all(outcome.error_hash for outcome in on_error_rows)
