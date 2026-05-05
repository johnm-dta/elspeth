"""ADR-019 discard-mode behaviour change integration tests."""

from __future__ import annotations

from collections import Counter

import pytest

from elspeth.contracts.audit import DISCARD_SINK_NAME
from elspeth.contracts.enums import RunStatus, TerminalOutcome, TerminalPath
from elspeth.core.landscape.factory import RecorderFactory
from tests.integration._helpers import build_test_pipeline_with_discard_sink, run_pipeline


@pytest.fixture
def pipeline_with_discard_and_success_sink(tmp_path, monkeypatch):
    """A pipeline where three rows are written and two are discarded."""
    return build_test_pipeline_with_discard_sink(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        success_row_count=3,
        discard_row_count=2,
    )


class TestDiscardModeRunStatusFlip:
    def test_discard_with_some_success_yields_completed_with_failures(
        self,
        pipeline_with_discard_and_success_sink,
    ) -> None:
        """Discard-mode sink diversions are predicate-input failures."""
        config, graph, db, store = pipeline_with_discard_and_success_sink
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_succeeded == 3
        assert result.rows_failed == 2
        assert result.rows_diverted == 2

        outcomes = RecorderFactory(db).query.get_all_token_outcomes_for_run(result.run_id)
        counts = Counter((outcome.outcome, outcome.path, outcome.completed) for outcome in outcomes)
        assert counts[(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, True)] == 3
        assert counts[(TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED, True)] == 2
        discard_rows = [outcome for outcome in outcomes if outcome.path == TerminalPath.SINK_DISCARDED]
        assert all(outcome.sink_name == DISCARD_SINK_NAME for outcome in discard_rows)
        assert all(outcome.error_hash for outcome in discard_rows)

    def test_all_discards_yields_failed(self, tmp_path, monkeypatch) -> None:
        """Pipelines with only discard-mode failures have no success indicator."""
        config, graph, db, store = build_test_pipeline_with_discard_sink(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            success_row_count=0,
            discard_row_count=3,
        )
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.FAILED
        assert result.rows_succeeded == 0
        assert result.rows_failed == 3
        assert result.rows_diverted == 3

        outcomes = RecorderFactory(db).query.get_all_token_outcomes_for_run(result.run_id)
        assert Counter((outcome.outcome, outcome.path, outcome.completed) for outcome in outcomes) == Counter(
            {
                (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED, True): 3,
            }
        )
