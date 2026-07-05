"""Tests for accumulate_row_outcomes DIVERTED invariant.

DIVERTED outcomes are counted in SinkExecutor (via write_pending_to_sinks
return value), NOT in the processing loop. If a DIVERTED outcome appears
in processing results, that's an orchestration bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.run_state import PendingTokenMap
from elspeth.engine.orchestrator.types import ExecutionCounters
from elspeth.testing import make_token_info


@dataclass(frozen=True)
class _ProcessingResult:
    outcome: TerminalOutcome | None
    path: TerminalPath
    token: TokenInfo
    sink_name: str | None
    scheduler_pending_sink: bool = False


def _make_result(
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    token: TokenInfo | None = None,
    sink_name: str | None = None,
) -> _ProcessingResult:
    """Create a result record with the given two-axis terminal pair."""
    return _ProcessingResult(
        outcome=outcome,
        path=path,
        token=token or make_token_info(),
        sink_name=sink_name,
    )


def _make_pending() -> PendingTokenMap:
    return {"sink1": [], "sink2": []}


class TestAccumulateDiverted:
    def test_diverted_raises_invariant_error(self) -> None:
        """DIVERTED in processing results is an orchestration bug."""
        counters = ExecutionCounters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED, sink_name="sink1")]
        with pytest.raises(OrchestrationInvariantError, match="Diversion path"):
            accumulate_row_outcomes(results, counters, pending)

    def test_diverted_after_completed_still_raises(self) -> None:
        """Even mixed with valid outcomes, DIVERTED crashes."""
        counters = ExecutionCounters()
        pending = _make_pending()
        results = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="sink1"),
            _make_result(TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED, sink_name="sink1"),
        ]
        with pytest.raises(OrchestrationInvariantError, match="Diversion path"):
            accumulate_row_outcomes(results, counters, pending)
