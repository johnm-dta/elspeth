"""Tests for accumulate_row_outcomes DIVERTED invariant.

DIVERTED outcomes are counted in SinkExecutor (via _write_pending_to_sinks
return value), NOT in the processing loop. If a DIVERTED outcome appears
in processing results, that's an orchestration bug.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.types import ExecutionCounters, PendingTokenMap
from elspeth.testing import make_token_info


def _make_result(
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    token: TokenInfo | None = None,
    sink_name: str | None = None,
) -> Mock:
    """Create a mock RowResult with the given two-axis terminal pair."""
    result = Mock()
    result.outcome = outcome
    result.path = path
    result.token = token or make_token_info()
    result.sink_name = sink_name
    # RowProcessingResult.scheduler_pending_sink defaults False; PendingOutcome's
    # offensive guard rejects the Mock a bare attribute access would vivify.
    result.scheduler_pending_sink = False
    return result


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
