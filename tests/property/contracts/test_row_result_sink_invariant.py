# tests/property/contracts/test_row_result_sink_invariant.py
"""Property-based tests for RowResult sink_name invariants.

RowResult enforces that sink-targeting outcomes (COMPLETED, ROUTED,
ROUTED_ON_ERROR, COALESCED) always require a sink_name, while non-sink
outcomes accept None. These invariants are critical for audit integrity:
every row that reaches a sink must record which sink it went to.

Properties tested:
- COMPLETED/ROUTED/ROUTED_ON_ERROR/COALESCED always require sink_name (raises without it)
- COMPLETED/ROUTED/ROUTED_ON_ERROR/COALESCED accept any non-empty sink_name string
  (ROUTED_ON_ERROR additionally requires a FailureInfo on .error)
- Non-sink outcomes accept sink_name=None
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.results import FailureInfo, RowResult
from elspeth.testing import make_pipeline_row

# =============================================================================
# Outcome Categories
# =============================================================================

SINK_OUTCOMES = {
    RowOutcome.COMPLETED,
    RowOutcome.ROUTED,
    RowOutcome.ROUTED_ON_ERROR,
    RowOutcome.COALESCED,
}

NON_SINK_OUTCOMES = [o for o in RowOutcome if o not in SINK_OUTCOMES]


def _error_for(outcome: RowOutcome) -> FailureInfo | None:
    """Return a real FailureInfo for outcomes that require it; None otherwise.

    elspeth-5069612f3c — ROUTED_ON_ERROR carries the originating transform
    error through to the audit trail. RowResult.__post_init__ rejects
    construction without a FailureInfo on .error, so the property tests must
    supply one for that outcome to exercise the positive sink-name property.
    """
    if outcome == RowOutcome.ROUTED_ON_ERROR:
        return FailureInfo(
            exception_type="ValueError",
            message="upstream transform raised",
        )
    return None


# =============================================================================
# Negative Properties: sink-targeting outcomes reject missing sink_name
# =============================================================================


class TestSinkTargetingOutcomeRequiresSinkName:
    """Property: COMPLETED/ROUTED/ROUTED_ON_ERROR/COALESCED always require sink_name."""

    @given(outcome=st.sampled_from(sorted(SINK_OUTCOMES, key=lambda o: o.value)))
    @settings(max_examples=20)
    def test_sink_targeting_outcome_rejects_none_sink_name(self, outcome: RowOutcome) -> None:
        """Sink-targeting outcomes raise OrchestrationInvariantError without sink_name."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}))
        with pytest.raises(OrchestrationInvariantError):
            RowResult(
                token=token,
                final_data=make_pipeline_row({}),
                outcome=outcome,
                sink_name=None,
                error=_error_for(outcome),
            )


# =============================================================================
# Positive Properties: sink-targeting outcomes accept valid sink_name
# =============================================================================


class TestSinkTargetingOutcomeAcceptsSinkName:
    """Property: COMPLETED/ROUTED/ROUTED_ON_ERROR/COALESCED accept any non-None sink_name."""

    @given(
        outcome=st.sampled_from(sorted(SINK_OUTCOMES, key=lambda o: o.value)),
        sink_name=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_sink_targeting_outcome_accepts_sink_name(self, outcome: RowOutcome, sink_name: str) -> None:
        """Sink-targeting outcomes successfully store the provided sink_name."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}))
        result = RowResult(
            token=token,
            final_data=make_pipeline_row({}),
            outcome=outcome,
            sink_name=sink_name,
            error=_error_for(outcome),
        )
        assert result.sink_name == sink_name


# =============================================================================
# Non-sink outcomes accept None sink_name
# =============================================================================


class TestNonSinkOutcomeAcceptsNoneSinkName:
    """Property: Non-sink outcomes don't require sink_name."""

    @given(outcome=st.sampled_from(NON_SINK_OUTCOMES))
    @settings(max_examples=30)
    def test_non_sink_outcome_accepts_none_sink_name(self, outcome: RowOutcome) -> None:
        """Non-sink outcomes can be created with sink_name=None."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}))
        result = RowResult(token=token, final_data=make_pipeline_row({}), outcome=outcome, sink_name=None)
        assert result.sink_name is None
