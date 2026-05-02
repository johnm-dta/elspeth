"""Tests for RowResult using RowOutcome enum.

This module tests the engine's RowResult dataclass with RowOutcome enum values.
RowOutcome enum contract tests are in tests/contracts/test_enums.py.
"""

from typing import ClassVar

from elspeth.contracts import RowOutcome, RowResult, TokenInfo
from elspeth.contracts.results import FailureInfo
from elspeth.testing import make_pipeline_row


def _error_for(outcome: RowOutcome) -> FailureInfo | None:
    """ROUTED_ON_ERROR (elspeth-5069612f3c) requires a real FailureInfo on .error.

    All other outcomes accept ``error=None`` per RowResult.__post_init__.
    """
    if outcome == RowOutcome.ROUTED_ON_ERROR:
        return FailureInfo(
            exception_type="ValueError",
            message="upstream transform raised",
        )
    return None


class TestRowResultOutcome:
    """Tests for RowResult.outcome as enum."""

    _SINK_OUTCOMES: ClassVar[set[RowOutcome]] = {
        RowOutcome.COMPLETED,
        RowOutcome.ROUTED,
        RowOutcome.ROUTED_ON_ERROR,
        RowOutcome.COALESCED,
    }

    def test_outcome_is_enum(self) -> None:
        """RowResult.outcome should be RowOutcome, not str."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)
        result = RowResult(
            token=token,
            final_data=make_pipeline_row({}),
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        assert isinstance(result.outcome, RowOutcome)

    def test_all_outcomes_accepted(self) -> None:
        """All RowOutcome values should work with RowResult."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)

        # Iterate over ALL enum members - not a hardcoded subset
        for outcome in RowOutcome:
            sink_name = "output" if outcome in self._SINK_OUTCOMES else None
            result = RowResult(
                token=token,
                final_data=make_pipeline_row({}),
                outcome=outcome,
                sink_name=sink_name,
                error=_error_for(outcome),
            )
            assert result.outcome == outcome
            assert result.outcome is outcome

    def test_row_result_preserves_outcome_identity(self) -> None:
        """RowResult should preserve exact enum identity, not just equality."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)

        for outcome in RowOutcome:
            sink_name = "output" if outcome in self._SINK_OUTCOMES else None
            result = RowResult(
                token=token,
                final_data=make_pipeline_row({}),
                outcome=outcome,
                sink_name=sink_name,
                error=_error_for(outcome),
            )
            # Use 'is' to verify identity, not just value equality
            assert result.outcome is outcome

    def test_outcome_equals_string_for_database_storage(self) -> None:
        """(str, Enum) values equal raw strings for database storage (AUD-001)."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)
        result = RowResult(
            token=token,
            final_data=make_pipeline_row({}),
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        # AUD-001: RowOutcome is now (str, Enum) for token_outcomes table storage.
        # The enum instance IS equal to the raw string for database serialization.
        # Access .value first to avoid mypy type narrowing from string comparison.
        outcome = result.outcome
        assert outcome.value == "completed"
        assert outcome == "completed"

    def test_consumed_in_batch_outcome(self) -> None:
        """CONSUMED_IN_BATCH maps to consumed_in_batch value."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)
        result = RowResult(
            token=token,
            final_data=make_pipeline_row({}),
            outcome=RowOutcome.CONSUMED_IN_BATCH,
        )
        assert result.outcome is RowOutcome.CONSUMED_IN_BATCH
        assert result.outcome.value == "consumed_in_batch"

    def test_all_terminal_outcomes_have_is_terminal_true(self) -> None:
        """All terminal outcomes should have is_terminal=True via RowResult."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)

        for outcome in (value for value in RowOutcome if value.is_terminal):
            sink_name = "output" if outcome in self._SINK_OUTCOMES else None
            result = RowResult(
                token=token,
                final_data=make_pipeline_row({}),
                outcome=outcome,
                sink_name=sink_name,
                error=_error_for(outcome),
            )
            assert result.outcome.is_terminal is True

    def test_buffered_outcome_is_not_terminal(self) -> None:
        """BUFFERED is the only non-terminal outcome."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)
        result = RowResult(
            token=token,
            final_data=make_pipeline_row({}),
            outcome=RowOutcome.BUFFERED,
        )
        assert result.outcome.is_terminal is False
