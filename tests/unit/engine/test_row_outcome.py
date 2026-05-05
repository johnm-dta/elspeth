"""Tests for RowResult's ADR-019 two-axis terminal contract."""

from elspeth.contracts import RowResult, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS
from elspeth.contracts.results import FailureInfo
from elspeth.testing import make_pipeline_row


def _token() -> TokenInfo:
    return TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}), branch_name=None)


def _sink_name_for(path: TerminalPath) -> str | None:
    if path in {TerminalPath.DEFAULT_FLOW, TerminalPath.GATE_ROUTED, TerminalPath.ON_ERROR_ROUTED, TerminalPath.COALESCED}:
        return "output"
    return None


def _error_for(path: TerminalPath) -> FailureInfo | None:
    if path == TerminalPath.ON_ERROR_ROUTED:
        return FailureInfo(
            exception_type="ValueError",
            message="upstream transform raised",
        )
    return None


class TestRowResultOutcome:
    """Tests for RowResult outcome/path pairs."""

    def test_outcome_is_terminal_outcome_enum(self) -> None:
        """RowResult.outcome should be TerminalOutcome, not a raw string."""
        result = RowResult(
            token=_token(),
            final_data=make_pipeline_row({}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        assert isinstance(result.outcome, TerminalOutcome)
        assert result.path == TerminalPath.DEFAULT_FLOW

    def test_all_legal_terminal_pairs_accepted(self) -> None:
        """All legal ADR-019 terminal pairs should work with RowResult."""
        for outcome, path in _LEGAL_TERMINAL_PAIRS:
            result = RowResult(
                token=_token(),
                final_data=make_pipeline_row({}),
                outcome=outcome,
                path=path,
                sink_name=_sink_name_for(path),
                error=_error_for(path),
            )
            assert result.outcome is outcome
            assert result.path is path

    def test_outcome_equals_string_for_database_storage(self) -> None:
        """StrEnum values equal raw strings for database storage (AUD-001)."""
        result = RowResult(
            token=_token(),
            final_data=make_pipeline_row({}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        outcome = result.outcome
        assert outcome is not None
        assert outcome.value == "success"
        assert outcome == "success"

    def test_consumed_in_batch_pair(self) -> None:
        """BATCH_CONSUMED is a transient terminal parent path."""
        result = RowResult(
            token=_token(),
            final_data=make_pipeline_row({}),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
        )
        assert result.outcome is TerminalOutcome.TRANSIENT
        assert result.path is TerminalPath.BATCH_CONSUMED

    def test_buffered_outcome_is_not_completed(self) -> None:
        """BUFFERED is represented by outcome=None and path=BUFFERED."""
        result = RowResult(
            token=_token(),
            final_data=make_pipeline_row({}),
            outcome=None,
            path=TerminalPath.BUFFERED,
        )
        assert result.outcome is None
        assert result.path is TerminalPath.BUFFERED
