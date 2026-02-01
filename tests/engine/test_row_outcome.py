"""Tests for RowResult using RowOutcome enum.

This module tests the engine's RowResult dataclass with RowOutcome enum values.
RowOutcome enum contract tests are in tests/contracts/test_enums.py.
"""

from elspeth.contracts import RowOutcome, RowResult, TokenInfo


class TestRowResultOutcome:
    """Tests for RowResult.outcome as enum."""

    def test_outcome_is_enum(self) -> None:
        """RowResult.outcome should be RowOutcome, not str."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.COMPLETED,
        )
        assert isinstance(result.outcome, RowOutcome)

    def test_all_outcomes_accepted(self) -> None:
        """All RowOutcome values should work with RowResult."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)

        # Iterate over ALL enum members - not a hardcoded subset
        for outcome in RowOutcome:
            result = RowResult(
                token=token,
                final_data={},
                outcome=outcome,
            )
            assert result.outcome == outcome
            assert result.outcome is outcome

    def test_row_result_preserves_outcome_identity(self) -> None:
        """RowResult should preserve exact enum identity, not just equality."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)

        for outcome in RowOutcome:
            result = RowResult(token=token, final_data={}, outcome=outcome)
            # Use 'is' to verify identity, not just value equality
            assert result.outcome is outcome

    def test_outcome_equals_string_for_database_storage(self) -> None:
        """(str, Enum) values equal raw strings for database storage (AUD-001)."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.COMPLETED,
        )
        # AUD-001: RowOutcome is now (str, Enum) for token_outcomes table storage.
        # The enum instance IS equal to the raw string for database serialization.
        # Access .value first to avoid mypy type narrowing from string comparison.
        outcome = result.outcome
        assert outcome.value == "completed"
        assert outcome == "completed"  # type: ignore[comparison-overlap]

    def test_consumed_in_batch_outcome(self) -> None:
        """CONSUMED_IN_BATCH maps to consumed_in_batch value."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.CONSUMED_IN_BATCH,
        )
        assert result.outcome is RowOutcome.CONSUMED_IN_BATCH
        assert result.outcome.value == "consumed_in_batch"

    def test_all_terminal_outcomes_have_is_terminal_true(self) -> None:
        """All terminal outcomes should have is_terminal=True via RowResult."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)

        terminal_outcomes = [
            RowOutcome.COMPLETED,
            RowOutcome.ROUTED,
            RowOutcome.FORKED,
            RowOutcome.FAILED,
            RowOutcome.QUARANTINED,
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.COALESCED,
            RowOutcome.EXPANDED,
        ]

        for outcome in terminal_outcomes:
            result = RowResult(token=token, final_data={}, outcome=outcome)
            assert result.outcome.is_terminal is True

    def test_buffered_outcome_is_not_terminal(self) -> None:
        """BUFFERED is the only non-terminal outcome."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.BUFFERED,
        )
        assert result.outcome.is_terminal is False
