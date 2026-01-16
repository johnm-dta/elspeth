"""Tests for RowResult using RowOutcome enum."""

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
        """All RowOutcome values should work."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        for outcome in [RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.FAILED]:
            result = RowResult(
                token=token,
                final_data={},
                outcome=outcome,
            )
            assert result.outcome == outcome

    def test_outcome_not_equal_to_string(self) -> None:
        """Enum values should not equal raw strings."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.COMPLETED,
        )
        # The enum instance is not equal to the raw string
        # (mypy correctly flags this as non-overlapping comparison - that's the point)
        assert result.outcome != "completed"  # type: ignore[comparison-overlap]
        # But value can be accessed
        assert result.outcome.value == "completed"

    def test_consumed_in_batch_outcome(self) -> None:
        """CONSUMED_IN_BATCH maps to consumed_in_batch value."""
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.CONSUMED_IN_BATCH,
        )
        assert result.outcome == RowOutcome.CONSUMED_IN_BATCH
        assert result.outcome.value == "consumed_in_batch"
