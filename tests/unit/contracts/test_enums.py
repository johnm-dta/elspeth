"""Tests for contracts enums."""

import pytest


class TestRowOutcome:
    """Tests for RowOutcome enum - stored in token_outcomes table (AUD-001)."""

    def test_terminal_mappings(self) -> None:
        """RowOutcome terminal/non-terminal mappings are correct."""
        from elspeth.contracts import RowOutcome

        terminal_outcomes = {
            RowOutcome.COMPLETED,
            RowOutcome.ROUTED,
            RowOutcome.ROUTED_ON_ERROR,
            RowOutcome.FORKED,
            RowOutcome.FAILED,
            RowOutcome.QUARANTINED,
            RowOutcome.DIVERTED,
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.DROPPED_BY_FILTER,
            RowOutcome.COALESCED,
            RowOutcome.EXPANDED,
        }
        non_terminal_outcomes = {RowOutcome.BUFFERED}

        assert {o for o in RowOutcome if o.is_terminal} == terminal_outcomes
        assert {o for o in RowOutcome if not o.is_terminal} == non_terminal_outcomes


class TestEnumCoercion:
    """Verify enums that ARE stored can be created from string values."""

    def test_run_status_from_string(self) -> None:
        """Can create RunStatus from string (for DB reads)."""
        from elspeth.contracts import RunStatus

        assert RunStatus("running") == RunStatus.RUNNING
        assert RunStatus("completed") == RunStatus.COMPLETED

    def test_invalid_value_raises(self) -> None:
        """Invalid string raises ValueError - no silent fallback."""
        from elspeth.contracts import RunStatus

        with pytest.raises(ValueError):
            RunStatus("invalid")


class TestTriggerType:
    """TriggerType must match the set of trigger causes the engine can actually emit."""

    def test_values_match_current_engine_producers(self) -> None:
        from elspeth.contracts import TriggerType

        assert {trigger.value for trigger in TriggerType} == {
            "count",
            "timeout",
            "condition",
            "end_of_source",
        }
