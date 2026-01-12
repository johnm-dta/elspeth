# tests/plugins/test_results.py
"""Tests for plugin result types."""

import pytest


class TestRowOutcome:
    """Terminal states for rows."""

    def test_all_terminal_states_exist(self) -> None:
        from elspeth.plugins.results import RowOutcome

        # Every row must reach exactly one terminal state
        assert RowOutcome.COMPLETED.value == "completed"
        assert RowOutcome.ROUTED.value == "routed"
        assert RowOutcome.FORKED.value == "forked"
        assert RowOutcome.CONSUMED_IN_BATCH.value == "consumed_in_batch"
        assert RowOutcome.COALESCED.value == "coalesced"
        assert RowOutcome.QUARANTINED.value == "quarantined"
        assert RowOutcome.FAILED.value == "failed"

    def test_outcome_is_enum(self) -> None:
        from enum import Enum

        from elspeth.plugins.results import RowOutcome

        assert issubclass(RowOutcome, Enum)
