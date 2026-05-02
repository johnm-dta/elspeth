"""Test DIVERTED RowOutcome and RunResult.rows_diverted."""

from __future__ import annotations

import pytest

from elspeth.contracts.enums import RowOutcome, RunStatus
from elspeth.contracts.run_result import RunResult


class TestDivertedOutcome:
    def test_diverted_in_terminal_set(self) -> None:
        """DIVERTED should be in the same category as QUARANTINED and ROUTED."""
        terminal_outcomes = [o for o in RowOutcome if o.is_terminal]
        assert RowOutcome.DIVERTED in terminal_outcomes


class TestRunResultDiverted:
    def test_rows_diverted_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="rows_diverted"):
            RunResult(
                run_id="test-1",
                status=RunStatus.COMPLETED,
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_diverted=-1,
            )
