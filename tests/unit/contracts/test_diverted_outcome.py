"""Test ADR-019 sink diversion pair and RunResult.rows_diverted."""

from __future__ import annotations

import pytest

from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.run_result import RunResult


class TestSinkDiversionPairs:
    def test_discard_mode_diversion_is_terminal_failure(self) -> None:
        """Discard-mode diversion is a terminal failure under ADR-019."""
        assert (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED) in _LEGAL_TERMINAL_PAIRS


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
