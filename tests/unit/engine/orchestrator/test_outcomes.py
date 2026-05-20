# tests/unit/engine/orchestrator/test_outcomes.py
"""Tests for terminal-pair accumulation and coalesce handling functions.

outcomes.py was extracted from duplicated terminal classification switch blocks in
_execute_run() and _process_resumed_rows(). These tests verify that:
1. Every terminal pair is correctly accumulated
2. Coalesce timeouts trigger downstream processing
3. End-of-source coalesce flush handles all paths
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from elspeth.contracts import PendingOutcome, TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError, OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    flush_coalesce_pending,
    handle_coalesce_timeouts,
    reconcile_sink_write_diversions,
)
from elspeth.engine.orchestrator.types import ExecutionCounters
from elspeth.testing import make_row, make_token_info

# =============================================================================
# Helpers
# =============================================================================


def _make_result(
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    token: TokenInfo | None = None,
    sink_name: str | None = None,
) -> Mock:
    """Create a mock RowResult with the two-axis terminal pair."""
    result_token = token or make_token_info()
    if path == TerminalPath.COALESCED and result_token.join_group_id is None:
        result_token = replace(result_token, join_group_id="join-1")
    result = Mock()
    result.outcome = outcome
    result.path = path
    result.token = result_token
    result.sink_name = sink_name
    result.error = None
    return result


def _make_counters() -> ExecutionCounters:
    return ExecutionCounters()


def _make_pending() -> dict[str, list[tuple[TokenInfo, PendingOutcome | None]]]:
    return {"output": [], "error_sink": []}


def _make_merged_coalesce_outcome(
    merged_token: TokenInfo,
    *,
    coalesce_name: str | None = None,
    consumed_tokens: tuple[TokenInfo, ...] | None = None,
) -> Mock:
    """Create a successful CoalesceOutcome-shaped mock."""
    outcome = Mock(spec=None)
    outcome.merged_token = merged_token
    outcome.failure_reason = None
    outcome.coalesce_name = coalesce_name
    outcome.consumed_tokens = consumed_tokens if consumed_tokens is not None else (make_token_info(token_id="consumed-token-1"),)
    return outcome


# =============================================================================
# accumulate_row_outcomes — Individual outcome types
# =============================================================================


class TestAccumulateTerminalPairsCompleted:
    """Tests for COMPLETED outcome handling."""

    def test_completed_increments_succeeded(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="output")]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_succeeded == 1

    def test_completed_appends_to_sink(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="output")]

        accumulate_row_outcomes(results, counters, pending)

        assert len(pending["output"]) == 1
        pending_outcome = pending["output"][0][1]
        assert pending_outcome is not None
        assert pending_outcome.outcome == TerminalOutcome.SUCCESS
        assert pending_outcome.path == TerminalPath.DEFAULT_FLOW

    def test_completed_ignores_branch_name_and_uses_result_sink(self) -> None:
        """COMPLETED routing uses result.sink_name, not token.branch_name."""
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "branch_a": []}
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="branch_a")
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")]

        accumulate_row_outcomes(results, counters, pending)

        assert len(pending["branch_a"]) == 0
        assert len(pending["output"]) == 1

    def test_completed_falls_back_to_sink_name_when_branch_not_in_sinks(self) -> None:
        """COMPLETED token with branch_name not in sinks uses result.sink_name."""
        counters = _make_counters()
        pending = _make_pending()
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="nonexistent")
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")]

        accumulate_row_outcomes(results, counters, pending)

        assert len(pending["output"]) == 1


class TestAccumulateTerminalPairsRouted:
    """Tests for ROUTED outcome handling."""

    def test_routed_increments_counter(self) -> None:
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "risk_sink": []}
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk_sink")]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_routed_success == 1
        assert counters.rows_succeeded == 1
        assert counters.rows_routed_failure == 0

    def test_routed_tracks_destination(self) -> None:
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "risk_sink": []}
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk_sink")]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.routed_destinations["risk_sink"] == 1

    def test_routed_appends_to_named_sink(self) -> None:
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "risk_sink": []}
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk_sink")]

        accumulate_row_outcomes(results, counters, pending)

        assert len(pending["risk_sink"]) == 1
        pending_outcome = pending["risk_sink"][0][1]
        assert pending_outcome is not None
        assert pending_outcome.outcome == TerminalOutcome.SUCCESS
        assert pending_outcome.path == TerminalPath.GATE_ROUTED

    def test_routed_without_sink_name_crashes(self) -> None:
        """ROUTED outcome without sink_name is a contract violation."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name=None)]

        with pytest.raises(OrchestrationInvariantError, match="missing sink_name"):
            accumulate_row_outcomes(results, counters, pending)


class TestAccumulateTerminalPairsRoutedOnError:
    """Tests for ROUTED_ON_ERROR outcome handling — DIVERT path."""

    def _make_routed_on_error_result(
        self,
        *,
        sink_name: str = "error_sink",
        token: TokenInfo | None = None,
    ) -> Mock:
        """ROUTED_ON_ERROR requires a real FailureInfo on .error."""
        from elspeth.contracts.results import FailureInfo

        result = Mock()
        result.outcome = TerminalOutcome.FAILURE
        result.path = TerminalPath.ON_ERROR_ROUTED
        result.token = token or make_token_info()
        result.sink_name = sink_name
        result.error = FailureInfo(
            exception_type="ValueError",
            message="upstream transform raised",
        )
        return result

    def test_routed_on_error_increments_routed_failure_counter(self) -> None:
        """ROUTED_ON_ERROR increments rows_routed_failure (NOT rows_routed_success)."""
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {
            "output": [],
            "error_sink": [],
        }
        results = [self._make_routed_on_error_result(sink_name="error_sink")]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_routed_failure == 1
        assert counters.rows_failed == 1
        assert counters.rows_routed_success == 0

    def test_routed_increments_routed_success_not_failure(self) -> None:
        """ROUTED (MOVE path) increments rows_routed_success (NOT rows_routed_failure)."""
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {
            "output": [],
            "risk_sink": [],
        }
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk_sink")]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_routed_success == 1
        assert counters.rows_succeeded == 1
        assert counters.rows_routed_failure == 0


class TestAccumulateTerminalPairsTerminal:
    """Tests for terminal outcome types that only increment counters."""

    def test_failed_increments_counter(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_failed == 1
        assert len(pending["output"]) == 0

    def test_quarantined_increments_counter(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_quarantined == 1
        assert counters.rows_failed == 1

    def test_forked_increments_counter(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_forked == 1

    def test_expanded_increments_counter(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_expanded == 1

    def test_buffered_increments_counter(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(None, TerminalPath.BUFFERED)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_buffered == 1

    def test_dropped_by_filter_counts_as_succeeded(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_succeeded == 1
        assert len(pending["output"]) == 0

    def test_consumed_in_batch_is_noop(self) -> None:
        """CONSUMED_IN_BATCH doesn't increment any counter (counted on batch flush)."""
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED)]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_succeeded == 0
        assert counters.rows_failed == 0


class TestAccumulateTerminalPairsCoalesced:
    """Tests for COALESCED outcome handling."""

    def test_coalesced_increments_both_counters(self) -> None:
        """COALESCED increments both rows_coalesced AND rows_succeeded."""
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name="output")]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_coalesced == 1
        assert counters.rows_succeeded == 1

    def test_coalesced_routes_to_sink(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        results = [_make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name="output")]

        accumulate_row_outcomes(results, counters, pending)

        assert len(pending["output"]) == 1
        pending_outcome = pending["output"][0][1]
        assert pending_outcome is not None
        assert pending_outcome.outcome == TerminalOutcome.SUCCESS
        assert pending_outcome.path == TerminalPath.COALESCED


class TestAccumulateTerminalPairsExclusiveCounters:
    """Mutation-killing tests: each terminal pair increments only its counters.

    Asserting every counter field per terminal pair catches accidental
    fallthrough in the accumulator switch.
    """

    def _assert_counters(
        self,
        counters: ExecutionCounters,
        *,
        succeeded: int = 0,
        failed: int = 0,
        quarantined: int = 0,
        routed_success: int = 0,
        routed_failure: int = 0,
        forked: int = 0,
        coalesced: int = 0,
        expanded: int = 0,
        buffered: int = 0,
    ) -> None:
        """Assert ALL counter fields match expected values."""
        assert counters.rows_succeeded == succeeded, f"rows_succeeded: expected {succeeded}, got {counters.rows_succeeded}"
        assert counters.rows_failed == failed, f"rows_failed: expected {failed}, got {counters.rows_failed}"
        assert counters.rows_quarantined == quarantined, f"rows_quarantined: expected {quarantined}, got {counters.rows_quarantined}"
        assert counters.rows_routed_success == routed_success, (
            f"rows_routed_success: expected {routed_success}, got {counters.rows_routed_success}"
        )
        assert counters.rows_routed_failure == routed_failure, (
            f"rows_routed_failure: expected {routed_failure}, got {counters.rows_routed_failure}"
        )
        assert counters.rows_forked == forked, f"rows_forked: expected {forked}, got {counters.rows_forked}"
        assert counters.rows_coalesced == coalesced, f"rows_coalesced: expected {coalesced}, got {counters.rows_coalesced}"
        assert counters.rows_expanded == expanded, f"rows_expanded: expected {expanded}, got {counters.rows_expanded}"
        assert counters.rows_buffered == buffered, f"rows_buffered: expected {buffered}, got {counters.rows_buffered}"

    def test_completed_only_increments_succeeded(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="output")],
            counters,
            pending,
        )
        self._assert_counters(counters, succeeded=1)
        assert len(pending["output"]) == 1

    def test_routed_increments_success_and_routed_success_only(self) -> None:
        """GATE_ROUTED is a success terminal with a reporting subset counter."""
        counters = _make_counters()
        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "risk": []}
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk")],
            counters,
            pending,
        )
        self._assert_counters(counters, succeeded=1, routed_success=1)
        assert counters.routed_destinations["risk"] == 1
        assert len(pending["risk"]) == 1
        assert len(pending["output"]) == 0

    def test_failed_only_increments_failed(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)],
            counters,
            pending,
        )
        self._assert_counters(counters, failed=1)
        assert len(pending["output"]) == 0

    def test_quarantined_only_increments_quarantined(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)],
            counters,
            pending,
        )
        self._assert_counters(counters, failed=1, quarantined=1)
        assert len(pending["output"]) == 0

    def test_forked_only_increments_forked(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT)],
            counters,
            pending,
        )
        self._assert_counters(counters, forked=1)
        assert len(pending["output"]) == 0

    def test_consumed_in_batch_increments_nothing(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED)],
            counters,
            pending,
        )
        self._assert_counters(counters)  # all zeros
        assert len(pending["output"]) == 0

    def test_coalesced_increments_coalesced_and_succeeded_only(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name="output")],
            counters,
            pending,
        )
        self._assert_counters(counters, coalesced=1, succeeded=1)
        assert len(pending["output"]) == 1

    def test_expanded_only_increments_expanded(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT)],
            counters,
            pending,
        )
        self._assert_counters(counters, expanded=1)
        assert len(pending["output"]) == 0

    def test_buffered_only_increments_buffered(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(None, TerminalPath.BUFFERED)],
            counters,
            pending,
        )
        self._assert_counters(counters, buffered=1)
        assert len(pending["output"]) == 0

    def test_completed_does_not_match_consumed_in_batch(self) -> None:
        """Guard against `== COMPLETED` -> `>= COMPLETED` catching CONSUMED_IN_BATCH.

        Alphabetically: "completed" < "consumed_in_batch", so `>= "completed"`
        would match "consumed_in_batch". This test sends CONSUMED_IN_BATCH and
        asserts rows_succeeded stays 0.
        """
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED)],
            counters,
            pending,
        )
        assert counters.rows_succeeded == 0
        assert len(pending["output"]) == 0

    def test_failed_does_not_match_forked(self) -> None:
        """Guard against `== FAILED` -> `>= FAILED` catching FORKED.

        Alphabetically: "failed" < "forked", so `>= "failed"` would match
        "forked". This test sends FORKED and asserts rows_failed stays 0.
        """
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT)],
            counters,
            pending,
        )
        assert counters.rows_failed == 0
        assert counters.rows_forked == 1


class TestCoalesceCountingOwnership:
    """Verify single-owner counting: accumulate_row_outcomes owns rows_coalesced.

    _process_merged_coalesce_outcome must NOT increment rows_coalesced.
    All counting happens in accumulate_row_outcomes based on the terminal
    terminal pair. This prevents double-counting for terminal coalesces and
    ensures non-terminal coalesces (which reach COMPLETED) are consistently
    not counted as COALESCED across all paths.
    """

    def test_terminal_timeout_coalesce_counted_exactly_once(self) -> None:
        """Terminal coalesce via timeout: COALESCED result counted by accumulate only."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token)

        coalesce_executor = Mock()
        coalesce_executor.get_registered_names.return_value = ["merge_1"]
        coalesce_executor.check_timeouts.return_value = [outcome]

        processor = Mock()
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, token=merged_token, sink_name="output"),
        ]
        processor.mark_blocked_barrier_terminal.return_value = 1

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        handle_coalesce_timeouts(
            coalesce_executor=coalesce_executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 1, f"Expected 1, got {counters.rows_coalesced} — double-counting if 2"
        assert counters.rows_succeeded == 1

    def test_terminal_flush_coalesce_counted_exactly_once(self) -> None:
        """Terminal coalesce via flush: COALESCED result counted by accumulate only."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token, coalesce_name="merge_1")

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = ["merge_1"]

        processor = Mock()
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, token=merged_token, sink_name="output"),
        ]
        processor.mark_blocked_barrier_terminal.return_value = 1

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 1, f"Expected 1, got {counters.rows_coalesced} — double-counting if 2"
        assert counters.rows_succeeded == 1

    def test_non_terminal_timeout_coalesce_not_counted_as_coalesced(self) -> None:
        """Non-terminal coalesce via timeout: COMPLETED result, rows_coalesced stays 0."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token)

        coalesce_executor = Mock()
        coalesce_executor.get_registered_names.return_value = ["merge_1"]
        coalesce_executor.check_timeouts.return_value = [outcome]

        processor = Mock()
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=merged_token, sink_name="output"),
        ]
        processor.mark_blocked_barrier_terminal.return_value = 1

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        handle_coalesce_timeouts(
            coalesce_executor=coalesce_executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 0, "Non-terminal coalesce should not increment rows_coalesced"
        assert counters.rows_succeeded == 1

    def test_timeout_coalesce_raises_when_scheduler_terminal_count_does_not_match_live_tokens(self) -> None:
        """Coalesce live/durable drift must fail before counters accept the merge."""
        merged_token = make_token_info(token_id="merged")
        consumed_tokens = (
            make_token_info(token_id="token-a"),
            make_token_info(token_id="token-b"),
            make_token_info(token_id="token-c"),
        )
        outcome = _make_merged_coalesce_outcome(merged_token, consumed_tokens=consumed_tokens)

        coalesce_executor = Mock()
        coalesce_executor.get_registered_names.return_value = ["merge_1"]
        coalesce_executor.check_timeouts.return_value = [outcome]

        processor = Mock()
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, token=merged_token, sink_name="output"),
        ]
        processor.mark_blocked_barrier_terminal.return_value = 2

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        with pytest.raises(AuditIntegrityError, match=r"live consumed 3 token.*durable scheduler terminalized 2"):
            handle_coalesce_timeouts(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=node_map,
                processor=processor,
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

        assert counters.rows_coalesced == 0
        assert counters.rows_succeeded == 0


class TestAccumulateTerminalPairsMixed:
    """Tests for multiple results in a single call."""

    def test_multiple_results_accumulated(self) -> None:
        """All results in the iterable are processed."""
        counters = _make_counters()
        pending = _make_pending()
        results = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="output"),
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="output"),
            _make_result(TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
            _make_result(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE),
        ]

        accumulate_row_outcomes(results, counters, pending)

        assert counters.rows_succeeded == 2
        assert counters.rows_failed == 2
        assert counters.rows_quarantined == 1
        assert len(pending["output"]) == 2

    def test_empty_results_is_noop(self) -> None:
        """Empty results list changes nothing."""
        counters = _make_counters()
        pending = _make_pending()

        accumulate_row_outcomes([], counters, pending)

        assert counters.rows_succeeded == 0
        assert counters.rows_failed == 0


class TestReconcileSinkWriteDiversions:
    """Tests for post-sink durable counter reconciliation."""

    def test_routed_diversion_decrements_routed_counters(self) -> None:
        """A routed-success sink diversion decrements success and routed subsets."""
        counters = _make_counters()
        counters.rows_succeeded = 7
        counters.rows_routed_success = 1
        counters.routed_destinations["risk_sink"] = 1

        reconcile_sink_write_diversions(
            counters,
            sink_name="risk_sink",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.GATE_ROUTED),
            diversion_count=1,
        )

        assert counters.rows_succeeded == 6
        assert counters.rows_routed_success == 0
        assert counters.rows_routed_failure == 0
        assert counters.routed_destinations == {}

    def test_terminal_coalesced_diversion_decrements_coalesced_and_success(self) -> None:
        counters = _make_counters()
        pending = _make_pending()
        accumulate_row_outcomes(
            [_make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name="output")],
            counters,
            pending,
        )

        reconcile_sink_write_diversions(
            counters,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.COALESCED),
            diversion_count=1,
        )

        assert counters.rows_coalesced == 0
        assert counters.rows_succeeded == 0


# =============================================================================
# handle_coalesce_timeouts
# =============================================================================


class TestHandleCoalesceTimeouts:
    """Tests for per-row coalesce timeout checks."""

    def _setup(
        self,
        *,
        timed_out_outcomes: list[Mock] | None = None,
        total_transforms: int = 2,
        coalesce_step: int = 1,
    ) -> tuple[Mock, Mock, ExecutionCounters, dict[str, list[tuple[TokenInfo, PendingOutcome | None]]], dict[CoalesceName, NodeID]]:
        coalesce_executor = Mock()
        coalesce_executor.get_registered_names.return_value = ["merge_1"]
        coalesce_executor.check_timeouts.return_value = timed_out_outcomes or []

        processor = Mock()
        processor.process_token.return_value = []
        processor.resolve_node_step.return_value = coalesce_step
        processor.mark_blocked_barrier_terminal.return_value = 1

        counters = _make_counters()
        pending = _make_pending()
        coalesce_node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        return coalesce_executor, processor, counters, pending, coalesce_node_map

    def test_no_timeouts_is_noop(self) -> None:
        """No timed-out coalesces means nothing happens."""
        executor, processor, counters, pending, node_map = self._setup()

        handle_coalesce_timeouts(
            coalesce_executor=executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 0
        processor.process_token.assert_not_called()

    def test_non_terminal_coalesce_counts_succeeded_not_coalesced(self) -> None:
        """Non-terminal coalesce: merged token continues through downstream transforms.

        The continuation returns COMPLETED (not COALESCED) because there are
        transforms after the coalesce. rows_coalesced is NOT incremented because
        the row's terminal state is COMPLETED, not COALESCED. Counting ownership
        for rows_coalesced belongs exclusively to accumulate_row_outcomes.
        """
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token)

        executor, processor, counters, pending, node_map = self._setup(
            timed_out_outcomes=[outcome],
            total_transforms=3,
            coalesce_step=1,
        )
        # Non-terminal: downstream transforms produce COMPLETED
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=merged_token, sink_name="output"),
        ]

        ctx = Mock()
        handle_coalesce_timeouts(
            coalesce_executor=executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=ctx,
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 0, "Non-terminal coalesce should not count as COALESCED"
        assert counters.rows_succeeded == 1
        processor.process_token.assert_called_once_with(
            token=merged_token,
            ctx=ctx,
            current_node_id=NodeID("coalesce::merge_1"),
            coalesce_node_id=NodeID("coalesce::merge_1"),
            coalesce_name=CoalesceName("merge_1"),
        )

    def test_merged_timeout_missing_node_map_entry_crashes_before_continuation(self) -> None:
        """Merged timeout with graph-map drift raises a typed invariant error."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token)

        executor, processor, counters, pending, _node_map = self._setup(timed_out_outcomes=[outcome])

        with pytest.raises(OrchestrationInvariantError, match="merge_1"):
            handle_coalesce_timeouts(
                coalesce_executor=executor,
                coalesce_node_map={},
                processor=processor,
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

        processor.process_token.assert_not_called()
        assert counters.rows_succeeded == 0
        assert counters.rows_coalesced == 0
        assert pending == _make_pending()

    def test_terminal_coalesce_counts_coalesced_and_succeeded(self) -> None:
        """Terminal coalesce: no downstream transforms, processor returns COALESCED."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token)

        executor, processor, counters, pending, node_map = self._setup(
            timed_out_outcomes=[outcome],
            total_transforms=1,
            coalesce_step=1,
        )
        # Terminal: processor returns COALESCED (no downstream transforms)
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, token=merged_token, sink_name="output"),
        ]

        ctx = Mock()
        handle_coalesce_timeouts(
            coalesce_executor=executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=ctx,
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 1
        assert counters.rows_succeeded == 1
        assert len(pending["output"]) == 1
        processor.process_token.assert_called_once_with(
            token=merged_token,
            ctx=ctx,
            current_node_id=NodeID("coalesce::merge_1"),
            coalesce_node_id=NodeID("coalesce::merge_1"),
            coalesce_name=CoalesceName("merge_1"),
        )

    def test_failure_increments_coalesce_failed(self) -> None:
        """Failed coalesce (missing branches) increments coalesce_failed and rows_failed.

        Regression: elspeth-045f60670c — rows_failed must reflect consumed tokens.
        """
        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = "quorum_not_met"
        outcome.consumed_tokens = (
            make_token_info(token_id="token-a"),
            make_token_info(token_id="token-b"),
            make_token_info(token_id="token-c"),
        )

        executor, processor, counters, pending, node_map = self._setup(
            timed_out_outcomes=[outcome],
        )
        processor.mark_blocked_barrier_terminal.return_value = 3

        handle_coalesce_timeouts(
            coalesce_executor=executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesce_failed == 1
        assert counters.rows_failed == 3  # one per consumed token
        assert counters.rows_coalesced == 0
        processor.mark_blocked_barrier_terminal.assert_called_once_with(
            "merge_1",
            ("token-a", "token-b", "token-c"),
        )

    def test_failed_timeout_raises_when_scheduler_terminal_count_does_not_match_live_tokens(self) -> None:
        """Failed timeout coalesces must reconcile durable scheduler rows before accepting counters."""
        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = "quorum_not_met"
        outcome.consumed_tokens = (
            make_token_info(token_id="token-a"),
            make_token_info(token_id="token-b"),
        )

        executor, processor, counters, pending, node_map = self._setup(
            timed_out_outcomes=[outcome],
        )
        processor.mark_blocked_barrier_terminal.return_value = 1

        with pytest.raises(AuditIntegrityError, match=r"live consumed 2 token.*durable scheduler terminalized 1"):
            handle_coalesce_timeouts(
                coalesce_executor=executor,
                coalesce_node_map=node_map,
                processor=processor,
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

        assert counters.rows_coalesce_failed == 0
        assert counters.rows_failed == 0
        processor.process_token.assert_not_called()

    def test_failure_emits_token_completed_telemetry_for_each_consumed_token(self) -> None:
        """Timeout-driven coalesce failures must surface in telemetry once per token."""
        tokens = (
            make_token_info(row_id="row-1", token_id="token-1"),
            make_token_info(row_id="row-2", token_id="token-2"),
        )
        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = "quorum_not_met"
        outcome.consumed_tokens = tokens

        executor, processor, counters, pending, node_map = self._setup(
            timed_out_outcomes=[outcome],
        )
        processor.mark_blocked_barrier_terminal.return_value = 2

        ctx = Mock()
        ctx.run_id = "run-1"
        ctx.telemetry_emit = Mock()

        handle_coalesce_timeouts(
            coalesce_executor=executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=ctx,
            counters=counters,
            pending_tokens=pending,
        )

        assert ctx.telemetry_emit.call_count == 2
        emitted = [call.args[0] for call in ctx.telemetry_emit.call_args_list]
        assert [event.token_id for event in emitted] == ["token-1", "token-2"]
        assert all(event.run_id == "run-1" for event in emitted)
        assert all(event.outcome == TerminalOutcome.FAILURE for event in emitted)
        assert all(event.path == TerminalPath.UNROUTED for event in emitted)


# =============================================================================
# flush_coalesce_pending
# =============================================================================


class TestFlushCoalescePending:
    """Tests for end-of-source coalesce flush."""

    def test_non_terminal_flush_counts_succeeded_not_coalesced(self) -> None:
        """Non-terminal flush: merged token continues through downstream transforms."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token, coalesce_name="merge_1")

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = ["merge_1"]

        processor = Mock()
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=merged_token, sink_name="output"),
        ]
        processor.resolve_node_step.return_value = 0
        processor.mark_blocked_barrier_terminal.return_value = 1

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        ctx = Mock()
        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=ctx,
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 0, "Non-terminal flush should not count as COALESCED"
        assert counters.rows_succeeded == 1
        processor.process_token.assert_called_once_with(
            token=merged_token,
            ctx=ctx,
            current_node_id=NodeID("coalesce::merge_1"),
            coalesce_node_id=NodeID("coalesce::merge_1"),
            coalesce_name=CoalesceName("merge_1"),
        )

    def test_terminal_flush_counts_coalesced_and_succeeded(self) -> None:
        """Terminal flush: no downstream transforms, processor returns COALESCED."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token, coalesce_name="merge_1")

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = ["merge_1"]

        processor = Mock()
        processor.resolve_node_step.return_value = 1
        processor.mark_blocked_barrier_terminal.return_value = 1
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, token=merged_token, sink_name="output"),
        ]
        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        ctx = Mock()
        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map=node_map,
            processor=processor,
            ctx=ctx,
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 1
        assert counters.rows_succeeded == 1
        assert len(pending["output"]) == 1
        processor.process_token.assert_called_once_with(
            token=merged_token,
            ctx=ctx,
            current_node_id=NodeID("coalesce::merge_1"),
            coalesce_node_id=NodeID("coalesce::merge_1"),
            coalesce_name=CoalesceName("merge_1"),
        )

    def test_merged_flush_missing_node_map_entry_crashes_before_continuation(self) -> None:
        """Merged flush with graph-map drift raises a typed invariant error."""
        merged_token = make_token_info()
        outcome = _make_merged_coalesce_outcome(merged_token, coalesce_name="merge_1")

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = ["merge_1"]

        processor = Mock()
        processor.process_token.return_value = [
            _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=merged_token, sink_name="output"),
        ]
        counters = _make_counters()
        pending = _make_pending()

        with pytest.raises(OrchestrationInvariantError, match="merge_1"):
            flush_coalesce_pending(
                coalesce_executor=coalesce_executor,
                coalesce_node_map={},
                processor=processor,
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

        processor.process_token.assert_not_called()
        assert counters.rows_succeeded == 0
        assert counters.rows_coalesced == 0
        assert pending == _make_pending()

    def test_failure_increments_coalesce_failed(self) -> None:
        """Failed flush outcomes increment coalesce_failed and rows_failed counters.

        Regression: elspeth-045f60670c — rows_failed must reflect consumed tokens.
        """
        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = "incomplete_branches"
        outcome.coalesce_name = None
        outcome.consumed_tokens = (Mock(), Mock())  # 2 consumed tokens

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = []

        counters = _make_counters()
        pending = _make_pending()

        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map={},
            processor=Mock(),
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesce_failed == 1
        assert counters.rows_failed == 2

    def test_failure_emits_token_completed_telemetry_for_each_consumed_token(self) -> None:
        """Flush-driven coalesce failures must surface in telemetry once per token."""
        tokens = (
            make_token_info(row_id="row-1", token_id="token-1"),
            make_token_info(row_id="row-2", token_id="token-2"),
        )
        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = "incomplete_branches"
        outcome.coalesce_name = None
        outcome.consumed_tokens = tokens

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = []

        counters = _make_counters()
        pending = _make_pending()
        ctx = Mock()
        ctx.run_id = "run-1"
        ctx.telemetry_emit = Mock()

        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map={},
            processor=Mock(),
            ctx=ctx,
            counters=counters,
            pending_tokens=pending,
        )

        assert ctx.telemetry_emit.call_count == 2
        emitted = [call.args[0] for call in ctx.telemetry_emit.call_args_list]
        assert [event.token_id for event in emitted] == ["token-1", "token-2"]
        assert all(event.run_id == "run-1" for event in emitted)
        assert all(event.outcome == TerminalOutcome.FAILURE for event in emitted)
        assert all(event.path == TerminalPath.UNROUTED for event in emitted)

    def test_empty_flush_is_noop(self) -> None:
        """No pending coalesces means nothing happens."""
        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = []

        counters = _make_counters()
        pending = _make_pending()
        processor = Mock()

        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map={},
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        assert counters.rows_coalesced == 0
        assert counters.rows_coalesce_failed == 0
        processor.mark_blocked_barrier_terminal.assert_not_called()
        coalesce_executor.get_registered_names.assert_not_called()

    def test_registered_barrier_without_live_outcome_does_not_terminalize_scheduler_work(self) -> None:
        """End-of-source sweep cannot turn durable BLOCKED rows terminal without live token evidence."""
        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = []
        coalesce_executor.get_registered_names.return_value = ["merge_1"]

        counters = _make_counters()
        pending = _make_pending()
        processor = Mock()

        flush_coalesce_pending(
            coalesce_executor=coalesce_executor,
            coalesce_node_map={CoalesceName("merge_1"): NodeID("coalesce::merge_1")},
            processor=processor,
            ctx=Mock(),
            counters=counters,
            pending_tokens=pending,
        )

        processor.mark_blocked_barrier_terminal.assert_not_called()
        coalesce_executor.get_registered_names.assert_not_called()


# =============================================================================
# CoalesceOutcome state validation
# BUG FIX: P1-2026-02-14 — invalid CoalesceOutcome states were silently
# ignored, potentially losing work or hiding bugs.
# =============================================================================


class TestCoalesceOutcomeValidation:
    """Tests that invalid CoalesceOutcome states crash instead of being silently ignored.

    BUG: handle_coalesce_timeouts and flush_coalesce_pending only checked
    two branches (merged_token / failure_reason) with no else clause.
    If a malformed outcome arrived (both None, or both set), it was silently
    dropped, violating fail-fast expectations.
    """

    def test_handle_coalesce_timeouts_rejects_both_none(self) -> None:
        """Outcome with both merged_token=None and failure_reason=None crashes."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = None

        coalesce_executor = Mock()
        coalesce_executor.get_registered_names.return_value = ["merge_1"]
        coalesce_executor.check_timeouts.return_value = [outcome]

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        with pytest.raises(OrchestrationInvariantError, match="Invalid CoalesceOutcome state"):
            handle_coalesce_timeouts(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=node_map,
                processor=Mock(),
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

    def test_handle_coalesce_timeouts_rejects_both_set(self) -> None:
        """Outcome with both merged_token and failure_reason set crashes."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        outcome = Mock()
        outcome.merged_token = make_token_info()
        outcome.failure_reason = "some_failure"

        coalesce_executor = Mock()
        coalesce_executor.get_registered_names.return_value = ["merge_1"]
        coalesce_executor.check_timeouts.return_value = [outcome]

        counters = _make_counters()
        pending = _make_pending()
        node_map = {CoalesceName("merge_1"): NodeID("coalesce::merge_1")}

        with pytest.raises(OrchestrationInvariantError, match="Invalid CoalesceOutcome state"):
            handle_coalesce_timeouts(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=node_map,
                processor=Mock(),
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

    def test_flush_coalesce_pending_rejects_both_none(self) -> None:
        """flush_coalesce_pending crashes on outcome with both fields None."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        outcome = Mock()
        outcome.merged_token = None
        outcome.failure_reason = None

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = []

        counters = _make_counters()
        pending = _make_pending()

        with pytest.raises(OrchestrationInvariantError, match="Invalid CoalesceOutcome state"):
            flush_coalesce_pending(
                coalesce_executor=coalesce_executor,
                coalesce_node_map={},
                processor=Mock(),
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

    def test_flush_coalesce_pending_rejects_both_set(self) -> None:
        """flush_coalesce_pending crashes on outcome with both fields set."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        outcome = Mock()
        outcome.merged_token = make_token_info()
        outcome.failure_reason = "contradictory"
        outcome.coalesce_name = "merge_1"

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = ["merge_1"]

        counters = _make_counters()
        pending = _make_pending()

        with pytest.raises(OrchestrationInvariantError, match="Invalid CoalesceOutcome state"):
            flush_coalesce_pending(
                coalesce_executor=coalesce_executor,
                coalesce_node_map={CoalesceName("merge_1"): NodeID("coalesce::merge_1")},
                processor=Mock(),
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )

    def test_flush_coalesce_pending_rejects_missing_coalesce_name(self) -> None:
        """flush_coalesce_pending crashes if merged_token present but coalesce_name is None."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        outcome = Mock()
        outcome.merged_token = make_token_info()
        outcome.failure_reason = None
        outcome.coalesce_name = None

        coalesce_executor = Mock()
        coalesce_executor.flush_pending.return_value = [outcome]
        coalesce_executor.get_registered_names.return_value = []

        counters = _make_counters()
        pending = _make_pending()

        with pytest.raises(OrchestrationInvariantError, match="coalesce_name is None"):
            flush_coalesce_pending(
                coalesce_executor=coalesce_executor,
                coalesce_node_map={},
                processor=Mock(),
                ctx=Mock(),
                counters=counters,
                pending_tokens=pending,
            )
