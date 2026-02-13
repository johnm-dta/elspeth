# tests/property/engine/test_orchestrator_lifecycle_properties.py
"""Property-based tests for orchestrator lifecycle types and accumulation logic.

The orchestrator uses a layered counter system:
1. AggregationFlushResult (frozen): immutable result from flushing aggregation buffers
2. ExecutionCounters (mutable): running totals accumulated during pipeline execution
3. RunResult (frozen): final snapshot produced from ExecutionCounters

Key invariants:
- AggregationFlushResult.__add__ forms a commutative monoid (associative, commutative, identity)
- ExecutionCounters.accumulate_flush_result is a fold: counters + result == expected
- ExecutionCounters.to_run_result preserves all counter fields faithfully
- accumulate_row_outcomes maps each RowOutcome to exactly the right counter(s)
- routed_destinations merge uses Counter semantics (additive per sink name)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts import PendingOutcome, RowOutcome, RunStatus, TokenInfo
from elspeth.contracts.results import RowResult
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.types import (
    AggregationFlushResult,
    ExecutionCounters,
    RunResult,
)
from elspeth.testing import make_pipeline_row

# Shared contract for test helpers — OBSERVED mode with no fixed fields
_TEST_CONTRACT = SchemaContract(mode="OBSERVED", fields=())


# =============================================================================
# Strategies
# =============================================================================

# Non-negative integers for counter fields
counter_values = st.integers(min_value=0, max_value=10_000)

# Sink names for routed_destinations
sink_names = st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_")

# Routed destination dicts
routed_dests = st.dictionaries(
    keys=sink_names,
    values=st.integers(min_value=1, max_value=1000),
    min_size=0,
    max_size=5,
)


@st.composite
def aggregation_flush_results(draw: st.DrawFn) -> AggregationFlushResult:
    """Generate arbitrary AggregationFlushResult instances."""
    return AggregationFlushResult(
        rows_succeeded=draw(counter_values),
        rows_failed=draw(counter_values),
        rows_routed=draw(counter_values),
        rows_quarantined=draw(counter_values),
        rows_coalesced=draw(counter_values),
        rows_forked=draw(counter_values),
        rows_expanded=draw(counter_values),
        rows_buffered=draw(counter_values),
        routed_destinations=draw(routed_dests),
    )


@st.composite
def execution_counters(draw: st.DrawFn) -> ExecutionCounters:
    """Generate arbitrary ExecutionCounters instances."""
    return ExecutionCounters(
        rows_processed=draw(counter_values),
        rows_succeeded=draw(counter_values),
        rows_failed=draw(counter_values),
        rows_routed=draw(counter_values),
        rows_quarantined=draw(counter_values),
        rows_forked=draw(counter_values),
        rows_coalesced=draw(counter_values),
        rows_coalesce_failed=draw(counter_values),
        rows_expanded=draw(counter_values),
        rows_buffered=draw(counter_values),
        routed_destinations=Counter(draw(routed_dests)),
    )


def _make_token(*, branch_name: str | None = None) -> TokenInfo:
    """Create a minimal TokenInfo for testing."""
    row = PipelineRow({"field": "value"}, _TEST_CONTRACT)
    return TokenInfo(
        row_id="row-1",
        token_id="tok-1",
        row_data=row,
        branch_name=branch_name,
    )


def _make_row_result(
    outcome: RowOutcome,
    *,
    sink_name: str | None = None,
    branch_name: str | None = None,
) -> RowResult:
    """Create a RowResult for outcome accumulation tests."""
    if outcome in (RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.COALESCED) and sink_name is None:
        sink_name = "default"
    token = _make_token(branch_name=branch_name)
    return RowResult(
        token=token,
        final_data=make_pipeline_row({"field": "value"}),
        outcome=outcome,
        sink_name=sink_name,
    )


# =============================================================================
# AggregationFlushResult.__add__ Monoid Properties
# =============================================================================


class TestFlushResultMonoidProperties:
    """AggregationFlushResult.__add__ must form a commutative monoid."""

    @given(a=aggregation_flush_results(), b=aggregation_flush_results())
    @settings(max_examples=200)
    def test_add_commutativity(self, a: AggregationFlushResult, b: AggregationFlushResult) -> None:
        """Property: a + b == b + a for all counter fields."""
        ab = a + b
        ba = b + a

        assert ab.rows_succeeded == ba.rows_succeeded
        assert ab.rows_failed == ba.rows_failed
        assert ab.rows_routed == ba.rows_routed
        assert ab.rows_quarantined == ba.rows_quarantined
        assert ab.rows_coalesced == ba.rows_coalesced
        assert ab.rows_forked == ba.rows_forked
        assert ab.rows_expanded == ba.rows_expanded
        assert ab.rows_buffered == ba.rows_buffered
        assert ab.routed_destinations == ba.routed_destinations

    @given(
        a=aggregation_flush_results(),
        b=aggregation_flush_results(),
        c=aggregation_flush_results(),
    )
    @settings(max_examples=200)
    def test_add_associativity(
        self,
        a: AggregationFlushResult,
        b: AggregationFlushResult,
        c: AggregationFlushResult,
    ) -> None:
        """Property: (a + b) + c == a + (b + c)."""
        left = (a + b) + c
        right = a + (b + c)

        assert left.rows_succeeded == right.rows_succeeded
        assert left.rows_failed == right.rows_failed
        assert left.rows_routed == right.rows_routed
        assert left.rows_quarantined == right.rows_quarantined
        assert left.rows_coalesced == right.rows_coalesced
        assert left.rows_forked == right.rows_forked
        assert left.rows_expanded == right.rows_expanded
        assert left.rows_buffered == right.rows_buffered
        assert left.routed_destinations == right.routed_destinations

    @given(a=aggregation_flush_results())
    @settings(max_examples=200)
    def test_add_identity(self, a: AggregationFlushResult) -> None:
        """Property: a + zero == a (right identity)."""
        zero = AggregationFlushResult()
        result = a + zero

        assert result.rows_succeeded == a.rows_succeeded
        assert result.rows_failed == a.rows_failed
        assert result.rows_routed == a.rows_routed
        assert result.rows_quarantined == a.rows_quarantined
        assert result.rows_coalesced == a.rows_coalesced
        assert result.rows_forked == a.rows_forked
        assert result.rows_expanded == a.rows_expanded
        assert result.rows_buffered == a.rows_buffered
        assert result.routed_destinations == a.routed_destinations

    @given(a=aggregation_flush_results())
    @settings(max_examples=200)
    def test_add_left_identity(self, a: AggregationFlushResult) -> None:
        """Property: zero + a == a (left identity)."""
        zero = AggregationFlushResult()
        result = zero + a

        assert result.rows_succeeded == a.rows_succeeded
        assert result.rows_failed == a.rows_failed
        assert result.rows_routed == a.rows_routed
        assert result.rows_quarantined == a.rows_quarantined
        assert result.rows_coalesced == a.rows_coalesced
        assert result.rows_forked == a.rows_forked
        assert result.rows_expanded == a.rows_expanded
        assert result.rows_buffered == a.rows_buffered
        assert result.routed_destinations == a.routed_destinations

    @given(a=aggregation_flush_results(), b=aggregation_flush_results())
    @settings(max_examples=100)
    def test_add_preserves_all_counter_fields(self, a: AggregationFlushResult, b: AggregationFlushResult) -> None:
        """Property: Addition sums each int field individually."""
        result = a + b

        assert result.rows_succeeded == a.rows_succeeded + b.rows_succeeded
        assert result.rows_failed == a.rows_failed + b.rows_failed
        assert result.rows_routed == a.rows_routed + b.rows_routed
        assert result.rows_quarantined == a.rows_quarantined + b.rows_quarantined
        assert result.rows_coalesced == a.rows_coalesced + b.rows_coalesced
        assert result.rows_forked == a.rows_forked + b.rows_forked
        assert result.rows_expanded == a.rows_expanded + b.rows_expanded
        assert result.rows_buffered == a.rows_buffered + b.rows_buffered

    @given(a=aggregation_flush_results(), b=aggregation_flush_results())
    @settings(max_examples=100)
    def test_routed_destinations_merge_is_additive(self, a: AggregationFlushResult, b: AggregationFlushResult) -> None:
        """Property: routed_destinations merge adds counts per sink name."""
        result = a + b

        # Every sink in the result must have the sum of both inputs
        all_sinks = set(a.routed_destinations) | set(b.routed_destinations)
        for sink in all_sinks:
            expected = a.routed_destinations.get(sink, 0) + b.routed_destinations.get(sink, 0)
            assert result.routed_destinations[sink] == expected

    @given(a=aggregation_flush_results())
    @settings(max_examples=100)
    def test_add_returns_new_instance(self, a: AggregationFlushResult) -> None:
        """Property: __add__ returns a new AggregationFlushResult (immutable)."""
        zero = AggregationFlushResult()
        result = a + zero

        # Must be a new object (frozen dataclass, not mutated in place)
        assert result is not a

    def test_zero_element_all_fields_zero(self) -> None:
        """Property: Default AggregationFlushResult has all counters at zero."""
        zero = AggregationFlushResult()
        assert zero.rows_succeeded == 0
        assert zero.rows_failed == 0
        assert zero.rows_routed == 0
        assert zero.rows_quarantined == 0
        assert zero.rows_coalesced == 0
        assert zero.rows_forked == 0
        assert zero.rows_expanded == 0
        assert zero.rows_buffered == 0
        assert zero.routed_destinations == {}


# =============================================================================
# ExecutionCounters.accumulate_flush_result Properties
# =============================================================================


class TestAccumulateFlushResultProperties:
    """accumulate_flush_result must correctly fold results into counters."""

    @given(counters=execution_counters(), flush=aggregation_flush_results())
    @settings(max_examples=200)
    def test_accumulate_adds_to_existing(self, counters: ExecutionCounters, flush: AggregationFlushResult) -> None:
        """Property: After accumulate, each field == original + flush."""
        # Snapshot before
        before_succeeded = counters.rows_succeeded
        before_failed = counters.rows_failed
        before_routed = counters.rows_routed
        before_quarantined = counters.rows_quarantined
        before_coalesced = counters.rows_coalesced
        before_forked = counters.rows_forked
        before_expanded = counters.rows_expanded
        before_buffered = counters.rows_buffered
        dict(counters.routed_destinations)

        counters.accumulate_flush_result(flush)

        assert counters.rows_succeeded == before_succeeded + flush.rows_succeeded
        assert counters.rows_failed == before_failed + flush.rows_failed
        assert counters.rows_routed == before_routed + flush.rows_routed
        assert counters.rows_quarantined == before_quarantined + flush.rows_quarantined
        assert counters.rows_coalesced == before_coalesced + flush.rows_coalesced
        assert counters.rows_forked == before_forked + flush.rows_forked
        assert counters.rows_expanded == before_expanded + flush.rows_expanded
        assert counters.rows_buffered == before_buffered + flush.rows_buffered

    @given(counters=execution_counters(), flush=aggregation_flush_results())
    @settings(max_examples=100)
    def test_accumulate_merges_routed_destinations(self, counters: ExecutionCounters, flush: AggregationFlushResult) -> None:
        """Property: routed_destinations are merged additively."""
        before_dests = dict(counters.routed_destinations)

        counters.accumulate_flush_result(flush)

        all_sinks = set(before_dests) | set(flush.routed_destinations)
        for sink in all_sinks:
            expected = before_dests.get(sink, 0) + flush.routed_destinations.get(sink, 0)
            assert counters.routed_destinations[sink] == expected

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_accumulate_zero_is_noop(self, counters: ExecutionCounters) -> None:
        """Property: Accumulating a zero-valued flush result changes nothing."""
        before_succeeded = counters.rows_succeeded
        before_failed = counters.rows_failed
        before_routed = counters.rows_routed
        before_dests = dict(counters.routed_destinations)

        counters.accumulate_flush_result(AggregationFlushResult())

        assert counters.rows_succeeded == before_succeeded
        assert counters.rows_failed == before_failed
        assert counters.rows_routed == before_routed
        assert dict(counters.routed_destinations) == before_dests

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_accumulate_does_not_touch_rows_processed(self, counters: ExecutionCounters) -> None:
        """Property: accumulate_flush_result never modifies rows_processed.

        rows_processed is only incremented by the main loop (per source row),
        not by aggregation flushes.
        """
        before = counters.rows_processed
        flush = AggregationFlushResult(rows_succeeded=5, rows_failed=3)
        counters.accumulate_flush_result(flush)
        assert counters.rows_processed == before

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_accumulate_does_not_touch_coalesce_failed(self, counters: ExecutionCounters) -> None:
        """Property: accumulate_flush_result never modifies rows_coalesce_failed.

        AggregationFlushResult has no coalesce_failed field — that counter
        is only managed by coalesce timeout/flush code, not aggregation flushes.
        """
        before = counters.rows_coalesce_failed
        flush = AggregationFlushResult(rows_succeeded=5, rows_coalesced=2)
        counters.accumulate_flush_result(flush)
        assert counters.rows_coalesce_failed == before

    @given(
        counters=execution_counters(),
        flushes=st.lists(aggregation_flush_results(), min_size=2, max_size=5),
    )
    @settings(max_examples=50)
    def test_sequential_accumulation_equals_summed(self, counters: ExecutionCounters, flushes: list[AggregationFlushResult]) -> None:
        """Property: Accumulating N flushes sequentially == accumulating their sum.

        This tests the fold property: accumulate(a); accumulate(b)
        gives the same result as accumulate(a + b).
        """
        # Strategy 1: sequential accumulation
        seq_counters = ExecutionCounters(
            rows_processed=counters.rows_processed,
            rows_succeeded=counters.rows_succeeded,
            rows_failed=counters.rows_failed,
            rows_routed=counters.rows_routed,
            rows_quarantined=counters.rows_quarantined,
            rows_forked=counters.rows_forked,
            rows_coalesced=counters.rows_coalesced,
            rows_coalesce_failed=counters.rows_coalesce_failed,
            rows_expanded=counters.rows_expanded,
            rows_buffered=counters.rows_buffered,
            routed_destinations=Counter(counters.routed_destinations),
        )
        for flush in flushes:
            seq_counters.accumulate_flush_result(flush)

        # Strategy 2: sum all flushes first, then accumulate once
        sum_counters = ExecutionCounters(
            rows_processed=counters.rows_processed,
            rows_succeeded=counters.rows_succeeded,
            rows_failed=counters.rows_failed,
            rows_routed=counters.rows_routed,
            rows_quarantined=counters.rows_quarantined,
            rows_forked=counters.rows_forked,
            rows_coalesced=counters.rows_coalesced,
            rows_coalesce_failed=counters.rows_coalesce_failed,
            rows_expanded=counters.rows_expanded,
            rows_buffered=counters.rows_buffered,
            routed_destinations=Counter(counters.routed_destinations),
        )
        total_flush = AggregationFlushResult()
        for flush in flushes:
            total_flush = total_flush + flush
        sum_counters.accumulate_flush_result(total_flush)

        # Must be identical
        assert seq_counters.rows_succeeded == sum_counters.rows_succeeded
        assert seq_counters.rows_failed == sum_counters.rows_failed
        assert seq_counters.rows_routed == sum_counters.rows_routed
        assert seq_counters.rows_quarantined == sum_counters.rows_quarantined
        assert seq_counters.rows_coalesced == sum_counters.rows_coalesced
        assert seq_counters.rows_forked == sum_counters.rows_forked
        assert seq_counters.rows_expanded == sum_counters.rows_expanded
        assert seq_counters.rows_buffered == sum_counters.rows_buffered
        assert dict(seq_counters.routed_destinations) == dict(sum_counters.routed_destinations)


# =============================================================================
# ExecutionCounters.to_run_result Field Mapping Properties
# =============================================================================


class TestToRunResultProperties:
    """to_run_result must faithfully project counters to RunResult."""

    @given(counters=execution_counters())
    @settings(max_examples=200)
    def test_all_counter_fields_preserved(self, counters: ExecutionCounters) -> None:
        """Property: Every counter field appears identically in RunResult."""
        result = counters.to_run_result("run-1")

        assert result.run_id == "run-1"
        assert result.rows_processed == counters.rows_processed
        assert result.rows_succeeded == counters.rows_succeeded
        assert result.rows_failed == counters.rows_failed
        assert result.rows_routed == counters.rows_routed
        assert result.rows_quarantined == counters.rows_quarantined
        assert result.rows_forked == counters.rows_forked
        assert result.rows_coalesced == counters.rows_coalesced
        assert result.rows_coalesce_failed == counters.rows_coalesce_failed
        assert result.rows_expanded == counters.rows_expanded
        assert result.rows_buffered == counters.rows_buffered

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_routed_destinations_snapshot(self, counters: ExecutionCounters) -> None:
        """Property: routed_destinations is a plain dict snapshot (not a Counter reference).

        This ensures mutations to counters after to_run_result() don't
        affect the RunResult.
        """
        result = counters.to_run_result("run-1")

        # Must be a plain dict, not a Counter reference
        assert type(result.routed_destinations) is dict
        assert result.routed_destinations == dict(counters.routed_destinations)

        # Mutating counters after snapshot must not affect result
        counters.routed_destinations["new_sink"] += 999
        assert "new_sink" not in result.routed_destinations or result.routed_destinations["new_sink"] != 999

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_default_status_is_running(self, counters: ExecutionCounters) -> None:
        """Property: Default status is RUNNING."""
        from elspeth.contracts import RunStatus

        result = counters.to_run_result("run-1")
        assert result.status == RunStatus.RUNNING

    @given(counters=execution_counters())
    @settings(max_examples=50)
    def test_status_override(self, counters: ExecutionCounters) -> None:
        """Property: Status can be overridden by caller."""
        from elspeth.contracts import RunStatus

        result = counters.to_run_result("run-1", status=RunStatus.COMPLETED)
        assert result.status == RunStatus.COMPLETED

        result = counters.to_run_result("run-1", status=RunStatus.FAILED)
        assert result.status == RunStatus.FAILED

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_to_run_result_is_snapshot(self, counters: ExecutionCounters) -> None:
        """Property: to_run_result produces a snapshot — further counter mutations
        do not affect the returned RunResult.
        """
        result = counters.to_run_result("run-1")
        original_succeeded = result.rows_succeeded

        # Mutate counters
        counters.rows_succeeded += 100

        # RunResult must not change
        assert result.rows_succeeded == original_succeeded


# =============================================================================
# accumulate_row_outcomes Mapping Properties
# =============================================================================


class TestAccumulateRowOutcomesProperties:
    """Each RowOutcome must map to exactly the correct counter(s)."""

    def _run_accumulation(
        self,
        outcomes: list[RowResult],
        sink_names: dict[str, object] | None = None,
    ) -> tuple[ExecutionCounters, dict[str, list[tuple[TokenInfo, PendingOutcome | None]]]]:
        """Helper: run accumulate_row_outcomes and return counters + pending_tokens."""
        counters = ExecutionCounters()
        if sink_names is None:
            sink_names = {"default": object(), "alerts": object()}
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {name: [] for name in sink_names}
        accumulate_row_outcomes(outcomes, counters, sink_names, pending_tokens)
        return counters, pending_tokens

    def test_completed_increments_succeeded(self) -> None:
        """Property: COMPLETED outcome increments rows_succeeded by 1."""
        result = _make_row_result(RowOutcome.COMPLETED)
        counters, pending = self._run_accumulation([result])

        assert counters.rows_succeeded == 1
        assert counters.rows_failed == 0
        assert counters.rows_routed == 0
        assert len(pending["default"]) == 1

    def test_completed_uses_explicit_sink_name(self) -> None:
        """Property: COMPLETED routing uses result.sink_name, not branch_name."""
        result = _make_row_result(RowOutcome.COMPLETED, sink_name="alerts", branch_name="ignored_branch")
        counters, pending = self._run_accumulation([result])

        assert counters.rows_succeeded == 1
        assert len(pending["alerts"]) == 1
        assert len(pending["default"]) == 0

    def test_completed_ignores_branch_name_when_sink_explicit(self) -> None:
        """Property: branch_name never determines COMPLETED sink routing."""
        result = _make_row_result(RowOutcome.COMPLETED, sink_name="default", branch_name="alerts")
        counters, pending = self._run_accumulation([result])

        assert counters.rows_succeeded == 1
        assert len(pending["default"]) == 1
        assert len(pending["alerts"]) == 0

    def test_routed_increments_routed(self) -> None:
        """Property: ROUTED outcome increments rows_routed."""
        result = _make_row_result(RowOutcome.ROUTED, sink_name="alerts")
        counters, pending = self._run_accumulation([result])

        assert counters.rows_routed == 1
        assert counters.routed_destinations["alerts"] == 1
        assert len(pending["alerts"]) == 1

    def test_routed_without_sink_name_raises(self) -> None:
        """Property: ROUTED without sink_name raises at construction time."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        with pytest.raises(OrchestrationInvariantError, match="ROUTED outcome requires sink_name"):
            RowResult(
                token=_make_token(),
                final_data=make_pipeline_row({"field": "value"}),
                outcome=RowOutcome.ROUTED,
                sink_name=None,
            )

    def test_failed_increments_failed(self) -> None:
        """Property: FAILED outcome increments rows_failed."""
        result = _make_row_result(RowOutcome.FAILED)
        counters, _ = self._run_accumulation([result])

        assert counters.rows_failed == 1
        assert counters.rows_succeeded == 0

    def test_quarantined_increments_quarantined(self) -> None:
        """Property: QUARANTINED outcome increments rows_quarantined."""
        result = _make_row_result(RowOutcome.QUARANTINED)
        counters, _ = self._run_accumulation([result])

        assert counters.rows_quarantined == 1

    def test_forked_increments_forked(self) -> None:
        """Property: FORKED outcome increments rows_forked."""
        result = _make_row_result(RowOutcome.FORKED)
        counters, _ = self._run_accumulation([result])

        assert counters.rows_forked == 1

    def test_consumed_in_batch_is_silent(self) -> None:
        """Property: CONSUMED_IN_BATCH increments no counters."""
        result = _make_row_result(RowOutcome.CONSUMED_IN_BATCH)
        counters, pending = self._run_accumulation([result])

        assert counters.rows_succeeded == 0
        assert counters.rows_failed == 0
        assert counters.rows_routed == 0
        assert counters.rows_quarantined == 0
        assert counters.rows_forked == 0
        assert counters.rows_coalesced == 0
        assert counters.rows_expanded == 0
        assert counters.rows_buffered == 0
        assert len(pending["default"]) == 0

    def test_coalesced_increments_both_coalesced_and_succeeded(self) -> None:
        """Property: COALESCED increments BOTH rows_coalesced AND rows_succeeded.

        This is the key subtlety: merged tokens proceed to the output sink,
        so they count as both coalesced AND succeeded.
        """
        result = _make_row_result(RowOutcome.COALESCED)
        counters, pending = self._run_accumulation([result])

        assert counters.rows_coalesced == 1
        assert counters.rows_succeeded == 1
        assert len(pending["default"]) == 1

    def test_expanded_increments_expanded(self) -> None:
        """Property: EXPANDED outcome increments rows_expanded."""
        result = _make_row_result(RowOutcome.EXPANDED)
        counters, _ = self._run_accumulation([result])

        assert counters.rows_expanded == 1

    def test_buffered_increments_buffered(self) -> None:
        """Property: BUFFERED outcome increments rows_buffered."""
        result = _make_row_result(RowOutcome.BUFFERED)
        counters, _ = self._run_accumulation([result])

        assert counters.rows_buffered == 1

    @given(
        completed=st.integers(min_value=0, max_value=10),
        failed=st.integers(min_value=0, max_value=10),
        routed=st.integers(min_value=0, max_value=10),
        quarantined=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_mixed_outcomes_conservation(self, completed: int, failed: int, routed: int, quarantined: int) -> None:
        """Property: Total counter increments == total number of results.

        For simple outcomes (no double-counting like COALESCED), the sum
        of all counter increments must equal the number of input results.
        """
        results: list[RowResult] = []
        results.extend(_make_row_result(RowOutcome.COMPLETED) for _ in range(completed))
        results.extend(_make_row_result(RowOutcome.FAILED) for _ in range(failed))
        results.extend(_make_row_result(RowOutcome.ROUTED, sink_name="alerts") for _ in range(routed))
        results.extend(_make_row_result(RowOutcome.QUARANTINED) for _ in range(quarantined))

        counters, _ = self._run_accumulation(results)

        total_increments = counters.rows_succeeded + counters.rows_failed + counters.rows_routed + counters.rows_quarantined
        assert total_increments == completed + failed + routed + quarantined

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_routed_destinations_count_per_sink(self, n: int) -> None:
        """Property: routed_destinations[sink] counts ROUTED outcomes to that sink."""
        results = [_make_row_result(RowOutcome.ROUTED, sink_name="alerts") for _ in range(n)]
        counters, _ = self._run_accumulation(results)

        assert counters.routed_destinations["alerts"] == n
        assert counters.rows_routed == n


# =============================================================================
# RunResult Field Completeness Properties
# =============================================================================


class TestRunResultFieldProperties:
    """RunResult must have all expected fields with correct defaults."""

    def test_default_optional_fields_are_zero(self) -> None:
        """Property: Optional counter fields default to zero."""
        result = RunResult(
            run_id="run-1",
            status=RunStatus.RUNNING,
            rows_processed=10,
            rows_succeeded=8,
            rows_failed=1,
            rows_routed=1,
        )
        assert result.rows_quarantined == 0
        assert result.rows_forked == 0
        assert result.rows_coalesced == 0
        assert result.rows_coalesce_failed == 0
        assert result.rows_expanded == 0
        assert result.rows_buffered == 0
        assert result.routed_destinations == {}

    @given(counters=execution_counters())
    @settings(max_examples=100)
    def test_run_result_field_coverage(self, counters: ExecutionCounters) -> None:
        """Property: to_run_result covers all RunResult counter fields.

        Every int field on RunResult (except run_id and status) must have a
        corresponding value from ExecutionCounters.
        """
        result = counters.to_run_result("run-1")
        run_result_fields = {f.name for f in fields(result)}
        # Non-counter fields
        meta_fields = {"run_id", "status", "routed_destinations"}
        counter_fields = run_result_fields - meta_fields

        for field_name in counter_fields:
            # Every counter field must have a matching value from ExecutionCounters
            run_value = getattr(result, field_name)
            counter_value = getattr(counters, field_name)
            assert run_value == counter_value, f"Field {field_name} mismatch: RunResult={run_value}, Counters={counter_value}"


# =============================================================================
# AggregationFlushResult Immutability Properties
# =============================================================================


class TestFlushResultImmutabilityProperties:
    """AggregationFlushResult must be truly frozen."""

    def test_fields_are_immutable(self) -> None:
        """Property: Cannot assign to fields of a frozen dataclass."""
        result = AggregationFlushResult(rows_succeeded=5)
        with pytest.raises(AttributeError):
            result.rows_succeeded = 10  # type: ignore[misc]

    @given(a=aggregation_flush_results(), b=aggregation_flush_results())
    @settings(max_examples=50)
    def test_add_does_not_mutate_operands(self, a: AggregationFlushResult, b: AggregationFlushResult) -> None:
        """Property: __add__ does not modify either operand."""
        a_succeeded_before = a.rows_succeeded
        b_succeeded_before = b.rows_succeeded
        a_dests_before = dict(a.routed_destinations)
        b_dests_before = dict(b.routed_destinations)

        _ = a + b

        assert a.rows_succeeded == a_succeeded_before
        assert b.rows_succeeded == b_succeeded_before
        assert a.routed_destinations == a_dests_before
        assert b.routed_destinations == b_dests_before
