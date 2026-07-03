# tests/unit/engine/orchestrator/test_terminal_pair_counter_parity.py
"""Terminal-pair counter effects: one table, three consumers (elspeth-feeb4482fc).

``TERMINAL_PAIR_COUNTER_EFFECTS`` is the single source of truth for how each
legal ``(TerminalOutcome, TerminalPath)`` pair moves ``ExecutionCounters``.
These tests pin the parity contract EXTENSIONALLY: for every pair, the live
accumulator (``accumulate_row_outcomes``), the sink-diversion reconciler
(``reconcile_sink_write_diversions``), and the audit derive
(``derive_terminal_status_from_audit``) must all agree with the table — so a
terminal-pair semantic change lands in exactly one place, and a new pair
added to ``contracts.enums._LEGAL_TERMINAL_PAIRS`` cannot reach one switch
without the others (the module's import-time lockstep guard plus these tests
fail first).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.contracts import PendingOutcome, TokenInfo
from elspeth.contracts.audit import TokenOutcome
from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.results import FailureInfo
from elspeth.engine.orchestrator.counter_classification import (
    TERMINAL_PAIR_COUNTER_EFFECTS,
    TerminalPairCounterEffect,
    apply_counter_increments,
)
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes, reconcile_sink_write_diversions
from elspeth.engine.orchestrator.run_status import derive_terminal_status_from_audit
from elspeth.engine.orchestrator.types import ExecutionCounters
from elspeth.testing import make_token_info

_RECORDED_AT = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)
_SINK = "output"

# Counter fields the table governs. rows_processed and rows_coalesce_failed
# are query-derived in the audit path (distinct source rows / FAILED coalesce
# node_states), never per-record increments, so they are outside the table.
_TABLE_GOVERNED_FIELDS = tuple(
    f.name for f in fields(ExecutionCounters) if f.name not in ("rows_processed", "rows_coalesce_failed", "routed_destinations")
)


def _expected_counters(effect: TerminalPairCounterEffect) -> ExecutionCounters:
    expected = ExecutionCounters()
    apply_counter_increments(expected, effect)
    if effect.counts_routed_destination:
        expected.routed_destinations[_SINK] += 1
    return expected


def _assert_governed_fields_match(actual: ExecutionCounters, expected: ExecutionCounters, *, context: str) -> None:
    for field_name in _TABLE_GOVERNED_FIELDS:
        assert getattr(actual, field_name) == getattr(expected, field_name), (
            f"{context}: {field_name} diverged from the counter-effect table"
        )
    assert dict(actual.routed_destinations) == dict(expected.routed_destinations), f"{context}: routed_destinations diverged"


# ---------------------------------------------------------------------------
# Audit derive fakes (same shape as test_run_status.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FakeQuery:
    outcomes: list[TokenOutcome]

    def get_all_token_outcomes_for_run(self, run_id: str) -> list[TokenOutcome]:
        return self.outcomes

    def count_distinct_source_rows_with_terminal_outcome(self, run_id: str) -> int:
        return len({o.token_id for o in self.outcomes if o.completed})

    def count_failed_coalesce_barrier_rows(self, run_id: str) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class _FakeFactory:
    query: _FakeQuery


def _token_outcome(
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None,
    completed: bool = True,
) -> TokenOutcome:
    return TokenOutcome(
        outcome_id="oc-1",
        run_id="run-1",
        token_id="tok-1",
        outcome=outcome,
        path=path,
        completed=completed,
        recorded_at=_RECORDED_AT,
        sink_name=sink_name,
    )


# ---------------------------------------------------------------------------
# Live accumulator fakes (attribute-shape of RowResult; no mocks)
# ---------------------------------------------------------------------------


def _row_result(
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None = None,
    error: FailureInfo | None = None,
    token: TokenInfo | None = None,
) -> Any:
    result_token = token or make_token_info()
    if path is TerminalPath.COALESCED and result_token.join_group_id is None:
        result_token = replace(result_token, join_group_id="join-1")
    return SimpleNamespace(
        outcome=outcome,
        path=path,
        token=result_token,
        sink_name=sink_name,
        error=error,
        scheduler_pending_sink=False,
    )


def _accumulate_inputs_for(pair: tuple[TerminalOutcome | None, TerminalPath]) -> Any:
    outcome, path = pair
    effect = TERMINAL_PAIR_COUNTER_EFFECTS[pair]
    sink_bound = effect.sink_reconcilable and path is not TerminalPath.QUARANTINED_AT_SOURCE
    sink_name = _SINK if (sink_bound or effect.counts_routed_destination) else None
    error = FailureInfo(exception_type="ValueError", message="boom") if path is TerminalPath.ON_ERROR_ROUTED else None
    return _row_result(outcome, path, sink_name=sink_name, error=error)


_ALL_TABLE_PAIRS = sorted(TERMINAL_PAIR_COUNTER_EFFECTS, key=str)
_PROCESSING_PAIRS = [pair for pair in _ALL_TABLE_PAIRS if not TERMINAL_PAIR_COUNTER_EFFECTS[pair].forbidden_in_processing_results]
_FORBIDDEN_PAIRS = [pair for pair in _ALL_TABLE_PAIRS if TERMINAL_PAIR_COUNTER_EFFECTS[pair].forbidden_in_processing_results]
_RECONCILABLE_PAIRS = [pair for pair in _ALL_TABLE_PAIRS if TERMINAL_PAIR_COUNTER_EFFECTS[pair].sink_reconcilable]


class TestTableLockstep:
    """The table's key set tracks the contract layer exactly."""

    def test_terminal_keys_match_legal_pairs(self) -> None:
        assert frozenset(k for k in TERMINAL_PAIR_COUNTER_EFFECTS if k[0] is not None) == _LEGAL_TERMINAL_PAIRS

    def test_only_non_terminal_key_is_buffered(self) -> None:
        assert frozenset(k for k in TERMINAL_PAIR_COUNTER_EFFECTS if k[0] is None) == frozenset({(None, TerminalPath.BUFFERED)})

    def test_increments_name_real_counter_fields(self) -> None:
        counter_fields = frozenset(f.name for f in fields(ExecutionCounters))
        for key, effect in TERMINAL_PAIR_COUNTER_EFFECTS.items():
            assert frozenset(effect.increments) <= counter_fields, key


class TestAuditDeriveMatchesTable:
    """derive_terminal_status_from_audit applies exactly the table's effects."""

    @pytest.mark.parametrize("pair", [p for p in _ALL_TABLE_PAIRS if p != (None, TerminalPath.BUFFERED)], ids=str)
    def test_terminal_pair_counters_match_table(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        effect = TERMINAL_PAIR_COUNTER_EFFECTS[pair]
        # COALESCED derive counts only the merged output (sink_name set);
        # routed pairs are contract-bound to carry sink_name.
        sink_name = _SINK if (effect.counts_routed_destination or pair[1] is TerminalPath.COALESCED) else None
        factory = _FakeFactory(query=_FakeQuery([_token_outcome(pair[0], pair[1], sink_name=sink_name)]))

        _status, counters = derive_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]

        _assert_governed_fields_match(counters, _expected_counters(effect), context=f"derive {pair!r}")

    def test_buffered_record_counts_rows_buffered(self) -> None:
        factory = _FakeFactory(query=_FakeQuery([_token_outcome(None, TerminalPath.BUFFERED, sink_name=None, completed=False)]))

        _status, counters = derive_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]

        expected = _expected_counters(TERMINAL_PAIR_COUNTER_EFFECTS[(None, TerminalPath.BUFFERED)])
        _assert_governed_fields_match(counters, expected, context="derive (None, BUFFERED)")

    def test_coalesced_consumed_input_counts_nothing(self) -> None:
        """A consumed branch input (sink_name None) delegates to the merged token."""
        factory = _FakeFactory(query=_FakeQuery([_token_outcome(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name=None)]))

        _status, counters = derive_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]

        _assert_governed_fields_match(counters, ExecutionCounters(), context="derive consumed COALESCED input")


class TestLiveAccumulatorMatchesTable:
    """accumulate_row_outcomes applies exactly the table's effects."""

    @pytest.mark.parametrize("pair", _PROCESSING_PAIRS, ids=str)
    def test_processing_pair_counters_match_table(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        counters = ExecutionCounters()
        pending_tokens: dict[str, list[Any]] = {_SINK: []}

        accumulate_row_outcomes([_accumulate_inputs_for(pair)], counters, pending_tokens)

        _assert_governed_fields_match(counters, _expected_counters(TERMINAL_PAIR_COUNTER_EFFECTS[pair]), context=f"accumulate {pair!r}")

    @pytest.mark.parametrize("pair", _FORBIDDEN_PAIRS, ids=str)
    def test_diversion_pairs_are_forbidden_in_processing_results(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        counters = ExecutionCounters()
        with pytest.raises(OrchestrationInvariantError, match="Diversion path"):
            accumulate_row_outcomes([_row_result(pair[0], pair[1], sink_name=_SINK)], counters, {_SINK: []})


class TestReconcileMatchesTable:
    """reconcile_sink_write_diversions subtracts exactly the table's effects."""

    @pytest.mark.parametrize("pair", _RECONCILABLE_PAIRS, ids=str)
    def test_reconcile_inverts_table_increments(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        effect = TERMINAL_PAIR_COUNTER_EFFECTS[pair]
        counters = ExecutionCounters()
        for _ in range(3):
            apply_counter_increments(counters, effect)
            if effect.counts_routed_destination:
                counters.routed_destinations[_SINK] += 1
        error_hash = "a" * 64 if pair[1] in PendingOutcome._REQUIRES_ERROR_HASH_PATHS else None
        pending = PendingOutcome(outcome=pair[0], path=pair[1], error_hash=error_hash)

        reconcile_sink_write_diversions(counters, sink_name=_SINK, pending_outcome=pending, diversion_count=3)

        _assert_governed_fields_match(counters, ExecutionCounters(), context=f"reconcile {pair!r}")

    @pytest.mark.parametrize(
        "pair", [p for p in _ALL_TABLE_PAIRS if not TERMINAL_PAIR_COUNTER_EFFECTS[p].sink_reconcilable and p[0] is not None], ids=str
    )
    def test_non_reconcilable_pairs_are_rejected(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        error_hash = "a" * 64 if pair[1] in PendingOutcome._REQUIRES_ERROR_HASH_PATHS else None
        pending = PendingOutcome(outcome=pair[0], path=pair[1], error_hash=error_hash)
        with pytest.raises(OrchestrationInvariantError, match="Unexpected sink-bound pending pair"):
            reconcile_sink_write_diversions(ExecutionCounters(), sink_name=_SINK, pending_outcome=pending, diversion_count=1)

    @pytest.mark.parametrize("pair", _RECONCILABLE_PAIRS, ids=str)
    def test_underflow_fails_closed(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        error_hash = "a" * 64 if pair[1] in PendingOutcome._REQUIRES_ERROR_HASH_PATHS else None
        pending = PendingOutcome(outcome=pair[0], path=pair[1], error_hash=error_hash)
        with pytest.raises(OrchestrationInvariantError, match="Cannot subtract"):
            reconcile_sink_write_diversions(ExecutionCounters(), sink_name=_SINK, pending_outcome=pending, diversion_count=1)
