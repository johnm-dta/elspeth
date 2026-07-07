"""Unit tests for resume-status aggregation invariants in ``run_status``.

Focus: the Tier-1 ``sink_name`` guard in
:func:`derive_resume_terminal_status_from_audit`.

A routed ``token_outcomes`` row (``GATE_ROUTED`` / ``ON_ERROR_ROUTED``) is
contract-bound to carry a non-NULL ``sink_name`` (write-side
``_TERMINAL_PAIR_FIELD_CONSTRAINTS``), and the read path enforces it — the
``TokenOutcome`` loader raises ``AuditIntegrityError`` on a NULL before the
aggregator ever runs. The guard in ``run_status.py`` is therefore
defense-in-depth *behind* the loader: it crashes loudly if a corrupt-our-data
NULL ever reaches the resume aggregator, rather than silently under-counting
``routed_destinations`` in the legal record.

Because the real query path raises ``AuditIntegrityError`` first, the only way
to exercise the guard is to inject a hand-built malformed record past a fake
query — which is precisely the Tier-1-illegal state the guard exists to crash
on. The fake isolates the pure aggregator; it is not bypassing production
wiring for an integration concern.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime

import pytest

from elspeth.contracts.audit import TokenOutcome
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.run_result import RunResult
from elspeth.engine.orchestrator import run_status
from elspeth.engine.orchestrator.run_status import (
    derive_resume_terminal_status_from_audit,
    is_counted_coalesced_output,
)
from elspeth.engine.orchestrator.types import ExecutionCounters

_RECORDED_AT = datetime(2026, 5, 29, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _FakeQuery:
    """Minimal stand-in for ``RecorderFactory.query`` returning canned outcomes."""

    outcomes: list[TokenOutcome]

    def get_all_token_outcomes_for_run(self, run_id: str) -> list[TokenOutcome]:
        return self.outcomes


@dataclass(frozen=True, slots=True)
class _FakeRunStatusProjection:
    """Minimal stand-in for the dedicated audit counter projection.

    These tests exercise the per-case ``routed_destinations`` tally and the
    Tier-1 ``sink_name`` guard, NOT ``rows_processed`` — the canned
    ``TokenOutcome`` records carry no ``row_id`` (the real distinct-``row_id``
    count is computed by the projection via a JOIN to the tokens table).
    ``count_distinct_source_rows_with_terminal_outcome`` therefore returns the
    count of distinct ``token_id`` among completed canned outcomes — sufficient
    to satisfy the helper's call and keep the RunResult biconditional happy
    without these unit tests asserting against ``rows_processed``.
    """

    outcomes: list[TokenOutcome]

    def count_distinct_source_rows_with_terminal_outcome(self, run_id: str) -> int:
        return len({o.token_id for o in self.outcomes if o.completed})

    def count_failed_coalesce_barrier_rows(self, run_id: str) -> int:
        # These tests exercise routed-destination tallies, not coalesce
        # failures; the real verb counts DISTINCT (coalesce node, row) pairs
        # over FAILED node_states, which a pure-outcome-list fake cannot
        # reproduce. No scenario here involves a coalesce, so 0 is faithful.
        return 0


@dataclass(frozen=True, slots=True)
class _FakeFactory:
    query: _FakeQuery
    run_status_projection: _FakeRunStatusProjection


def _fake_factory(outcomes: list[TokenOutcome]) -> _FakeFactory:
    return _FakeFactory(query=_FakeQuery(outcomes), run_status_projection=_FakeRunStatusProjection(outcomes))


def _routed_outcome(path: TerminalPath, outcome: TerminalOutcome, *, sink_name: str | None) -> TokenOutcome:
    # TokenOutcome.__post_init__ validates the completed/outcome XOR invariant
    # but NOT the routed-pair sink_name constraint (that lives in the loader),
    # so a NULL-sink_name routed row is constructible here on purpose.
    return TokenOutcome(
        outcome_id="oc-1",
        run_id="run-1",
        token_id="tok-1",
        outcome=outcome,
        path=path,
        completed=True,
        recorded_at=_RECORDED_AT,
        sink_name=sink_name,
    )


@pytest.mark.parametrize(
    ("path", "outcome"),
    [
        (TerminalPath.GATE_ROUTED, TerminalOutcome.SUCCESS),
        (TerminalPath.ON_ERROR_ROUTED, TerminalOutcome.FAILURE),
    ],
)
def test_routed_outcome_missing_sink_name_crashes(path: TerminalPath, outcome: TerminalOutcome) -> None:
    """A routed token_outcomes row with NULL sink_name must crash the resume
    aggregator — a Tier-1 audit-integrity violation, never a silent skip."""
    factory = _fake_factory([_routed_outcome(path, outcome, sink_name=None)])
    with pytest.raises(OrchestrationInvariantError, match="missing sink_name"):
        derive_resume_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "outcome", "destination"),
    [
        (TerminalPath.GATE_ROUTED, TerminalOutcome.SUCCESS, "approved_sink"),
        (TerminalPath.ON_ERROR_ROUTED, TerminalOutcome.FAILURE, "error_sink"),
    ],
)
def test_routed_outcome_tallies_destination(path: TerminalPath, outcome: TerminalOutcome, destination: str) -> None:
    """The happy path still tallies routed_destinations by sink_name."""
    factory = _fake_factory([_routed_outcome(path, outcome, sink_name=destination)])
    _status, counters = derive_resume_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]
    assert counters.routed_destinations[destination] == 1


def _coalesced_outcome(token_id: str, *, sink_name: str | None) -> TokenOutcome:
    """A ``(SUCCESS, COALESCED)`` record. ``sink_name=None`` is a CONSUMED branch
    input (must NOT be counted); ``sink_name`` set is the MERGED output (counted
    once). The CoalesceExecutor hard-codes ``sink_name=None`` on consumed inputs
    and the merged token carries its terminal sink name — see
    ``is_counted_coalesced_output``."""
    return TokenOutcome(
        outcome_id=f"oc-{token_id}",
        run_id="run-1",
        token_id=token_id,
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.COALESCED,
        completed=True,
        recorded_at=_RECORDED_AT,
        sink_name=sink_name,
    )


def test_is_counted_coalesced_output_discriminates_on_sink_name() -> None:
    """The helper names the invariant: merged output (sink_name set) is counted;
    consumed branch input (sink_name None) is not."""
    assert is_counted_coalesced_output(_coalesced_outcome("merged", sink_name="out_sink")) is True
    assert is_counted_coalesced_output(_coalesced_outcome("consumed", sink_name=None)) is False


def test_flat_coalesce_success_counts_merged_output_once() -> None:
    """A 2-branch coalesce-success writes two consumed inputs (sink_name=None)
    plus one merged output (sink_name set). The derive must count the merged
    output ONCE — counting every COALESCED record would report 3."""
    factory = _fake_factory(
        [
            _coalesced_outcome("branch-1", sink_name=None),
            _coalesced_outcome("branch-2", sink_name=None),
            _coalesced_outcome("merged", sink_name="out_sink"),
        ]
    )
    _status, counters = derive_resume_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]
    assert counters.rows_coalesced == 1
    assert counters.rows_succeeded == 1


def test_nested_coalesce_counts_only_final_merged_output() -> None:
    """An inner merged token consumed by an outer coalesce is itself recorded as
    a consumed input (sink_name=None), so it is NOT counted at the inner level.
    Only the OUTER merged output (which reaches the final sink) is counted."""
    factory = _fake_factory(
        [
            _coalesced_outcome("inner-branch-1", sink_name=None),
            _coalesced_outcome("inner-branch-2", sink_name=None),
            # inner merged token, absorbed by the outer coalesce -> sink_name None
            _coalesced_outcome("inner-merged", sink_name=None),
            # outer merged token reaches the final sink -> the only counted record
            _coalesced_outcome("outer-merged", sink_name="final_sink"),
        ]
    )
    _status, counters = derive_resume_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]
    assert counters.rows_coalesced == 1
    assert counters.rows_succeeded == 1


def test_terminal_counter_parity_fields_follow_execution_counters() -> None:
    """Every ExecutionCounters field is compared or explicitly excluded."""
    execution_counter_fields = tuple(field.name for field in fields(ExecutionCounters))
    run_result_fields = {field.name for field in fields(RunResult)}
    strict_fields = run_status._PARITY_STRICT_FIELDS
    excluded_fields = run_status._PARITY_EXCLUDED_FIELDS

    assert excluded_fields == frozenset({"rows_coalesce_failed", "routed_destinations"})
    assert set(strict_fields).isdisjoint(excluded_fields)
    assert set(execution_counter_fields) == set(strict_fields) | excluded_fields
    assert strict_fields == tuple(field for field in execution_counter_fields if field not in excluded_fields)
    assert not [field for field in strict_fields if field not in run_result_fields]
