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

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from elspeth.contracts.audit import TokenOutcome
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.orchestrator.run_status import derive_resume_terminal_status_from_audit

_RECORDED_AT = datetime(2026, 5, 29, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _FakeQuery:
    """Minimal stand-in for ``RecorderFactory.query`` returning canned outcomes.

    These tests exercise the per-case ``routed_destinations`` tally and the
    Tier-1 ``sink_name`` guard, NOT ``rows_processed`` — the canned
    ``TokenOutcome`` records carry no ``row_id`` (the real distinct-``row_id``
    count is computed by ``QueryRepository`` via a JOIN to the tokens table).
    ``count_distinct_source_rows_with_terminal_outcome`` therefore returns the
    count of distinct ``token_id`` among completed canned outcomes — sufficient
    to satisfy the helper's call and keep the RunResult biconditional happy
    without these unit tests asserting against ``rows_processed``.
    """

    outcomes: list[TokenOutcome]

    def get_all_token_outcomes_for_run(self, run_id: str) -> list[TokenOutcome]:
        return self.outcomes

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
    factory = _FakeFactory(query=_FakeQuery([_routed_outcome(path, outcome, sink_name=None)]))
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
    factory = _FakeFactory(query=_FakeQuery([_routed_outcome(path, outcome, sink_name=destination)]))
    _status, counters = derive_resume_terminal_status_from_audit(factory, "run-1")  # type: ignore[arg-type]
    assert counters.routed_destinations[destination] == 1
