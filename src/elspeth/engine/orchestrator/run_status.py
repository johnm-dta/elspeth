"""Terminal run-status resolution functions for the orchestrator.

This module contains pure functions for:
- Deriving the truthful terminal :class:`RunStatus` (and the matching
  :class:`ExecutionCounters`) of a resumed run from the Landscape audit DB
- Mapping a terminal :class:`RunStatus` to the CLI ``RunCompletionStatus``
  and process exit code

All functions operate on external state passed via parameters - they don't
maintain internal state. This follows the same pattern as aggregation.py and
outcomes.py: pure delegation targets for the Orchestrator.

These functions were extracted from ``Orchestrator`` (where they lived as
``_derive_resume_terminal_status_from_audit`` and ``_cli_completion_for``)
to shrink ``core.py`` and to make the status-resolution logic independently
testable without constructing an Orchestrator instance.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from elspeth.contracts import RunStatus
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.events import RunCompletionStatus
from elspeth.contracts.run_result import derive_terminal_run_status
from elspeth.engine.orchestrator.counter_classification import TERMINAL_PAIR_COUNTER_EFFECTS, apply_counter_increments
from elspeth.engine.orchestrator.types import ExecutionCounters

if TYPE_CHECKING:
    from elspeth.contracts.audit import TokenOutcome
    from elspeth.contracts.run_result import RunResult
    from elspeth.core.landscape.factory import RecorderFactory


def _require_routed_sink_name(outcome_record: TokenOutcome, pair: tuple[TerminalOutcome | None, TerminalPath]) -> str:
    """Require a non-NULL ``sink_name`` on a routed ``token_outcomes`` row.

    ``GATE_ROUTED`` / ``ON_ERROR_ROUTED`` outcomes are bound by the write-side
    Tier-1 contract (``_TERMINAL_PAIR_FIELD_CONSTRAINTS`` in
    :mod:`elspeth.contracts.audit`) to carry a non-NULL ``sink_name``, and the
    read path enforces it (``model_loaders.py`` raises ``AuditIntegrityError``
    on a NULL). A NULL reaching this resume aggregator is therefore a Tier-1
    audit-integrity violation — our own data is corrupt — so we crash loudly
    rather than silently under-count ``routed_destinations`` in the legal
    record. This mirrors the live accumulator's ``_require_sink_name``
    (:mod:`elspeth.engine.orchestrator.outcomes`); the two paths must agree on
    the audit truth.
    """
    name = outcome_record.sink_name
    if name is None:
        raise OrchestrationInvariantError(
            f"Routed token_outcomes row for token {outcome_record.token_id!r} has terminal pair "
            f"{pair} but missing sink_name — a Tier-1 audit-integrity violation (sink_name is "
            f"contract-required for routed outcomes and enforced on read by the TokenOutcome loader)."
        )
    return name


def is_counted_coalesced_output(outcome_record: TokenOutcome) -> bool:
    """Return True when a ``(SUCCESS, COALESCED)`` record is the MERGED output
    that the counters must tally, False when it is a CONSUMED branch input.

    A coalesce produces TWO kinds of ``(SUCCESS, COALESCED)`` record:

    1. the MERGED/output token, recorded with ``sink_name`` SET when it reaches
       its terminal sink (live: ``outcomes.py`` accumulate_row_outcomes →
       ``_route_to_sink``). This is the row that "coalesced", and the LIVE
       accumulator counts it once.
    2. each CONSUMED branch input, recorded by :class:`CoalesceExecutor` with
       ``sink_name`` HARD-CODED to None (``coalesce_executor.py`` ~1016-1022) —
       these are absorbed INTO the merged token and never routed through
       ``accumulate_row_outcomes``, so the live path counts only the merged
       output.

    The derive must mirror that: counting every ``(SUCCESS, COALESCED)`` record
    double-counts — it reported ``rows_coalesced``/``rows_succeeded`` == 3 for a
    2-branch coalesce-success where the live RunResult reports 1 (a
    resume-independent bug that leaked into the audit-derived RunResult of every
    resumed coalesce-success run). The ``sink_name`` discriminator is an
    invariant, not a topology coincidence: a consumed input ALWAYS has
    ``sink_name=None`` and the merged output ALWAYS carries its sink name. This
    also keeps NESTED coalesces correct — an inner merged token absorbed by an
    outer coalesce is itself recorded as a consumed input with ``sink_name=None``,
    so it is not counted at the inner level either.

    This mirrors how ``BATCH_CONSUMED`` delegates its success to the aggregate
    result and fork children are counted at their own sinks: the consumed inputs
    DELEGATE their success/coalesce tally to the merged token.
    """
    return outcome_record.sink_name is not None


def derive_terminal_status_from_audit(factory: RecorderFactory, run_id: str) -> tuple[RunStatus, ExecutionCounters]:
    """Recover the truthful cumulative terminal status + counters of a
    run from the Landscape audit DB.

    ADR-030 §D: no longer resume-only — the NORMAL completion arm
    (``Orchestrator.run``) also finalizes from this derive, so every path
    produces an audit-derived terminal record (single bookkeeper; the
    historical name ``derive_resume_terminal_status_from_audit`` remains as
    an alias). The live loop counters are demoted to a cross-check
    (:func:`assert_terminal_counter_parity`).

    Phase 2.2 (elspeth-0de989c56d) introduced this for the
    "all-rows-already-processed" resume branch (resume found no
    unprocessed rows): the resume's local counters are 0 because nothing
    was reprocessed, but the audit DB carries the truth in
    ``token_outcomes``.  Pre-Phase-2.2 the engine wrote
    ``RunStatus.COMPLETED`` there unconditionally, masking runs that
    actually failed.

    F2 (resume-fork-reemit) made this helper the SINGLE finalization
    path for BOTH resume branches.  The "with-unprocessed-rows"
    (fork-re-drive) branch previously returned *resume-only* local
    counters — only what that resume call reprocessed — so a resumed
    run's RunResult counter fields disagreed with an uninterrupted run
    (e.g. a resumed 1-row 2-branch fork reported ``rows_succeeded=1,
    rows_forked=0`` instead of the cumulative ``2, 1``).  A single
    RunResult type whose counter semantics depended on an invisible
    branch is a latent correctness trap; both branches now derive the
    same cumulative ``(status, counters)`` from the audit trail, which is
    the source of truth.  This call MUST run only after every outcome the
    resume wrote (including the end-of-source aggregation / coalesce
    flushes and any sink diversions) has been committed and swept
    (``sweep_deferred_invariants_or_crash``) — otherwise it undercounts.

    ``rows_processed`` is reconstructed per *source row* (distinct
    ``row_id`` reaching a terminal outcome) rather than per terminal
    token; see the inline comment and
    ``QueryRepository.count_distinct_source_rows_with_terminal_outcome``.

    Returns:
        ``(terminal_status, counters)``.  ``counters`` feeds the local
        RunResult so its row counts match the chosen status (otherwise
        the biconditional in :class:`elspeth.contracts.run_result.RunResult`
        would crash) and so structural counters are not fabricated as 0.
    """
    outcomes = factory.query.get_all_token_outcomes_for_run(run_id)
    counters = ExecutionCounters()
    # ``rows_processed`` is per *source row*, not per terminal token — see
    # ``QueryRepository.count_distinct_source_rows_with_terminal_outcome``.  It
    # MUST NOT be accumulated per-case below: a 1-source-row fork emits two leaf
    # tokens, a 3-source-row aggregation emits one result token, a 1-source-row
    # expand emits N children, yet each contributes exactly its source rows
    # (1, 3, 1) to ``rows_processed``.  A per-leaf tally over-counts forks and
    # expands and under-counts aggregations; the distinct-``row_id`` count is the
    # unique value that matches an uninterrupted run (F2 reconciliation). All
    # OTHER counters below are genuine per-terminal-token tallies and stay
    # per-case.
    counters.rows_processed = factory.query.count_distinct_source_rows_with_terminal_outcome(run_id)
    # ``rows_coalesce_failed`` has no token_outcomes arm: a failed coalesce
    # records per-branch (FAILURE, UNROUTED) outcomes (so ``rows_failed``
    # reconstructs below) but those carry no node attribution, and a naive
    # per-outcome tally would over-report a multi-branch barrier failure.
    # The durable evidence is the FAILED node_states ``_fail_pending`` writes
    # at the run's coalesce nodes; one failed barrier == one DISTINCT
    # (coalesce node, row_id) pair regardless of branch fan-in.  This is THE
    # value (elspeth-7294de558e) — it is cumulative over run-1 AND resume
    # re-drives (same run_id), replacing the resume-only live-counter graft
    # that forgot run-1 failures.
    counters.rows_coalesce_failed = factory.query.count_failed_coalesce_barrier_rows(run_id)
    for outcome_record in outcomes:
        if not outcome_record.completed:
            if (outcome_record.outcome, outcome_record.path) == (None, TerminalPath.BUFFERED):
                apply_counter_increments(counters, TERMINAL_PAIR_COUNTER_EFFECTS[(None, TerminalPath.BUFFERED)])
            continue
        pair = (outcome_record.outcome, outcome_record.path)
        effect = TERMINAL_PAIR_COUNTER_EFFECTS.get(pair)
        if effect is None:
            raise AssertionError(
                f"Unhandled (outcome, path) pair in resume aggregation: {pair!r}. "
                "Add it to TERMINAL_PAIR_COUNTER_EFFECTS; see ADR-019 mapping table."
            )
        if pair == (TerminalOutcome.SUCCESS, TerminalPath.COALESCED) and not is_counted_coalesced_output(outcome_record):
            # Consumed coalesce branch input — it delegates its success/coalesce
            # tally to the merged output token (which carries sink_name), exactly
            # as the live accumulator does. See is_counted_coalesced_output for
            # the full invariant and the double-count it prevents.
            continue
        # Counter movement comes from the shared table (elspeth-feeb4482fc);
        # the live accumulator and the sink-diversion reconciler consume the
        # SAME entries, so assert_terminal_counter_parity cannot be broken by
        # pair-by-pair drift between the two bookkeepers.
        apply_counter_increments(counters, effect)
        if effect.counts_routed_destination:
            counters.routed_destinations[_require_routed_sink_name(outcome_record, pair)] += 1
    terminal_status = derive_terminal_run_status(
        rows_processed=counters.rows_processed,
        rows_succeeded=counters.rows_succeeded,
        rows_failed=counters.rows_failed,
        rows_routed_success=counters.rows_routed_success,
        rows_routed_failure=counters.rows_routed_failure,
        rows_quarantined=counters.rows_quarantined,
        rows_coalesce_failed=counters.rows_coalesce_failed,
    )
    return terminal_status, counters


# Historical name (pre-ADR-030 §D the derive was resume-only). Kept so
# existing callers/tests keep working; new code uses the unprefixed name.
derive_resume_terminal_status_from_audit = derive_terminal_status_from_audit


# Live-vs-audit counter fields compared strictly by assert_terminal_counter_parity.
# ExecutionCounters is the authoritative field list. Every field is strict by
# default; add an entry here only when the exception is documented and handled
# below.
# rows_coalesce_failed is EXCLUDED — its two documented divergences (ADR-030 §D,
# bug elspeth-ff6d48c180) are tolerated and logged instead:
#   1. arrival-time barrier failures (branch-lost cascades, merge-exception
#      cleanup) write FAILED node_states the derive counts but the live
#      accumulator misses (it only counts the timeout/EOF sweeps) — audit MAY
#      EXCEED live, and the audit value is the owned improvement;
#   2. a zero-arrival best_effort_timeout_no_arrivals or
#      first_timeout_no_arrivals failure consumes no tokens and writes no
#      node_states — live counts it, the derive cannot, so live MAY EXCEED
#      audit (accepted, audit-is-truth doctrine).
#
# routed_destinations is compared separately as a plain dict below because
# RunResult stores a frozen Mapping while ExecutionCounters stores a Counter.
_PARITY_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "rows_coalesce_failed",
        "routed_destinations",
    }
)
_PARITY_STRICT_FIELDS: tuple[str, ...] = tuple(
    field.name for field in fields(ExecutionCounters) if field.name not in _PARITY_EXCLUDED_FIELDS
)


def assert_terminal_counter_parity(*, live: RunResult, audit: ExecutionCounters, run_id: str) -> None:
    """Cross-check the demoted live loop counters against the audit derive.

    ADR-030 §D: the audit-derived counters ARE the terminal record; the live
    accumulator survives only as this assertion. Any divergence outside the
    two documented ``rows_coalesce_failed`` arms (see
    ``_PARITY_STRICT_FIELDS``) means one of the two bookkeepers is broken —
    crash loudly rather than record an unexplained terminal status.

    Raises:
        OrchestrationInvariantError: on any strict-field mismatch.
    """
    mismatches = {
        field: {"live": getattr(live, field), "audit": getattr(audit, field)}
        for field in _PARITY_STRICT_FIELDS
        if getattr(live, field) != getattr(audit, field)
    }
    if dict(live.routed_destinations) != dict(audit.routed_destinations):
        mismatches["routed_destinations"] = {
            "live": dict(live.routed_destinations),
            "audit": dict(audit.routed_destinations),
        }
    if mismatches:
        raise OrchestrationInvariantError(
            f"Live-vs-audit terminal counter mismatch for run {run_id!r}: {mismatches!r}. "
            "The audit derive is the terminal record (ADR-030 §D); an unexplained divergence "
            "from the live loop counters means one of the two bookkeepers is broken."
        )
    if live.rows_coalesce_failed != audit.rows_coalesce_failed:
        import structlog

        structlog.get_logger(__name__).warning(
            "rows_coalesce_failed live/audit divergence (documented, tolerated)",
            run_id=run_id,
            live=live.rows_coalesce_failed,
            audit=audit.rows_coalesce_failed,
        )


def cli_completion_for(status: RunStatus) -> tuple[RunCompletionStatus, int]:
    """Phase 2.2 (elspeth-0de989c56d) — map terminal RunStatus to the
    CLI ``RunCompletionStatus`` + exit code.

    ``COMPLETED`` and ``EMPTY`` both map to ``COMPLETED`` / exit 0:
    a run that ingested zero rows is operationally a clean exit at the
    CLI surface (the Web layer carries the structural distinction).
    ``COMPLETED_WITH_FAILURES`` reuses the existing ``PARTIAL``
    ceremony — same exit-code-1 semantics as the post-run export-failure
    path that already emits ``PARTIAL``.
    """
    match status:
        case RunStatus.COMPLETED | RunStatus.EMPTY:
            return RunCompletionStatus.COMPLETED, 0
        case RunStatus.COMPLETED_WITH_FAILURES:
            return RunCompletionStatus.PARTIAL, 1
        case RunStatus.FAILED:
            return RunCompletionStatus.FAILED, 2
        case RunStatus.INTERRUPTED:
            return RunCompletionStatus.INTERRUPTED, 3
        case _:
            raise OrchestrationInvariantError(
                f"Cannot map RunStatus {status!r} to RunCompletionStatus — "
                f"this is a terminal-status-only mapping; RUNNING and other "
                f"non-terminal values must not reach this site."
            )
