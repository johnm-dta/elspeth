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

from typing import TYPE_CHECKING

from elspeth.contracts import RunStatus
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.events import RunCompletionStatus
from elspeth.contracts.run_result import derive_terminal_run_status
from elspeth.engine.orchestrator.types import ExecutionCounters

if TYPE_CHECKING:
    from elspeth.contracts.audit import TokenOutcome
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


def derive_resume_terminal_status_from_audit(factory: RecorderFactory, run_id: str) -> tuple[RunStatus, ExecutionCounters]:
    """Recover the truthful cumulative terminal status + counters of a
    resumed run from the Landscape audit DB.

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
    for outcome_record in outcomes:
        if not outcome_record.completed:
            if (outcome_record.outcome, outcome_record.path) == (None, TerminalPath.BUFFERED):
                counters.rows_buffered += 1
            continue
        pair = (outcome_record.outcome, outcome_record.path)
        match pair:
            case (
                (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
                | (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED)
                | (TerminalOutcome.SUCCESS, TerminalPath.GATE_DISCARDED)
            ):
                counters.rows_succeeded += 1
            case (TerminalOutcome.SUCCESS, TerminalPath.COALESCED):
                counters.rows_coalesced += 1
                counters.rows_succeeded += 1
            case (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED):
                counters.rows_routed_success += 1
                counters.rows_succeeded += 1
                counters.routed_destinations[_require_routed_sink_name(outcome_record, pair)] += 1
            case (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED):
                counters.rows_failed += 1
                counters.rows_routed_failure += 1
                counters.routed_destinations[_require_routed_sink_name(outcome_record, pair)] += 1
            case (TerminalOutcome.FAILURE, TerminalPath.UNROUTED):
                counters.rows_failed += 1
            case (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE):
                counters.rows_quarantined += 1
                counters.rows_failed += 1
            case (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
                counters.rows_diverted += 1
                counters.rows_failed += 1
            case (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK):
                counters.rows_diverted += 1
            case (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT):
                # Parent tokens delegate predicate counters to their
                # children, but the structural fork count belongs here.
                counters.rows_forked += 1
            case (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT):
                # Deaggregation parents behave like fork parents for the
                # success/failure tally, with their own structural count.
                counters.rows_expanded += 1
            case (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED):
                # Batch-consumed tokens do not have a dedicated RunResult
                # counter; the BUFFERED record captures the structural row.
                pass
            case _:
                raise AssertionError(
                    f"Unhandled (outcome, path) pair in resume aggregation: {pair!r}. "
                    "Add a case here; see ADR-019 mapping table and the live accumulator."
                )
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
