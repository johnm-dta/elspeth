"""Row outcome accumulation functions for the orchestrator.

This module contains functions for:
- Accumulating row processing outcomes into ExecutionCounters
- Handling coalesce timeout checks per-row
- Flushing pending coalesce operations at end-of-source

All functions operate on external state passed via parameters - they don't
maintain internal state. This follows the same pattern as aggregation.py:
pure delegation targets for the Orchestrator.

These functions were extracted from _execute_run() and _process_resumed_rows()
to eliminate ~400 lines of duplicated code. The extraction also fixed bugs
where the resume path was missing `rows_succeeded += 1` for coalesce
timeout continuations (lines 2124, 2139-2143 in the original).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING

from elspeth.contracts import PendingOutcome, RowOutcome, TokenInfo
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.engine.orchestrator.types import ExecutionCounters, PendingTokenMap

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.results import RowResult
    from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome
    from elspeth.engine.processor import RowProcessor


def _require_sink_name(result: RowResult) -> str:
    """Require sink_name for outcomes that must route to a sink.

    Replaces cast(str, result.sink_name) which is a no-op at runtime.
    If sink_name is None, this is a Tier 1 invariant violation (our data).
    """
    name: str | None = result.sink_name
    if name is None:
        raise OrchestrationInvariantError(f"Result with outcome {result.outcome} missing sink_name. Token: {result.token}")
    return name


def _route_to_sink(
    sink_name: str,
    pending_tokens: PendingTokenMap,
    token: TokenInfo,
    pending_outcome: RowOutcome,
    *,
    error_hash: str | None = None,
) -> None:
    """Validate sink exists in pending_tokens and append the token.

    Extracted from accumulate_row_outcomes where multiple outcome branches
    (COMPLETED, ROUTED, ROUTED_ON_ERROR, COALESCED) had identical
    validate+append logic.

    The `error_hash` keyword is required when ``pending_outcome`` is
    ``ROUTED_ON_ERROR`` (mirror of DIVERTED's contract — both outcomes are
    failure-handling redirects with an originating error that must be
    captured on the eventual outcome record). It must be None for ROUTED
    (intentional MOVE — no triggering error exists; passing a hash would
    be fabrication per CLAUDE.md Tier-1).

    Args:
        sink_name: Target sink name from result.sink_name
        pending_tokens: Sink-keyed accumulator to append to
        token: The token to route
        pending_outcome: The RowOutcome variant for the PendingOutcome
            (note: COALESCED tokens use COMPLETED here since they're
            finished from the sink's perspective)
        error_hash: 16-char sha256 prefix capturing the originating error;
            required for ROUTED_ON_ERROR, forbidden for ROUTED.
    """
    if sink_name not in pending_tokens:
        raise OrchestrationInvariantError(
            f"Sink '{sink_name}' not in configured sinks. Available: {sorted(pending_tokens.keys())}. Token: {token}"
        )
    # Offensive: ROUTED_ON_ERROR requires error_hash; ROUTED forbids it.
    # The PendingOutcome __post_init__ also enforces this via _REQUIRES_ERROR_HASH_OUTCOMES,
    # but a clearer error message at the routing site catches producer bugs
    # before they hit the dataclass invariant.
    if pending_outcome == RowOutcome.ROUTED_ON_ERROR and error_hash is None:
        raise OrchestrationInvariantError(
            f"_route_to_sink: ROUTED_ON_ERROR requires error_hash, got None. Token: {token}, sink: {sink_name}"
        )
    if pending_outcome == RowOutcome.ROUTED and error_hash is not None:
        raise OrchestrationInvariantError(
            f"_route_to_sink: ROUTED (intentional MOVE) must not carry error_hash; got {error_hash!r}. Token: {token}, sink: {sink_name}"
        )
    pending_tokens[sink_name].append((token, PendingOutcome(pending_outcome, error_hash=error_hash)))


def reconcile_sink_write_diversions(
    counters: ExecutionCounters,
    *,
    sink_name: str,
    pending_outcome: PendingOutcome | None,
    diversion_count: int,
) -> None:
    """Reconcile provisional counters after sink write reveals diversions.

    The processing loop counts sink-bound outcomes before durability is known so
    progress can remain responsive. Once SinkExecutor.write() partitions the
    batch into primary vs diverted rows, the final counters must remove the
    diverted rows from any provisional success/routing/quarantine totals.
    """
    if diversion_count == 0 or pending_outcome is None:
        return

    if pending_outcome.outcome == RowOutcome.COMPLETED:
        if counters.rows_succeeded < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_succeeded={counters.rows_succeeded} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_succeeded -= diversion_count
        return

    if pending_outcome.outcome == RowOutcome.ROUTED:
        if counters.rows_routed_success < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_routed_success={counters.rows_routed_success} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_routed_success -= diversion_count
        current_destination_count = counters.routed_destinations[sink_name]
        if current_destination_count < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted routed rows from "
                f"routed_destinations[{sink_name!r}]={current_destination_count} "
                f"(pending_outcome={pending_outcome.outcome.value!r}). "
                "This indicates counter drift between processing and sink-write phases."
            )
        remaining = current_destination_count - diversion_count
        if remaining == 0:
            del counters.routed_destinations[sink_name]
        else:
            counters.routed_destinations[sink_name] = remaining
        return

    if pending_outcome.outcome == RowOutcome.ROUTED_ON_ERROR:
        if counters.rows_routed_failure < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_routed_failure={counters.rows_routed_failure} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_routed_failure -= diversion_count
        current_destination_count = counters.routed_destinations[sink_name]
        if current_destination_count < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted routed rows from "
                f"routed_destinations[{sink_name!r}]={current_destination_count} "
                f"(pending_outcome={pending_outcome.outcome.value!r}). "
                "This indicates counter drift between processing and sink-write phases."
            )
        remaining = current_destination_count - diversion_count
        if remaining == 0:
            del counters.routed_destinations[sink_name]
        else:
            counters.routed_destinations[sink_name] = remaining
        return

    if pending_outcome.outcome == RowOutcome.QUARANTINED:
        if counters.rows_quarantined < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_quarantined={counters.rows_quarantined} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_quarantined -= diversion_count


def _emit_failed_token_completed(ctx: PluginContext, token: TokenInfo) -> None:
    """Emit TokenCompleted(FAILED) after audit recording succeeds."""
    from datetime import UTC, datetime

    from elspeth.contracts import TokenCompleted

    ctx.telemetry_emit(
        TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id=ctx.run_id,
            row_id=token.row_id,
            token_id=token.token_id,
            outcome=RowOutcome.FAILED,
            sink_name=None,
        )
    )


def _emit_failed_coalesce_telemetry(ctx: PluginContext, tokens: tuple[TokenInfo, ...]) -> None:
    """Emit failure telemetry for coalesce branches already failed in audit."""
    for token in tokens:
        _emit_failed_token_completed(ctx, token)


def accumulate_row_outcomes(
    results: Iterable[RowResult],
    counters: ExecutionCounters,
    pending_tokens: PendingTokenMap,
) -> None:
    """Accumulate row processing outcomes into counters and pending_tokens.

    Replaces the RowOutcome switch block that was duplicated 4 times in
    _execute_run() and _process_resumed_rows() (main loop, coalesce timeout
    continuations, coalesce flush continuations).

    This single implementation ensures consistent counting across all paths.
    In particular, COALESCED outcomes always increment rows_succeeded (fixing
    the bug where the resume path omitted this).

    Routing is determined by result.sink_name (set by on_success routing in
    the processor) rather than a default_sink_name parameter.

    Args:
        results: Iterable of RowProcessingResult from processor.process_row/process_token
        counters: Mutable ExecutionCounters to update
        pending_tokens: Dict of sink_name -> list of (token, pending_outcome) pairs
    """
    for result in results:
        if result.outcome == RowOutcome.COMPLETED:
            counters.rows_succeeded += 1
            sink_name = _require_sink_name(result)
            _route_to_sink(sink_name, pending_tokens, result.token, RowOutcome.COMPLETED)
        elif result.outcome == RowOutcome.ROUTED:
            counters.rows_routed_success += 1
            sink_name = _require_sink_name(result)
            counters.routed_destinations[sink_name] += 1
            _route_to_sink(sink_name, pending_tokens, result.token, RowOutcome.ROUTED)
        elif result.outcome == RowOutcome.ROUTED_ON_ERROR:
            # ROUTED_ON_ERROR carries the originating transform error through to
            # the outcome record. The producer site (processor.py) sets
            # result.error to a FailureInfo capturing the upstream exception;
            # here we convert FailureInfo.message to a 16-char sha256 prefix
            # matching the existing pattern used by FAILED, QUARANTINED, and
            # DIVERTED at the recorder layer.
            if result.error is None:
                # Offensive: RowResult.__post_init__ should have caught this
                # at construction time. Reaching this branch with error=None
                # means the producer bypassed the invariant — Tier 1 violation.
                # This guard must run before ANY counter or routed_destinations
                # mutation, otherwise a malformed Tier-1 RowResult partially
                # mutates audit counters before crashing.
                raise OrchestrationInvariantError(f"ROUTED_ON_ERROR result missing error (FailureInfo). Token: {result.token}")
            sink_name = _require_sink_name(result)
            error_hash = hashlib.sha256(result.error.message.encode()).hexdigest()[:16]
            counters.rows_routed_failure += 1
            counters.routed_destinations[sink_name] += 1
            _route_to_sink(
                sink_name,
                pending_tokens,
                result.token,
                RowOutcome.ROUTED_ON_ERROR,
                error_hash=error_hash,
            )
        elif result.outcome == RowOutcome.FAILED:
            counters.rows_failed += 1
        elif result.outcome == RowOutcome.QUARANTINED:
            counters.rows_quarantined += 1
        elif result.outcome == RowOutcome.DIVERTED:
            raise OrchestrationInvariantError(
                f"DIVERTED outcome should not appear in processing results — "
                f"diversions are counted in SinkExecutor, not the processing loop. "
                f"Token: {result.token}"
            )
        elif result.outcome == RowOutcome.FORKED:
            counters.rows_forked += 1
            # Children are counted separately when they reach terminal state
        elif result.outcome == RowOutcome.CONSUMED_IN_BATCH:
            # Aggregated - will be counted when batch flushes
            pass
        elif result.outcome == RowOutcome.DROPPED_BY_FILTER:
            counters.rows_succeeded += 1
        elif result.outcome == RowOutcome.COALESCED:
            sink_name = _require_sink_name(result)
            counters.rows_coalesced += 1
            counters.rows_succeeded += 1
            _route_to_sink(sink_name, pending_tokens, result.token, RowOutcome.COMPLETED)
        elif result.outcome == RowOutcome.EXPANDED:
            # Deaggregation parent token - children counted separately
            counters.rows_expanded += 1
        elif result.outcome == RowOutcome.BUFFERED:
            # Non-terminal: token held in aggregation buffer (passthrough or transform mode).
            # Terminal outcome deferred to flush time (count trigger, timeout, or end-of-source).
            # Post-flush assertion in _post_source_iteration_work verifies no tokens remain buffered.
            counters.rows_buffered += 1
        else:
            raise OrchestrationInvariantError(f"Unhandled RowOutcome variant: {result.outcome!r}. Token: {result.token}")


def _validate_coalesce_outcome(outcome: CoalesceOutcome) -> bool:
    """Validate CoalesceOutcome invariant and return whether it has a merged token.

    Raises OrchestrationInvariantError if the outcome has both or neither of
    merged_token and failure_reason — exactly one must be set.

    Returns:
        True if outcome has a merged token, False if it has a failure.
    """
    has_merged = outcome.merged_token is not None
    has_failure = outcome.failure_reason is not None
    if has_merged == has_failure:
        raise OrchestrationInvariantError(
            f"Invalid CoalesceOutcome state: merged={has_merged}, "
            f"failure_reason={outcome.failure_reason!r}. "
            f"Outcome must have exactly one of merged_token or failure_reason."
        )
    return has_merged


def _process_merged_coalesce_outcome(
    outcome: CoalesceOutcome,
    coalesce_name: CoalesceName,
    coalesce_node_map: dict[CoalesceName, NodeID],
    processor: RowProcessor,
    ctx: PluginContext,
    counters: ExecutionCounters,
    pending_tokens: PendingTokenMap,
) -> None:
    """Process a successfully merged CoalesceOutcome through the processor.

    Extracted from handle_coalesce_timeouts and flush_coalesce_pending which
    had identical merge routing logic.

    Does NOT increment rows_coalesced. Counting ownership belongs exclusively
    to accumulate_row_outcomes (COALESCED branch). Terminal coalesces produce
    RowOutcome.COALESCED which accumulate_row_outcomes counts. Non-terminal
    coalesces produce COMPLETED — the row's terminal state is "completed after
    coalesce", not "coalesced", so rows_coalesced is correctly not incremented.
    """
    merged_token = outcome.merged_token
    if merged_token is None:
        raise OrchestrationInvariantError("CoalesceOutcome has_merged=True but merged_token is None")
    coalesce_node_id = coalesce_node_map[coalesce_name]
    continuation_results: list[RowResult] = list(
        processor.process_token(
            token=merged_token,
            ctx=ctx,
            current_node_id=coalesce_node_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )
    )
    accumulate_row_outcomes(
        continuation_results,
        counters,
        pending_tokens,
    )


def handle_coalesce_timeouts(
    coalesce_executor: CoalesceExecutor,
    coalesce_node_map: dict[CoalesceName, NodeID],
    processor: RowProcessor,
    ctx: PluginContext,
    counters: ExecutionCounters,
    pending_tokens: PendingTokenMap,
) -> None:
    """Check and handle coalesce timeouts after processing each row.

    Extracted from the per-row coalesce timeout block that was duplicated in
    _execute_run() (lines 1286-1340) and _process_resumed_rows() (lines 2102-2145).

    Uses accumulate_row_outcomes() for downstream continuation handling, which
    fixes the bug where the resume path omitted `rows_succeeded += 1` for
    COMPLETED coalesce continuations.

    Args:
        coalesce_executor: CoalesceExecutor managing join barriers
        coalesce_node_map: Maps CoalesceName -> coalesce node ID in graph
        processor: RowProcessor for downstream processing
        ctx: Plugin context for transform execution
        counters: Mutable ExecutionCounters to update
        pending_tokens: Dict of sink_name -> tokens to append results to
    """
    for coalesce_name_str in coalesce_executor.get_registered_names():
        coalesce_name = CoalesceName(coalesce_name_str)
        timed_out = coalesce_executor.check_timeouts(
            coalesce_name=coalesce_name_str,
        )
        for outcome in timed_out:
            if _validate_coalesce_outcome(outcome):
                _process_merged_coalesce_outcome(
                    outcome,
                    coalesce_name,
                    coalesce_node_map,
                    processor,
                    ctx,
                    counters,
                    pending_tokens,
                )
            else:
                counters.rows_coalesce_failed += 1
                counters.rows_failed += len(outcome.consumed_tokens)
                _emit_failed_coalesce_telemetry(ctx, outcome.consumed_tokens)


def flush_coalesce_pending(
    coalesce_executor: CoalesceExecutor,
    coalesce_node_map: dict[CoalesceName, NodeID],
    processor: RowProcessor,
    ctx: PluginContext,
    counters: ExecutionCounters,
    pending_tokens: PendingTokenMap,
) -> None:
    """Flush pending coalesce operations at end-of-source.

    Extracted from the end-of-source coalesce flush that was duplicated in
    _execute_run() (lines 1420-1476) and _process_resumed_rows() (lines 2172-2221).

    Uses accumulate_row_outcomes() for consistent downstream outcome handling.

    Args:
        coalesce_executor: CoalesceExecutor managing join barriers
        coalesce_node_map: Maps CoalesceName -> coalesce node ID in graph
        processor: RowProcessor for downstream processing
        ctx: Plugin context for transform execution
        counters: Mutable ExecutionCounters to update
        pending_tokens: Dict of sink_name -> tokens to append results to
    """
    pending_outcomes = coalesce_executor.flush_pending()

    for outcome in pending_outcomes:
        if _validate_coalesce_outcome(outcome):
            # flush_pending outcomes carry coalesce_name on the outcome itself
            if outcome.coalesce_name is None:
                raise OrchestrationInvariantError(
                    "CoalesceOutcome has merged_token but coalesce_name is None. This indicates a bug in CoalesceExecutor.flush_pending()."
                )
            coalesce_name = CoalesceName(outcome.coalesce_name)
            _process_merged_coalesce_outcome(
                outcome,
                coalesce_name,
                coalesce_node_map,
                processor,
                ctx,
                counters,
                pending_tokens,
            )
        else:
            counters.rows_coalesce_failed += 1
            counters.rows_failed += len(outcome.consumed_tokens)
            _emit_failed_coalesce_telemetry(ctx, outcome.consumed_tokens)
