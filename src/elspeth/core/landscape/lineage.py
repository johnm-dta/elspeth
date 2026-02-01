# src/elspeth/core/landscape/lineage.py
"""Lineage query functionality for ELSPETH Landscape.

Provides the explain() function to compose query results into
complete lineage for a token or row.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from elspeth.contracts import (
    Call,
    NodeState,
    RoutingEvent,
    RowLineage,
    Token,
    TokenOutcome,
    TransformErrorRecord,
    ValidationErrorRecord,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


@dataclass
class LineageResult:
    """Complete lineage for a token.

    Contains all information needed to explain how a row
    was processed through the pipeline, including any errors
    encountered during validation or transformation.
    """

    token: Token
    """The token being explained."""

    source_row: RowLineage
    """The original source row with resolved payload."""

    node_states: list[NodeState]
    """All node states visited by this token, in order."""

    routing_events: list[RoutingEvent]
    """All routing events for this token's states."""

    calls: list[Call]
    """All external calls made during processing."""

    parent_tokens: list[Token]
    """Parent tokens (for tokens created by fork/coalesce)."""

    validation_errors: list[ValidationErrorRecord] = field(default_factory=list)
    """Validation errors for this row (from source validation)."""

    transform_errors: list[TransformErrorRecord] = field(default_factory=list)
    """Transform errors for this token (from transform processing)."""

    outcome: TokenOutcome | None = None
    """Terminal outcome for this token (COMPLETED, ROUTED, FAILED, etc.)."""


def explain(
    recorder: "LandscapeRecorder",
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
    sink: str | None = None,
) -> LineageResult | None:
    """Query complete lineage for a token or row.

    Args:
        recorder: LandscapeRecorder with query methods.
        run_id: Run ID to query.
        token_id: Token ID for precise lineage (preferred for DAGs with forks).
        row_id: Row ID (requires disambiguation if multiple terminal tokens exist).
        sink: Sink name to disambiguate when row has multiple terminal tokens.

    Returns:
        LineageResult with complete lineage, or None if:
        - Token/row not found
        - No terminal tokens exist yet (e.g., all tokens BUFFERED)
        - Specified sink has no terminal tokens from this row

    Raises:
        ValueError: If neither token_id nor row_id provided.
        ValueError: If row_id has multiple terminal tokens and sink not specified.
        ValueError: If row_id with sink has multiple tokens (pipeline config issue).
    """
    if token_id is None and row_id is None:
        raise ValueError("Must provide either token_id or row_id")

    # Resolve token_id from row_id if needed
    if token_id is None and row_id is not None:
        # BATCH QUERY: Get all outcomes for this row in one query (avoid N+1)
        outcomes = recorder.get_token_outcomes_for_row(run_id, row_id)

        if not outcomes:
            return None  # Row not found or no outcomes recorded yet

        # Filter to terminal outcomes only
        terminal_outcomes = [o for o in outcomes if o.is_terminal]

        if not terminal_outcomes:
            # All tokens are non-terminal (e.g., BUFFERED awaiting aggregation)
            return None

        # If sink specified, filter to that sink
        if sink is not None:
            matching_outcomes = [o for o in terminal_outcomes if o.sink_name == sink]

            if not matching_outcomes:
                return None  # No tokens reached this sink

            if len(matching_outcomes) > 1:
                # Multiple tokens to same sink - ambiguous, fail explicitly
                token_ids = [o.token_id for o in matching_outcomes]
                raise ValueError(
                    f"Row {row_id} has {len(matching_outcomes)} tokens at sink '{sink}'. "
                    f"This may indicate fork paths reaching the same sink. "
                    f"Use token_id for precision: {token_ids[:5]}"
                    f"{'...' if len(token_ids) > 5 else ''}"
                )

            token_id = matching_outcomes[0].token_id
        else:
            # No sink - require exactly one terminal token
            if len(terminal_outcomes) > 1:
                sink_names = {o.sink_name for o in terminal_outcomes if o.sink_name}
                raise ValueError(
                    f"Row {row_id} has {len(terminal_outcomes)} terminal tokens "
                    f"across sinks: {sink_names}. "
                    f"Provide sink parameter to disambiguate, or use token_id for precision."
                )

            token_id = terminal_outcomes[0].token_id

    # At this point token_id is guaranteed to be set (either passed in or resolved from row_id)
    # The token_id is guaranteed non-None at this point by control flow
    token_id = cast(str, token_id)

    # Get the token
    token = recorder.get_token(token_id)
    if token is None:
        return None

    # Get source row with resolved payload via explain_row
    source_row = recorder.explain_row(run_id, token.row_id)
    if source_row is None:
        return None

    # Get node states for this token
    node_states = recorder.get_node_states_for_token(token_id)
    node_states.sort(key=lambda s: s.step_index)

    # Get routing events for each state
    routing_events: list[RoutingEvent] = []
    for state in node_states:
        events = recorder.get_routing_events(state.state_id)
        routing_events.extend(events)

    # Get external calls for each state
    calls: list[Call] = []
    for state in node_states:
        state_calls = recorder.get_calls(state.state_id)
        calls.extend(state_calls)

    # Get parent tokens
    # TIER 1 TRUST: token_parents is audit data - crash on any anomaly
    parent_tokens: list[Token] = []
    parents = recorder.get_token_parents(token_id)
    for parent in parents:
        parent_token = recorder.get_token(parent.parent_token_id)
        if parent_token is None:
            # This indicates audit DB corruption - a token_parents record
            # references a parent that doesn't exist. This should be impossible
            # with FK constraints enabled, but we crash as defense-in-depth.
            raise ValueError(
                f"Audit integrity violation: parent token '{parent.parent_token_id}' "
                f"not found for token '{token_id}'. The token_parents table references "
                f"a non-existent parent. This indicates database corruption."
            )
        parent_tokens.append(parent_token)

    # Get validation errors for this row (by hash)
    validation_errors = recorder.get_validation_errors_for_row(run_id, source_row.source_data_hash)

    # Get transform errors for this token
    transform_errors = recorder.get_transform_errors_for_token(token_id)

    # Get token outcome if recorded
    outcome = recorder.get_token_outcome(token_id)

    return LineageResult(
        token=token,
        source_row=source_row,
        node_states=node_states,
        routing_events=routing_events,
        calls=calls,
        parent_tokens=parent_tokens,
        validation_errors=validation_errors,
        transform_errors=transform_errors,
        outcome=outcome,
    )
