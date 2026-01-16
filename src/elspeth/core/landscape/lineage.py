# src/elspeth/core/landscape/lineage.py
"""Lineage query functionality for ELSPETH Landscape.

Provides the explain() function to compose query results into
complete lineage for a token or row.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from elspeth.contracts import RowLineage
from elspeth.core.landscape.models import (
    Call,
    NodeState,
    RoutingEvent,
    Token,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


@dataclass
class LineageResult:
    """Complete lineage for a token.

    Contains all information needed to explain how a row
    was processed through the pipeline.
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


def explain(
    recorder: "LandscapeRecorder",
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
) -> LineageResult | None:
    """Query complete lineage for a token or row.

    Args:
        recorder: LandscapeRecorder with query methods.
        run_id: Run ID to query.
        token_id: Token ID for precise lineage (preferred).
        row_id: Row ID (will use first token for that row).

    Returns:
        LineageResult with complete lineage, or None if not found.

    Raises:
        ValueError: If neither token_id nor row_id provided.
    """
    if token_id is None and row_id is None:
        raise ValueError("Must provide either token_id or row_id")

    # Resolve token_id from row_id if needed
    if token_id is None and row_id is not None:
        tokens = recorder.get_tokens(row_id)
        if not tokens:
            return None
        token_id = tokens[0].token_id

    # At this point token_id is guaranteed to be set (either passed in or resolved from row_id)
    assert token_id is not None  # for type narrowing

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
    parent_tokens: list[Token] = []
    parents = recorder.get_token_parents(token_id)
    for parent in parents:
        parent_token = recorder.get_token(parent.parent_token_id)
        if parent_token is not None:
            parent_tokens.append(parent_token)

    return LineageResult(
        token=token,
        source_row=source_row,
        node_states=node_states,
        routing_events=routing_events,
        calls=calls,
        parent_tokens=parent_tokens,
    )
