"""Lineage query functionality for ELSPETH Landscape.

Provides the explain() function to compose query results into
complete lineage for a token or row.
"""

from dataclasses import dataclass
from typing import cast

from elspeth.contracts import (
    Call,
    NodeState,
    RoutingEvent,
    RowLineage,
    SchedulerEvent,
    Token,
    TokenOutcome,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.query_repository import QueryRepository


@dataclass(frozen=True, slots=True)
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

    node_states: tuple[NodeState, ...]
    """All node states visited by this token, in order."""

    routing_events: tuple[RoutingEvent, ...]
    """All routing events for this token's states."""

    calls: tuple[Call, ...]
    """All external calls made during processing."""

    parent_tokens: tuple[Token, ...]
    """Parent tokens (for tokens created by fork/coalesce)."""

    validation_errors: tuple[ValidationErrorRecord, ...] = ()
    """Validation errors for this row (from source validation)."""

    transform_errors: tuple[TransformErrorRecord, ...] = ()
    """Transform errors for this token (from transform processing)."""

    scheduler_events: tuple[SchedulerEvent, ...] = ()
    """Scheduler lease and work-item transitions for this token."""

    outcome: TokenOutcome | None = None
    """Terminal outcome for this token (COMPLETED, ROUTED, FAILED, etc.)."""

    def __post_init__(self) -> None:
        if self.token.row_id != self.source_row.row_id:
            raise AuditIntegrityError(
                f"Token row_id mismatch: token '{self.token.token_id}' has row_id "
                f"'{self.token.row_id}' but source_row has row_id '{self.source_row.row_id}'. "
                f"This indicates an audit integrity violation."
            )
        if self.token.run_id != self.source_row.run_id:
            raise AuditIntegrityError(
                f"Token run_id mismatch: token '{self.token.token_id}' belongs to run "
                f"'{self.token.run_id}' but source_row belongs to run '{self.source_row.run_id}'. "
                f"This indicates an audit integrity violation."
            )


def explain(
    query: QueryRepository,
    data_flow: DataFlowRepository,
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
    sink: str | None = None,
) -> LineageResult | None:
    """Query complete lineage for a token or row.

    Args:
        query: QueryRepository for token/row/state lookups.
        data_flow: DataFlowRepository for outcomes and error lookups.
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
        AuditIntegrityError: If a token resolved from token_outcomes is missing
            from the tokens table, or if a row referenced by a token doesn't
            exist (Tier 1 database corruption).
    """
    if token_id is None and row_id is None:
        raise ValueError("Must provide either token_id or row_id")

    # Track whether we resolved token_id ourselves from Tier 1 data (token_outcomes).
    # If so, the token MUST exist — a missing token is database corruption, not "not found".
    caller_provided_token_id = token_id is not None

    # Resolve token_id from row_id if needed
    if token_id is None and row_id is not None:
        # BATCH QUERY: Get all outcomes for this row in one query (avoid N+1)
        outcomes = data_flow.get_token_outcomes_for_row(run_id, row_id)

        if not outcomes:
            return None  # Row not found or no outcomes recorded yet

        # Filter to terminal outcomes only
        terminal_outcomes = [o for o in outcomes if o.completed]

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

    # Caller-provided tokens are external lookup keys, so scope them to the
    # requested run before loading row lineage. Tokens resolved from run-scoped
    # token_outcomes remain Tier 1 data and keep fail-closed corruption checks.
    token = query.get_token_for_run(run_id, token_id) if caller_provided_token_id else query.get_token(token_id)
    if token is None:
        if not caller_provided_token_id:
            # Token was resolved from our own token_outcomes table — it MUST exist
            raise AuditIntegrityError(
                f"Token '{token_id}' resolved from token_outcomes for row '{row_id}' "
                f"but does not exist in tokens table — database corruption (Tier 1 violation)"
            )
        return None  # Caller-provided token_id, genuinely not found

    # Get source row with resolved payload via explain_row
    source_row = query.explain_row(run_id, token.row_id)
    if source_row is None:
        # token.row_id is Tier 1 data — the row MUST exist
        raise AuditIntegrityError(
            f"Row '{token.row_id}' for token '{token_id}' does not exist in rows table "
            f"— foreign key violation, database corruption (Tier 1 violation)"
        )

    # Get node states for this token, sorted by step_index
    node_states = sorted(query.get_node_states_for_token(token_id), key=lambda s: s.step_index)

    # Batch query: Get routing events and calls for all states at once (N+1 fix)
    state_ids = [s.state_id for s in node_states]
    routing_events = query.get_routing_events_for_states(state_ids)
    calls = query.get_calls_for_states(state_ids)

    # Get parent tokens
    # TIER 1 TRUST: token_parents is audit data - crash on any anomaly
    parent_tokens: list[Token] = []
    parents = query.get_token_parents(token_id)

    # Validate parent relationships consistency using strict `is not None` checks.
    # Empty-string group IDs are audit corruption (UUIDs are never empty).
    group_ids = {
        "fork": token.fork_group_id,
        "join": token.join_group_id,
        "expand": token.expand_group_id,
    }
    # Reject empty-string group IDs — they're corruption, not valid values
    for gtype, gval in group_ids.items():
        if gval is not None and gval == "":
            raise AuditIntegrityError(
                f"Audit integrity violation: token '{token_id}' has empty {gtype}_group_id. "
                f"Group IDs must be non-empty UUIDs or NULL. This indicates database corruption."
            )
    set_groups = [k for k, v in group_ids.items() if v is not None]
    # At most one group type should be set (fork XOR join XOR expand)
    if len(set_groups) > 1:
        raise AuditIntegrityError(
            f"Audit integrity violation: token '{token_id}' has multiple group IDs set: "
            f"{set_groups}. A token can belong to exactly one lineage operation."
        )
    # Bidirectional consistency: group_id ↔ parents
    if set_groups and not parents:
        group_type = set_groups[0]
        group_id = group_ids[group_type]
        raise AuditIntegrityError(
            f"Audit integrity violation: token '{token_id}' has {group_type}_group_id='{group_id}' "
            f"but no parent relationships in token_parents table. Tokens with group IDs must have "
            f"parent lineage recorded. This indicates missing {group_type} metadata or audit corruption."
        )
    if parents and not set_groups:
        parent_ids = [p.parent_token_id for p in parents]
        raise AuditIntegrityError(
            f"Audit integrity violation: token '{token_id}' has parent relationships "
            f"{parent_ids} but no group ID (fork/join/expand) is set. Parent tokens must "
            f"be associated with a lineage operation."
        )

    parent_token_ids = [parent.parent_token_id for parent in parents]
    parent_tokens_by_id = {parent.token_id: parent for parent in query.get_tokens_by_ids(parent_token_ids)}
    for parent in parents:
        parent_token = parent_tokens_by_id.get(parent.parent_token_id)
        if parent_token is None:
            # This indicates audit DB corruption - a token_parents record
            # references a parent that doesn't exist. This should be impossible
            # with FK constraints enabled, but we crash as defense-in-depth.
            raise AuditIntegrityError(
                f"Audit integrity violation: parent token '{parent.parent_token_id}' "
                f"not found for token '{token_id}'. The token_parents table references "
                f"a non-existent parent. This indicates database corruption."
            )
        # Parent must belong to the same run. Cross-run parent references are
        # corruption — lineage operations (fork/coalesce/expand) are always
        # within a single run. (Note: cross-row parents ARE valid for coalesce,
        # where multiple rows merge into one output token.)
        if parent_token.run_id != run_id:
            raise AuditIntegrityError(
                f"Audit integrity violation: parent token '{parent.parent_token_id}' "
                f"belongs to run '{parent_token.run_id}' but child token '{token_id}' "
                f"belongs to run '{run_id}'. Cross-run parent lineage is impossible — "
                f"this indicates database corruption in token_parents."
            )
        parent_tokens.append(parent_token)

    # Get validation errors for this row (by hash)
    validation_errors = data_flow.get_validation_errors_for_row(
        run_id,
        source_row.source_data_hash,
        row_id=source_row.row_id,
    )

    # Get transform errors for this token
    transform_errors = data_flow.get_transform_errors_for_token(token_id)

    # Get durable scheduler transition history for this token.
    scheduler_events = query.get_scheduler_events(run_id=run_id, token_id=token_id)

    # Get token outcome if recorded
    outcome = data_flow.get_token_outcome(token_id)

    return LineageResult(
        token=token,
        source_row=source_row,
        node_states=tuple(node_states),
        routing_events=tuple(routing_events),
        calls=tuple(calls),
        parent_tokens=tuple(parent_tokens),
        validation_errors=tuple(validation_errors),
        transform_errors=tuple(transform_errors),
        scheduler_events=tuple(scheduler_events),
        outcome=outcome,
    )
