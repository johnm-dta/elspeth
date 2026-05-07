"""Core CRUD query functions for the Landscape audit database.

Functions: list_runs, get_run, list_rows, list_nodes, list_tokens,
list_operations, get_operation_calls, get_node_states, get_calls, query.

All functions accept (db, factory) as their first two parameters.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from elspeth.contracts import NodeStateStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.formatters import dataclass_to_dict, serialize_datetime
from elspeth.mcp.types import (
    CallDetail,
    CollisionFieldRecord,
    CollisionRecord,
    NodeDetail,
    NodeStateRecord,
    OperationCallRecord,
    OperationRecord,
    RowRecord,
    RunDetail,
    RunRecord,
    TokenChildRecord,
    TokenRecord,
)

_serialize_datetime = serialize_datetime
_dataclass_to_dict = dataclass_to_dict


def list_runs(db: LandscapeDB, factory: RecorderFactory, limit: int = 50, status: str | None = None) -> list[RunRecord]:
    """List pipeline runs.

    Args:
        db: Database connection
        factory: Recorder factory
        limit: Maximum number of runs to return (default 50)
        status: Filter by status (PENDING, RUNNING, COMPLETED, FAILED)

    Returns:
        List of run records with id, status, timestamps
    """
    from sqlalchemy import select

    from elspeth.contracts import RunStatus
    from elspeth.core.landscape.schema import runs_table

    with db.connection() as conn:
        query = select(runs_table).order_by(runs_table.c.started_at.desc()).limit(limit)

        if status is not None:
            # Validate status
            try:
                RunStatus(status)
            except ValueError as exc:
                valid = [s.value for s in RunStatus]
                raise ValueError(f"Invalid status '{status}'. Valid: {valid}") from exc
            query = query.where(runs_table.c.status == status)

        rows = conn.execute(query).fetchall()

    return [
        {
            "run_id": row.run_id,
            "status": row.status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "config_hash": row.config_hash,
            "export_status": row.export_status,
        }
        for row in rows
    ]


def get_run(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> RunDetail | None:
    """Get details of a specific run.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: The run ID to retrieve

    Returns:
        Run record or None if not found
    """
    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return None
    return cast(RunDetail, _dataclass_to_dict(run))


def list_rows(db: LandscapeDB, factory: RecorderFactory, run_id: str, limit: int = 100, offset: int = 0) -> list[RowRecord]:
    """List source rows for a run.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (default 0)

    Returns:
        List of row records
    """
    from sqlalchemy import select

    from elspeth.core.landscape.schema import rows_table

    with db.connection() as conn:
        query = select(rows_table).where(rows_table.c.run_id == run_id).order_by(rows_table.c.row_index).limit(limit).offset(offset)
        rows = conn.execute(query).fetchall()

    return [
        {
            "row_id": row.row_id,
            "run_id": row.run_id,
            "source_node_id": row.source_node_id,
            "row_index": row.row_index,
            "source_data_hash": row.source_data_hash,
            "source_data_ref": row.source_data_ref,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def list_nodes(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> list[NodeDetail]:
    """List all nodes (plugin instances) for a run.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query

    Returns:
        List of node records with plugin info
    """
    nodes = factory.data_flow.get_nodes(run_id)
    return [_dataclass_to_dict(node) for node in nodes]


def list_tokens(
    db: LandscapeDB,
    factory: RecorderFactory,
    run_id: str,
    row_id: str | None = None,
    limit: int = 100,
) -> list[TokenRecord]:
    """List tokens for a run or specific row.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        row_id: Optional row ID to filter by
        limit: Maximum tokens to return

    Returns:
        List of token records
    """
    from sqlalchemy import select

    from elspeth.core.landscape.schema import tokens_table

    with db.connection() as conn:
        query = select(tokens_table).where(tokens_table.c.run_id == run_id).order_by(tokens_table.c.created_at).limit(limit)

        if row_id is not None:
            query = query.where(tokens_table.c.row_id == row_id)

        rows = conn.execute(query).fetchall()

    return [
        {
            "token_id": row.token_id,
            "row_id": row.row_id,
            "branch_name": row.branch_name,
            "fork_group_id": row.fork_group_id,
            "join_group_id": row.join_group_id,
            "step_in_pipeline": row.step_in_pipeline,
            "expand_group_id": row.expand_group_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def get_token_children(
    db: LandscapeDB,
    factory: RecorderFactory,
    parent_token_id: str,
) -> list[TokenChildRecord]:
    """Get child tokens created from a parent (forward lineage).

    This closes the audit trail gap for COALESCED tokens: given a token
    that was consumed in a coalesce operation, find what it merged into.

    Args:
        db: Database connection
        factory: Recorder factory
        parent_token_id: Token ID to find children for

    Returns:
        List of TokenChildRecord entries. Each record shows a child token
        that was created from this parent (via coalesce), along with the
        parent's ordinal position in that child's parent list.
    """
    children = factory.query.get_token_children(parent_token_id)
    return [
        {
            "child_token_id": c.token_id,
            "parent_token_id": c.parent_token_id,
            "ordinal": c.ordinal,
        }
        for c in children
    ]


def list_operations(
    db: LandscapeDB,
    factory: RecorderFactory,
    run_id: str,
    operation_type: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[OperationRecord]:
    """List source/sink operations for a run.

    Operations are the source/sink equivalent of node_states. They track
    external I/O operations (blob downloads, file writes, database inserts)
    and provide a parent context for external calls made during source.load()
    or sink.write().

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        operation_type: Filter by type ('source_load' or 'sink_write')
        status: Filter by status ('open', 'completed', 'failed', 'pending')
        limit: Maximum operations to return

    Returns:
        List of operation records with node info
    """
    from sqlalchemy import select

    from elspeth.core.landscape.schema import nodes_table, operations_table

    with db.connection() as conn:
        query = (
            select(
                operations_table.c.operation_id,
                operations_table.c.run_id,
                operations_table.c.node_id,
                operations_table.c.operation_type,
                operations_table.c.started_at,
                operations_table.c.completed_at,
                operations_table.c.status,
                operations_table.c.error_message,
                operations_table.c.duration_ms,
                nodes_table.c.plugin_name,
            )
            .join(
                nodes_table,
                (operations_table.c.node_id == nodes_table.c.node_id) & (operations_table.c.run_id == nodes_table.c.run_id),
            )
            .where(operations_table.c.run_id == run_id)
            .order_by(operations_table.c.started_at.desc())
            .limit(limit)
        )

        if operation_type is not None:
            if operation_type not in ("source_load", "sink_write"):
                raise ValueError(f"Invalid operation_type '{operation_type}'. Valid: source_load, sink_write")
            query = query.where(operations_table.c.operation_type == operation_type)

        if status is not None:
            valid_statuses = ("open", "completed", "failed", "pending")
            if status not in valid_statuses:
                raise ValueError(f"Invalid status '{status}'. Valid: {valid_statuses}")
            query = query.where(operations_table.c.status == status)

        rows = conn.execute(query).fetchall()

    return [
        {
            "operation_id": row.operation_id,
            "run_id": row.run_id,
            "node_id": row.node_id,
            "plugin_name": row.plugin_name,
            "operation_type": row.operation_type,
            "status": row.status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "duration_ms": row.duration_ms,
            "error_message": row.error_message,
        }
        for row in rows
    ]


def get_operation_calls(db: LandscapeDB, factory: RecorderFactory, operation_id: str) -> list[OperationCallRecord]:
    """Get external calls for a source/sink operation.

    Unlike get_calls() which takes a state_id for transform calls, this
    method returns calls made during source.load() or sink.write().

    Args:
        db: Database connection
        factory: Recorder factory
        operation_id: Operation ID to query

    Returns:
        List of call records (HTTP, SQL, etc.)
    """
    from sqlalchemy import select

    from elspeth.core.landscape.schema import calls_table

    with db.connection() as conn:
        query = select(calls_table).where(calls_table.c.operation_id == operation_id).order_by(calls_table.c.call_index)
        rows = conn.execute(query).fetchall()

    return [
        {
            "call_id": row.call_id,
            "operation_id": row.operation_id,
            "call_index": row.call_index,
            "call_type": row.call_type,
            "status": row.status,
            "latency_ms": row.latency_ms,
            "request_hash": row.request_hash,
            "response_hash": row.response_hash,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def explain_token(
    db: LandscapeDB,
    factory: RecorderFactory,
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
    sink: str | None = None,
) -> dict[str, Any] | None:
    """Get complete lineage for a token or row.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        token_id: Token ID for precise lineage (preferred for DAGs with forks)
        row_id: Row ID (requires disambiguation if multiple terminal tokens)
        sink: Sink name to disambiguate when row has multiple terminal tokens

    Returns:
        Complete lineage including source row, node states, calls, errors, outcome,
        or None if not found
    """
    from elspeth.core.landscape.lineage import explain

    result = explain(factory.query, factory.data_flow, run_id, token_id=token_id, row_id=row_id, sink=sink)
    if result is None:
        return None
    result_dict = cast(dict[str, Any], _dataclass_to_dict(result))

    # Annotate routing_events with flow_type convenience field
    for event in result_dict["routing_events"]:
        event["flow_type"] = "divert" if event["mode"] == "divert" else "normal"

    # Build divert_summary from routing events
    divert_events = [e for e in result_dict["routing_events"] if e["mode"] == "divert"]

    if divert_events:
        divert_event = divert_events[-1]  # Last divert event is the terminal one
        edge = factory.data_flow.get_edge(divert_event["edge_id"])
        result_dict["divert_summary"] = {
            "diverted": True,
            "divert_type": "quarantine" if "__quarantine__" in edge.label else "error",
            "from_node": edge.from_node_id,
            "to_sink": edge.to_node_id,
            "edge_label": edge.label,
            "reason_hash": divert_event["reason_hash"],
        }
    else:
        result_dict["divert_summary"] = None

    return result_dict


def get_errors(
    db: LandscapeDB,
    factory: RecorderFactory,
    run_id: str,
    error_type: str = "all",
    limit: int = 100,
) -> dict[str, Any]:
    """Get validation and/or transform errors for a run.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        error_type: "validation", "transform", or "all" (default)
        limit: Maximum errors to return per type

    Returns:
        Errors grouped by type

    Raises:
        ValueError: If error_type is not "all", "validation", or "transform"
    """
    valid_error_types = {"all", "validation", "transform"}
    if error_type not in valid_error_types:
        raise ValueError(f"Invalid error_type '{error_type}'. Must be one of: {', '.join(sorted(valid_error_types))}")

    from sqlalchemy import select

    from elspeth.core.landscape.schema import transform_errors_table, validation_errors_table

    result: dict[str, Any] = {"run_id": run_id}

    with db.connection() as conn:
        if error_type in ("all", "validation"):
            query = (
                select(validation_errors_table)
                .where(validation_errors_table.c.run_id == run_id)
                .order_by(validation_errors_table.c.created_at.desc())
                .limit(limit)
            )
            rows = conn.execute(query).fetchall()
            result["validation_errors"] = [
                {
                    "error_id": row.error_id,
                    "node_id": row.node_id,
                    "row_hash": row.row_hash,
                    "row_data": json.loads(row.row_data_json) if row.row_data_json else None,
                    "schema_mode": row.schema_mode,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

        if error_type in ("all", "transform"):
            query = (
                select(transform_errors_table)
                .where(transform_errors_table.c.run_id == run_id)
                .order_by(transform_errors_table.c.created_at.desc())
                .limit(limit)
            )
            rows = conn.execute(query).fetchall()
            result["transform_errors"] = [
                {
                    "error_id": row.error_id,
                    "token_id": row.token_id,
                    "transform_id": row.transform_id,
                    "row_data": json.loads(row.row_data_json) if row.row_data_json else None,
                    "error_details": json.loads(row.error_details_json) if row.error_details_json else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    return result


def get_node_states(
    db: LandscapeDB,
    factory: RecorderFactory,
    run_id: str,
    node_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    include_context: bool = False,
) -> list[NodeStateRecord]:
    """Get node states (processing records) for a run.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        node_id: Optional filter by node ID
        status: Optional filter by status (PENDING, RUNNING, COMPLETED, FAILED)
        limit: Maximum states to return
        include_context: Include context_after, error, and success_reason JSON fields.
            These are large and expensive to parse; disabled by default.

    Returns:
        List of node state records
    """
    from sqlalchemy import select

    from elspeth.contracts import NodeStateStatus
    from elspeth.core.landscape.schema import node_states_table

    # Core columns always needed for NodeStateRecord
    core_columns = [
        node_states_table.c.state_id,
        node_states_table.c.token_id,
        node_states_table.c.node_id,
        node_states_table.c.step_index,
        node_states_table.c.attempt,
        node_states_table.c.status,
        node_states_table.c.input_hash,
        node_states_table.c.output_hash,
        node_states_table.c.duration_ms,
        node_states_table.c.started_at,
        node_states_table.c.completed_at,
    ]

    # Context columns are large JSON blobs — only fetch when requested.
    # This reduces I/O and memory for the common case where callers only
    # need structural metadata, not the full context payloads.
    if include_context:
        columns = [
            *core_columns,
            node_states_table.c.context_after_json,
            node_states_table.c.error_json,
            node_states_table.c.success_reason_json,
        ]
    else:
        columns = core_columns

    with db.connection() as conn:
        query = select(*columns).where(node_states_table.c.run_id == run_id).limit(limit)

        if node_id is not None:
            query = query.where(node_states_table.c.node_id == node_id)

        if status is not None:
            try:
                NodeStateStatus(status)
            except ValueError as exc:
                valid = [s.value for s in NodeStateStatus]
                raise ValueError(f"Invalid status '{status}'. Valid: {valid}") from exc
            query = query.where(node_states_table.c.status == status)

        query = query.order_by(
            node_states_table.c.step_index,
            node_states_table.c.attempt,
            node_states_table.c.token_id,
        )
        rows = conn.execute(query).fetchall()

    results: list[NodeStateRecord] = []
    for row in rows:
        record: NodeStateRecord = {
            "state_id": row.state_id,
            "token_id": row.token_id,
            "node_id": row.node_id,
            "step_index": row.step_index,
            "attempt": row.attempt,
            "status": row.status,
            "input_hash": row.input_hash,
            "output_hash": row.output_hash,
            "duration_ms": row.duration_ms,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }
        if include_context:
            record["context_after"] = json.loads(row.context_after_json) if row.context_after_json else None
            record["error"] = json.loads(row.error_json) if row.error_json else None
            record["success_reason"] = json.loads(row.success_reason_json) if row.success_reason_json else None
        results.append(record)

    return results


def get_calls(db: LandscapeDB, factory: RecorderFactory, state_id: str) -> list[CallDetail]:
    """Get external calls for a node state.

    Args:
        db: Database connection
        factory: Recorder factory
        state_id: Node state ID to query

    Returns:
        List of call records (LLM calls, HTTP requests, etc.)
    """
    calls = factory.query.get_calls(state_id)
    return [_dataclass_to_dict(call) for call in calls]


def _canonicalize_for_comparison(value: Any) -> str:
    """Convert a value to canonical string for structural equality comparison.

    Uses RFC 8785 (JCS) for deterministic JSON serialization. This ensures
    that structurally equal dicts with different key ordering compare as equal.

    Falls back to repr() for non-JSON-serializable types — these will compare
    by identity, which is conservative (may report false collisions).
    """
    import rfc8785

    try:
        return rfc8785.dumps(value).decode("utf-8")
    except (TypeError, ValueError):
        # Non-JSON-serializable type — fall back to repr
        return repr(value)


def list_collisions(
    db: LandscapeDB,
    factory: RecorderFactory,
    run_id: str,
    limit: int = 100,
) -> list[CollisionRecord]:
    """List coalesce collision events for a run.

    Finds all coalesce node states where union_field_collision_values is present,
    indicating that fields had conflicting values from different branches.

    This is essential for debugging production coalesce failures — without this,
    operators must use raw SQL to find why a merged row has unexpected values.

    Note: A single coalesce merge produces multiple node_states rows (one per
    consumed branch token). This function returns one record per node_states row
    that contains actual collisions (differing values). Callers who want to count
    unique collision patterns can group by (node_id, collision_fields) themselves.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        limit: Maximum collision records to return (applied AFTER filtering
            overlap-only rows that don't contain real collisions)

    Returns:
        List of collision records with field-level details including winner/loser values.
        Only fields with genuinely differing values are reported as collisions.
    """
    from sqlalchemy import or_, select

    from elspeth.core.landscape.schema import node_states_table, nodes_table

    # Chunked fetching: overlap-only filtering happens in Python (we can't tell
    # in SQL whether values are structurally identical), so we can't use a simple
    # LIMIT. Instead, fetch in batches with an over-fetch factor to bound memory
    # while ensuring we find enough real collisions.
    #
    # Over-fetch factor of 3 means: if we want 10 results, fetch 30 rows at a time.
    # Most coalesce rows that have union_field_collision_values also have real
    # collisions (not just overlap), so this is usually sufficient in one batch.
    batch_size = max(50, limit * 3)
    offset = 0

    # Base query without LIMIT/OFFSET — we'll add those per-batch
    base_query = (
        select(
            node_states_table.c.token_id,
            node_states_table.c.node_id,
            node_states_table.c.status,
            node_states_table.c.completed_at,
            node_states_table.c.context_after_json,
            nodes_table.c.plugin_name,
        )
        .select_from(
            node_states_table.join(
                nodes_table,
                (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(node_states_table.c.run_id == run_id)
        # Match both named coalesce nodes (coalesce:name) and plain 'coalesce'
        # from older/manual runs. Plain 'coalesce' is valid for manually assembled
        # pipelines or historical runs before named coalesce was standard.
        .where(
            or_(
                nodes_table.c.plugin_name.like("coalesce:%"),
                nodes_table.c.plugin_name == "coalesce",
            )
        )
        .where(node_states_table.c.context_after_json.isnot(None))
        .where(node_states_table.c.context_after_json.like("%union_field_collision_values%"))
        # state_id as tie-breaker ensures stable LIMIT/OFFSET pagination when
        # multiple rows share the same completed_at timestamp. Without this,
        # row order between batches is undefined and pagination can skip/duplicate.
        .order_by(node_states_table.c.completed_at.desc(), node_states_table.c.state_id)
    )

    results: list[CollisionRecord] = []

    with db.connection() as conn:
        while len(results) < limit:
            # Fetch next batch
            batch_query = base_query.limit(batch_size).offset(offset)
            rows = conn.execute(batch_query).fetchall()

            if not rows:
                # No more rows in database — done
                break

            offset += len(rows)

            for row in rows:
                context = json.loads(row.context_after_json)

                # Extract collision values: {field: [[branch, value], ...]}
                collision_values = context.get("union_field_collision_values", {})
                field_origins = context.get("union_field_origins", {})

                collision_fields: list[CollisionFieldRecord] = []
                for field, entries in collision_values.items():
                    if not entries:
                        continue

                    # Filter out overlap-only fields: union_field_collision_values contains
                    # all overlapping fields, even when values are identical. Only report
                    # fields where at least two branches provided different values.
                    #
                    # Use canonical JSON serialization for structural comparison. This
                    # ensures dicts with same key/value pairs but different insertion
                    # order compare as equal, avoiding false collision reports.
                    values = [e[1] for e in entries]
                    canonical_values = {_canonicalize_for_comparison(v) for v in values}
                    if len(canonical_values) < 2:
                        continue

                    # Determine winner from union_field_origins, not from entry order.
                    # entry order is merge order, but the actual winner depends on
                    # union_collision_policy (first_wins, last_wins, or fail).
                    # union_field_origins records which branch's value was kept.
                    #
                    # IMPORTANT: When status is FAILED (e.g., union_collision_policy='fail'),
                    # no winner was selected — the merge aborted. The metadata still contains
                    # union_field_origins from the pre-failure state, but reporting those as
                    # winners would be misleading. Set winner fields to None for failed merges.
                    # Compare against stored value (lowercase) per NodeStateStatus StrEnum.
                    is_failed = row.status == NodeStateStatus.FAILED.value
                    winner_branch: str | None = None
                    winner_value = None
                    if not is_failed:
                        winner_branch = field_origins.get(field)
                        if winner_branch is not None:
                            # Find the value from the winning branch
                            for branch, val in entries:
                                if branch == winner_branch:
                                    winner_value = val
                                    break

                    collision_fields.append(
                        {
                            "field": field,
                            "winner_branch": winner_branch,
                            "winner_value": winner_value,
                            "competing_values": [(e[0], e[1]) for e in entries],
                        }
                    )

                # Only emit a record if there are actual collisions (differing values)
                if not collision_fields:
                    continue

                # Check limit AFTER filtering to ensure we return up to `limit` real collisions
                if len(results) >= limit:
                    break

                results.append(
                    {
                        "run_id": run_id,
                        "token_id": row.token_id,
                        "node_id": row.node_id,
                        "plugin_name": row.plugin_name,
                        "status": row.status,
                        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                        "collision_fields": collision_fields,
                        "union_field_origins": field_origins,
                    }
                )

            # If we reached the limit within this batch, stop fetching more
            if len(results) >= limit:
                break

    return results


def _strip_sql_comments(sql: str) -> str:
    """Strip SQL comments (block and line) from a query string.

    Removes ``/* ... */`` block comments and ``-- ...`` line comments.
    Handles string literals so comment markers inside quotes are preserved.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_single_quote:
            out.append(ch)
            if ch == "'":
                # SQL escape for single quote inside single-quoted literal.
                if nxt == "'":
                    out.append(nxt)
                    i += 2
                    continue
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            out.append(ch)
            if ch == '"':
                # SQL escape for double quote inside quoted identifier.
                if nxt == '"':
                    out.append(nxt)
                    i += 2
                    continue
                in_double_quote = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            out.append(" ")
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            out.append(" ")
            i += 2
            continue

        if ch == "'":
            in_single_quote = True
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double_quote = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _strip_sql_string_literals(sql: str) -> str:
    """Strip string and quoted-identifier contents from SQL.

    Keeps quote delimiters while dropping inner content so keyword checks
    do not flag words that only appear inside quoted values/identifiers.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single_quote = False
    in_double_quote = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_single_quote:
            if ch == "'":
                out.append(ch)
                if nxt == "'":
                    out.append(nxt)
                    i += 2
                    continue
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == '"':
                out.append(ch)
                if nxt == '"':
                    out.append(nxt)
                    i += 2
                    continue
                in_double_quote = False
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double_quote = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


# Keyword blocklist: statements that are NOT read-only.
# Uses word-boundary matching to avoid false positives (e.g., created_at vs CREATE).
_FORBIDDEN_KEYWORDS = frozenset(
    {
        # Standard DML/DDL
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "MERGE",
        "REPLACE",
        # Transaction control
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "RELEASE",
        # SQLite-specific
        "PRAGMA",
        "ATTACH",
        "DETACH",
        "VACUUM",
        "REINDEX",
        # PostgreSQL-specific
        "COPY",
        "SET",
        # Procedure execution
        "EXEC",
        "EXECUTE",
        "CALL",
        # Data loading
        "LOAD",
        "IMPORT",
        "EXPORT",
    }
)


def _validate_readonly_sql(sql: str) -> None:
    """Validate that a SQL string is a single read-only SELECT or CTE.

    Raises ValueError if the query is empty, contains multiple statements,
    uses a non-SELECT/WITH prefix, or contains forbidden keywords.

    Note: Semicolons inside string literals will be rejected. This is an
    acceptable limitation for the MCP analysis server, which receives
    machine-generated queries from LLMs.
    """
    # Step 1: Strip comments to prevent bypass via comment tricks
    stripped = _strip_sql_comments(sql)
    normalized = stripped.strip()

    if not normalized:
        raise ValueError("Query is empty")

    # Step 2: Reject multi-statement payloads (semicolons).
    # Allow a single trailing semicolon (common in SQL tools).
    without_trailing = normalized.rstrip("; \t\n\r")
    if ";" in without_trailing:
        raise ValueError("Multiple statements are not allowed (semicolons found)")

    # Step 3: Require SELECT or WITH prefix (case-insensitive)
    upper = normalized.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT queries are allowed (must start with SELECT or WITH)")

    # Step 4: Reject forbidden keywords (word-boundary match).
    # Strip string literal contents first so that keywords inside quotes
    # (e.g., WHERE status = 'INSERT') don't trigger false positives.
    upper_stripped = _strip_sql_string_literals(upper)
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_stripped):
            raise ValueError(f"Query contains forbidden keyword: {keyword}")


def query(db: LandscapeDB, factory: RecorderFactory, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a read-only SQL query.

    Args:
        db: Database connection
        factory: Recorder factory
        sql: SQL query (must be a single SELECT or WITH...SELECT)
        params: Optional query parameters

    Returns:
        Query results as list of dicts

    Raises:
        ValueError: If query fails read-only validation
    """
    _validate_readonly_sql(sql)

    from sqlalchemy import text

    with db.read_only_connection() as conn:
        result = conn.execute(text(sql), params or {})
        columns = list(result.keys())
        rows = result.fetchall()

    if len(columns) != len(set(columns)):
        from collections import Counter

        dupes = [name for name, count in Counter(columns).items() if count > 1]
        raise ValueError(
            f"Query returns duplicate column names: {dupes}. "
            f"Use AS aliases to disambiguate (e.g., SELECT a.id AS a_id, b.id AS b_id). "
            f"Duplicate columns cause silent data loss in dict conversion."
        )

    return [dict(zip(columns, [_serialize_datetime(v) for v in row], strict=True)) for row in rows]
