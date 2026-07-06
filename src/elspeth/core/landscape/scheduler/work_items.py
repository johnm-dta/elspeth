"""Shared work-item row plumbing for the scheduler components.

Deterministic work-item identity, ``RowMapping`` -> ``TokenWorkItem``
hydration, READY-row value construction, Tier-1 insert helpers, and
cross-table reference validation. Module-level functions over a
caller-supplied connection; no single component owns them. Extracted from
``TokenSchedulerRepository`` (filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import nodes_table, rows_table, token_work_items_table, tokens_table


def work_item_id(run_id: str, token_id: str, node_id: str | None, attempt: int) -> str:
    node_key = "<terminal>" if node_id is None else node_id
    raw = f"{run_id}:{token_id}:{node_key}:{attempt}".encode()
    return hashlib.sha256(raw).hexdigest()


def work_item_identity(values: dict[str, object]) -> str:
    return (
        f"run_id={values['run_id']!r} token_id={values['token_id']!r} row_id={values['row_id']!r} "
        f"node_id={values['node_id']!r} attempt={values['attempt']!r}"
    )


def item_from_mapping(row: RowMapping) -> TokenWorkItem:
    data = dict(row)
    for key in ("available_at", "created_at", "updated_at", "lease_expires_at", "barrier_blocked_at"):
        value = data[key]
        if type(value) is datetime and value.tzinfo is None:
            data[key] = value.replace(tzinfo=UTC)
    return TokenWorkItem(
        work_item_id=data["work_item_id"],
        run_id=data["run_id"],
        token_id=data["token_id"],
        row_id=data["row_id"],
        node_id=data["node_id"],
        step_index=data["step_index"],
        ingest_sequence=data["ingest_sequence"],
        row_payload_json=data["row_payload_json"],
        status=TokenWorkStatus(data["status"]),
        queue_key=data["queue_key"],
        barrier_key=data["barrier_key"],
        on_success_sink=data["on_success_sink"],
        pending_sink_name=data["pending_sink_name"],
        pending_outcome=data["pending_outcome"],
        pending_path=data["pending_path"],
        pending_error_hash=data["pending_error_hash"],
        pending_error_message=data["pending_error_message"],
        branch_name=data["branch_name"],
        fork_group_id=data["fork_group_id"],
        join_group_id=data["join_group_id"],
        expand_group_id=data["expand_group_id"],
        coalesce_node_id=data["coalesce_node_id"],
        coalesce_name=data["coalesce_name"],
        attempt=data["attempt"],
        lease_owner=data["lease_owner"],
        lease_expires_at=data["lease_expires_at"],
        available_at=data["available_at"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        barrier_blocked_at=data["barrier_blocked_at"],
        barrier_adopted_epoch=data["barrier_adopted_epoch"],
    )


def ready_work_item_values(
    *,
    run_id: str,
    token_id: str,
    row_id: str,
    node_id: str | None,
    step_index: int,
    ingest_sequence: int,
    row_payload_json: str,
    available_at: datetime,
    attempt: int,
    queue_key: str | None,
    barrier_key: str | None,
    on_success_sink: str | None,
    branch_name: str | None,
    fork_group_id: str | None,
    join_group_id: str | None,
    expand_group_id: str | None,
    coalesce_node_id: str | None,
    coalesce_name: str | None,
) -> dict[str, object]:
    return {
        "work_item_id": work_item_id(run_id, token_id, node_id, attempt),
        "run_id": run_id,
        "token_id": token_id,
        "row_id": row_id,
        "node_id": node_id,
        "step_index": step_index,
        "ingest_sequence": ingest_sequence,
        "row_payload_json": row_payload_json,
        "status": TokenWorkStatus.READY.value,
        "queue_key": queue_key,
        "barrier_key": barrier_key,
        "on_success_sink": on_success_sink,
        "pending_sink_name": None,
        "pending_outcome": None,
        "pending_path": None,
        "pending_error_hash": None,
        "pending_error_message": None,
        "branch_name": branch_name,
        "fork_group_id": fork_group_id,
        "join_group_id": join_group_id,
        "expand_group_id": expand_group_id,
        "coalesce_node_id": coalesce_node_id,
        "coalesce_name": coalesce_name,
        "attempt": attempt,
        "lease_owner": None,
        "lease_expires_at": None,
        "available_at": available_at,
        "created_at": available_at,
        "updated_at": available_at,
    }


def validate_node_cursor(conn: Connection, *, run_id: str, node_id: str | None, label: str) -> None:
    if node_id is None:
        return
    exists = conn.execute(
        select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id, nodes_table.c.node_id == node_id)
    ).scalar_one_or_none()
    if exists is None:
        raise AuditIntegrityError(
            f"Scheduler work item references {label}={node_id!r} outside run_id={run_id!r}; "
            "node cursors must be owned by the scheduled run."
        )


def validate_work_item_references(
    conn: Connection,
    *,
    run_id: str,
    token_id: str,
    row_id: str,
    ingest_sequence: int,
    node_id: str | None,
    coalesce_node_id: str | None,
) -> None:
    token_row_id = conn.execute(
        select(tokens_table.c.row_id).where(tokens_table.c.run_id == run_id, tokens_table.c.token_id == token_id)
    ).scalar_one_or_none()
    if token_row_id is None:
        raise AuditIntegrityError(
            f"Scheduler work item references token_id={token_id!r} outside run_id={run_id!r}; "
            "token cursors must be owned by the scheduled run."
        )
    if token_row_id != row_id:
        raise AuditIntegrityError(
            f"Scheduler work item token_id={token_id!r} in run_id={run_id!r} belongs to row_id={token_row_id!r}, "
            f"not scheduled row_id={row_id!r}."
        )
    row_ingest_sequence = conn.execute(
        select(rows_table.c.ingest_sequence).where(rows_table.c.run_id == run_id, rows_table.c.row_id == row_id)
    ).scalar_one_or_none()
    if row_ingest_sequence is None:
        raise AuditIntegrityError(
            f"Scheduler work item references row_id={row_id!r} outside run_id={run_id!r}; row cursors must be owned by the scheduled run."
        )
    if row_ingest_sequence != ingest_sequence:
        raise AuditIntegrityError(
            f"Scheduler work item row_id={row_id!r} in run_id={run_id!r} has ingest_sequence={row_ingest_sequence}, "
            f"not scheduled ingest_sequence={ingest_sequence}."
        )
    validate_node_cursor(conn, run_id=run_id, node_id=node_id, label="node_id")
    validate_node_cursor(
        conn,
        run_id=run_id,
        node_id=coalesce_node_id,
        label="coalesce_node_id",
    )


def insert_work_item(conn: Connection, *, values: dict[str, object], operation: str) -> None:
    identity = work_item_identity(values)
    try:
        result = conn.execute(token_work_items_table.insert().values(**values))
    except SQLAlchemyError as exc:
        raise LandscapeRecordError(
            f"Scheduler {operation} failed for {identity} — database rejected audit write: {type(exc).__name__}"
        ) from exc
    if result.rowcount != 1:
        raise LandscapeRecordError(f"Scheduler {operation} affected {result.rowcount} rows for {identity}; expected exactly one audit row.")


def insert_work_item_idempotent(conn: Connection, *, values: dict[str, object], operation: str) -> bool:
    """Insert a work item or accept an exact existing continuation.

    Child continuations are persisted before their parent claim is marked
    complete. A crash in that window can replay the parent and attempt to
    enqueue the same child again. The deterministic work_item_id lets us
    reconcile that replay, but only when the existing durable row carries
    the same resume cursor and token lineage.
    """
    try:
        insert_work_item(conn, values=values, operation=operation)
        return True
    except LandscapeRecordError as exc:
        cause = exc.__cause__
        if type(cause) is not IntegrityError:
            raise

    existing = (
        conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == values["work_item_id"]))
        .mappings()
        .one_or_none()
    )
    if existing is None:
        raise LandscapeRecordError(
            f"Scheduler {operation} failed for {work_item_identity(values)} — "
            "database reported duplicate identity but no matching row could be read back."
        )

    comparable_fields = (
        "work_item_id",
        "run_id",
        "token_id",
        "row_id",
        "node_id",
        "step_index",
        "ingest_sequence",
        "row_payload_json",
        "queue_key",
        "barrier_key",
        "on_success_sink",
        "pending_sink_name",
        "pending_outcome",
        "pending_path",
        "pending_error_hash",
        "pending_error_message",
        "branch_name",
        "fork_group_id",
        "join_group_id",
        "expand_group_id",
        "coalesce_node_id",
        "coalesce_name",
        "attempt",
    )
    mismatches = {
        field_name: {"expected": values[field_name], "actual": existing[field_name]}
        for field_name in comparable_fields
        if existing[field_name] != values[field_name]
    }
    if mismatches:
        raise LandscapeRecordError(
            f"Scheduler {operation} found incompatible existing work item for {work_item_identity(values)}: {mismatches!r}"
        )
    return False
