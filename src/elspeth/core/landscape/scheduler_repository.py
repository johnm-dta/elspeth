"""Durable token scheduler repository.

The v1 scheduler is intentionally embedded-database backed and single-process
friendly. It records enough state for a run to resume without rediscovering
source rows: ready work, active leases, delayed retry availability, queue or
barrier blocking keys, and terminal/failed states.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, and_, func, select, update
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import nodes_table, rows_table, token_work_items_table, tokens_table


class TokenSchedulerRepository:
    """Persistence boundary for token scheduler work items."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def enqueue_ready(
        self,
        *,
        run_id: str,
        token_id: str,
        row_id: str,
        node_id: str | None,
        step_index: int,
        ingest_sequence: int,
        row_payload_json: str,
        available_at: datetime,
        attempt: int = 1,
        queue_key: str | None = None,
        barrier_key: str | None = None,
        on_success_sink: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
        expand_group_id: str | None = None,
        coalesce_node_id: str | None = None,
        coalesce_name: str | None = None,
    ) -> TokenWorkItem:
        """Persist a READY token continuation.

        Token lineage and coalesce cursor fields are durable resume metadata:
        workers that claim this row later must be able to rebuild the same
        ``TokenInfo`` and ``WorkItem`` without any live in-memory queue state.
        """
        work_item_id = self._work_item_id(run_id, token_id, node_id, attempt)
        values: dict[str, object] = {
            "work_item_id": work_item_id,
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
        with self._engine.begin() as conn:
            self._validate_work_item_references(
                conn,
                run_id=run_id,
                token_id=token_id,
                row_id=row_id,
                ingest_sequence=ingest_sequence,
                node_id=node_id,
                coalesce_node_id=coalesce_node_id,
            )
            self._insert_work_item(conn, values=values, operation="enqueue READY scheduler work")
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        return self._item_from_mapping(row)

    def ensure_blocked_barrier_work_item(
        self,
        *,
        run_id: str,
        token_id: str,
        row_id: str,
        node_id: str | None,
        step_index: int,
        ingest_sequence: int,
        row_payload_json: str,
        barrier_key: str,
        available_at: datetime,
        attempt: int = 1,
        on_success_sink: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
        expand_group_id: str | None = None,
        coalesce_node_id: str | None = None,
        coalesce_name: str | None = None,
    ) -> TokenWorkItem:
        """Idempotently materialize a checkpoint-restored barrier block.

        Resume can restore aggregation buffers from checkpoint state before any
        READY scheduler work is claimed. Those tokens are already waiting at a
        barrier, so the scheduler row must be BLOCKED immediately rather than
        briefly leased through the normal advancement path.
        """
        work_item_id = self._work_item_id(run_id, token_id, node_id, attempt)
        values: dict[str, object] = {
            "work_item_id": work_item_id,
            "run_id": run_id,
            "token_id": token_id,
            "row_id": row_id,
            "node_id": node_id,
            "step_index": step_index,
            "ingest_sequence": ingest_sequence,
            "row_payload_json": row_payload_json,
            "status": TokenWorkStatus.BLOCKED.value,
            "queue_key": None,
            "barrier_key": barrier_key,
            "on_success_sink": on_success_sink,
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
        with self._engine.begin() as conn:
            self._validate_work_item_references(
                conn,
                run_id=run_id,
                token_id=token_id,
                row_id=row_id,
                ingest_sequence=ingest_sequence,
                node_id=node_id,
                coalesce_node_id=coalesce_node_id,
            )
            row = (
                conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id))
                .mappings()
                .one_or_none()
            )
            if row is None:
                self._insert_work_item(conn, values=values, operation="restore BLOCKED scheduler work")
                row = (
                    conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id))
                    .mappings()
                    .one()
                )

        item = self._item_from_mapping(row)
        if (
            item.run_id != run_id
            or item.token_id != token_id
            or item.row_id != row_id
            or item.node_id != node_id
            or item.attempt != attempt
            or item.status is not TokenWorkStatus.BLOCKED
            or item.barrier_key != barrier_key
        ):
            raise AuditIntegrityError(
                f"Scheduler checkpoint restore found incompatible existing work_item_id={work_item_id!r}: "
                f"expected BLOCKED run_id={run_id!r} token_id={token_id!r} row_id={row_id!r} "
                f"node_id={node_id!r} barrier_key={barrier_key!r} attempt={attempt}, got {item!r}."
            )
        expected_resume_metadata = {
            "step_index": step_index,
            "ingest_sequence": ingest_sequence,
            "row_payload_json": row_payload_json,
            "queue_key": None,
            "on_success_sink": on_success_sink,
            "branch_name": branch_name,
            "fork_group_id": fork_group_id,
            "join_group_id": join_group_id,
            "expand_group_id": expand_group_id,
            "coalesce_node_id": coalesce_node_id,
            "coalesce_name": coalesce_name,
        }
        actual_resume_metadata = {field_name: getattr(item, field_name) for field_name in expected_resume_metadata}
        mismatches = {
            field_name: {"expected": expected_value, "actual": actual_resume_metadata[field_name]}
            for field_name, expected_value in expected_resume_metadata.items()
            if actual_resume_metadata[field_name] != expected_value
        }
        if mismatches:
            raise AuditIntegrityError(
                f"Scheduler checkpoint restore found stale existing work_item_id={work_item_id!r} "
                f"for run_id={run_id!r} token_id={token_id!r} row_id={row_id!r} node_id={node_id!r} "
                f"barrier_key={barrier_key!r}: resume metadata mismatches={mismatches!r}."
            )
        return item

    @staticmethod
    def _validate_work_item_references(
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
                f"Scheduler work item references row_id={row_id!r} outside run_id={run_id!r}; "
                "row cursors must be owned by the scheduled run."
            )
        if row_ingest_sequence != ingest_sequence:
            raise AuditIntegrityError(
                f"Scheduler work item row_id={row_id!r} in run_id={run_id!r} has ingest_sequence={row_ingest_sequence}, "
                f"not scheduled ingest_sequence={ingest_sequence}."
            )
        TokenSchedulerRepository._validate_node_cursor(conn, run_id=run_id, node_id=node_id, label="node_id")
        TokenSchedulerRepository._validate_node_cursor(
            conn,
            run_id=run_id,
            node_id=coalesce_node_id,
            label="coalesce_node_id",
        )

    @staticmethod
    def _validate_node_cursor(conn: Connection, *, run_id: str, node_id: str | None, label: str) -> None:
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

    @staticmethod
    def _insert_work_item(conn: Connection, *, values: dict[str, object], operation: str) -> None:
        identity = TokenSchedulerRepository._work_item_identity(values)
        try:
            result = conn.execute(token_work_items_table.insert().values(**values))
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"Scheduler {operation} failed for {identity} — database rejected audit write: {type(exc).__name__}"
            ) from exc
        if result.rowcount != 1:
            raise LandscapeRecordError(
                f"Scheduler {operation} affected {result.rowcount} rows for {identity}; expected exactly one audit row."
            )

    @staticmethod
    def _work_item_identity(values: dict[str, object]) -> str:
        return (
            f"run_id={values['run_id']!r} token_id={values['token_id']!r} row_id={values['row_id']!r} "
            f"node_id={values['node_id']!r} attempt={values['attempt']!r}"
        )

    @staticmethod
    def serialize_row_payload(row: PipelineRow) -> str:
        """Serialize the current token row and its contract for durable resume."""
        return canonical_json(
            {
                "row": row.to_checkpoint_format(),
                "contract": row.contract.to_checkpoint_format(),
            }
        )

    @staticmethod
    def deserialize_row_payload(row_payload_json: str) -> PipelineRow:
        """Restore a scheduler row payload written by ``serialize_row_payload``."""
        try:
            payload = json.loads(row_payload_json)
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError(f"Corrupt scheduler row payload JSON: {exc}") from exc
        if type(payload) is not dict:
            raise AuditIntegrityError(f"Corrupt scheduler row payload: expected object, got {type(payload).__name__}")

        try:
            row_checkpoint = payload["row"]
            contract_checkpoint = payload["contract"]
        except KeyError as exc:
            raise AuditIntegrityError(f"Corrupt scheduler row payload: missing {exc}. Available keys: {sorted(payload.keys())}") from exc
        if type(row_checkpoint) is not dict:
            raise AuditIntegrityError(f"Corrupt scheduler row payload: row must be object, got {type(row_checkpoint).__name__}")
        if type(contract_checkpoint) is not dict:
            raise AuditIntegrityError(f"Corrupt scheduler row payload: contract must be object, got {type(contract_checkpoint).__name__}")

        contract = SchemaContract.from_checkpoint(contract_checkpoint)
        return PipelineRow.from_checkpoint(row_checkpoint, {contract.version_hash(): contract})

    def claim_ready(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> TokenWorkItem | None:
        """Claim the next available READY work item for a bounded lease."""
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._engine.begin() as conn:
            row = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.READY.value)
                    .where(token_work_items_table.c.available_at <= now)
                    .order_by(
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.created_at,
                    )
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None

            result = conn.execute(
                update(token_work_items_table)
                .where(
                    and_(
                        token_work_items_table.c.work_item_id == row["work_item_id"],
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.status == TokenWorkStatus.READY.value,
                    )
                )
                .values(
                    status=TokenWorkStatus.LEASED.value,
                    lease_owner=lease_owner,
                    lease_expires_at=lease_expires_at,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise AuditIntegrityError(
                    f"Scheduler claim_ready lost race on run_id={run_id!r} "
                    f"work_item_id={row['work_item_id']!r}: SELECT saw READY but UPDATE matched "
                    f"{result.rowcount} rows for lease_owner={lease_owner!r}. Concurrent claim by another worker."
                )
            claimed = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
                    .where(token_work_items_table.c.lease_owner == lease_owner)
                )
                .mappings()
                .one_or_none()
            )
            if claimed is None:
                raise AuditIntegrityError(
                    f"Scheduler claim invariant violated for work_item_id={row['work_item_id']!r}: leased row was not owned by the claimant"
                )
        return self._item_from_mapping(claimed)

    def recover_expired_leases(self, *, run_id: str, now: datetime) -> int:
        """Return expired LEASED work to READY for retry by another worker."""
        with self._engine.begin() as conn:
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
                .where(token_work_items_table.c.lease_expires_at < now)
                .values(
                    status=TokenWorkStatus.READY.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
        return result.rowcount

    def mark_waiting(
        self,
        *,
        work_item_id: str,
        available_at: datetime,
        now: datetime,
        expected_lease_owner: str,
    ) -> TokenWorkItem:
        """Move a claimed item to WAITING until ``available_at``."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.WAITING,
            expected_lease_owner=expected_lease_owner,
            available_at=available_at,
            lease_owner=None,
            lease_expires_at=None,
        )

    def release_waiting(self, *, run_id: str, now: datetime) -> int:
        """Release due WAITING items back to READY."""
        with self._engine.begin() as conn:
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status == TokenWorkStatus.WAITING.value)
                .where(token_work_items_table.c.available_at <= now)
                .values(
                    status=TokenWorkStatus.READY.value,
                    updated_at=now,
                )
            )
        return result.rowcount

    def mark_blocked(
        self,
        *,
        work_item_id: str,
        queue_key: str | None,
        barrier_key: str | None,
        now: datetime,
        expected_lease_owner: str,
    ) -> TokenWorkItem:
        """Move an item to BLOCKED at a queue or barrier."""
        if queue_key is None and barrier_key is None:
            raise AuditIntegrityError(
                f"Scheduler cannot block work_item_id={work_item_id!r} without a queue_key or barrier_key; "
                "the work item would be unreleasable on resume."
            )
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.BLOCKED,
            expected_lease_owner=expected_lease_owner,
            queue_key=queue_key,
            barrier_key=barrier_key,
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_terminal(self, *, work_item_id: str, now: datetime, expected_lease_owner: str) -> TokenWorkItem:
        """Mark a leased work item terminal."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.TERMINAL,
            expected_lease_owner=expected_lease_owner,
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_failed(self, *, work_item_id: str, now: datetime, expected_lease_owner: str | None = None) -> TokenWorkItem:
        """Mark a work item failed after retries are exhausted."""
        expected_statuses = (TokenWorkStatus.READY,) if expected_lease_owner is None else (TokenWorkStatus.LEASED,)
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.FAILED,
            expected_statuses=expected_statuses,
            expected_lease_owner=expected_lease_owner,
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_blocked_barrier_terminal(
        self,
        *,
        run_id: str,
        barrier_key: str,
        token_ids: tuple[str, ...],
        now: datetime,
    ) -> int:
        """Mark BLOCKED work consumed by a resolved barrier as terminal."""
        requested_token_ids = frozenset(token_ids)
        if not token_ids:
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization for run_id={run_id!r} barrier_key={barrier_key!r} "
                "requires at least one live token_id; refusing to terminalize durable BLOCKED rows without "
                "live barrier evidence."
            )
        if len(requested_token_ids) != len(token_ids):
            duplicates = sorted(token_id for token_id in requested_token_ids if token_ids.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization received duplicate live token_ids for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        with self._engine.begin() as conn:
            if requested_token_ids:
                durable_token_ids = frozenset(
                    conn.execute(
                        select(token_work_items_table.c.token_id)
                        .where(token_work_items_table.c.run_id == run_id)
                        .where(token_work_items_table.c.barrier_key == barrier_key)
                        .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                    )
                    .scalars()
                    .all()
                )
                missing_token_ids = requested_token_ids - durable_token_ids
                if missing_token_ids:
                    matching_count = len(requested_token_ids & durable_token_ids)
                    raise AuditIntegrityError(
                        f"Scheduler barrier terminalization mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                        f"live consumed {len(requested_token_ids)} token(s), but durable BLOCKED rows contained "
                        f"{matching_count} matching token(s) out of {len(durable_token_ids)} blocked token(s). "
                        f"missing token_ids={sorted(missing_token_ids)!r}; durable token_ids={sorted(durable_token_ids)!r}."
                    )
            statement = (
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.barrier_key == barrier_key)
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .where(token_work_items_table.c.token_id.in_(token_ids))
            )
            result = conn.execute(
                statement.values(
                    status=TokenWorkStatus.TERMINAL.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            rowcount = result.rowcount
            if rowcount is None or rowcount < 0:
                raise AuditIntegrityError(
                    f"Scheduler barrier terminalization for run_id={run_id!r} barrier_key={barrier_key!r} "
                    f"returned unsupported rowcount {rowcount!r}; durable/live reconciliation cannot be proven."
                )
            if requested_token_ids and rowcount != len(requested_token_ids):
                raise AuditIntegrityError(
                    f"Scheduler barrier terminalization mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                    f"live consumed {len(requested_token_ids)} token(s), but durable scheduler terminalized {rowcount}."
                )
        return int(rowcount)

    def count_active_work(self, *, run_id: str) -> int:
        """Count non-terminal scheduler work for a run."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.WAITING.value,
            TokenWorkStatus.BLOCKED.value,
        )
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status.in_(active_statuses))
            ).scalar_one()
        return int(result)

    def active_row_ids(self, *, run_id: str) -> frozenset[str]:
        """Return row IDs represented by non-terminal scheduler work."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.WAITING.value,
            TokenWorkStatus.BLOCKED.value,
        )
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table.c.row_id)
                    .distinct()
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status.in_(active_statuses))
                )
                .scalars()
                .all()
            )
        return frozenset(rows)

    def summarize_active_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize active work grouped by status and blocking keys."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.WAITING.value,
            TokenWorkStatus.BLOCKED.value,
        )
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                        func.count(),
                    )
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status.in_(active_statuses))
                    .group_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                    )
                    .order_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                    )
                )
                .tuples()
                .all()
            )
        return tuple(
            f"status={status}, queue={queue_key}, barrier={barrier_key}, count={count}" for status, queue_key, barrier_key, count in rows
        )

    def _transition(
        self,
        *,
        work_item_id: str,
        now: datetime,
        status: TokenWorkStatus,
        expected_statuses: tuple[TokenWorkStatus, ...] = (TokenWorkStatus.LEASED,),
        expected_lease_owner: str | None = None,
        **values: object,
    ) -> TokenWorkItem:
        update_values = {"status": status.value, "updated_at": now, **values}
        expected_status_values = tuple(candidate.value for candidate in expected_statuses)
        expected_status_text = " or ".join(candidate.name for candidate in expected_statuses)
        predicates = [
            token_work_items_table.c.work_item_id == work_item_id,
            token_work_items_table.c.status.in_(expected_status_values),
        ]
        if expected_lease_owner is not None:
            predicates.append(token_work_items_table.c.lease_owner == expected_lease_owner)
        with self._engine.begin() as conn:
            result = conn.execute(update(token_work_items_table).where(and_(*predicates)).values(**update_values))
            if result.rowcount != 1:
                actual = (
                    conn.execute(
                        select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                            token_work_items_table.c.work_item_id == work_item_id
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if actual is None:
                    actual_message = "missing"
                else:
                    actual_message = f"actual status {actual['status']}, actual lease_owner {actual['lease_owner']!r}"
                expected_owner_message = "" if expected_lease_owner is None else f" and expected lease_owner {expected_lease_owner!r}"
                raise AuditIntegrityError(
                    f"Scheduler transition to {status.name!r} for work_item_id={work_item_id!r} "
                    f"affected {result.rowcount} rows; expected exactly 1 row with expected status {expected_status_text}"
                    f"{expected_owner_message}. Caller assumed ownership but the row is missing or in an unexpected "
                    f"state ({actual_message})."
                )
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        return self._item_from_mapping(row)

    @staticmethod
    def _work_item_id(run_id: str, token_id: str, node_id: str | None, attempt: int) -> str:
        node_key = "<terminal>" if node_id is None else node_id
        raw = f"{run_id}:{token_id}:{node_key}:{attempt}".encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _item_from_mapping(row: RowMapping) -> TokenWorkItem:
        data = dict(row)
        for key in ("available_at", "created_at", "updated_at", "lease_expires_at"):
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
        )
