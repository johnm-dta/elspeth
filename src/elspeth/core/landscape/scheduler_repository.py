"""Durable token scheduler repository.

The v1 scheduler is intentionally embedded-database backed and single-process
friendly. It records enough state for a run to resume without rediscovering
source rows: ready work, active leases, delayed retry availability, queue or
barrier blocking keys, and terminal/failed states.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from sqlalchemy import ColumnElement, and_, func, or_, select, update
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from elspeth.contracts import TokenInfo
from elspeth.contracts.errors import AuditIntegrityError, SchedulerLeaseLostError
from elspeth.contracts.scheduler import (
    BarrierEmission,
    BlockedPendingSinkHandoff,
    SchedulerEventType,
    TokenWorkItem,
    TokenWorkStatus,
)
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.database import Tier1Engine
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import (
    blocked_barrier_hold_clause,
    nodes_table,
    rows_table,
    scheduler_events_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)


def token_from_journal_item(
    item: TokenWorkItem,
    *,
    attempt_offset: int,
    resume_checkpoint_id: str,
) -> TokenInfo:
    """Rebuild a ``TokenInfo`` from a journal BLOCKED row (F1 resume path).

    Shared by the aggregation and coalesce executors' ``restore_from_journal``:
    the journal row is authoritative for the payload and token lineage, while
    the resume provenance (``attempt_offset`` = audit-derived max_attempt + 1,
    ``resume_checkpoint_id``) is supplied by the restoring caller.

    Lives next to ``serialize_row_payload`` / ``deserialize_row_payload``
    because the payload round-trip is the heart of the mapping.
    """
    row_data = TokenSchedulerRepository.deserialize_row_payload(item.row_payload_json)
    return TokenInfo(
        row_id=item.row_id,
        token_id=item.token_id,
        row_data=row_data,
        branch_name=item.branch_name,
        fork_group_id=item.fork_group_id,
        join_group_id=item.join_group_id,
        expand_group_id=item.expand_group_id,
        resume_attempt_offset=attempt_offset,
        resume_checkpoint_id=resume_checkpoint_id,
    )


class TokenSchedulerRepository:
    """Persistence boundary for token scheduler work items."""

    _SCRUBBED_ROW_PAYLOAD_JSON = canonical_json({"row_payload": "purged", "payload_hash": None})
    _TRANSITION_EVENT_TYPES: ClassVar[dict[TokenWorkStatus, SchedulerEventType]] = {
        TokenWorkStatus.BLOCKED: SchedulerEventType.MARK_BLOCKED,
        TokenWorkStatus.TERMINAL: SchedulerEventType.MARK_TERMINAL,
        TokenWorkStatus.FAILED: SchedulerEventType.MARK_FAILED,
        TokenWorkStatus.PENDING_SINK: SchedulerEventType.MARK_PENDING_SINK,
    }

    def __init__(self, engine: Tier1Engine) -> None:
        # Runtime PRAGMA probe — defence in depth against a caller that slips
        # past the type checker (e.g. a ``cast()`` in test code or a mypy
        # suppression).  Tier-1 doctrine: the scheduler touches the audit DB;
        # we must refuse to proceed if the engine's SQLite guarantees are unmet.
        #
        # The probe mirrors :meth:`LandscapeDB._verify_sqlite_pragmas`.  We
        # check only ``foreign_keys`` and ``journal_mode`` here — they are the
        # invariants most likely to be missing on a bare ``create_engine()``
        # call that bypassed ``LandscapeDB._configure_sqlite``.  If either is
        # wrong the scheduler would silently operate without referential
        # integrity or without crash-safe journalling, which is unacceptable
        # for the audit record.
        with engine.connect() as conn:
            fk_result = conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one_or_none()
            jm_result = conn.exec_driver_sql("PRAGMA journal_mode").scalar_one_or_none()

        foreign_keys = "" if fk_result is None else str(fk_result).lower()
        journal_mode = "" if jm_result is None else str(jm_result).lower()

        violations: list[str] = []
        if foreign_keys != "1":
            violations.append(f"PRAGMA foreign_keys: expected '1', observed {foreign_keys!r}")
        if journal_mode not in ("wal", "memory"):
            violations.append(f"PRAGMA journal_mode: expected 'wal' (or 'memory' for :memory: DBs), observed {journal_mode!r}")

        if violations:
            raise AuditIntegrityError(
                "TokenSchedulerRepository received an engine that does not meet Tier-1 audit-integrity "
                "requirements; the engine was not opened through LandscapeDB. " + "; ".join(violations)
            )

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
        values = self._ready_work_item_values(
            run_id=run_id,
            token_id=token_id,
            row_id=row_id,
            node_id=node_id,
            step_index=step_index,
            ingest_sequence=ingest_sequence,
            row_payload_json=row_payload_json,
            available_at=available_at,
            attempt=attempt,
            queue_key=queue_key,
            barrier_key=barrier_key,
            on_success_sink=on_success_sink,
            branch_name=branch_name,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )
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
            inserted = self._insert_work_item_idempotent(conn, values=values, operation="enqueue READY scheduler work")
            if inserted:
                self._record_scheduler_event(
                    conn,
                    event_type=SchedulerEventType.ENQUEUE,
                    run_id=run_id,
                    token_id=token_id,
                    work_item_id=work_item_id,
                    node_id=node_id,
                    from_status=None,
                    to_status=TokenWorkStatus.READY,
                    from_lease_owner=None,
                    to_lease_owner=None,
                    from_attempt=None,
                    to_attempt=attempt,
                    recorded_at=available_at,
                )
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        return self._item_from_mapping(row)

    def enqueue_ready_claimed(
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
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
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
        """Persist a READY continuation and claim it in the same audit transaction."""
        work_item_id = self._work_item_id(run_id, token_id, node_id, attempt)
        values = self._ready_work_item_values(
            run_id=run_id,
            token_id=token_id,
            row_id=row_id,
            node_id=node_id,
            step_index=step_index,
            ingest_sequence=ingest_sequence,
            row_payload_json=row_payload_json,
            available_at=available_at,
            attempt=attempt,
            queue_key=queue_key,
            barrier_key=barrier_key,
            on_success_sink=on_success_sink,
            branch_name=branch_name,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )
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
            inserted = self._insert_work_item_idempotent(conn, values=values, operation="enqueue and claim READY scheduler work")
            if inserted:
                self._record_scheduler_event(
                    conn,
                    event_type=SchedulerEventType.ENQUEUE,
                    run_id=run_id,
                    token_id=token_id,
                    work_item_id=work_item_id,
                    node_id=node_id,
                    from_status=None,
                    to_status=TokenWorkStatus.READY,
                    from_lease_owner=None,
                    to_lease_owner=None,
                    from_attempt=None,
                    to_attempt=attempt,
                    recorded_at=available_at,
                )
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
            if row["status"] == TokenWorkStatus.READY.value:
                claimed = self._claim_ready_row(
                    conn,
                    row=row,
                    run_id=run_id,
                    lease_owner=lease_owner,
                    lease_seconds=lease_seconds,
                    now=now,
                )
                if claimed is not None:
                    row = claimed
        return self._item_from_mapping(row)

    def _ready_work_item_values(
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
            "work_item_id": self._work_item_id(run_id, token_id, node_id, attempt),
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

    def _record_scheduler_event(
        self,
        conn: Connection,
        *,
        event_type: SchedulerEventType,
        run_id: str,
        token_id: str,
        work_item_id: str,
        node_id: str | None,
        from_status: TokenWorkStatus | None,
        to_status: TokenWorkStatus,
        from_lease_owner: str | None,
        to_lease_owner: str | None,
        from_attempt: int | None,
        to_attempt: int,
        recorded_at: datetime,
        from_lease_expires_at: datetime | None = None,
        to_lease_expires_at: datetime | None = None,
        caller_owner: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        context_json = canonical_json({} if context is None else dict(context))
        event_identity = canonical_json(
            {
                "caller_owner": caller_owner,
                "context_json": context_json,
                "event_type": event_type.value,
                "from_attempt": from_attempt,
                "from_lease_expires_at": None if from_lease_expires_at is None else from_lease_expires_at.isoformat(),
                "from_lease_owner": from_lease_owner,
                "from_status": None if from_status is None else from_status.value,
                "node_id": node_id,
                "recorded_at": recorded_at.isoformat(),
                "run_id": run_id,
                "to_attempt": to_attempt,
                "to_lease_expires_at": None if to_lease_expires_at is None else to_lease_expires_at.isoformat(),
                "to_lease_owner": to_lease_owner,
                "to_status": to_status.value,
                "token_id": token_id,
                "work_item_id": work_item_id,
            }
        )
        event_id = hashlib.sha256(event_identity.encode()).hexdigest()
        values = {
            "event_id": event_id,
            "run_id": run_id,
            "token_id": token_id,
            "work_item_id": work_item_id,
            "node_id": node_id,
            "event_type": event_type.value,
            "from_status": None if from_status is None else from_status.value,
            "to_status": to_status.value,
            "from_lease_owner": from_lease_owner,
            "to_lease_owner": to_lease_owner,
            "from_lease_expires_at": from_lease_expires_at,
            "to_lease_expires_at": to_lease_expires_at,
            "from_attempt": from_attempt,
            "to_attempt": to_attempt,
            "recorded_at": recorded_at,
            "caller_owner": caller_owner,
            "context_json": context_json,
        }
        try:
            result = conn.execute(scheduler_events_table.insert().values(**values))
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"Scheduler event {event_type.value!r} failed for run_id={run_id!r} "
                f"work_item_id={work_item_id!r} — database rejected audit write: {type(exc).__name__}"
            ) from exc
        if result.rowcount != 1:
            raise LandscapeRecordError(
                f"Scheduler event {event_type.value!r} affected {result.rowcount} rows for "
                f"run_id={run_id!r} work_item_id={work_item_id!r}; expected exactly one audit row."
            )

    @staticmethod
    def _insert_work_item_idempotent(conn: Connection, *, values: dict[str, object], operation: str) -> bool:
        """Insert a work item or accept an exact existing continuation.

        Child continuations are persisted before their parent claim is marked
        complete. A crash in that window can replay the parent and attempt to
        enqueue the same child again. The deterministic work_item_id lets us
        reconcile that replay, but only when the existing durable row carries
        the same resume cursor and token lineage.
        """
        try:
            TokenSchedulerRepository._insert_work_item(conn, values=values, operation=operation)
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
                f"Scheduler {operation} failed for {TokenSchedulerRepository._work_item_identity(values)} — "
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
                f"Scheduler {operation} found incompatible existing work item for "
                f"{TokenSchedulerRepository._work_item_identity(values)}: {mismatches!r}"
            )
        return False

    @staticmethod
    def _work_item_identity(values: dict[str, object]) -> str:
        return (
            f"run_id={values['run_id']!r} token_id={values['token_id']!r} row_id={values['row_id']!r} "
            f"node_id={values['node_id']!r} attempt={values['attempt']!r}"
        )

    @staticmethod
    def serialize_row_payload(row: PipelineRow) -> str:
        """Serialize the current token row and its contract for durable resume.

        Uses the type-preserving checkpoint serializer (NOT canonical_json):
        journal row payloads are re-driven as live PipelineRows on resume
        (PENDING_SINK re-drive, F1 barrier-buffer rebuild), so typed values
        (datetime, Decimal, date, time, bytes, UUID) must round-trip with
        full fidelity — canonical_json flattens them to bare strings.
        """
        # Deferred import: module-level would cycle —
        # core.checkpoint.__init__ → recovery → core.landscape.factory → this module.
        from elspeth.core.checkpoint.serialization import checkpoint_dumps

        return checkpoint_dumps(
            {
                "row": row.to_checkpoint_format(),
                "contract": row.contract.to_checkpoint_format(),
            }
        )

    @staticmethod
    def deserialize_row_payload(row_payload_json: str) -> PipelineRow:
        """Restore a scheduler row payload written by ``serialize_row_payload``."""
        # Deferred import: see serialize_row_payload.
        from elspeth.core.checkpoint.serialization import checkpoint_loads

        try:
            payload = checkpoint_loads(row_payload_json)
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
                        # Stable last-resort tiebreaker for cross-source same-tick
                        # collisions where ingest_sequence/step_index/created_at
                        # are not jointly disambiguating (filigree elspeth-6cb89db535,
                        # G3 determinism-reviewer M1).
                        token_work_items_table.c.work_item_id,
                    )
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            claimed = self._claim_ready_row(
                conn,
                row=row,
                run_id=run_id,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
                now=now,
            )
            if claimed is None:
                return None
        return self._item_from_mapping(claimed)

    def _claim_ready_row(
        self,
        conn: Connection,
        *,
        row: RowMapping,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> RowMapping | None:
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        result = conn.execute(
            update(token_work_items_table)
            .where(
                and_(
                    token_work_items_table.c.work_item_id == row["work_item_id"],
                    token_work_items_table.c.run_id == run_id,
                    token_work_items_table.c.status == TokenWorkStatus.READY.value,
                    token_work_items_table.c.available_at <= now,
                )
            )
            .values(
                status=TokenWorkStatus.LEASED.value,
                lease_owner=lease_owner,
                lease_expires_at=lease_expires_at,
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            return None
        if result.rowcount != 1:
            raise AuditIntegrityError(
                f"Scheduler claim_ready lost race on run_id={run_id!r} "
                f"work_item_id={row['work_item_id']!r}: SELECT saw READY but UPDATE matched "
                f"{result.rowcount} rows for lease_owner={lease_owner!r}. Concurrent claim by another worker."
            )
        self._record_scheduler_event(
            conn,
            event_type=SchedulerEventType.CLAIM_READY,
            run_id=run_id,
            token_id=row["token_id"],
            work_item_id=row["work_item_id"],
            node_id=row["node_id"],
            from_status=TokenWorkStatus.READY,
            to_status=TokenWorkStatus.LEASED,
            from_lease_owner=row["lease_owner"],
            to_lease_owner=lease_owner,
            from_attempt=row["attempt"],
            to_attempt=row["attempt"],
            recorded_at=now,
            from_lease_expires_at=row["lease_expires_at"],
            to_lease_expires_at=lease_expires_at,
            caller_owner=lease_owner,
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
        return claimed

    def claim_pending_sink(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> TokenWorkItem | None:
        """Claim a sink-bound token whose transform work is already durable."""
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._engine.begin() as conn:
            row = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.PENDING_SINK.value)
                    .order_by(
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.created_at,
                        # Stable last-resort tiebreaker for cross-source same-tick
                        # collisions (filigree elspeth-6cb89db535, G3 M1).
                        token_work_items_table.c.work_item_id,
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
                        token_work_items_table.c.status == TokenWorkStatus.PENDING_SINK.value,
                    )
                )
                .values(
                    status=TokenWorkStatus.LEASED.value,
                    lease_owner=lease_owner,
                    lease_expires_at=lease_expires_at,
                    updated_at=now,
                )
            )
            if result.rowcount == 0:
                return None
            if result.rowcount != 1:
                raise AuditIntegrityError(
                    f"Scheduler claim_pending_sink lost race on run_id={run_id!r} work_item_id={row['work_item_id']!r}: "
                    f"UPDATE matched {result.rowcount} rows for lease_owner={lease_owner!r}."
                )
            self._record_scheduler_event(
                conn,
                event_type=SchedulerEventType.CLAIM_PENDING_SINK,
                run_id=run_id,
                token_id=row["token_id"],
                work_item_id=row["work_item_id"],
                node_id=row["node_id"],
                from_status=TokenWorkStatus.PENDING_SINK,
                to_status=TokenWorkStatus.LEASED,
                from_lease_owner=row["lease_owner"],
                to_lease_owner=lease_owner,
                from_attempt=row["attempt"],
                to_attempt=row["attempt"],
                recorded_at=now,
                from_lease_expires_at=row["lease_expires_at"],
                to_lease_expires_at=lease_expires_at,
                caller_owner=lease_owner,
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
                    f"Scheduler pending-sink claim invariant violated for work_item_id={row['work_item_id']!r}: "
                    "leased row was not owned by the claimant"
                )
        return self._item_from_mapping(claimed)

    def recover_expired_leases(self, *, run_id: str, now: datetime, caller_owner: str) -> int:
        """Return expired LEASED work to READY for retry by another worker.

        Recovery advances the durable scheduler attempt so any node state
        created before the crashed worker lost its lease is not replayed under
        the same ``(token_id, node_id, attempt)`` audit identity.

        ``caller_owner`` is the lease_owner of the caller making the recovery
        sweep (every RowProcessor instance owns a unique
        ``row-processor:<run_id>:<uuid>`` identity). Leases owned by this caller
        are skipped: a worker must never reap its own still-running lease.
        Any token processing step that exceeds the lease window (LLM/HTTP
        pipelines exceed the default with regularity) would otherwise have its
        in-flight lease silently transitioned back to READY by the next
        iteration of the same worker's drain loop, leaving the worker holding
        a stale ``expected_lease_owner`` that fails CAS on the subsequent
        ``mark_terminal`` / ``mark_pending_sink`` / ``mark_blocked`` write and
        kills the run. The function's job is to reap leases held by *other*
        workers (a previous crashed RowProcessor with a different uuid; in
        future a peer worker), not the caller's own work. See
        filigree elspeth-941f1508f5.
        """
        # Predicate symmetric across the SELECT and UPDATE to close two
        # multi-worker race classes (filigree elspeth-28aaa36a62, G1 P2):
        #
        # 1. PENDING_SINK ABA window. ``next_work_item_id`` for the
        #    PENDING_SINK-recovery branch is the row's existing
        #    ``work_item_id`` (see below — ``pending_sink_name is not None``
        #    keeps the work-item identity stable). Between this SELECT and
        #    the per-row UPDATE, a peer reaper may have already returned the
        #    row to PENDING_SINK and a peer claimant may have leased it
        #    again with a fresh ``lease_expires_at``. Without the
        #    ``lease_expires_at < now`` predicate on the UPDATE, the caller
        #    would clobber the peer's live lease. The READY-recovery branch
        #    is incidentally protected by ``work_item_id`` rotation at the
        #    next-attempt bump, but symmetry costs nothing and survives the
        #    G27 connection-fan-out transition.
        #
        # 2. ``lease_owner`` NULL-blindness. SQLite three-valued logic
        #    makes ``NULL != caller_owner`` evaluate to NULL (not TRUE),
        #    so a row wedged with ``status=LEASED`` and ``lease_owner=NULL``
        #    (a Tier-1 invariant violation we want to recover from, not
        #    leak) would be invisible to the sweep. ``(lease_owner IS NULL
        #    OR lease_owner != caller_owner)`` recovers those rows while
        #    still excluding the caller's own active leases.
        lease_owner_not_caller = or_(
            token_work_items_table.c.lease_owner.is_(None),
            token_work_items_table.c.lease_owner != caller_owner,
        )
        with self._engine.begin() as conn:
            expired = conn.execute(
                select(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
                .where(token_work_items_table.c.lease_expires_at < now)
                .where(lease_owner_not_caller)
                .order_by(
                    token_work_items_table.c.ingest_sequence,
                    token_work_items_table.c.step_index,
                    # Stable last-resort tiebreaker for cross-source same-tick
                    # collisions (filigree elspeth-6cb89db535, G3 M1).
                    token_work_items_table.c.work_item_id,
                )
            ).mappings()
            recovered = 0
            for row in expired:
                pending_sink_name = row["pending_sink_name"]
                next_attempt = row["attempt"] if pending_sink_name is not None else row["attempt"] + 1
                next_work_item_id = (
                    row["work_item_id"]
                    if pending_sink_name is not None
                    else self._work_item_id(run_id, row["token_id"], row["node_id"], next_attempt)
                )
                recovered_status = TokenWorkStatus.PENDING_SINK.value if pending_sink_name is not None else TokenWorkStatus.READY.value
                result = conn.execute(
                    update(token_work_items_table)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
                    .where(token_work_items_table.c.lease_expires_at < now)
                    .where(
                        or_(
                            token_work_items_table.c.lease_owner.is_(None),
                            token_work_items_table.c.lease_owner != caller_owner,
                        )
                    )
                    .values(
                        work_item_id=next_work_item_id,
                        attempt=next_attempt,
                        status=recovered_status,
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._record_scheduler_event(
                        conn,
                        event_type=SchedulerEventType.RECOVER_EXPIRED_LEASE,
                        run_id=run_id,
                        token_id=row["token_id"],
                        work_item_id=next_work_item_id,
                        node_id=row["node_id"],
                        from_status=TokenWorkStatus.LEASED,
                        to_status=TokenWorkStatus(recovered_status),
                        from_lease_owner=row["lease_owner"],
                        to_lease_owner=None,
                        from_attempt=row["attempt"],
                        to_attempt=next_attempt,
                        recorded_at=now,
                        from_lease_expires_at=row["lease_expires_at"],
                        to_lease_expires_at=None,
                        caller_owner=caller_owner,
                        context=({"previous_work_item_id": row["work_item_id"]} if next_work_item_id != row["work_item_id"] else None),
                    )
                recovered += result.rowcount
        return recovered

    @staticmethod
    def _recovery_event_for_previous_work_item(
        conn: Connection,
        *,
        run_id: str,
        previous_work_item_id: str,
    ) -> RowMapping | None:
        recovery_events = (
            conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == run_id)
                .where(scheduler_events_table.c.event_type == SchedulerEventType.RECOVER_EXPIRED_LEASE.value)
                .order_by(
                    scheduler_events_table.c.recorded_at.desc(),
                    scheduler_events_table.c.event_id.desc(),
                )
            )
            .mappings()
            .all()
        )
        for event in recovery_events:
            try:
                context = json.loads(event["context_json"])
            except json.JSONDecodeError as exc:
                raise AuditIntegrityError(
                    f"Corrupt scheduler recovery event context_json for event_id={event['event_id']!r}: {exc}"
                ) from exc
            if type(context) is not dict:
                raise AuditIntegrityError(
                    f"Corrupt scheduler recovery event context_json for event_id={event['event_id']!r}: "
                    f"expected object, got {type(context).__name__}"
                )
            if "previous_work_item_id" not in context:
                continue
            previous = context["previous_work_item_id"]
            if type(previous) is not str:
                raise AuditIntegrityError(
                    f"Corrupt scheduler recovery event context_json for event_id={event['event_id']!r}: "
                    f"previous_work_item_id must be str, got {type(previous).__name__}"
                )
            if previous == previous_work_item_id:
                return event
        return None

    def heartbeat_lease(
        self,
        *,
        run_id: str,
        work_item_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> datetime:
        """Extend a held lease's ``lease_expires_at`` to ``now + lease_seconds``.

        Single-timestamp heartbeat for ADR-026 RC6 multi-worker (filigree
        elspeth-ddde8144b6). A worker mid-processing calls this periodically
        from inside the slow work loop so a peer's ``recover_expired_leases``
        sweep does NOT reap an alive-but-slow worker. ``peer_active_leases``
        already filters on ``lease_expires_at > now``, so extending that
        timestamp is sufficient — no second clock and no inconsistency window
        between heartbeat-fresh-but-lease-expired vs reaper semantics.

        CAS contract (Tier-1 strictness on the write boundary):

        - The UPDATE matches on ``(work_item_id, run_id, status=LEASED,
          lease_owner)``. If rowcount != 1, the lease has been reaped or
          reassigned by a peer reaper (``recover_expired_leases`` rewrites the
          ``work_item_id`` for bumped attempts), and this worker no longer owns
          the row. Raise ``SchedulerLeaseLostError`` so the caller can abandon
          its in-flight work cleanly — issuing a follow-up ``mark_*`` would
          CAS-fail and cascade into a Tier-1 ``AuditIntegrityError``, which is
          the exact failure mode this primitive exists to eliminate.

        Returns the new ``lease_expires_at`` so the caller can update its
        local last-heartbeat-attempt clock without re-reading the row.
        """
        new_expires_at = now + timedelta(seconds=lease_seconds)
        lease_lost = False
        with self._engine.connect() as conn:
            transaction = conn.begin()
            try:
                result = conn.execute(
                    update(token_work_items_table)
                    .where(
                        and_(
                            token_work_items_table.c.work_item_id == work_item_id,
                            token_work_items_table.c.run_id == run_id,
                            token_work_items_table.c.status == TokenWorkStatus.LEASED.value,
                            token_work_items_table.c.lease_owner == lease_owner,
                        )
                    )
                    .values(
                        lease_expires_at=new_expires_at,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    current = (
                        conn.execute(
                            select(token_work_items_table)
                            .where(token_work_items_table.c.run_id == run_id)
                            .where(token_work_items_table.c.work_item_id == work_item_id)
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if current is not None:
                        self._record_scheduler_event(
                            conn,
                            event_type=SchedulerEventType.LEASE_LOST,
                            run_id=run_id,
                            token_id=current["token_id"],
                            work_item_id=work_item_id,
                            node_id=current["node_id"],
                            from_status=TokenWorkStatus.LEASED,
                            to_status=TokenWorkStatus(current["status"]),
                            from_lease_owner=lease_owner,
                            to_lease_owner=current["lease_owner"],
                            from_attempt=current["attempt"],
                            to_attempt=current["attempt"],
                            recorded_at=now,
                            from_lease_expires_at=None,
                            to_lease_expires_at=current["lease_expires_at"],
                            caller_owner=lease_owner,
                            context={"reason": "heartbeat_cas_miss"},
                        )
                    else:
                        recovery_event = self._recovery_event_for_previous_work_item(
                            conn,
                            run_id=run_id,
                            previous_work_item_id=work_item_id,
                        )
                        if recovery_event is not None:
                            self._record_scheduler_event(
                                conn,
                                event_type=SchedulerEventType.LEASE_LOST,
                                run_id=run_id,
                                token_id=recovery_event["token_id"],
                                work_item_id=work_item_id,
                                node_id=recovery_event["node_id"],
                                from_status=TokenWorkStatus.LEASED,
                                to_status=TokenWorkStatus(recovery_event["to_status"]),
                                from_lease_owner=lease_owner,
                                to_lease_owner=recovery_event["to_lease_owner"],
                                from_attempt=recovery_event["from_attempt"],
                                to_attempt=recovery_event["to_attempt"],
                                recorded_at=now,
                                from_lease_expires_at=recovery_event["from_lease_expires_at"],
                                to_lease_expires_at=recovery_event["to_lease_expires_at"],
                                caller_owner=lease_owner,
                                context={
                                    "reason": "heartbeat_cas_miss_after_recovery",
                                    "recovered_work_item_id": recovery_event["work_item_id"],
                                    "recovery_event_id": recovery_event["event_id"],
                                },
                            )
                    transaction.commit()
                    lease_lost = True
                else:
                    transaction.commit()
            except Exception:
                if transaction.is_active:
                    transaction.rollback()
                raise
        if lease_lost:
            raise SchedulerLeaseLostError(
                work_item_id=work_item_id,
                lease_owner=lease_owner,
                run_id=run_id,
            )
        return new_expires_at

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
            # F1: barrier holds are restored from the journal using this
            # absolute timestamp. Queue-holds (ADR-028) get stamped too —
            # harmless; nothing reads the column on that arm, and a single
            # UPDATE shape keeps this verb the column's only writer.
            barrier_blocked_at=now,
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
            row_payload_json=self._scrubbed_row_payload_json(work_item_id),
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_failed(self, *, work_item_id: str, now: datetime, expected_lease_owner: str) -> TokenWorkItem:
        """Mark a leased work item failed after retries are exhausted."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.FAILED,
            expected_lease_owner=expected_lease_owner,
            row_payload_json=self._scrubbed_row_payload_json(work_item_id),
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_pending_sink(
        self,
        *,
        work_item_id: str,
        row_payload_json: str,
        sink_name: str,
        outcome: str,
        path: str,
        error_hash: str | None,
        error_message: str | None,
        now: datetime,
        expected_lease_owner: str,
    ) -> TokenWorkItem:
        """Move a claimed item to a durable sink handoff state."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.PENDING_SINK,
            expected_lease_owner=expected_lease_owner,
            row_payload_json=row_payload_json,
            pending_sink_name=sink_name,
            pending_outcome=outcome,
            pending_path=path,
            pending_error_hash=error_hash,
            pending_error_message=error_message,
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_pending_sink_terminal(
        self,
        *,
        run_id: str,
        token_id: str,
        now: datetime,
        expected_lease_owner: str | None = None,
    ) -> int:
        """Terminalize pending sink scheduler work after token outcome durability."""
        predicates = [
            token_work_items_table.c.run_id == run_id,
            token_work_items_table.c.token_id == token_id,
            token_work_items_table.c.status.in_((TokenWorkStatus.PENDING_SINK.value, TokenWorkStatus.LEASED.value)),
            token_work_items_table.c.pending_sink_name.is_not(None),
        ]
        if expected_lease_owner is not None:
            predicates.append(
                (token_work_items_table.c.lease_owner.is_(None)) | (token_work_items_table.c.lease_owner == expected_lease_owner)
            )
        with self._engine.begin() as conn:
            rows = (
                conn.execute(select(token_work_items_table).where(and_(*predicates)).order_by(token_work_items_table.c.work_item_id))
                .mappings()
                .all()
            )
            terminalized = 0
            for row in rows:
                result = conn.execute(
                    update(token_work_items_table)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(and_(*predicates))
                    .values(
                        status=TokenWorkStatus.TERMINAL.value,
                        row_payload_json=self._scrubbed_row_payload_json(token_id),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._record_scheduler_event(
                        conn,
                        event_type=SchedulerEventType.MARK_PENDING_SINK_TERMINAL,
                        run_id=run_id,
                        token_id=row["token_id"],
                        work_item_id=row["work_item_id"],
                        node_id=row["node_id"],
                        from_status=TokenWorkStatus(row["status"]),
                        to_status=TokenWorkStatus.TERMINAL,
                        from_lease_owner=row["lease_owner"],
                        to_lease_owner=None,
                        from_attempt=row["attempt"],
                        to_attempt=row["attempt"],
                        recorded_at=now,
                        from_lease_expires_at=row["lease_expires_at"],
                        to_lease_expires_at=None,
                        caller_owner=expected_lease_owner,
                    )
                    terminalized += 1
                elif result.rowcount not in (0, None):
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink terminalization affected {result.rowcount} rows for "
                        f"run_id={run_id!r} token_id={token_id!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                    )
        return terminalized

    def complete_barrier(
        self,
        *,
        run_id: str,
        barrier_key: str,
        consumed_token_ids: Sequence[str],
        emitted_pending_sink: Sequence[BarrierEmission],
        emitted_ready: Sequence[BarrierEmission],
        now: datetime,
        leased_exclusion_token_id: str | None = None,
        require_exhaustive_release: bool = True,
        scope_row_id: str | None = None,
    ) -> int:
        """Complete a barrier atomically: consume BLOCKED inputs and emit outputs.

        ONE journal transaction (F1, elspeth-ae5183307b) performs:

        - validation of the consumed set against the durable BLOCKED set under
          ``(run_id, barrier_key)`` — both directions: every consumed/handed-off
          token must hold a BLOCKED row, and every BLOCKED row must be consumed
          or handed off (``leased_exclusion_token_id`` exempts the in-claim
          triggering token, matching the processor's ``leased_token_id``
          exclusions);

          ``scope_row_id`` narrows that BLOCKED universe — both the
          missing-token cross-check and the exhaustiveness check — to one
          pending group. COALESCE barriers share ONE ``barrier_key`` (the
          coalesce_name) across ALL row_ids pending at the coalesce, and
          fork/expand children inherit their parent's ``row_id``, so a firing
          group's identity is exactly ``(run_id, barrier_key, row_id)``.
          Coalesce call sites MUST pass the firing group's ``row_id`` —
          otherwise every other row's held branches would be spuriously
          reported as orphaned. Aggregation call sites omit it
          (``barrier_key`` = the node id; one batch consumes ALL BLOCKED rows
          of the node, so whole-key exhaustiveness is correct);
        - consumed rows -> TERMINAL with payload scrub and per-row
          ``MARK_BLOCKED_BARRIER_TERMINAL`` events (``{"barrier_key"}`` context,
          the existing event shape);
        - ``emitted_pending_sink``: buffered passthrough tokens (those with a
          BLOCKED row under this barrier) transition BLOCKED -> PENDING_SINK in
          place with the handoff bundle; others INSERT a fresh PENDING_SINK row
          on the node_id-NULL terminal lane (deterministic work_item_id);
        - ``emitted_ready``: INSERT READY continuations with ``ENQUEUE`` events;
        - every emission event's context carries
          ``{"barrier_key": ..., "consumed_count": N}`` so the atomic completion
          is reconstructable from scheduler_events alone.

        ``require_exhaustive_release=False`` is the legacy wrapper arm used by
        ``mark_blocked_barrier_terminal`` / ``mark_blocked_barrier_pending_sink_many``
        (their partial-release semantics and ``{"barrier_key"}``-only emission
        context are pinned by existing tests and the lifecycle state machine):
        BLOCKED rows not named stay BLOCKED, every pending-sink emission must be
        a passthrough (fresh inserts are refused), and emission events keep the
        legacy context. New callers must not pass it; the processor's flush
        sites were migrated onto the strict default (Task 3.4, landed).

        Returns the number of consumed rows terminalized.
        """
        if scope_row_id is not None and not require_exhaustive_release:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                f"scope_row_id={scope_row_id!r} with require_exhaustive_release=False; row scoping narrows the "
                "exhaustiveness universe and is meaningless on the legacy partial-release arm."
            )
        if leased_exclusion_token_id is not None and not require_exhaustive_release:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                f"leased_exclusion_token_id={leased_exclusion_token_id!r} with require_exhaustive_release=False; "
                "the exclusion only exempts a token from the exhaustiveness check and is meaningless on the "
                "legacy partial-release arm."
            )
        consumed = tuple(consumed_token_ids)
        consumed_set = frozenset(consumed)
        if len(consumed_set) != len(consumed):
            duplicates = sorted(token_id for token_id in consumed_set if consumed.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization received duplicate live token_ids for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        pending_token_ids = tuple(emission.token_id for emission in emitted_pending_sink)
        if len(frozenset(pending_token_ids)) != len(pending_token_ids):
            duplicates = sorted(token_id for token_id in frozenset(pending_token_ids) if pending_token_ids.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier completion received duplicate pending-sink emissions for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        ready_token_ids = tuple(emission.token_id for emission in emitted_ready)
        if len(frozenset(ready_token_ids)) != len(ready_token_ids):
            duplicates = sorted(token_id for token_id in frozenset(ready_token_ids) if ready_token_ids.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier completion received duplicate ready emissions for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        pending_overlap = consumed_set & frozenset(pending_token_ids)
        if pending_overlap:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                f"token_ids both consumed and emitted to pending-sink: {sorted(pending_overlap)!r}; a BLOCKED row "
                "cannot be terminalized and handed off in the same completion."
            )
        for emission in emitted_pending_sink:
            if emission.sink_name is None or emission.outcome is None or emission.path is None:
                raise AuditIntegrityError(
                    f"Scheduler barrier completion pending-sink emission for run_id={run_id!r} "
                    f"barrier_key={barrier_key!r} token_id={emission.token_id!r} requires sink_name, outcome and path."
                )
        if leased_exclusion_token_id is not None:
            emitted_token_ids = frozenset(pending_token_ids) | {emission.token_id for emission in emitted_ready}
            if leased_exclusion_token_id in consumed_set or leased_exclusion_token_id in emitted_token_ids:
                raise AuditIntegrityError(
                    f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                    f"leased_exclusion_token_id={leased_exclusion_token_id!r} that is itself consumed or emitted; "
                    "the in-claim triggering token must be excluded, not completed."
                )
        if scope_row_id is not None:
            for emission in (*emitted_pending_sink, *emitted_ready):
                if emission.row_id is not None and emission.row_id != scope_row_id:
                    raise AuditIntegrityError(
                        f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                        f"emission token_id={emission.token_id!r} with row_id={emission.row_id!r} outside the scoped "
                        f"pending group scope_row_id={scope_row_id!r}; a scoped completion must not emit into "
                        "another row's group."
                    )

        emission_context: dict[str, object] = {"barrier_key": barrier_key}
        if require_exhaustive_release:
            emission_context["consumed_count"] = len(consumed_set)

        # Narrowed column set: no arm reads the blocked rows' row_payload_json
        # (terminalization scrubs it, passthrough overwrites it with the
        # handoff payload), so fetching it would drag every held row's full
        # payload into memory on large barrier sets for nothing.
        blocked_select = (
            select(
                token_work_items_table.c.work_item_id,
                token_work_items_table.c.token_id,
                token_work_items_table.c.node_id,
                token_work_items_table.c.attempt,
                token_work_items_table.c.lease_owner,
                token_work_items_table.c.lease_expires_at,
            )
            .where(token_work_items_table.c.run_id == run_id)
            .where(token_work_items_table.c.barrier_key == barrier_key)
            .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
        )
        if scope_row_id is not None:
            blocked_select = blocked_select.where(token_work_items_table.c.row_id == scope_row_id)

        with self._engine.begin() as conn:
            blocked_rows = (
                conn.execute(
                    blocked_select.order_by(
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
            blocked_by_token: dict[str, list[RowMapping]] = {}
            for row in blocked_rows:
                blocked_by_token.setdefault(row["token_id"], []).append(row)
            durable_token_ids = frozenset(blocked_by_token)

            scope_note = "" if scope_row_id is None else f" scope_row_id={scope_row_id!r}."
            missing_token_ids = consumed_set - durable_token_ids
            if missing_token_ids:
                matching_count = len(consumed_set & durable_token_ids)
                raise AuditIntegrityError(
                    f"Scheduler barrier terminalization mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                    f"live consumed {len(consumed_set)} token(s), but durable BLOCKED rows contained "
                    f"{matching_count} matching token(s) out of {len(durable_token_ids)} blocked token(s). "
                    f"missing token_ids={sorted(missing_token_ids)!r}; durable token_ids={sorted(durable_token_ids)!r}."
                    f"{scope_note}"
                )

            passthrough_emissions: list[BarrierEmission] = []
            fresh_emissions: list[BarrierEmission] = []
            for emission in emitted_pending_sink:
                matching_rows = blocked_by_token.get(emission.token_id, [])
                if not matching_rows:
                    # Strict/legacy partition: on the strict arm (the F1 verb)
                    # an unbuffered sink-bound emission is a fresh terminal-lane
                    # INSERT; on the legacy wrapper arm it is a refused partial
                    # handoff (mark_blocked_barrier_pending_sink_many's pinned
                    # behavior).
                    if require_exhaustive_release:
                        fresh_emissions.append(emission)
                        continue
                    raise AuditIntegrityError(
                        f"Scheduler barrier pending-sink handoff for run_id={run_id!r} barrier_key={barrier_key!r} "
                        f"is missing token_id={emission.token_id!r}; refusing partial sink handoff."
                    )
                if len(matching_rows) != 1:
                    raise AuditIntegrityError(
                        f"Scheduler barrier pending-sink handoff for run_id={run_id!r} barrier_key={barrier_key!r} "
                        f"token_id={emission.token_id!r} found {len(matching_rows)} matching rows; expected exactly one."
                    )
                passthrough_emissions.append(emission)

            if require_exhaustive_release:
                handed_off_token_ids = frozenset(emission.token_id for emission in passthrough_emissions)
                required_token_ids = durable_token_ids - frozenset(
                    () if leased_exclusion_token_id is None else (leased_exclusion_token_id,)
                )
                uncovered_token_ids = required_token_ids - consumed_set - handed_off_token_ids
                if uncovered_token_ids:
                    raise AuditIntegrityError(
                        f"Scheduler barrier completion mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                        f"durable BLOCKED rows hold {len(uncovered_token_ids)} token(s) neither consumed nor handed "
                        f"off; the completion would orphan them. uncovered token_ids={sorted(uncovered_token_ids)!r}; "
                        f"consumed token_ids={sorted(consumed_set)!r}; handoff token_ids={sorted(handed_off_token_ids)!r}."
                        f"{scope_note}"
                    )

            terminalized = self._terminalize_consumed_barrier_rows(
                conn,
                run_id=run_id,
                barrier_key=barrier_key,
                consumed=consumed,
                blocked_by_token=blocked_by_token,
                now=now,
            )
            self._transition_passthrough_pending_sink(
                conn,
                run_id=run_id,
                barrier_key=barrier_key,
                blocked_rows=blocked_rows,
                passthrough_emissions=passthrough_emissions,
                emission_context=emission_context,
                now=now,
            )
            for emission in fresh_emissions:
                self._insert_fresh_pending_sink_emission(
                    conn,
                    run_id=run_id,
                    barrier_key=barrier_key,
                    emission=emission,
                    emission_context=emission_context,
                    now=now,
                )
            for emission in emitted_ready:
                self._insert_ready_emission(
                    conn,
                    run_id=run_id,
                    barrier_key=barrier_key,
                    emission=emission,
                    emission_context=emission_context,
                    now=now,
                )
        return terminalized

    def _terminalize_consumed_barrier_rows(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        consumed: tuple[str, ...],
        blocked_by_token: Mapping[str, list[RowMapping]],
        now: datetime,
    ) -> int:
        """BLOCKED -> TERMINAL for the consumed set (legacy terminalization arm)."""
        consumed_set = frozenset(consumed)
        candidate_rows = sorted(
            (row for token_id in consumed_set for row in blocked_by_token.get(token_id, ())),
            key=lambda row: (row["token_id"], row["work_item_id"]),
        )
        terminalized = 0
        for row in candidate_rows:
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.barrier_key == barrier_key)
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .where(token_work_items_table.c.token_id.in_(consumed))
                .values(
                    status=TokenWorkStatus.TERMINAL.value,
                    row_payload_json=self._scrubbed_row_payload_json(barrier_key),
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            if result.rowcount == 1:
                self._record_scheduler_event(
                    conn,
                    event_type=SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL,
                    run_id=run_id,
                    token_id=row["token_id"],
                    work_item_id=row["work_item_id"],
                    node_id=row["node_id"],
                    from_status=TokenWorkStatus.BLOCKED,
                    to_status=TokenWorkStatus.TERMINAL,
                    from_lease_owner=row["lease_owner"],
                    to_lease_owner=None,
                    from_attempt=row["attempt"],
                    to_attempt=row["attempt"],
                    recorded_at=now,
                    from_lease_expires_at=row["lease_expires_at"],
                    to_lease_expires_at=None,
                    context={"barrier_key": barrier_key},
                )
                terminalized += 1
            elif result.rowcount not in (0, None):
                raise AuditIntegrityError(
                    f"Scheduler barrier terminalization affected {result.rowcount} rows for "
                    f"run_id={run_id!r} barrier_key={barrier_key!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                )
        if consumed_set and terminalized != len(consumed_set):
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                f"live consumed {len(consumed_set)} token(s), but durable scheduler terminalized {terminalized}."
            )
        return terminalized

    def _transition_passthrough_pending_sink(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        blocked_rows: Sequence[RowMapping],
        passthrough_emissions: Sequence[BarrierEmission],
        emission_context: Mapping[str, object],
        now: datetime,
    ) -> None:
        """BLOCKED -> PENDING_SINK in place for buffered passthrough tokens (legacy handoff arm)."""
        emission_by_token = {emission.token_id: emission for emission in passthrough_emissions}
        if not emission_by_token:
            return
        transitioned = 0
        for row in blocked_rows:
            emission = emission_by_token.get(row["token_id"])
            if emission is None:
                continue
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.barrier_key == barrier_key)
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .where(token_work_items_table.c.token_id == row["token_id"])
                .values(
                    status=TokenWorkStatus.PENDING_SINK.value,
                    row_payload_json=emission.row_payload_json,
                    pending_sink_name=emission.sink_name,
                    pending_outcome=emission.outcome,
                    pending_path=emission.path,
                    pending_error_hash=emission.error_hash,
                    pending_error_message=emission.error_message,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            if result.rowcount == 1:
                self._record_scheduler_event(
                    conn,
                    event_type=SchedulerEventType.MARK_PENDING_SINK,
                    run_id=run_id,
                    token_id=row["token_id"],
                    work_item_id=row["work_item_id"],
                    node_id=row["node_id"],
                    from_status=TokenWorkStatus.BLOCKED,
                    to_status=TokenWorkStatus.PENDING_SINK,
                    from_lease_owner=row["lease_owner"],
                    to_lease_owner=None,
                    from_attempt=row["attempt"],
                    to_attempt=row["attempt"],
                    recorded_at=now,
                    from_lease_expires_at=row["lease_expires_at"],
                    to_lease_expires_at=None,
                    context=emission_context,
                )
                transitioned += 1
            elif result.rowcount not in (0, None):
                raise AuditIntegrityError(
                    f"Scheduler barrier pending-sink handoff affected {result.rowcount} rows for "
                    f"run_id={run_id!r} barrier_key={barrier_key!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                )
        if transitioned != len(emission_by_token):
            raise AuditIntegrityError(
                f"Scheduler barrier pending-sink handoff mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                f"requested {len(emission_by_token)} token(s), transitioned {transitioned}."
            )

    def _insert_fresh_pending_sink_emission(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        emission: BarrierEmission,
        emission_context: Mapping[str, object],
        now: datetime,
    ) -> None:
        """INSERT a fresh PENDING_SINK row on the node_id-NULL terminal lane."""
        if emission.node_id is not None:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                f"received fresh pending-sink emission token_id={emission.token_id!r} with "
                f"node_id={emission.node_id!r}; fresh sink-bound emissions live on the node_id-NULL terminal lane."
            )
        if emission.row_id is None or emission.step_index is None or emission.ingest_sequence is None:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                f"fresh pending-sink emission token_id={emission.token_id!r} requires row_id, step_index "
                "and ingest_sequence; the inserted journal row must be a complete resume cursor."
            )
        self._validate_work_item_references(
            conn,
            run_id=run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            ingest_sequence=emission.ingest_sequence,
            node_id=None,
            coalesce_node_id=emission.coalesce_node_id,
        )
        work_item_id = self._work_item_id(run_id, emission.token_id, None, emission.attempt)
        values: dict[str, object] = {
            "work_item_id": work_item_id,
            "run_id": run_id,
            "token_id": emission.token_id,
            "row_id": emission.row_id,
            "node_id": None,
            "step_index": emission.step_index,
            "ingest_sequence": emission.ingest_sequence,
            "row_payload_json": emission.row_payload_json,
            "status": TokenWorkStatus.PENDING_SINK.value,
            "queue_key": emission.queue_key,
            "barrier_key": emission.barrier_key,
            "on_success_sink": emission.on_success_sink,
            "pending_sink_name": emission.sink_name,
            "pending_outcome": emission.outcome,
            "pending_path": emission.path,
            "pending_error_hash": emission.error_hash,
            "pending_error_message": emission.error_message,
            "branch_name": emission.branch_name,
            "fork_group_id": emission.fork_group_id,
            "join_group_id": emission.join_group_id,
            "expand_group_id": emission.expand_group_id,
            "coalesce_node_id": emission.coalesce_node_id,
            "coalesce_name": emission.coalesce_name,
            "attempt": emission.attempt,
            "lease_owner": None,
            "lease_expires_at": None,
            "available_at": now,
            "created_at": now,
            "updated_at": now,
        }
        self._insert_work_item(conn, values=values, operation="barrier-completion PENDING_SINK emission")
        self._record_scheduler_event(
            conn,
            event_type=SchedulerEventType.MARK_PENDING_SINK,
            run_id=run_id,
            token_id=emission.token_id,
            work_item_id=work_item_id,
            node_id=None,
            from_status=None,
            to_status=TokenWorkStatus.PENDING_SINK,
            from_lease_owner=None,
            to_lease_owner=None,
            from_attempt=None,
            to_attempt=emission.attempt,
            recorded_at=now,
            context=emission_context,
        )

    def _insert_ready_emission(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        emission: BarrierEmission,
        emission_context: Mapping[str, object],
        now: datetime,
    ) -> None:
        """INSERT a READY continuation emitted by a barrier completion."""
        if emission.row_id is None or emission.step_index is None or emission.ingest_sequence is None:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                f"ready emission token_id={emission.token_id!r} requires row_id, step_index "
                "and ingest_sequence; the inserted journal row must be a complete resume cursor."
            )
        self._validate_work_item_references(
            conn,
            run_id=run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            ingest_sequence=emission.ingest_sequence,
            node_id=emission.node_id,
            coalesce_node_id=emission.coalesce_node_id,
        )
        values = self._ready_work_item_values(
            run_id=run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            node_id=emission.node_id,
            step_index=emission.step_index,
            ingest_sequence=emission.ingest_sequence,
            row_payload_json=emission.row_payload_json,
            available_at=now,
            attempt=emission.attempt,
            queue_key=emission.queue_key,
            barrier_key=emission.barrier_key,
            on_success_sink=emission.on_success_sink,
            branch_name=emission.branch_name,
            fork_group_id=emission.fork_group_id,
            join_group_id=emission.join_group_id,
            expand_group_id=emission.expand_group_id,
            coalesce_node_id=emission.coalesce_node_id,
            coalesce_name=emission.coalesce_name,
        )
        self._insert_work_item(conn, values=values, operation="barrier-completion READY emission")
        self._record_scheduler_event(
            conn,
            event_type=SchedulerEventType.ENQUEUE,
            run_id=run_id,
            token_id=emission.token_id,
            work_item_id=str(values["work_item_id"]),
            node_id=emission.node_id,
            from_status=None,
            to_status=TokenWorkStatus.READY,
            from_lease_owner=None,
            to_lease_owner=None,
            from_attempt=None,
            to_attempt=emission.attempt,
            recorded_at=now,
            context=emission_context,
        )

    def mark_blocked_barrier_pending_sink_many(
        self,
        *,
        run_id: str,
        barrier_key: str,
        handoffs: Mapping[str, BlockedPendingSinkHandoff],
        now: datetime,
    ) -> int:
        """Move BLOCKED barrier work to PENDING_SINK before external sink writes.

        Delegating wrapper over :meth:`complete_barrier` (F1 Task 2.3). The
        legacy arm (``require_exhaustive_release=False``) preserves the pinned
        semantics: BLOCKED rows not named in ``handoffs`` stay BLOCKED, a
        handoff token without a BLOCKED row is refused (no fresh inserts), and
        handoff events keep the ``{"barrier_key"}``-only context.
        """
        requested_token_ids = tuple(handoffs.keys())
        if not requested_token_ids:
            return 0
        emissions = tuple(
            BarrierEmission(
                token_id=token_id,
                row_payload_json=handoff.row_payload_json,
                sink_name=handoff.sink_name,
                outcome=handoff.outcome,
                path=handoff.path,
                error_hash=handoff.error_hash,
                error_message=handoff.error_message,
            )
            for token_id, handoff in handoffs.items()
        )
        self.complete_barrier(
            run_id=run_id,
            barrier_key=barrier_key,
            consumed_token_ids=(),
            emitted_pending_sink=emissions,
            emitted_ready=(),
            now=now,
            require_exhaustive_release=False,
        )
        # complete_barrier raised unless every requested handoff transitioned.
        return len(requested_token_ids)

    def mark_pending_sink_terminal_many(
        self,
        *,
        run_id: str,
        token_ids: tuple[str, ...],
        now: datetime,
        expected_lease_owner: str | None = None,
    ) -> int:
        """Terminalize sink-bound scheduler work for a durable sink batch.

        This preserves the audit contract of one scheduler event per terminalized
        work item while avoiding one transaction and one indexed SELECT per token
        after large sink writes.
        """
        requested_token_ids = token_ids
        if not requested_token_ids:
            return 0
        seen_token_ids: set[str] = set()
        for token_id in requested_token_ids:
            if token_id in seen_token_ids:
                raise AuditIntegrityError(
                    f"Scheduler pending-sink batch terminalization for run_id={run_id!r} received duplicate token_id={token_id!r}."
                )
            seen_token_ids.add(token_id)

        predicates = [
            token_work_items_table.c.run_id == run_id,
            token_work_items_table.c.token_id.in_(requested_token_ids),
            token_work_items_table.c.status.in_((TokenWorkStatus.PENDING_SINK.value, TokenWorkStatus.LEASED.value)),
            token_work_items_table.c.pending_sink_name.is_not(None),
        ]
        with self._engine.begin() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table)
                    .where(and_(*predicates))
                    .order_by(
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
            rows_by_token_id: dict[str, list[RowMapping]] = {}
            for row in rows:
                row_token_id = row["token_id"]
                if row_token_id not in rows_by_token_id:
                    rows_by_token_id[row_token_id] = []
                rows_by_token_id[row_token_id].append(row)
            for token_id in requested_token_ids:
                matching_rows = rows_by_token_id[token_id] if token_id in rows_by_token_id else []
                if not matching_rows:
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink batch terminalization for run_id={run_id!r} is missing token_id={token_id!r}; "
                        "refusing partial terminalization."
                    )
                if len(matching_rows) != 1:
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink batch terminalization for run_id={run_id!r} token_id={token_id!r} found "
                        f"{len(matching_rows)} matching rows; expected exactly one."
                    )
                row_lease_owner = matching_rows[0]["lease_owner"]
                if expected_lease_owner is not None and row_lease_owner not in (None, expected_lease_owner):
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink batch terminalization for run_id={run_id!r} token_id={token_id!r} found "
                        f"lease_owner={row_lease_owner!r}; expected lease_owner={expected_lease_owner!r} or NULL."
                    )

            terminalized = 0
            for row in rows:
                lease_owner_predicate = (
                    token_work_items_table.c.lease_owner.is_(None)
                    if row["lease_owner"] is None
                    else token_work_items_table.c.lease_owner == row["lease_owner"]
                )
                result = conn.execute(
                    update(token_work_items_table)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.token_id == row["token_id"])
                    .where(token_work_items_table.c.status == row["status"])
                    .where(token_work_items_table.c.pending_sink_name.is_not(None))
                    .where(lease_owner_predicate)
                    .values(
                        status=TokenWorkStatus.TERMINAL.value,
                        row_payload_json=self._scrubbed_row_payload_json(row["token_id"]),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._record_scheduler_event(
                        conn,
                        event_type=SchedulerEventType.MARK_PENDING_SINK_TERMINAL,
                        run_id=run_id,
                        token_id=row["token_id"],
                        work_item_id=row["work_item_id"],
                        node_id=row["node_id"],
                        from_status=TokenWorkStatus(row["status"]),
                        to_status=TokenWorkStatus.TERMINAL,
                        from_lease_owner=row["lease_owner"],
                        to_lease_owner=None,
                        from_attempt=row["attempt"],
                        to_attempt=row["attempt"],
                        recorded_at=now,
                        from_lease_expires_at=row["lease_expires_at"],
                        to_lease_expires_at=None,
                        caller_owner=expected_lease_owner,
                    )
                    terminalized += 1
                elif result.rowcount not in (0, None):
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink batch terminalization affected {result.rowcount} rows for "
                        f"run_id={run_id!r} token_id={row['token_id']!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                    )
        return terminalized

    def terminalize_pending_sinks_with_terminal_outcomes(self, *, run_id: str, now: datetime, caller_owner: str) -> int:
        """Repair PENDING_SINK work whose terminal token outcome is already durable.

        A crash can land after sink outcome durability but before the scheduler
        handoff row is marked terminal. Resume must not claim and re-emit those
        rows externally; the terminal token outcome is the authoritative witness.
        """
        terminal_outcome_exists = (
            select(token_outcomes_table.c.outcome_id)
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.token_id == token_work_items_table.c.token_id)
            .where(token_outcomes_table.c.completed == 1)
            .exists()
        )
        with self._engine.begin() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.PENDING_SINK.value)
                    .where(token_work_items_table.c.pending_sink_name.is_not(None))
                    .where(terminal_outcome_exists)
                    .order_by(
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
            terminalized = 0
            for row in rows:
                result = conn.execute(
                    update(token_work_items_table)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.PENDING_SINK.value)
                    .where(token_work_items_table.c.pending_sink_name.is_not(None))
                    .values(
                        status=TokenWorkStatus.TERMINAL.value,
                        row_payload_json=self._scrubbed_row_payload_json(row["token_id"]),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._record_scheduler_event(
                        conn,
                        event_type=SchedulerEventType.MARK_PENDING_SINK_TERMINAL,
                        run_id=run_id,
                        token_id=row["token_id"],
                        work_item_id=row["work_item_id"],
                        node_id=row["node_id"],
                        from_status=TokenWorkStatus.PENDING_SINK,
                        to_status=TokenWorkStatus.TERMINAL,
                        from_lease_owner=row["lease_owner"],
                        to_lease_owner=None,
                        from_attempt=row["attempt"],
                        to_attempt=row["attempt"],
                        recorded_at=now,
                        from_lease_expires_at=row["lease_expires_at"],
                        to_lease_expires_at=None,
                        caller_owner=caller_owner,
                    )
                    terminalized += 1
                elif result.rowcount not in (0, None):
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink terminal-outcome repair affected {result.rowcount} rows for "
                        f"run_id={run_id!r} token_id={row['token_id']!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                    )
        return terminalized

    def mark_blocked_barrier_terminal(
        self,
        *,
        run_id: str,
        barrier_key: str,
        token_ids: tuple[str, ...],
        now: datetime,
    ) -> int:
        """Mark BLOCKED work consumed by a resolved barrier as terminal.

        Delegating wrapper over :meth:`complete_barrier` (F1 Task 2.3). The
        legacy arm (``require_exhaustive_release=False``) preserves the pinned
        partial-release semantics: BLOCKED rows under the barrier that are not
        named in ``token_ids`` stay BLOCKED.
        """
        if not token_ids:
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization for run_id={run_id!r} barrier_key={barrier_key!r} "
                "requires at least one live token_id; refusing to terminalize durable BLOCKED rows without "
                "live barrier evidence."
            )
        return self.complete_barrier(
            run_id=run_id,
            barrier_key=barrier_key,
            consumed_token_ids=token_ids,
            emitted_pending_sink=(),
            emitted_ready=(),
            now=now,
            require_exhaustive_release=False,
        )

    def peer_active_leases(
        self,
        *,
        run_id: str,
        caller_owner: str,
        now: datetime,
    ) -> tuple[str, ...]:
        """Return distinct lease_owners of unexpired LEASED rows held by peers.

        A "peer" is any ``lease_owner`` other than ``caller_owner``. A lease is
        "active" if ``lease_expires_at > now``; rows whose lease has expired are
        recoverable via ``recover_expired_leases`` and do not contribute to the
        peer set.

        This is the single-active-resume precondition surface for
        ``drain_scheduled_work``: a resume drain run while a peer holds an
        unexpired lease on the same ``run_id`` would race against the peer's
        sink-bound RowResult emission (PENDING_SINK can transition to LEASED
        under the peer's identity and the helper would re-emit a duplicate
        RowResult on a later iteration once the lease expires). See
        filigree elspeth-66be4216cd (G3 single-active-resume invariant).

        Under ADR-026 Precondition #9 (multi-worker deployment-shape ADR not
        yet authored), no code path exists today that spawns concurrent
        ``RowProcessor`` instances on the same ``run_id``; this method returns
        ``()`` for every supported deployment. The check exists as an offensive
        guard so that any future code that violates the precondition crashes
        with a Tier-1 audit-integrity error instead of silently emitting
        duplicate RowResults into the audit trail.
        """
        with self._engine.connect() as conn:
            owners = (
                conn.execute(
                    select(token_work_items_table.c.lease_owner)
                    .distinct()
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
                    .where(token_work_items_table.c.lease_owner != caller_owner)
                    .where(token_work_items_table.c.lease_expires_at > now)
                    .order_by(token_work_items_table.c.lease_owner)
                )
                .scalars()
                .all()
            )
        return tuple(owner for owner in owners if owner is not None)

    def count_active_work(self, *, run_id: str) -> int:
        """Count non-terminal scheduler work for a run."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
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
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
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

    def list_blocked_barrier_items(self, *, run_id: str) -> list[TokenWorkItem]:
        """Return BLOCKED barrier holds for a run in deterministic order.

        The dual-use BLOCKED partition (D1) lives in the shared
        ``blocked_barrier_hold_clause`` predicate — see its docstring.

        Iteration order is ``(barrier_key, ingest_sequence, work_item_id)``
        for determinism only; buffer ORDER at restore comes from
        ``batch_members.ordinal``, not from this verb. Read-only: no
        scheduler_event is recorded.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(blocked_barrier_hold_clause())
                    .order_by(
                        token_work_items_table.c.barrier_key,
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
        return [self._item_from_mapping(row) for row in rows]

    @staticmethod
    def _unresolved_work_predicate() -> ColumnElement[bool]:
        """Predicate for work that has not reached a durable sink handoff.

        PENDING_SINK rows — and LEASED rows carrying a ``pending_sink_name``
        (a pending sink re-claimed during resume recovery) — are excluded:
        their producer work is durably complete and they are terminalized
        only after sink durability via ``mark_pending_sink_terminal``.
        """
        return or_(
            token_work_items_table.c.status.in_(
                (
                    TokenWorkStatus.READY.value,
                    TokenWorkStatus.BLOCKED.value,
                )
            ),
            and_(
                token_work_items_table.c.status == TokenWorkStatus.LEASED.value,
                token_work_items_table.c.pending_sink_name.is_(None),
            ),
        )

    def count_unresolved_work(self, *, run_id: str) -> int:
        """Count scheduler work not yet resolved into a durable sink handoff."""
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(self._unresolved_work_predicate())
            ).scalar_one()
        return int(result)

    def summarize_unresolved_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize unresolved work grouped by status and blocking keys."""
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
                    .where(self._unresolved_work_predicate())
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

    def summarize_active_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize active work grouped by status and blocking keys."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
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
            before = (
                conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id))
                .mappings()
                .one_or_none()
            )
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
            if before is None:
                raise AuditIntegrityError(
                    f"Scheduler transition to {status.name!r} for work_item_id={work_item_id!r} "
                    "updated a row that could not be read before the write; audit transition history cannot be proven."
                )
            next_lease_owner = update_values["lease_owner"] if "lease_owner" in update_values else before["lease_owner"]
            next_lease_expires_at = update_values["lease_expires_at"] if "lease_expires_at" in update_values else before["lease_expires_at"]
            next_attempt = update_values["attempt"] if "attempt" in update_values else before["attempt"]
            if next_lease_owner is not None and type(next_lease_owner) is not str:
                raise AuditIntegrityError(
                    f"Scheduler transition to {status.name!r} for work_item_id={work_item_id!r} "
                    f"produced invalid lease_owner type {type(next_lease_owner).__name__}; audit transition history cannot be proven."
                )
            if next_lease_expires_at is not None and type(next_lease_expires_at) is not datetime:
                raise AuditIntegrityError(
                    f"Scheduler transition to {status.name!r} for work_item_id={work_item_id!r} "
                    f"produced invalid lease_expires_at type {type(next_lease_expires_at).__name__}; audit transition history cannot be proven."
                )
            if type(next_attempt) is not int:
                raise AuditIntegrityError(
                    f"Scheduler transition to {status.name!r} for work_item_id={work_item_id!r} "
                    f"produced invalid attempt type {type(next_attempt).__name__}; audit transition history cannot be proven."
                )
            self._record_scheduler_event(
                conn,
                event_type=self._TRANSITION_EVENT_TYPES[status],
                run_id=before["run_id"],
                token_id=before["token_id"],
                work_item_id=work_item_id,
                node_id=before["node_id"],
                from_status=TokenWorkStatus(before["status"]),
                to_status=status,
                from_lease_owner=before["lease_owner"],
                to_lease_owner=next_lease_owner,
                from_attempt=before["attempt"],
                to_attempt=next_attempt,
                recorded_at=now,
                from_lease_expires_at=before["lease_expires_at"],
                to_lease_expires_at=next_lease_expires_at,
                caller_owner=expected_lease_owner,
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
        )

    @staticmethod
    def _scrubbed_row_payload_json(anchor: str) -> str:
        """Return non-row scheduler payload retained after terminalization."""
        return canonical_json({"row_payload": "purged", "payload_hash": hashlib.sha256(anchor.encode()).hexdigest()})
