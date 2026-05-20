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
from sqlalchemy.engine import RowMapping

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.schema import token_work_items_table


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
        node_id: str,
        step_index: int,
        ingest_sequence: int,
        row_payload_json: str,
        available_at: datetime,
        attempt: int = 1,
        queue_key: str | None = None,
        barrier_key: str | None = None,
    ) -> TokenWorkItem:
        """Persist a READY token continuation."""
        work_item_id = self._work_item_id(run_id, token_id, node_id, attempt)
        values = {
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
            "attempt": attempt,
            "lease_owner": None,
            "lease_expires_at": None,
            "available_at": available_at,
            "created_at": available_at,
            "updated_at": available_at,
        }
        with self._engine.begin() as conn:
            conn.execute(token_work_items_table.insert().values(**values))
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        return self._item_from_mapping(row)

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

            conn.execute(
                update(token_work_items_table)
                .where(
                    and_(
                        token_work_items_table.c.work_item_id == row["work_item_id"],
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
            claimed = (
                conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == row["work_item_id"]))
                .mappings()
                .one()
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
    ) -> TokenWorkItem:
        """Move a claimed item to WAITING until ``available_at``."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.WAITING,
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
    ) -> TokenWorkItem:
        """Move an item to BLOCKED at a queue or barrier."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.BLOCKED,
            queue_key=queue_key,
            barrier_key=barrier_key,
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_terminal(self, *, work_item_id: str, now: datetime) -> TokenWorkItem:
        """Mark a work item terminal. Re-marking TERMINAL is idempotent."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.TERMINAL,
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_failed(self, *, work_item_id: str, now: datetime) -> TokenWorkItem:
        """Mark a work item failed after retries are exhausted."""
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.FAILED,
            lease_owner=None,
            lease_expires_at=None,
        )

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

    def _transition(
        self,
        *,
        work_item_id: str,
        now: datetime,
        status: TokenWorkStatus,
        **values: object,
    ) -> TokenWorkItem:
        update_values = {"status": status.value, "updated_at": now, **values}
        with self._engine.begin() as conn:
            conn.execute(
                update(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id).values(**update_values)
            )
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        return self._item_from_mapping(row)

    @staticmethod
    def _work_item_id(run_id: str, token_id: str, node_id: str, attempt: int) -> str:
        raw = f"{run_id}:{token_id}:{node_id}:{attempt}".encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _item_from_mapping(row: RowMapping) -> TokenWorkItem:
        data = dict(row)
        for key in ("available_at", "created_at", "updated_at", "lease_expires_at"):
            value = data[key]
            if isinstance(value, datetime) and value.tzinfo is None:
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
            attempt=data["attempt"],
            lease_owner=data["lease_owner"],
            lease_expires_at=data["lease_expires_at"],
            available_at=data["available_at"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
