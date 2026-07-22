"""Durable scheduler event store.

Writes one ``scheduler_events`` audit row per state transition on the
caller's connection. Every scheduler component records through ONE shared
``SchedulerEventStore`` instance (composed by the facade), so a test can
intercept the whole event plane at a single seam. Extracted from
``TokenSchedulerRepository`` (filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import scheduler_events_table


class SchedulerEventStore:
    """Append-only writer (and recovery-event reader) for ``scheduler_events``."""

    def record(
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
            inserted_event_id = conn.execute(
                scheduler_events_table.insert().values(**values).returning(scheduler_events_table.c.event_id)
            ).scalar_one()
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"Scheduler event {event_type.value!r} failed for run_id={run_id!r} "
                f"work_item_id={work_item_id!r} — database rejected audit write: {type(exc).__name__}"
            ) from exc
        if inserted_event_id != event_id:
            raise LandscapeRecordError(
                f"Scheduler event {event_type.value!r} returned unexpected event_id={inserted_event_id!r} for "
                f"run_id={run_id!r} work_item_id={work_item_id!r}; expected {event_id!r}."
            )

    @staticmethod
    def recovery_event_for_previous_work_item(
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
