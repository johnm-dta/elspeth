"""Fail-closed recovery for the legacy TS-02 source-completion crash seam."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.engine import Connection, RowMapping

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.node_states import NodeStateRepository
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.scheduler.payload_codec import deserialize_row_payload
from elspeth.core.landscape.schema import (
    rows_table,
    scheduler_events_table,
    token_parents_table,
    token_work_items_table,
    tokens_table,
)


class SourceCompletionReconciler:
    """Synthesize only source states proven by the exact pre-fix TS-02 image."""

    def __init__(self, db: LandscapeDB, *, node_states: NodeStateRepository) -> None:
        self._db = db
        self._node_states = node_states

    @staticmethod
    def _event_mismatches(event: RowMapping, expected: dict[str, object]) -> dict[str, tuple[object, object]]:
        return {field: (event[field], value) for field, value in expected.items() if event[field] != value}

    def _validate_exact_scheduler_witness(
        self,
        conn: Connection,
        *,
        run_id: str,
        witness: RowMapping,
    ) -> None:
        """Require the two events emitted by initial enqueue-and-claim, exactly."""
        token_id = str(witness["token_id"])
        work_item_id = str(witness["work_item_id"])
        if witness["attempt"] != 1 or witness["step_index"] != 1:
            raise AuditIntegrityError(
                f"Source completion reconciliation for token {token_id!r} found a missing source state on "
                f"work item {work_item_id!r} at attempt={witness['attempt']!r}, step_index={witness['step_index']!r}; "
                "the legacy TS-02 witness requires attempt=1 and step_index=1."
            )

        events = (
            conn.execute(
                select(scheduler_events_table).where(
                    scheduler_events_table.c.run_id == run_id,
                    scheduler_events_table.c.token_id == token_id,
                    scheduler_events_table.c.work_item_id == work_item_id,
                )
            )
            .mappings()
            .all()
        )
        if len(events) != 2:
            raise AuditIntegrityError(
                f"Source completion reconciliation for token {token_id!r} expected exactly two scheduler events "
                f"for initial work item {work_item_id!r}; found {len(events)}."
            )
        events_by_type: dict[str, list[RowMapping]] = {}
        for event in events:
            events_by_type.setdefault(str(event["event_type"]), []).append(event)
        expected_types = {SchedulerEventType.ENQUEUE.value, SchedulerEventType.CLAIM_READY.value}
        if set(events_by_type) != expected_types or any(len(group) != 1 for group in events_by_type.values()):
            raise AuditIntegrityError(
                f"Source completion reconciliation for token {token_id!r} found ambiguous scheduler event types "
                f"for initial work item {work_item_id!r}: {sorted(event['event_type'] for event in events)!r}."
            )

        node_id = witness["node_id"]
        lease_owner = witness["lease_owner"]
        enqueue = events_by_type[SchedulerEventType.ENQUEUE.value][0]
        enqueue_expected: dict[str, object] = {
            "node_id": node_id,
            "from_status": None,
            "to_status": TokenWorkStatus.READY.value,
            "from_lease_owner": None,
            "to_lease_owner": None,
            "from_lease_expires_at": None,
            "to_lease_expires_at": None,
            "from_attempt": None,
            "to_attempt": 1,
            "recorded_at": witness["available_at"],
            "caller_owner": None,
            "context_json": "{}",
        }
        claim = events_by_type[SchedulerEventType.CLAIM_READY.value][0]
        claim_expected: dict[str, object] = {
            "node_id": node_id,
            "from_status": TokenWorkStatus.READY.value,
            "to_status": TokenWorkStatus.LEASED.value,
            "from_lease_owner": None,
            "to_lease_owner": lease_owner,
            "from_lease_expires_at": None,
            "to_lease_expires_at": witness["lease_expires_at"],
            "from_attempt": 1,
            "to_attempt": 1,
            "recorded_at": witness["updated_at"],
            "caller_owner": lease_owner,
            "context_json": "{}",
        }
        mismatches = {
            SchedulerEventType.ENQUEUE.value: self._event_mismatches(enqueue, enqueue_expected),
            SchedulerEventType.CLAIM_READY.value: self._event_mismatches(claim, claim_expected),
        }
        mismatches = {event_type: fields for event_type, fields in mismatches.items() if fields}
        if mismatches:
            raise AuditIntegrityError(
                f"Source completion reconciliation for token {token_id!r} found conflicting scheduler evidence "
                f"for initial work item {work_item_id!r}: {mismatches!r}."
            )

    def reconcile(
        self,
        *,
        run_id: str,
        coordination_token: CoordinationToken,
        at: datetime,
    ) -> int:
        """Repair pre-fix TS-02 gaps atomically before any plugin can run.

        Existing source states are validated as exact source witnesses. A
        missing witness is repaired only for a root token whose current
        LEASED attempt-1 work item has the exact two-event initial scheduler
        history. Every ambiguity is an audit-integrity failure.
        """
        repaired = 0
        with fenced_leader_transaction(
            self._db.engine,
            token=coordination_token,
            now=at,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="reconcile_source_completions_from_scheduler",
        ) as conn:
            witnesses = (
                conn.execute(
                    select(
                        token_work_items_table.c.work_item_id,
                        token_work_items_table.c.token_id,
                        token_work_items_table.c.row_id,
                        token_work_items_table.c.node_id,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.row_payload_json,
                        token_work_items_table.c.attempt,
                        token_work_items_table.c.lease_owner,
                        token_work_items_table.c.lease_expires_at,
                        token_work_items_table.c.available_at,
                        token_work_items_table.c.updated_at,
                        rows_table.c.source_node_id,
                        rows_table.c.source_data_hash,
                        rows_table.c.ingest_sequence.label("row_ingest_sequence"),
                    )
                    .join(
                        tokens_table,
                        (tokens_table.c.token_id == token_work_items_table.c.token_id)
                        & (tokens_table.c.row_id == token_work_items_table.c.row_id)
                        & (tokens_table.c.run_id == token_work_items_table.c.run_id),
                    )
                    .join(
                        rows_table,
                        (rows_table.c.row_id == token_work_items_table.c.row_id) & (rows_table.c.run_id == token_work_items_table.c.run_id),
                    )
                    .where(
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.status == TokenWorkStatus.LEASED.value,
                        tokens_table.c.step_in_pipeline.is_(None),
                        ~select(token_parents_table.c.token_id)
                        .where(
                            token_parents_table.c.token_id == token_work_items_table.c.token_id,
                            token_parents_table.c.run_id == run_id,
                        )
                        .exists(),
                    )
                )
                .mappings()
                .all()
            )

            for witness in witnesses:
                token_id = str(witness["token_id"])
                if witness["ingest_sequence"] != witness["row_ingest_sequence"]:
                    raise AuditIntegrityError(
                        f"Source completion reconciliation for token {token_id!r} found scheduler/row ingest-sequence mismatch."
                    )
                source_node_id = str(witness["source_node_id"])
                source_data_hash = str(witness["source_data_hash"])
                if self._node_states.validate_existing_source_completed_node_state_on(
                    conn,
                    token_id=token_id,
                    source_node_id=source_node_id,
                    run_id=run_id,
                    expected_hash=source_data_hash,
                ):
                    continue

                self._validate_exact_scheduler_witness(conn, run_id=run_id, witness=witness)
                source_data = deserialize_row_payload(str(witness["row_payload_json"])).to_dict()
                source_hash = stable_hash(source_data)
                if source_hash != source_data_hash:
                    raise AuditIntegrityError(
                        f"Source completion reconciliation for token {token_id!r} found scheduler payload hash {source_hash!r} "
                        f"but rows.source_data_hash is {source_data_hash!r}."
                    )
                repaired += int(
                    self._node_states.ensure_source_completed_node_state_on(
                        conn,
                        token_id=token_id,
                        source_node_id=source_node_id,
                        run_id=run_id,
                        source_data=source_data,
                    )
                )
        return repaired
