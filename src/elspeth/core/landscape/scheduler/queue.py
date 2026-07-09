"""Queue intake for scheduler work items: enqueue verbs and fenced ingest.

Persisting READY continuations (plain, claimed-in-transaction, and the
composed fenced leader INGEST). Depends on the lease repository for the
in-transaction claim CAS. Extracted from ``TokenSchedulerRepository``
(filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.engine import Connection, RowMapping

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, RunWorkerEvictedError
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.core.landscape.database import Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.scheduler.events import SchedulerEventStore
from elspeth.core.landscape.scheduler.leases import SchedulerLeaseRepository
from elspeth.core.landscape.scheduler.work_items import (
    insert_work_item_idempotent,
    item_from_mapping,
    ready_work_item_values,
    validate_work_item_references,
)
from elspeth.core.landscape.scheduler.work_items import (
    work_item_id as make_work_item_id,
)
from elspeth.core.landscape.schema import active_worker_fence_clause, token_work_items_table

if TYPE_CHECKING:
    from elspeth.contracts.audit import Row, Token


class SchedulerQueueRepository:
    """Persists READY token continuations into the scheduler queue."""

    def __init__(self, engine: Tier1Engine, *, events: SchedulerEventStore, leases: SchedulerLeaseRepository) -> None:
        self._engine = engine
        self._events = events
        self._leases = leases

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
        worker_id: str | None = None,
    ) -> TokenWorkItem:
        """Persist a READY token continuation.

        Token lineage and coalesce cursor fields are durable resume metadata:
        workers that claim this row later must be able to rebuild the same
        ``TokenInfo`` and ``WorkItem`` without any live in-memory queue state.

        ``worker_id`` (ADR-030 §G, slice 4): when provided, the membership fence
        is checked BEFORE the idempotent INSERT — an evicted or departed caller
        raises :class:`~elspeth.contracts.errors.RunWorkerEvictedError` rather
        than inserting a READY row that no active worker will ever claim.  The
        dedup path (``insert_work_item_idempotent`` returns ``False``) re-reads
        the existing row without the fence to preserve idempotency on resume; the
        fence only guards the FIRST write.

        ``None`` preserves the unfenced legacy behavior for direct
        repository-level callers (tests, tooling, barrier completion) that do not
        carry a worker identity.  This verb stays Optional because several
        important callers — ``complete_barrier`` (fires inside a larger
        fenced transaction) and tests — legitimately have no registry row.
        """
        work_item_id = make_work_item_id(run_id, token_id, node_id, attempt)
        values = ready_work_item_values(
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
        with begin_write(self._engine) as conn:
            # Membership fence (ADR-030 §G, slice 4): checked BEFORE the INSERT
            # and BEFORE the reference validation — an evicted caller must not
            # leave a READY orphan, and the fence is the outer guard (reference
            # validation is an integrity check that assumes the caller is valid).
            # Only applied when worker_id is provided (see docstring for who
            # stays Optional).
            if worker_id is not None:
                fence_holds = conn.execute(select(active_worker_fence_clause(worker_id=worker_id, run_id=run_id))).scalar()
                if not fence_holds:
                    raise RunWorkerEvictedError(worker_id=worker_id, run_id=run_id)
            validate_work_item_references(
                conn,
                run_id=run_id,
                token_id=token_id,
                row_id=row_id,
                ingest_sequence=ingest_sequence,
                node_id=node_id,
                coalesce_node_id=coalesce_node_id,
            )
            inserted = insert_work_item_idempotent(conn, values=values, operation="enqueue READY scheduler work")
            if inserted:
                self._events.record(
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
        return item_from_mapping(row)

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
        with begin_write(self._engine) as conn:
            row = self.enqueue_ready_claimed_on(
                conn,
                run_id=run_id,
                token_id=token_id,
                row_id=row_id,
                node_id=node_id,
                step_index=step_index,
                ingest_sequence=ingest_sequence,
                row_payload_json=row_payload_json,
                available_at=available_at,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
                now=now,
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
        return item_from_mapping(row)

    def enqueue_ready_claimed_on(
        self,
        conn: Connection,
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
    ) -> RowMapping:
        """Connection-accepting enqueue-and-claim: composes into the caller's transaction.

        Extracted from :meth:`enqueue_ready_claimed` so the fenced ingest verb
        (:meth:`ingest_row_with_initial_claim`, ADR-030 §C.4 row 9) can
        compose the reference validation + idempotent insert + ENQUEUE/CLAIM
        events onto ONE connection with the rows/tokens inserts.
        """
        work_item_id = make_work_item_id(run_id, token_id, node_id, attempt)
        values = ready_work_item_values(
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
        validate_work_item_references(
            conn,
            run_id=run_id,
            token_id=token_id,
            row_id=row_id,
            ingest_sequence=ingest_sequence,
            node_id=node_id,
            coalesce_node_id=coalesce_node_id,
        )
        inserted = insert_work_item_idempotent(conn, values=values, operation="enqueue and claim READY scheduler work")
        if inserted:
            self._events.record(
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
            claimed = self._leases.claim_ready_row(
                conn,
                row=row,
                run_id=run_id,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
                now=now,
            )
            if claimed is not None:
                row = claimed
        return row

    def ingest_row_with_initial_claim(
        self,
        *,
        coordination_token: CoordinationToken,
        now: datetime,
        insert_row_and_token: Callable[[Connection], tuple[Row, Token]],
        token_id: str,
        row_id: str,
        node_id: str | None,
        step_index: int,
        ingest_sequence: int,
        row_payload_json: str,
        lease_owner: str,
        lease_seconds: int,
        queue_key: str | None = None,
        barrier_key: str | None = None,
        on_success_sink: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
        expand_group_id: str | None = None,
        coalesce_node_id: str | None = None,
        coalesce_name: str | None = None,
    ) -> tuple[Row, Token, TokenWorkItem]:
        """Fenced leader INGEST (ADR-030 §C.4 row 9): one IMMEDIATE transaction.

        Composes (1) the verify-and-extend epoch fence, (2) the ``rows`` +
        ``tokens`` inserts (via the injected ``insert_row_and_token``
        callable, a closure over ``DataFlowRepository.insert_row_with_token_on``),
        and (3) the initial enqueue-and-claim — on ONE connection. A stale
        epoch refuses the WHOLE ingest: the rows insert rolls back with
        everything else, so a deposed leader woken mid-ingest leaves no
        orphan ``rows`` row (crash-walk step 8). The UNIQUE
        ``(run_id, ingest_sequence)`` constraint becomes a true backstop.

        ``coordination_token`` is REQUIRED — this verb has no legacy callers.
        Raises :class:`~elspeth.contracts.errors.RunLeadershipLostError` on a
        fence miss (``fence_refusal`` evented on a fresh connection).
        """
        run_id = coordination_token.run_id
        with fenced_leader_transaction(
            self._engine,
            token=coordination_token,
            now=now,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="ingest_row_with_initial_claim",
        ) as conn:
            row_record, token_record = insert_row_and_token(conn)
            if row_record.row_id != row_id or token_record.token_id != token_id:
                raise AuditIntegrityError(
                    f"Fenced ingest for run_id={run_id!r} inserted row_id={row_record.row_id!r} / "
                    f"token_id={token_record.token_id!r} but the scheduler enqueue was declared for "
                    f"row_id={row_id!r} / token_id={token_id!r}; the composed transaction would "
                    "journal a cursor for identities it did not insert."
                )
            scheduled = self.enqueue_ready_claimed_on(
                conn,
                run_id=run_id,
                token_id=token_id,
                row_id=row_id,
                node_id=node_id,
                step_index=step_index,
                ingest_sequence=ingest_sequence,
                row_payload_json=row_payload_json,
                available_at=now,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
                now=now,
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
        return row_record, token_record, item_from_mapping(scheduled)
