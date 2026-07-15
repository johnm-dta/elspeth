"""Disposition verbs: transitions out of LEASED and pending-sink terminalization.

The lease-owner-CAS ``mark_*`` verbs (BLOCKED / TERMINAL / FAILED /
PENDING_SINK), the strict post-sink terminalizers, and the crash-repair
terminalization sweep, all built on the shared ``_transition`` CAS.
Extracted from ``TokenSchedulerRepository`` (filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from sqlalchemy import and_, select, update
from sqlalchemy.engine import RowMapping

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, RunWorkerEvictedError
from elspeth.contracts.scheduler import BranchLossSpec, SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.core.landscape.database import Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.scheduler.branch_losses import record_coalesce_branch_loss
from elspeth.core.landscape.scheduler.events import SchedulerEventStore
from elspeth.core.landscape.scheduler.fencing import fenced_write
from elspeth.core.landscape.scheduler.payload_codec import scrubbed_row_payload_json
from elspeth.core.landscape.scheduler.work_items import item_from_mapping
from elspeth.core.landscape.schema import (
    claim_verb_fence_clause,
    run_workers_table,
    token_outcomes_table,
    token_work_items_table,
)


class SchedulerDispositionRepository:
    """Lease-fenced dispositions and pending-sink terminalization."""

    _TRANSITION_EVENT_TYPES: ClassVar[dict[TokenWorkStatus, SchedulerEventType]] = {
        TokenWorkStatus.BLOCKED: SchedulerEventType.MARK_BLOCKED,
        TokenWorkStatus.TERMINAL: SchedulerEventType.MARK_TERMINAL,
        TokenWorkStatus.FAILED: SchedulerEventType.MARK_FAILED,
        TokenWorkStatus.PENDING_SINK: SchedulerEventType.MARK_PENDING_SINK,
    }

    def __init__(self, engine: Tier1Engine, *, events: SchedulerEventStore) -> None:
        self._engine = engine
        self._events = events

    def mark_blocked(
        self,
        *,
        work_item_id: str,
        queue_key: str | None,
        barrier_key: str | None,
        now: datetime,
        expected_lease_owner: str,
        worker_id: str | None = None,
    ) -> TokenWorkItem:
        """Move an item to BLOCKED at a queue or barrier.

        ``worker_id`` (optional): membership-fence identity — see
        :meth:`mark_terminal`.
        """
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
            fenced_worker_id=worker_id,
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

    def mark_terminal(
        self,
        *,
        work_item_id: str,
        now: datetime,
        expected_lease_owner: str,
        branch_loss: BranchLossSpec | None = None,
        worker_id: str | None = None,
    ) -> TokenWorkItem:
        """Mark a leased work item terminal.

        ``branch_loss`` (§E.5): a non-failure lossy disposition of a
        fork-lineage branch feeding a coalesce (filter-drop / gate-discard)
        records its durable loss in the SAME transaction (record-then-notify
        uniformity rule).

        ``worker_id`` (optional): membership-fence identity (ADR-030 §G
        parity, filigree elspeth-ba7b2cc25d). When supplied, the disposition
        UPDATE also requires the worker to be an active member of the row's
        run (LENIENT ``claim_verb_fence_clause`` — N=0 runs pass); an
        evicted/departed worker is refused with ``RunWorkerEvictedError``
        and zero mutation.
        """
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.TERMINAL,
            expected_lease_owner=expected_lease_owner,
            fenced_worker_id=worker_id,
            branch_loss=branch_loss,
            row_payload_json=scrubbed_row_payload_json(work_item_id),
            lease_owner=None,
            lease_expires_at=None,
        )

    def mark_failed(
        self,
        *,
        work_item_id: str,
        now: datetime,
        expected_lease_owner: str,
        branch_loss: BranchLossSpec | None = None,
        worker_id: str | None = None,
    ) -> TokenWorkItem:
        """Mark a leased work item failed after retries are exhausted.

        ``branch_loss`` (§E.5): when the failed item is a fork-lineage branch
        feeding a coalesce, the durable loss record commits in the SAME
        transaction as this disposition (record-then-notify uniformity rule).

        ``worker_id`` (optional): membership-fence identity — see
        :meth:`mark_terminal`.
        """
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.FAILED,
            expected_lease_owner=expected_lease_owner,
            fenced_worker_id=worker_id,
            branch_loss=branch_loss,
            row_payload_json=scrubbed_row_payload_json(work_item_id),
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
        branch_loss: BranchLossSpec | None = None,
        worker_id: str | None = None,
    ) -> TokenWorkItem:
        """Move a claimed item to a durable sink handoff state.

        ``worker_id`` (optional): membership-fence identity — see
        :meth:`mark_terminal`.

        Attributed park (ADR-030 strict pending-sink terminalization): the
        parked row KEEPS ``lease_owner=expected_lease_owner`` with
        ``lease_expires_at=None`` — "parked, owner-attributed, not leased"
        (schema-legal: the lease CHECK constrains only LEASED rows). The
        post-sink terminalization (:meth:`mark_pending_sink_terminal`) then
        CASes strictly on that owner; the historical NULL park forced a
        NULL-acceptance arm there that a takeover could slip through.

        ``branch_loss`` (§E.5): a divert arm that lossy-disposes a
        fork-lineage branch records its durable loss in the SAME transaction.
        """
        return self._transition(
            work_item_id=work_item_id,
            now=now,
            status=TokenWorkStatus.PENDING_SINK,
            expected_lease_owner=expected_lease_owner,
            branch_loss=branch_loss,
            fenced_worker_id=worker_id,
            row_payload_json=row_payload_json,
            pending_sink_name=sink_name,
            pending_outcome=outcome,
            pending_path=path,
            pending_error_hash=error_hash,
            pending_error_message=error_message,
            lease_owner=expected_lease_owner,
            lease_expires_at=None,
        )

    def mark_pending_sink_terminal(
        self,
        *,
        run_id: str,
        token_id: str,
        now: datetime,
        expected_lease_owner: str,
        coordination_token: CoordinationToken,
    ) -> int:
        """Terminalize pending sink scheduler work after token outcome durability.

        ``expected_lease_owner`` is REQUIRED and the owner CAS is STRICT
        (ADR-030 §C.4 row 7): the historical NULL-owner acceptance arm is
        deleted. Every path that parks a row into PENDING_SINK now attributes
        the owner (``mark_pending_sink`` / ``complete_barrier``'s emission
        arms stamp it; ``recover_expired_leases``' reap arm deliberately
        parks NULL because the prior owner is deposed — reaped handoffs are
        always re-claimed via ``claim_pending_sink``, which overwrites the
        owner, before terminalization). A row whose owner does not match —
        including NULL — is simply not terminalized (returns 0 for the
        caller's loud invariant check).

        ``coordination_token`` (ADR-030 §C.4 row 7, slice-4 ratchet:
        REQUIRED): the verify-and-extend leader epoch fence is the FIRST
        statement of this verb's transaction — a deposed leader cannot
        terminalize the new leader's ledger even with a matching owner.
        The epoch fence stacks on top of the owner CAS: both must pass.
        """
        predicates = [
            token_work_items_table.c.run_id == run_id,
            token_work_items_table.c.token_id == token_id,
            token_work_items_table.c.status.in_((TokenWorkStatus.PENDING_SINK.value, TokenWorkStatus.LEASED.value)),
            token_work_items_table.c.pending_sink_name.is_not(None),
            token_work_items_table.c.lease_owner == expected_lease_owner,
        ]
        with fenced_leader_transaction(
            self._engine,
            token=coordination_token,
            now=now,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="mark_pending_sink_terminal",
        ) as conn:
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
                        row_payload_json=scrubbed_row_payload_json(token_id),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._events.record(
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

    def mark_pending_sink_terminal_many(
        self,
        *,
        run_id: str,
        token_ids: tuple[str, ...],
        now: datetime,
        expected_lease_owner: str,
        coordination_token: CoordinationToken,
    ) -> int:
        """Terminalize sink-bound scheduler work for a durable sink batch.

        This preserves the audit contract of one scheduler event per terminalized
        work item while avoiding one transaction and one indexed SELECT per token
        after large sink writes.

        ``expected_lease_owner`` is REQUIRED and the owner CAS is STRICT
        (ADR-030 §C.4 row 7; see :meth:`mark_pending_sink_terminal` for the
        attributed-park co-change): a row whose owner does not match —
        including NULL — refuses the whole batch with
        :class:`~elspeth.contracts.errors.AuditIntegrityError`.

        ``coordination_token`` (ADR-030 §C.4 row 7, slice-4 ratchet:
        REQUIRED): the verify-and-extend leader epoch fence is the FIRST
        statement of the batch transaction; a deposed leader's batch is
        refused with :class:`RunLeadershipLostError` (and a
        ``fence_refusal`` event) before any row is touched. Mirrors the
        singleton sibling :meth:`mark_pending_sink_terminal`.
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
        with fenced_leader_transaction(
            self._engine,
            token=coordination_token,
            now=now,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="mark_pending_sink_terminal_many",
        ) as conn:
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
                if row_lease_owner != expected_lease_owner:
                    raise AuditIntegrityError(
                        f"Scheduler pending-sink batch terminalization for run_id={run_id!r} token_id={token_id!r} found "
                        f"lease_owner={row_lease_owner!r}; expected lease_owner={expected_lease_owner!r} "
                        "(strict owner CAS — NULL-owner acceptance removed, ADR-030)."
                    )

            terminalized = 0
            for row in rows:
                result = conn.execute(
                    update(token_work_items_table)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.token_id == row["token_id"])
                    .where(token_work_items_table.c.status == row["status"])
                    .where(token_work_items_table.c.pending_sink_name.is_not(None))
                    .where(token_work_items_table.c.lease_owner == expected_lease_owner)
                    .values(
                        status=TokenWorkStatus.TERMINAL.value,
                        row_payload_json=scrubbed_row_payload_json(row["token_id"]),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._events.record(
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

    def terminalize_pending_sinks_with_terminal_outcomes(
        self,
        *,
        run_id: str,
        now: datetime,
        caller_owner: str,
        coordination_token: CoordinationToken,
    ) -> int:
        """Repair PENDING_SINK work whose terminal token outcome is already durable.

        A crash can land after sink outcome durability but before the scheduler
        handoff row is marked terminal. Resume must not claim and re-emit those
        rows externally; the terminal token outcome is the authoritative witness.

        ``coordination_token`` (ADR-030 §G, slice-4 ratchet: REQUIRED): this
        verb deliberately terminalizes REGARDLESS of owner — it is the
        crash-repair verb and the terminal token outcome is the witness — so
        its protection is the leader epoch fence (first statement of the
        transaction), not the owner CAS. Owner-blindness is unchanged;
        ``caller_owner`` remains the event attribution only.
        """
        terminal_outcome_exists = (
            select(token_outcomes_table.c.outcome_id)
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.token_id == token_work_items_table.c.token_id)
            .where(token_outcomes_table.c.completed == 1)
            .exists()
        )
        with fenced_write(
            self._engine, coordination_token=coordination_token, now=now, verb="terminalize_pending_sinks_with_terminal_outcomes"
        ) as conn:
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
                        row_payload_json=scrubbed_row_payload_json(row["token_id"]),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    self._events.record(
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

    def _transition(
        self,
        *,
        work_item_id: str,
        now: datetime,
        status: TokenWorkStatus,
        expected_statuses: tuple[TokenWorkStatus, ...] = (TokenWorkStatus.LEASED,),
        expected_lease_owner: str | None = None,
        branch_loss: BranchLossSpec | None = None,
        fenced_worker_id: str | None = None,
        **values: object,
    ) -> TokenWorkItem:
        update_values = {"status": status.value, "updated_at": now, **values}
        expected_status_values = tuple(candidate.value for candidate in expected_statuses)
        expected_status_text = " or ".join(candidate.name for candidate in expected_statuses)
        predicates = [
            token_work_items_table.c.work_item_id == work_item_id,
            token_work_items_table.c.status.in_(expected_status_values),
            # TS-07 through TS-10 are transform-work dispositions.  A sink
            # handoff reclaimed by claim_pending_sink is also LEASED, but its
            # non-NULL pending_sink_name makes it the sink-redrive subtype;
            # only the dedicated pending-sink terminalizers may consume it.
            token_work_items_table.c.pending_sink_name.is_(None),
        ]
        if expected_lease_owner is not None:
            predicates.append(token_work_items_table.c.lease_owner == expected_lease_owner)
        if fenced_worker_id is not None:
            # Membership fence, ADR-030 §G parity (filigree elspeth-ba7b2cc25d):
            # disposition verbs carry the same LENIENT fence as claim_ready —
            # an evicted/departed worker cannot commit a disposition, while the
            # N=0 OR-branch keeps runs with no registered workers unfenced.
            # Correlated on the row's own run_id (the work_item_id predicate
            # already pins the row).
            predicates.append(claim_verb_fence_clause(worker_id=fenced_worker_id, run_id=token_work_items_table.c.run_id))
        with begin_write(self._engine) as conn:
            before = (
                conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id))
                .mappings()
                .one_or_none()
            )
            result = conn.execute(update(token_work_items_table).where(and_(*predicates)).values(**update_values))
            if result.rowcount != 1:
                if fenced_worker_id is not None and before is not None:
                    # The base predicates may still match — then the membership
                    # fence was the blocker. Mirror claim_ready's re-probe:
                    # a registered-but-not-active worker raises the canonical
                    # eviction signal (zero mutation) instead of the generic
                    # audit-integrity crash.
                    base_predicates_match = (
                        before["status"] in expected_status_values
                        and before["pending_sink_name"] is None
                        and (expected_lease_owner is None or before["lease_owner"] == expected_lease_owner)
                    )
                    if base_predicates_match:
                        worker_status = conn.execute(
                            select(run_workers_table.c.status)
                            .where(run_workers_table.c.worker_id == fenced_worker_id)
                            .where(run_workers_table.c.run_id == before["run_id"])
                        ).scalar()
                        if worker_status is not None and str(worker_status) != "active":
                            raise RunWorkerEvictedError(worker_id=fenced_worker_id, run_id=str(before["run_id"]))
                actual = (
                    conn.execute(
                        select(
                            token_work_items_table.c.status,
                            token_work_items_table.c.lease_owner,
                            token_work_items_table.c.pending_sink_name,
                        ).where(token_work_items_table.c.work_item_id == work_item_id)
                    )
                    .mappings()
                    .one_or_none()
                )
                if actual is None:
                    actual_message = "missing"
                else:
                    actual_subtype = "transform" if actual["pending_sink_name"] is None else "sink-redrive"
                    actual_message = (
                        f"actual status {actual['status']}, actual subtype {actual_subtype}, actual lease_owner {actual['lease_owner']!r}"
                    )
                expected_owner_message = "" if expected_lease_owner is None else f" and expected lease_owner {expected_lease_owner!r}"
                fence_message = "" if fenced_worker_id is None else f" under membership fence for worker {fenced_worker_id!r}"
                raise AuditIntegrityError(
                    f"Scheduler transition to {status.name!r} for work_item_id={work_item_id!r} "
                    f"affected {result.rowcount} rows; expected exactly 1 transform-lease row with expected status {expected_status_text}"
                    f"{expected_owner_message}{fence_message}. Caller assumed ownership but the row is missing or in an "
                    f"unexpected state ({actual_message})."
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
            self._events.record(
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
            if branch_loss is not None:
                # §E.5 record-then-notify: the durable loss record commits iff
                # this disposition commits (one transaction).
                record_coalesce_branch_loss(
                    conn,
                    run_id=before["run_id"],
                    coalesce_name=branch_loss.coalesce_name,
                    row_id=branch_loss.row_id,
                    branch_name=branch_loss.branch_name,
                    token_id=branch_loss.token_id,
                    reason=branch_loss.reason,
                    recorded_by=branch_loss.recorded_by,
                    now=now,
                )
            row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        return item_from_mapping(row)
