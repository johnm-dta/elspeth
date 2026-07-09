"""Lease lifecycle for scheduler work items: claim, heartbeat, recovery.

The CAS claim verbs (READY and PENDING_SINK), the single-timestamp lease
heartbeat, the liveness-aware expired-lease recovery sweep, and the
peer-lease probe. Extracted from ``TokenSchedulerRepository``
(filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import ColumnElement, and_, literal, or_, select, true, update
from sqlalchemy.engine import Connection, RowMapping

from elspeth.contracts.coordination import (
    DEFAULT_ITEM_STALL_BUDGET_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.errors import AuditIntegrityError, RunWorkerEvictedError, SchedulerLeaseLostError
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.core.landscape.database import WRITE_INTENT_OPTION, Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import record_coordination_event
from elspeth.core.landscape.scheduler.events import SchedulerEventStore
from elspeth.core.landscape.scheduler.fencing import fenced_or_plain_write
from elspeth.core.landscape.scheduler.work_items import item_from_mapping, work_item_id
from elspeth.core.landscape.schema import claim_verb_fence_clause, run_workers_table, token_work_items_table


class SchedulerLeaseRepository:
    """Claiming, heartbeating and recovering leases on token work items."""

    def __init__(self, engine: Tier1Engine, *, events: SchedulerEventStore) -> None:
        self._engine = engine
        self._events = events

    def claim_ready(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> TokenWorkItem | None:
        """Claim the next available READY work item for a bounded lease."""
        with begin_write(self._engine) as conn:
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
            claimed = self.claim_ready_row(
                conn,
                row=row,
                run_id=run_id,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
                now=now,
                # Membership fence: the public claim_ready verb applies the
                # claim_verb_fence_clause so an evicted/departed claimant
                # is refused with RunWorkerEvictedError (ADR-030 §G, slice 4).
                membership_fenced=True,
            )
            if claimed is None:
                return None
        return item_from_mapping(claimed)

    def claim_ready_row(
        self,
        conn: Connection,
        *,
        row: RowMapping,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
        membership_fenced: bool = False,
    ) -> RowMapping | None:
        """CAS update to claim a READY row under a lease.

        ``membership_fenced=True`` (set only by the public ``claim_ready``
        verb) adds the ``claim_verb_fence_clause`` to the UPDATE WHERE.
        That clause passes when this worker is active OR when no workers are
        registered for the run at all (N=0/unit-test mode). When workers ARE
        registered, an absent caller gets rowcount=0 with zero mutation and an
        evicted/departed caller gets rowcount=0 plus the re-probe below raises
        ``RunWorkerEvictedError``. Internal callers
        (``SchedulerQueueRepository.enqueue_ready_claimed_on``, ``ingest_row_with_initial_claim``) pass
        the default ``False`` because they operate inside a broader fenced
        transaction whose leadership CAS is the outer guard.
        """
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        where_clauses = and_(
            token_work_items_table.c.work_item_id == row["work_item_id"],
            token_work_items_table.c.run_id == run_id,
            token_work_items_table.c.status == TokenWorkStatus.READY.value,
            token_work_items_table.c.available_at <= now,
        )
        if membership_fenced:
            # Membership fence (ADR-030 §G, slice 4): the claimant must hold
            # an active run_workers row OR the run has no registered workers
            # at all (N=0 / unit-test mode — see claim_verb_fence_clause).
            # An absent caller is refused once any worker has registered for
            # the run; an evicted/departed caller is re-probed below and raises
            # RunWorkerEvictedError.
            where_clauses = and_(
                where_clauses,
                claim_verb_fence_clause(worker_id=lease_owner, run_id=run_id),
            )
        result = conn.execute(
            update(token_work_items_table)
            .where(where_clauses)
            .values(
                status=TokenWorkStatus.LEASED.value,
                lease_owner=lease_owner,
                lease_expires_at=lease_expires_at,
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            if membership_fenced:
                # Distinguish "empty queue" (row raced away — legitimate None)
                # from "membership fence failed" (this worker is no longer active).
                # The SELECT above found the row; re-probe the fence only.
                still_ready = conn.execute(
                    select(token_work_items_table.c.work_item_id)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.READY.value)
                ).first()
                if still_ready is not None:
                    # Row is still READY but the UPDATE matched 0 rows.  Two cases:
                    #
                    # (a) Worker EXISTS with status != 'active' (evicted/departed):
                    #     the worker was registered, then evicted — raise
                    #     RunWorkerEvictedError (the canonical multi-worker signal).
                    #
                    # (b) Worker is ABSENT while this run has registered workers:
                    #     in production every worker registers before claiming;
                    #     an absent worker means the caller bypassed registration.
                    #     Return None rather than raising; the fence clause in
                    #     the UPDATE WHERE prevents any data mutation.
                    worker_status = conn.execute(
                        select(run_workers_table.c.status)
                        .where(run_workers_table.c.worker_id == lease_owner)
                        .where(run_workers_table.c.run_id == run_id)
                    ).scalar()
                    if worker_status is not None and str(worker_status) != "active":
                        raise RunWorkerEvictedError(worker_id=lease_owner, run_id=run_id)
            return None
        if result.rowcount != 1:
            raise AuditIntegrityError(
                f"Scheduler claim_ready lost race on run_id={run_id!r} "
                f"work_item_id={row['work_item_id']!r}: SELECT saw READY but UPDATE matched "
                f"{result.rowcount} rows for lease_owner={lease_owner!r}. Concurrent claim by another worker."
            )
        self._events.record(
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
        with begin_write(self._engine) as conn:
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
                        # Membership fence (ADR-030 §G, slice 4): same discipline
                        # as claim_ready — absent claimants are refused once the
                        # run has any registered worker; evicted/departed claimants
                        # are re-probed below and raise RunWorkerEvictedError.
                        # Uses the lenient variant (passes only in N=0/unit-test mode).
                        claim_verb_fence_clause(worker_id=lease_owner, run_id=run_id),
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
                # Distinguish "row raced away" from "membership fence failed".
                # Same absent-vs-evicted logic as _claim_ready_row: absent
                # unregistered claimants return None with zero mutation, while
                # registered-then-evicted claimants raise the multi-worker signal.
                still_pending = conn.execute(
                    select(token_work_items_table.c.work_item_id)
                    .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == TokenWorkStatus.PENDING_SINK.value)
                ).first()
                if still_pending is not None:
                    worker_status = conn.execute(
                        select(run_workers_table.c.status)
                        .where(run_workers_table.c.worker_id == lease_owner)
                        .where(run_workers_table.c.run_id == run_id)
                    ).scalar()
                    if worker_status is not None and str(worker_status) != "active":
                        raise RunWorkerEvictedError(worker_id=lease_owner, run_id=run_id)
                return None
            if result.rowcount != 1:
                raise AuditIntegrityError(
                    f"Scheduler claim_pending_sink lost race on run_id={run_id!r} work_item_id={row['work_item_id']!r}: "
                    f"UPDATE matched {result.rowcount} rows for lease_owner={lease_owner!r}."
                )
            self._events.record(
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
        return item_from_mapping(claimed)

    def recover_expired_leases(
        self,
        *,
        run_id: str,
        now: datetime,
        caller_owner: str,
        coordination_token: CoordinationToken | None = None,
        grace_seconds: float = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        stall_budget_seconds: float = DEFAULT_ITEM_STALL_BUDGET_SECONDS,
    ) -> int:
        """Return expired LEASED work to READY for retry by another worker.

        ``coordination_token`` (ADR-030 §C.4 row 8): the repair sweep is a
        LEADER verb — the verify-and-extend epoch fence runs as the first
        statement of the transaction so a deposed leader cannot rotate
        attempts under the new one.

        **Liveness-aware reap (slice 4, §A.5/§C.1):** An expired item lease is
        reapable ONLY under one of two conditions:

        1. ``owner_registry_dead`` — the ``run_workers`` row for the lease
           owner is ABSENT, has status IN ('evicted','departed'), or has
           ``status='active'`` AND ``heartbeat_expires_at < now - grace``.
           All three arms collapse into a single NOT EXISTS predicate so
           absent, non-active, and active-but-stale rows all match (no row
           → no EXISTS match → dead). This is the primary protection: a
           registered-live owner's expired item lease is LEFT LEASED so the
           owner's next ``heartbeat_lease`` revives it. Long LLM calls no
           longer become reap targets on every maintenance sweep.

        2. ``lease_stalled`` — the owner IS registry-live BUT the lease has
           been expired for longer than ``stall_budget_seconds`` (default
           2x scheduler_lease). A genuinely wedged worker (heartbeat thread
           alive, drain loop stuck) triggers this arm. The rotation emits a
           ``worker_stalled`` coordination event in the SAME transaction,
           naming the owner and the reaped item (§A.5 :145).

        **UNFENCED/TEST ARM CONTRACT (re-pin):** when ``coordination_token``
        is None the lease_owner has no registry row → ``owner_registry_dead``
        is TRUE for every row → reap behaves exactly as the pre-slice-4 form.
        Preserves all slice 1-3 tests that call without a token.

        **N=1 IMPROVEMENT (re-pin):** a single live leader beating its own
        seat+row keeps ``owner_registry_dead`` FALSE for its own in-flight
        items, so a long LLM call's expired item lease is NO LONGER reapable
        by a racing recover sweep. The self-sweep guard (lease_owner_not_caller)
        is unchanged; the NEW protection is against a peer/successor sweep
        reaping a registered-live owner.

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
        #
        # 3. Liveness-aware gate (slice 4, §A.5/§C.1): only reap when the
        #    owner is registry-DEAD or far past the stall budget.
        #    NOT EXISTS(active + fresh heartbeat row) folds ABSENT/non-active/
        #    stale into a single correlated subquery — portable and keeps the
        #    per-row CAS shape unchanged.
        lease_owner_not_caller = or_(
            token_work_items_table.c.lease_owner.is_(None),
            token_work_items_table.c.lease_owner != caller_owner,
        )

        # §A.5/§C.1: liveness-aware gate — ONLY applied when
        # ``coordination_token`` is not None (the fenced/leader path).
        #
        # UNFENCED/TEST ARM (coordination_token is None): the caller operates
        # outside the coordination substrate.  This path is used by:
        #   (a) resume sweeps (recover expired leases before acquiring
        #       leadership — no token yet);
        #   (b) integration tests that inject a MockClock with timestamps in a
        #       different epoch than the real-clock ``heartbeat_expires_at`` rows
        #       written by ``begin_run``/``worker_heartbeat``.
        #   (c) direct repository-level construction in tests with no
        #       run_workers rows at all.
        # In all cases, the time-domain mismatch between the ``now`` argument
        # and the stored ``heartbeat_expires_at`` values would cause the liveness
        # predicate to fire spuriously.  We therefore SKIP the predicate
        # entirely on the unfenced path and reap all expired non-caller leases
        # unconditionally — the pre-slice-4 ("legacy") behavior.
        #
        # FENCED PATH (coordination_token is not None): apply the full
        # liveness-aware gate.  Arms of owner_registry_dead:
        #   (a) absent row → no EXISTS match → dead;
        #   (b) status in ('evicted','departed') → status!='active' → dead;
        #   (c) status='active' + stale heartbeat → heartbeat<now-grace → dead;
        #   (d) status='active' + fresh heartbeat → MATCH → LIVE → excluded.
        if coordination_token is not None:
            _grace_threshold = now - timedelta(seconds=grace_seconds)
            owner_registry_dead: ColumnElement[bool] = ~(
                select(run_workers_table.c.worker_id)
                .where(
                    run_workers_table.c.worker_id == token_work_items_table.c.lease_owner,
                    run_workers_table.c.status == "active",
                    run_workers_table.c.heartbeat_expires_at >= _grace_threshold,
                )
                .exists()
            )
            # §A.5: stall arm — owner IS registry-live but drain loop is wedged.
            # The item has been expired far past the stall budget, so we reap it
            # and emit worker_stalled in the same transaction.
            _stall_threshold = now - timedelta(seconds=stall_budget_seconds)
            lease_stalled: ColumnElement[bool] = token_work_items_table.c.lease_expires_at < _stall_threshold
            reap_eligible: ColumnElement[bool] = or_(owner_registry_dead, lease_stalled)
        else:
            # Legacy/unfenced arm: unconditionally reap all expired non-caller
            # leases; liveness predicate skipped to avoid epoch mismatch.
            owner_registry_dead = literal(True)
            lease_stalled = literal(False)
            reap_eligible = true()

        with fenced_or_plain_write(self._engine, coordination_token=coordination_token, now=now, verb="recover_expired_leases") as conn:
            expired_rows = conn.execute(
                select(
                    token_work_items_table,
                    owner_registry_dead.label("owner_is_dead"),
                )
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
                .where(token_work_items_table.c.lease_expires_at < now)
                .where(lease_owner_not_caller)
                .where(reap_eligible)
                .order_by(
                    token_work_items_table.c.ingest_sequence,
                    token_work_items_table.c.step_index,
                    # Stable last-resort tiebreaker for cross-source same-tick
                    # collisions (filigree elspeth-6cb89db535, G3 M1).
                    token_work_items_table.c.work_item_id,
                )
            ).mappings()
            # Materialise into a list: the connection must remain open for the
            # per-row UPDATEs below, but lazy iteration over the SELECT cursor
            # would be invalidated by the first write on the same connection
            # (SQLite WAL mode). Collect all eligible rows first, then update.
            expired = list(expired_rows)

            recovered = 0
            for row in expired:
                pending_sink_name = row["pending_sink_name"]
                next_attempt = row["attempt"] if pending_sink_name is not None else row["attempt"] + 1
                next_work_item_id = (
                    row["work_item_id"]
                    if pending_sink_name is not None
                    else work_item_id(run_id, row["token_id"], row["node_id"], next_attempt)
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
                    .where(reap_eligible)
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
                    self._events.record(
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
                    # §A.5 :145: emit worker_stalled ONLY when the owner is
                    # registry-LIVE but stalled (stall arm, not dead-owner arm).
                    # ``owner_is_dead`` is materialised per-row by the SELECT
                    # (correlated EXISTS evaluated at query time). A dead-owner
                    # reap is already explained by the worker_evict event on
                    # that path; emitting worker_stalled for it is redundant
                    # and violates the invariant (every non-evicted rotation
                    # is explained by stalled, not double-evented).
                    owner_is_dead = bool(row["owner_is_dead"])
                    if not owner_is_dead and row["lease_owner"] is not None and coordination_token is not None:
                        record_coordination_event(
                            conn,
                            run_id=run_id,
                            event_type="worker_stalled",
                            worker_id=row["lease_owner"],
                            leader_epoch=coordination_token.leader_epoch,
                            recorded_at=now,
                            context={
                                "reaped_work_item_id": row["work_item_id"],
                                "previous_work_item_id": row["work_item_id"],
                                "reason": "item_stall_budget",
                            },
                        )
                recovered += result.rowcount
        return recovered

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
            # Write intent must be declared BEFORE conn.begin() — the begin
            # event reads the execution option to choose BEGIN IMMEDIATE.
            conn.execution_options(**{WRITE_INTENT_OPTION: True})
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
                        self._events.record(
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
                        recovery_event = self._events.recovery_event_for_previous_work_item(
                            conn,
                            run_id=run_id,
                            previous_work_item_id=work_item_id,
                        )
                        if recovery_event is not None:
                            self._events.record(
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
