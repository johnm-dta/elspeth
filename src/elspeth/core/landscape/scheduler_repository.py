"""Durable token scheduler repository — compatibility facade.

The v1 scheduler is intentionally embedded-database backed and single-process
friendly. It records enough state for a run to resume without rediscovering
source rows: ready work, active leases, delayed retry availability, queue or
barrier blocking keys, and terminal/failed states.

The behaviour lives in the cohesive components under
:mod:`elspeth.core.landscape.scheduler` (filigree elspeth-ef9c36d767):
queueing (:class:`SchedulerQueueRepository`), leasing
(:class:`SchedulerLeaseRepository`), dispositions
(:class:`SchedulerDispositionRepository`), the barrier journal
(:class:`BarrierJournalRepository`), the branch-loss ledger
(:class:`CoalesceBranchLossRepository`), scheduler events
(:class:`SchedulerEventStore`), read models (:class:`SchedulerReadModel`)
and the pure payload codec. :class:`TokenSchedulerRepository` composes them
behind the historical surface so call sites can migrate incrementally —
new code should prefer the component attributes (``.queue``, ``.leases``,
``.dispositions``, ``.barriers``, ``.branch_losses``, ``.reads``,
``.events``) over the flat delegators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.coordination import (
    DEFAULT_ITEM_STALL_BUDGET_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.scheduler import (
    BarrierEmission,
    BatchMembershipSpec,
    BlockedPendingSinkHandoff,
    BranchLossSpec,
    BufferedOutcomeSpec,
    SchedulerEventType,
    TokenWorkItem,
    TokenWorkStatus,
)
from elspeth.core.landscape.database import Tier1Engine, verify_sqlite_tier1_pragmas
from elspeth.core.landscape.scheduler import (
    BarrierAdoptionResult,
    BarrierJournalRepository,
    CoalesceBranchLoss,
    CoalesceBranchLossRepository,
    SchedulerDispositionRepository,
    SchedulerEventStore,
    SchedulerLeaseRepository,
    SchedulerQueueRepository,
    SchedulerReadModel,
    record_coalesce_branch_loss,
    token_from_journal_item,
)
from elspeth.core.landscape.scheduler.payload_codec import deserialize_row_payload, serialize_row_payload
from elspeth.core.landscape.scheduler.work_items import ready_work_item_values

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from sqlalchemy.engine import Connection

    from elspeth.contracts.audit import Row, Token
    from elspeth.contracts.schema_contract import PipelineRow

__all__ = [
    "BarrierAdoptionResult",
    "BarrierEmission",
    "BatchMembershipSpec",
    "BlockedPendingSinkHandoff",
    "BranchLossSpec",
    "BufferedOutcomeSpec",
    "CoalesceBranchLoss",
    "SchedulerEventType",
    "TokenSchedulerRepository",
    "TokenWorkItem",
    "TokenWorkStatus",
    "record_coalesce_branch_loss",
    "token_from_journal_item",
]


class TokenSchedulerRepository:
    """Persistence boundary for token scheduler work items.

    Compatibility facade over the scheduler components: every historical
    verb delegates to exactly one component, and the components share ONE
    :class:`SchedulerEventStore` so the audit event plane stays a single
    seam. Construction runs the Tier-1 PRAGMA probe once, here.
    """

    def __init__(self, engine: Tier1Engine) -> None:
        # Runtime SQLite PRAGMA probe — defence in depth against a caller that
        # slips a bare SQLite engine past the type checker. Non-SQLite Tier-1
        # engines skip this SQLite-only syntax. Probed ONCE at the composition
        # root; the components trust the branded engine handed to them.
        verify_sqlite_tier1_pragmas(engine, owner="TokenSchedulerRepository")
        self._engine = engine
        self.events = SchedulerEventStore()
        self.leases = SchedulerLeaseRepository(engine, events=self.events)
        self.queue = SchedulerQueueRepository(engine, events=self.events, leases=self.leases)
        self.dispositions = SchedulerDispositionRepository(engine, events=self.events)
        self.barriers = BarrierJournalRepository(engine, events=self.events)
        self.branch_losses = CoalesceBranchLossRepository(engine)
        self.reads = SchedulerReadModel(engine)

    # ------------------------------------------------------------------
    # Queue intake (SchedulerQueueRepository)
    # ------------------------------------------------------------------

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
        """Persist a READY token continuation (see :meth:`SchedulerQueueRepository.enqueue_ready`)."""
        return self.queue.enqueue_ready(
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
            worker_id=worker_id,
        )

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
        """Persist and claim READY work for an active registered worker."""
        return self.queue.enqueue_ready_claimed(
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

    def enqueue_ready_claimed_legacy_unfenced(
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
        """Compatibility enqueue-and-claim for N=0 repository test fixtures."""
        return self.queue.enqueue_ready_claimed_legacy_unfenced(
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
        """Fenced leader INGEST (see :meth:`SchedulerQueueRepository.ingest_row_with_initial_claim`)."""
        return self.queue.ingest_row_with_initial_claim(
            coordination_token=coordination_token,
            now=now,
            insert_row_and_token=insert_row_and_token,
            token_id=token_id,
            row_id=row_id,
            node_id=node_id,
            step_index=step_index,
            ingest_sequence=ingest_sequence,
            row_payload_json=row_payload_json,
            lease_owner=lease_owner,
            lease_seconds=lease_seconds,
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
        """Historical test seam over :func:`ready_work_item_values`."""
        return ready_work_item_values(
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

    # ------------------------------------------------------------------
    # Payload codec (pure functions)
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_row_payload(row: PipelineRow) -> str:
        """Serialize a token row for durable resume (see :func:`serialize_row_payload`)."""
        return serialize_row_payload(row)

    @staticmethod
    def deserialize_row_payload(row_payload_json: str) -> PipelineRow:
        """Restore a scheduler row payload written by :meth:`serialize_row_payload`."""
        return deserialize_row_payload(row_payload_json)

    # ------------------------------------------------------------------
    # Leases (SchedulerLeaseRepository)
    # ------------------------------------------------------------------

    def claim_ready(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> TokenWorkItem | None:
        """Claim the next available READY work item for a bounded lease."""
        return self.leases.claim_ready(run_id=run_id, lease_owner=lease_owner, lease_seconds=lease_seconds, now=now)

    def claim_pending_sink(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> TokenWorkItem | None:
        """Claim a sink-bound token whose transform work is already durable."""
        return self.leases.claim_pending_sink(run_id=run_id, lease_owner=lease_owner, lease_seconds=lease_seconds, now=now)

    def recover_expired_leases(
        self,
        *,
        now: datetime,
        coordination_token: CoordinationToken,
        grace_seconds: float = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        stall_budget_seconds: float = DEFAULT_ITEM_STALL_BUDGET_SECONDS,
    ) -> int:
        """Return expired LEASED work to READY (see :meth:`SchedulerLeaseRepository.recover_expired_leases`)."""
        return self.leases.recover_expired_leases(
            now=now,
            coordination_token=coordination_token,
            grace_seconds=grace_seconds,
            stall_budget_seconds=stall_budget_seconds,
        )

    def recover_expired_leases_legacy_unfenced(
        self,
        *,
        run_id: str,
        now: datetime,
        caller_owner: str,
    ) -> int:
        """Recover expired leases for pre-coordination direct harnesses only."""
        return self.leases.recover_expired_leases_legacy_unfenced(
            run_id=run_id,
            now=now,
            caller_owner=caller_owner,
        )

    def heartbeat_lease(
        self,
        *,
        run_id: str,
        work_item_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
        membership_fenced: bool,
    ) -> datetime:
        """Extend a held lease (see :meth:`SchedulerLeaseRepository.heartbeat_lease`)."""
        return self.leases.heartbeat_lease(
            run_id=run_id,
            work_item_id=work_item_id,
            lease_owner=lease_owner,
            lease_seconds=lease_seconds,
            now=now,
            membership_fenced=membership_fenced,
        )

    def peer_active_leases(
        self,
        *,
        run_id: str,
        caller_owner: str,
        now: datetime,
    ) -> tuple[str, ...]:
        """Distinct lease_owners of unexpired LEASED rows held by peers."""
        return self.leases.peer_active_leases(run_id=run_id, caller_owner=caller_owner, now=now)

    # ------------------------------------------------------------------
    # Dispositions (SchedulerDispositionRepository)
    # ------------------------------------------------------------------

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
        """Move an item to BLOCKED at a queue or barrier."""
        return self.dispositions.mark_blocked(
            work_item_id=work_item_id,
            queue_key=queue_key,
            barrier_key=barrier_key,
            now=now,
            expected_lease_owner=expected_lease_owner,
            worker_id=worker_id,
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
        """Mark a leased work item terminal."""
        return self.dispositions.mark_terminal(
            work_item_id=work_item_id,
            now=now,
            expected_lease_owner=expected_lease_owner,
            branch_loss=branch_loss,
            worker_id=worker_id,
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
        """Mark a leased work item failed after retries are exhausted."""
        return self.dispositions.mark_failed(
            work_item_id=work_item_id,
            now=now,
            expected_lease_owner=expected_lease_owner,
            branch_loss=branch_loss,
            worker_id=worker_id,
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
        """Move a claimed item to a durable sink handoff state."""
        return self.dispositions.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=row_payload_json,
            sink_name=sink_name,
            outcome=outcome,
            path=path,
            error_hash=error_hash,
            error_message=error_message,
            now=now,
            expected_lease_owner=expected_lease_owner,
            branch_loss=branch_loss,
            worker_id=worker_id,
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
        """Terminalize pending sink scheduler work after token outcome durability."""
        return self.dispositions.mark_pending_sink_terminal(
            run_id=run_id,
            token_id=token_id,
            now=now,
            expected_lease_owner=expected_lease_owner,
            coordination_token=coordination_token,
        )

    def mark_pending_sink_terminal_many(
        self,
        *,
        run_id: str,
        token_ids: tuple[str, ...],
        now: datetime,
        expected_lease_owner: str,
        coordination_token: CoordinationToken,
    ) -> int:
        """Terminalize sink-bound scheduler work for a durable sink batch."""
        return self.dispositions.mark_pending_sink_terminal_many(
            run_id=run_id,
            token_ids=token_ids,
            now=now,
            expected_lease_owner=expected_lease_owner,
            coordination_token=coordination_token,
        )

    def terminalize_pending_sinks_with_terminal_outcomes(
        self,
        *,
        run_id: str,
        now: datetime,
        caller_owner: str,
        coordination_token: CoordinationToken,
    ) -> int:
        """Repair PENDING_SINK work whose terminal token outcome is already durable."""
        return self.dispositions.terminalize_pending_sinks_with_terminal_outcomes(
            run_id=run_id,
            now=now,
            caller_owner=caller_owner,
            coordination_token=coordination_token,
        )

    # ------------------------------------------------------------------
    # Barrier journal (BarrierJournalRepository)
    # ------------------------------------------------------------------

    def complete_barrier(
        self,
        *,
        run_id: str,
        barrier_key: str,
        consumed_token_ids: Sequence[str],
        emitted_pending_sink: Sequence[BarrierEmission],
        emitted_ready: Sequence[BarrierEmission],
        now: datetime,
        require_exhaustive_release: bool = True,
        scope_row_id: str | None = None,
        intake_snapshot_token_ids: frozenset[str] | None = None,
        release_context: Mapping[str, object] | None = None,
        coordination_token: CoordinationToken,
        pending_sink_lease_owner: str | None = None,
        branch_losses: Sequence[BranchLossSpec] = (),
    ) -> int:
        """Complete a barrier atomically (see :meth:`BarrierJournalRepository.complete_barrier`)."""
        return self.barriers.complete_barrier(
            run_id=run_id,
            barrier_key=barrier_key,
            consumed_token_ids=consumed_token_ids,
            emitted_pending_sink=emitted_pending_sink,
            emitted_ready=emitted_ready,
            now=now,
            require_exhaustive_release=require_exhaustive_release,
            scope_row_id=scope_row_id,
            intake_snapshot_token_ids=intake_snapshot_token_ids,
            release_context=release_context,
            coordination_token=coordination_token,
            pending_sink_lease_owner=pending_sink_lease_owner,
            branch_losses=branch_losses,
        )

    def mark_blocked_barrier_pending_sink_many(
        self,
        *,
        run_id: str,
        barrier_key: str,
        handoffs: Mapping[str, BlockedPendingSinkHandoff],
        now: datetime,
        coordination_token: CoordinationToken,
        pending_sink_lease_owner: str | None = None,
    ) -> int:
        """Move BLOCKED barrier work to PENDING_SINK before external sink writes."""
        return self.barriers.mark_blocked_barrier_pending_sink_many(
            run_id=run_id,
            barrier_key=barrier_key,
            handoffs=handoffs,
            now=now,
            coordination_token=coordination_token,
            pending_sink_lease_owner=pending_sink_lease_owner,
        )

    def mark_blocked_barrier_terminal(
        self,
        *,
        run_id: str,
        barrier_key: str,
        token_ids: tuple[str, ...],
        now: datetime,
        coordination_token: CoordinationToken,
        release_context: Mapping[str, object] | None = None,
    ) -> int:
        """Mark BLOCKED work consumed by a resolved barrier as terminal."""
        return self.barriers.mark_blocked_barrier_terminal(
            run_id=run_id,
            barrier_key=barrier_key,
            token_ids=token_ids,
            now=now,
            coordination_token=coordination_token,
            release_context=release_context,
        )

    def adopt_blocked_barrier_item(
        self,
        *,
        run_id: str,
        work_item_id: str,
        token_id: str,
        barrier_key: str,
        membership: BatchMembershipSpec | None,
        buffered_outcome: BufferedOutcomeSpec | None,
        now: datetime,
        coordination_token: CoordinationToken,
    ) -> BarrierAdoptionResult:
        """Fenced, backdated adoption of one durable BLOCKED barrier hold (SE.2)."""
        return self.barriers.adopt_blocked_barrier_item(
            run_id=run_id,
            work_item_id=work_item_id,
            token_id=token_id,
            barrier_key=barrier_key,
            membership=membership,
            buffered_outcome=buffered_outcome,
            now=now,
            coordination_token=coordination_token,
        )

    def list_blocked_barrier_items(self, *, run_id: str) -> list[TokenWorkItem]:
        """Return BLOCKED barrier holds for a run in deterministic order."""
        return self.barriers.list_blocked_barrier_items(run_id=run_id)

    def blocked_barrier_token_ids(self, *, run_id: str) -> frozenset[str]:
        """Return token IDs currently held by journal BLOCKED barrier rows."""
        return self.barriers.blocked_barrier_token_ids(run_id=run_id)

    def count_blocked_barrier_items(self, *, run_id: str) -> int:
        """Count journal BLOCKED barrier holds for a run."""
        return self.barriers.count_blocked_barrier_items(run_id=run_id)

    def list_pending_blocked_barrier_items(self, *, run_id: str) -> list[TokenWorkItem]:
        """Return intake-pending BLOCKED barrier holds (``barrier_adopted_epoch IS NULL``)."""
        return self.barriers.list_pending_blocked_barrier_items(run_id=run_id)

    def reset_adoption_marker_to_pending(
        self,
        *,
        work_item_ids: Sequence[str],
        run_id: str,
    ) -> int:
        """Reset ``barrier_adopted_epoch`` to NULL for crash-window BLOCKED rows."""
        return self.barriers.reset_adoption_marker_to_pending(work_item_ids=work_item_ids, run_id=run_id)

    # ------------------------------------------------------------------
    # Branch-loss ledger (CoalesceBranchLossRepository)
    # ------------------------------------------------------------------

    def list_unadopted_coalesce_branch_losses(self, *, run_id: str) -> list[CoalesceBranchLoss]:
        """Branch-loss rows not yet replayed into leader memory (SE.5 intake read)."""
        return self.branch_losses.list_unadopted_coalesce_branch_losses(run_id=run_id)

    def list_coalesce_branch_losses(self, *, run_id: str) -> list[CoalesceBranchLoss]:
        """ALL branch-loss rows, adopted or not (SE.4 takeover restore read)."""
        return self.branch_losses.list_coalesce_branch_losses(run_id=run_id)

    def adopt_coalesce_branch_losses(
        self,
        *,
        run_id: str,
        loss_ids: Sequence[str],
        now: datetime,
        coordination_token: CoordinationToken,
    ) -> int:
        """Fenced replay-cursor mark: ``adopted_epoch NULL -> epoch`` (SE.5)."""
        return self.branch_losses.adopt_coalesce_branch_losses(
            run_id=run_id,
            loss_ids=loss_ids,
            now=now,
            coordination_token=coordination_token,
        )

    # ------------------------------------------------------------------
    # Read models (SchedulerReadModel)
    # ------------------------------------------------------------------

    def count_ready_in_set(self, *, run_id: str, work_item_ids: Sequence[str]) -> int:
        """Count how many of the given work item IDs are in READY status."""
        return self.reads.count_ready_in_set(run_id=run_id, work_item_ids=work_item_ids)

    def count_failed_in_set(self, *, run_id: str, work_item_ids: Sequence[str]) -> int:
        """Count how many of the given work item IDs are in FAILED status."""
        return self.reads.count_failed_in_set(run_id=run_id, work_item_ids=work_item_ids)

    def has_peer_owned_work(self, *, run_id: str, caller_owner: str) -> bool:
        """Return True if any non-terminal row is owned by a DIFFERENT worker."""
        return self.reads.has_peer_owned_work(run_id=run_id, caller_owner=caller_owner)

    def count_active_work(self, *, run_id: str) -> int:
        """Count non-terminal scheduler work for a run."""
        return self.reads.count_active_work(run_id=run_id)

    def active_row_ids(self, *, run_id: str) -> frozenset[str]:
        """Return row IDs represented by non-terminal scheduler work."""
        return self.reads.active_row_ids(run_id=run_id)

    def count_unquiesced_work(self, *, run_id: str) -> int:
        """Count work items still able to deposit new barrier arrivals (SD step 2)."""
        return self.reads.count_unquiesced_work(run_id=run_id)

    def summarize_unquiesced_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize SD step-2 unquiesced work for invariant diagnostics."""
        return self.reads.summarize_unquiesced_work(run_id=run_id)

    def count_unresolved_work(self, *, run_id: str) -> int:
        """Count scheduler work not yet resolved into a durable sink handoff."""
        return self.reads.count_unresolved_work(run_id=run_id)

    def summarize_unresolved_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize unresolved work grouped by status and blocking keys."""
        return self.reads.summarize_unresolved_work(run_id=run_id)

    def summarize_active_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize active work grouped by status and blocking keys."""
        return self.reads.summarize_active_work(run_id=run_id)
