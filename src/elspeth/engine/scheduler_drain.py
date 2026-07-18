"""Durable scheduler claim/drain subsystem extracted from RowProcessor.

The SchedulerDrainCoordinator owns the claim -> process -> disposition loop
over the durable token scheduler (elspeth-c49f33d6e4, component 3): lease
claims and their heartbeat, the per-iteration journal-first barrier intake
ordering, the four disposition arms (BLOCKED / PENDING_SINK / FAILED /
TERMINAL) with their membership-fence identities, pending-sink crash
recovery, scheduler maintenance cadence, and READY work-item persistence.

Boundary rules (mirrors engine/barrier_coordination.py):

- The coordinator holds the drain-only mutable state
  (``_active_claim_work_item_id`` / ``_last_heartbeat_at`` /
  ``_scheduler_drains_since_maintenance``) and SHARES two mutable objects by
  reference with the processor and the barrier subsystem:
  ``live_barrier_holds`` (also injected into BarrierIntakeCoordinator) and
  ``pending_branch_losses`` (appended by the processor's coalesce loss path,
  consumed by both this drain's dispositions and the flush path). Copying
  either would silently break §E.2 intake parity / §E.5 loss-riding
  transactions.
- Token processing itself stays behind the ``SchedulerDrainHost`` port and is
  resolved at CALL time (``self._processor._process_single_token``), not
  bound at construction: the traversal engine is component (4) of the same
  split and the existing test net patches ``_process_single_token`` on the
  processor instance.
- RowProcessor keeps thin delegates for the load-bearing private names
  (``_drain_scheduler_claims`` is called by the ADR-030 invariant-guard
  tests). orchestrator/follower.py enters through the PUBLIC follower
  surface (``RowProcessor.drain_follower_ready_work``), which fences on the
  explicit :class:`ProcessorMode` stored at construction.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from elspeth.contracts import RowResult, TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    AuditIntegrityError,
    OrchestrationInvariantError,
    RunWorkerEvictedError,
    SchedulerLeaseLostError,
)
from elspeth.contracts.results import FailureInfo
from elspeth.contracts.scheduler import BranchLossSpec, TokenWorkItem, TokenWorkStatus
from elspeth.engine._error_hash import compute_error_hash

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from typing import TypeIs

    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.types import CoalesceName, NodeID
    from elspeth.core.landscape.execution_repository import ExecutionRepository
    from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
    from elspeth.core.landscape.scheduler import BarrierRestoreReadModel
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
    from elspeth.engine.barrier_coordination import _LiveBarrierHold
    from elspeth.engine.clock import Clock
    from elspeth.engine.scheduler_work_codec import SchedulerWorkCodec
    from elspeth.engine.work_items import WorkItem

# Iteration guard to prevent infinite loops from bugs.
# This counts dequeued work items, so it must exceed the largest supported
# single-row fan-out; otherwise legal expansion trips the safety valve before
# the final child runs.
MAX_WORK_QUEUE_ITERATIONS = 100_000
SCHEDULER_MAINTENANCE_INTERVAL = 64
logger = logging.getLogger(__name__)


class ProcessorMode(enum.Enum):
    """Explicit processor role for the mode-gated drain policy (elspeth-577179bba1).

    Replaces the old triple-None inference (``coordination_token is None and
    run_coordination is None and scheduler_lease_owner_registered``) that
    reverse-derived "is this a follower" from constructor sentinels — an
    invisible-default hazard: any future change to the None-sentinel meaning
    would have silently flipped a follower into leader maintenance.

    LEADER (the default) is the production maintenance role. Lease recovery
    always requires ``coordination_token`` and uses the strict fenced API;
    ``run_coordination`` presence only controls whether the §C.2 dead-member
    eviction sweep precedes recovery. Pre-coordination repository and crash-
    image harnesses bypass ``ProcessorMode`` and call the explicitly named
    ``recover_expired_leases_legacy_unfenced`` adapter directly.

    FOLLOWER is drain-only (ADR-030 §C.3): ``claim_ready`` only, never
    pending-sink recovery, never the §C.2 housekeeping sweep, never
    ``recover_expired_leases`` — lease recovery and eviction are the
    leader's responsibility. RowProcessor validates the FOLLOWER wiring
    invariants fail-closed at construction.
    """

    LEADER = "leader"
    FOLLOWER = "follower"


def _is_result_tuple(result: RowResult | tuple[RowResult, ...]) -> TypeIs[tuple[RowResult, ...]]:
    """Narrow a scheduler result to its fan-out tuple form.

    A ``TypeIs`` guard (PEP 742) so mypy narrows BOTH branches: callers that
    fall through get ``RowResult``, not the unnarrowed union. Strict
    ``type() is`` on purpose — a tuple subclass here would be a framework bug.
    """
    return type(result) is tuple


def is_buffered_scheduler_result(result: RowResult | tuple[RowResult, ...] | None) -> bool:
    """Return whether a scheduler result is an active aggregation buffer."""
    if result is None:
        return False
    if _is_result_tuple(result):
        return bool(result) and all(item.outcome is None and item.path is TerminalPath.BUFFERED for item in result)
    return result.outcome is None and result.path is TerminalPath.BUFFERED


def scheduler_result_failed_claimed_token(result: RowResult | tuple[RowResult, ...] | None, claimed_token_id: str) -> bool:
    """Return whether the claimed scheduler token itself reached FAILURE."""
    if result is None:
        return False
    result_items = result if _is_result_tuple(result) else (result,)
    return any(item.token.token_id == claimed_token_id and item.outcome is TerminalOutcome.FAILURE for item in result_items)


def is_scheduler_sink_bound_result(result: RowResult) -> bool:
    """Return whether a result carries a sink-bound terminal for handoff."""
    return result.sink_name is not None and result.path in {
        TerminalPath.DEFAULT_FLOW,
        TerminalPath.GATE_ROUTED,
        TerminalPath.ON_ERROR_ROUTED,
        TerminalPath.COALESCED,
    }


def scheduler_sink_bound_result_for_claimed_token(
    result: RowResult | tuple[RowResult, ...] | None,
    claimed_token_id: str,
) -> RowResult | None:
    """Return the claimed token's sink-bound result, if any."""
    if result is None:
        return None
    result_items = result if _is_result_tuple(result) else (result,)
    for item in result_items:
        if item.token.token_id != claimed_token_id:
            continue
        if is_scheduler_sink_bound_result(item):
            return item
    return None


def with_scheduler_pending_sink_handoffs(
    result: RowResult | tuple[RowResult, ...] | None,
    token_ids: frozenset[str],
) -> RowResult | tuple[RowResult, ...]:
    """Mark every requested token result as scheduler pending-sink backed."""
    if result is None:
        raise OrchestrationInvariantError(
            f"Cannot mark scheduler pending-sink handoffs for token_ids={sorted(token_ids)!r}: result is None."
        )
    if not token_ids:
        return result
    if isinstance(result, tuple):
        tagged: list[RowResult] = []
        matched_token_ids: set[str] = set()
        for item in result:
            if item.token.token_id in token_ids:
                tagged.append(replace(item, scheduler_pending_sink=True))
                matched_token_ids.add(item.token.token_id)
            else:
                tagged.append(item)
        missing_token_ids = token_ids - matched_token_ids
        if missing_token_ids:
            raise OrchestrationInvariantError(
                f"Cannot mark scheduler pending-sink handoffs: token_ids={sorted(missing_token_ids)!r} were not present in tuple result."
            )
        return tuple(tagged)
    if result.token.token_id not in token_ids:
        raise OrchestrationInvariantError(
            "Cannot mark scheduler pending-sink handoff: "
            f"token_ids={sorted(token_ids)!r} do not include result token {result.token.token_id!r}."
        )
    return replace(result, scheduler_pending_sink=True)


def with_scheduler_pending_sink_handoff(
    result: RowResult | tuple[RowResult, ...] | None,
    claimed_token_id: str,
) -> RowResult | tuple[RowResult, ...]:
    """Mark only the claimed token result as scheduler pending-sink backed."""
    return with_scheduler_pending_sink_handoffs(result, frozenset((claimed_token_id,)))


def require_scheduler_sink_name(result: RowResult) -> str:
    """Return the sink name a sink-bound result must carry."""
    if result.sink_name is None:
        raise OrchestrationInvariantError(f"Scheduler sink-bound result missing sink_name for token {result.token.token_id!r}")
    return result.sink_name


def require_scheduler_outcome(result: RowResult) -> TerminalOutcome:
    """Return the terminal outcome a sink-bound result must carry."""
    if result.outcome is None:
        raise OrchestrationInvariantError(f"Scheduler sink-bound result missing terminal outcome for token {result.token.token_id!r}")
    return result.outcome


def scheduler_error_hash(result: RowResult) -> str | None:
    """Return the audited error hash for an ON_ERROR_ROUTED handoff."""
    if result.path is not TerminalPath.ON_ERROR_ROUTED:
        return None
    if result.error is None:
        raise OrchestrationInvariantError(f"Scheduler ON_ERROR_ROUTED result missing error for token {result.token.token_id!r}")
    return compute_error_hash(result.error.message, exception_type=result.error.exception_type)


def scheduler_error_message(result: RowResult) -> str | None:
    """Return the audited error message for an ON_ERROR_ROUTED handoff."""
    if result.path is not TerminalPath.ON_ERROR_ROUTED:
        return None
    if result.error is None:
        raise OrchestrationInvariantError(f"Scheduler ON_ERROR_ROUTED result missing error for token {result.token.token_id!r}")
    return result.error.message


class SchedulerDrainHost(Protocol):
    """The RowProcessor surface the drain drives, resolved at call time.

    Attribute lookup happens per call (``self._processor.<seam>``) so tests
    that patch these methods on the processor instance keep working, and so
    component (4) of the split (the token traversal engine) can replace
    ``_process_single_token`` without rewiring this coordinator.
    """

    def _process_single_token(
        self,
        token: TokenInfo,
        ctx: PluginContext,
        current_node_id: NodeID | None,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
        on_success_sink: str | None = None,
        attempt_offset: int = 0,
    ) -> tuple[RowResult | tuple[RowResult, ...] | None, list[WorkItem]]: ...

    def _run_barrier_intake_pass(self, ctx: PluginContext) -> tuple[list[RowResult], list[WorkItem]]: ...

    def resolve_sink_step(self) -> int: ...

    def _require_coordination_token(self) -> CoordinationToken: ...

    def _queue_key_for_blocked_item(self, item: WorkItem) -> str | None: ...

    def _barrier_key_for_blocked_item(self, item: WorkItem) -> str | None: ...


class SchedulerDrainCoordinator:
    """Owns the durable scheduler claim/drain loop for one RowProcessor."""

    def __init__(
        self,
        *,
        processor: SchedulerDrainHost,
        mode: ProcessorMode,
        run_id: str,
        scheduler: TokenSchedulerRepository,
        work_codec: SchedulerWorkCodec,
        execution: ExecutionRepository,
        barrier_restore_reads: BarrierRestoreReadModel | ExecutionRepository,
        clock: Clock,
        run_coordination: RunCoordinationRepository | None,
        coordination_token: CoordinationToken | None,
        scheduler_lease_owner: str,
        scheduler_lease_seconds: int,
        scheduler_heartbeat_seconds: int,
        scheduler_lease_owner_registered: bool,
        resume_checkpoint_id: str | None,
        live_barrier_holds: dict[str, _LiveBarrierHold],
        pending_branch_losses: list[BranchLossSpec],
    ) -> None:
        self._processor = processor
        # Explicit role decided at construction (elspeth-577179bba1): the
        # maintenance skip below keys on THIS, never on re-deriving
        # follower-ness from the coordination sentinels.
        self._mode = mode
        self._run_id = run_id
        self._scheduler = scheduler
        self._work_codec = work_codec
        self._execution = execution
        self._barrier_restore_reads = barrier_restore_reads
        self._clock = clock
        self._run_coordination = run_coordination
        self._coordination_token = coordination_token
        self._scheduler_lease_owner = scheduler_lease_owner
        self._scheduler_lease_seconds = scheduler_lease_seconds
        self._scheduler_heartbeat_seconds = scheduler_heartbeat_seconds
        self._scheduler_lease_owner_registered = scheduler_lease_owner_registered
        self._resume_checkpoint_id = resume_checkpoint_id
        # SHARED references (never copy): the live-token stash is written by
        # the processor's block-deciding producers and read by both the
        # barrier intake and this drain; the branch-loss stage is appended by
        # the processor's coalesce loss path and consumed by this drain's
        # dispositions and the processor's flush path.
        self._live_barrier_holds = live_barrier_holds
        self._pending_branch_losses = pending_branch_losses
        # Active scheduler claim state for in-loop heartbeat refresh
        # (ADR-026 RC6 multi-worker, filigree elspeth-ddde8144b6). These
        # fields are non-None only inside ``drain_claims`` between
        # ``claim_ready``/``claim_pending_sink`` and the terminal ``mark_*``.
        # ``_process_single_token`` calls the processor's
        # ``_heartbeat_active_claim`` delegate on each node-iteration
        # boundary so an alive-but-slow worker's lease does not expire under
        # a peer reaper. The drain is single-threaded per row, so this
        # instance state has no concurrent access.
        self._active_claim_work_item_id: str | None = None
        self._last_heartbeat_at: datetime | None = None
        self._scheduler_drains_since_maintenance = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Maintenance cadence
    # ─────────────────────────────────────────────────────────────────────────

    def run_maintenance(self, now: datetime) -> int:
        """Evict dead members then recover expired peer leases (§C.2 path 1).

        Ordering: evict-before-reap (§C.2 :232) ensures that when we rotate
        an expired item lease the owner's registry row already carries
        status='evicted' (arm b of owner_registry_dead), so the reap is
        silent (no worker_stalled emitted for already-evicted workers).

        The stall-arm reap (item_stall_budget) is the documented exception:
        it may rotate BEFORE eviction for a live-heartbeat-but-wedged-drain
        worker, and emits worker_stalled in the same transaction (§A.5 :145).

        ADR-030 §C.2/§C.3 (slice 5): followers run NO maintenance at all —
        the explicit ``ProcessorMode.FOLLOWER`` stored at construction
        (elspeth-577179bba1) returns 0 up front.  Followers must not run
        ``recover_expired_leases``: the strict API now refuses their missing
        token before any transaction, and the explicit mode guard preserves
        the stronger policy that followers do not attempt maintenance at all.
        Followers are drain-only workers; lease recovery and eviction are the
        leader's responsibility (§C.2 path 1, §C.3: "followers drain what is
        claimable, then idle/exit").

        Within LEADER mode the token is required before either maintenance
        write. Pre-coordination repository/crash-image harnesses use the
        explicitly named legacy recovery adapter directly; production
        maintenance never selects an unfenced write from an optional token.
        """
        if self._mode is ProcessorMode.FOLLOWER:
            # Identical to the old post-evict-sweep skip: a follower's
            # coordination_token/run_coordination are None (validated
            # fail-closed by RowProcessor), so the evict sweep below was
            # already unreachable for it.
            self._scheduler_drains_since_maintenance = 0
            return 0

        coordination_token = self._processor._require_coordination_token()

        # §C.2 path 1: leader evicts dead non-leader members before reaping.
        # Individual, not bulk — one evict_worker call per dead member (§B.4,
        # §C.2 :233). evict_worker is idempotent (benign skip on CAS miss).
        if self._run_coordination is not None:
            from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS

            grace = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
            dead_members = self._run_coordination.dead_non_leader_workers(
                run_id=self._run_id,
                leader_worker_id=coordination_token.worker_id,
                now=now,
                grace_seconds=grace,
            )
            for target_worker_id in dead_members:
                self._run_coordination.evict_worker(
                    token=coordination_token,
                    target_worker_id=target_worker_id,
                    now=now,
                    grace_seconds=grace,
                    window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
                )

        recovered = self._scheduler.recover_expired_leases(
            now=now,
            coordination_token=coordination_token,
        )
        self._scheduler_drains_since_maintenance = 0
        return recovered

    def _maintenance_due(self, *, recover_pending_sinks: bool) -> bool:
        """Return whether this drain should run scheduler maintenance up front."""
        if recover_pending_sinks:
            return True
        self._scheduler_drains_since_maintenance += 1
        return self._scheduler_drains_since_maintenance >= SCHEDULER_MAINTENANCE_INTERVAL

    # ─────────────────────────────────────────────────────────────────────────
    # The claim/drain loop
    # ─────────────────────────────────────────────────────────────────────────

    def drain_claims(
        self,
        *,
        ctx: PluginContext,
        pending_items: dict[str, WorkItem],
        recover_pending_sinks: bool,
        preclaimed_items: list[TokenWorkItem] | None = None,
        before_claim: Callable[[], None] | None = None,
    ) -> list[RowResult]:
        """Claim and advance scheduler work until no READY work remains.

        When ``recover_pending_sinks=True`` (the resume entry point), any
        PENDING_SINK rows left durable by a prior crashed worker are drained
        up front via ``claim_pending_sink`` BEFORE the main claim_ready loop
        runs. This is the only path that re-claims PENDING_SINK rows inside
        a drain.

        PENDING_SINK rows produced by the main loop itself (via
        ``mark_pending_sink`` after a transform success at sink handoff) are
        NOT re-claimed by this drain — their RowResult is already in
        ``results`` from the ``claim_ready`` path that produced them, and
        their durable scheduler row is terminalized later by
        ``mark_sink_bound_scheduler_terminal`` via the sink-write callback
        (``_make_checkpoint_after_sink_factory`` in orchestrator/core.py).
        Re-claiming them here would emit a duplicate
        ``row_result_from_pending_sink`` for the same token_id. See
        filigree elspeth-5c5e88b071 (G3).

        Note: if a prior worker's lease on a sink-bound row is still active
        (not yet expired), ``claim_pending_sink`` won't find it (status is
        LEASED, not PENDING_SINK) and ``recover_expired_leases`` won't touch
        it (lease not yet expired). It will be stranded until the lease
        ages out, at which point a subsequent ``drain_scheduled_work`` call
        from a later resume attempt recovers it. That is correct behavior:
        the prior worker may still be alive and finishing the sink write.
        Forcing recovery here would race against an in-flight sink writer
        and risk duplicate sink emission.
        """
        results: list[RowResult] = []

        if recover_pending_sinks:
            self._drain_preexisting_pending_sinks(results)

        iterations = 0
        maintenance_iteration = 0
        preclaimed_queue = list(preclaimed_items or ())

        if self._maintenance_due(recover_pending_sinks=recover_pending_sinks):
            self.run_maintenance(self._clock.now_utc())

        while True:
            iterations += 1
            if iterations > MAX_WORK_QUEUE_ITERATIONS:
                raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")
            if before_claim is not None:
                before_claim()

            # ADR-030 §E.2 (slice 3): per-iteration journal-first intake —
            # adopt intake-pending BLOCKED barrier rows into executor memory,
            # replay durable branch losses (§E.5), release late arrivals
            # (§E.3a) and evaluate count/condition triggers over post-intake
            # memory, ALL before this iteration's claim. Flush/merge outputs
            # append to the drain's results; continuation items enqueue into
            # the current loop (their READY rows were inserted atomically by
            # complete_barrier; the enqueue reconciles idempotently).
            intake_results, intake_child_items = self._processor._run_barrier_intake_pass(ctx)
            results.extend(intake_results)
            for intake_child in intake_child_items:
                self.enqueue_work_item(intake_child, pending_items)

            # The claim timestamp is read AFTER intake: rows the intake just
            # emitted carry available_at stamps later than an iteration-top
            # reading, and claim_ready's available_at <= now predicate must
            # see them.
            now = self._clock.now_utc()
            if iterations - maintenance_iteration >= SCHEDULER_MAINTENANCE_INTERVAL:
                self.run_maintenance(now)
                maintenance_iteration = iterations

            claimed: TokenWorkItem | None
            if preclaimed_queue:
                claimed = preclaimed_queue.pop(0)
            else:
                claimed = self._scheduler.claim_ready(
                    run_id=self._run_id,
                    lease_owner=self._scheduler_lease_owner,
                    lease_seconds=self._scheduler_lease_seconds,
                    now=now,
                )
            if claimed is None:
                if recover_pending_sinks:
                    recovered = self.run_maintenance(self._clock.now_utc())
                    if recovered:
                        continue
                if pending_items:
                    # ADR-030 multi-worker: peer followers may have claimed some of
                    # the READY children enqueued by this leader (e.g. json_explode
                    # or per-LLM-call continuations).  Once a peer claims a child, the
                    # leader's in-memory pending entry can no longer be claimed READY:
                    # the peer drives it LEASED → PENDING_SINK (lease_owner kept) →
                    # TERMINAL/FAILED.  The leader may relinquish such peer-owned
                    # continuations, but ONLY under a discriminator that keeps the
                    # single-worker invariant and the audit backstops intact:
                    #
                    #   (1) NONE of the pending items are still READY
                    #       (count_ready_in_set == 0) — a still-READY item is a
                    #       genuine stranded continuation and must raise; and
                    #   (2) NONE of the pending items are FAILED
                    #       (count_failed_in_set == 0) — FAILED is the ONLY status
                    #       absent from BOTH backstops (count_active_work AND
                    #       complete_run's quiescence CAS cover READY/LEASED/BLOCKED/
                    #       PENDING_SINK but NOT FAILED), so a self-FAILED stray would
                    #       be silently lost; refusing on any FAILED pending row keeps
                    #       it loud (this is the M1 residual fix, and it lands at N=1
                    #       where the leader's OWN item is FAILED with no peer); and
                    #   (3) a peer is/was carrying work — has_peer_owned_work: some
                    #       OTHER lease_owner holds a LEASED or PENDING_SINK row on
                    #       this run.  PENDING_SINK is included because
                    #       mark_pending_sink KEEPS the claimant's lease_owner, so a
                    #       follower that has already parked all its claims as
                    #       PENDING_SINK and let its active LEASES lapse is STILL
                    #       detectable (the in-claim drain races AGAINST this: by the
                    #       time the leader's claim_ready returns None, the follower
                    #       may hold zero active leases yet own many PENDING_SINK
                    #       rows).  An N=1 leader's own rows carry the leader's owner
                    #       (BLOCKED rows carry none), so has_peer_owned_work is False
                    #       for a solo leader → it never relinquishes → its
                    #       self-stranded LEASED/BLOCKED/PENDING_SINK rows still hit
                    #       the raise here or the run-level backstop — never silent.
                    #
                    # The surviving relinquish set is therefore exactly "non-READY,
                    # non-FAILED continuations a peer is carrying" — every one of
                    # which is covered by the run-level and quiescence backstops if
                    # the peer somehow fails to finish it.
                    if self._scheduler_lease_owner_registered:
                        pending_ids = list(pending_items.keys())
                        ready_count = self._scheduler.count_ready_in_set(run_id=self._run_id, work_item_ids=pending_ids)
                        failed_count = self._scheduler.count_failed_in_set(run_id=self._run_id, work_item_ids=pending_ids)
                        if (
                            ready_count == 0
                            and failed_count == 0
                            and self._scheduler.has_peer_owned_work(run_id=self._run_id, caller_owner=self._scheduler_lease_owner)
                        ):
                            relinquished = {
                                work_item_id: f"{item.token.token_id}@{item.current_node_id}"
                                for work_item_id, item in pending_items.items()
                            }
                            logger.info(
                                "Relinquishing %d non-READY/non-FAILED pending continuation(s) to a peer worker "
                                "(run_id=%r, work_item_ids=%r, observed_statuses=%r)",
                                len(pending_items),
                                self._run_id,
                                list(relinquished.keys()),
                                self._scheduler.summarize_active_work(run_id=self._run_id),
                            )
                            pending_items.clear()
                            break
                    stranded = ", ".join(f"{item.token.token_id}@{item.current_node_id}" for item in pending_items.values())
                    active = "; ".join(self._scheduler.summarize_active_work(run_id=self._run_id)) or "<none>"
                    raise OrchestrationInvariantError(
                        f"Scheduler has {len(pending_items)} in-memory continuations for run {self._run_id!r} "
                        f"but no READY work item could be claimed. Stranded: {stranded}. Active journal work: {active}."
                    )
                break

            if claimed.work_item_id in pending_items:
                item = pending_items.pop(claimed.work_item_id)
            else:
                item = self._work_codec.work_item_from_scheduler(claimed)
            claimed_lease_owner = self._claimed_scheduler_lease_owner(claimed)
            # Mark this claim active so ``_process_single_token``'s per-node
            # heartbeat refreshes the right lease (filigree elspeth-ddde8144b6).
            # Initial last_heartbeat_at is now: claim_ready just set
            # lease_expires_at = now + lease_seconds, so the first heartbeat
            # only needs to fire once heartbeat_seconds has elapsed.
            self._active_claim_work_item_id = claimed.work_item_id
            self._last_heartbeat_at = self._clock.now_utc()
            try:
                try:
                    result, child_items = self._processor._process_single_token(
                        token=item.token,
                        ctx=ctx,
                        current_node_id=item.current_node_id,
                        coalesce_node_id=item.coalesce_node_id,
                        coalesce_name=item.coalesce_name,
                        on_success_sink=item.on_success_sink,
                        attempt_offset=max(claimed.attempt - 1, 0),
                    )
                except SchedulerLeaseLostError as exc:
                    # The lease was reaped by a peer mid-processing. The
                    # original ``work_item_id`` no longer exists (peer rewrote
                    # it under a bumped attempt) or no longer carries this
                    # worker's ``lease_owner``. Issuing ``mark_failed`` would
                    # CAS-fail and cascade into Tier-1 AuditIntegrityError —
                    # the exact failure mode this primitive exists to
                    # eliminate. Abandon the in-flight work, do NOT emit the
                    # (lost) result, and return the results that were already
                    # proven before this lease was lost. The caller's
                    # post-sink scheduler invariant check will refuse run
                    # completion if active work remains. Staged §E.5 loss
                    # records are discarded with the abandoned claim (the
                    # peer's re-drive re-stages them with its own disposition).
                    self._pending_branch_losses.clear()
                    exc.add_note("scheduler lease lost during row processing; in-flight token result was abandoned")
                    return results
                except RunWorkerEvictedError as exc:
                    # Membership loss is a coordination signal, not a plugin
                    # processing failure.  Propagate it directly: the generic
                    # arm below performs mark_failed bookkeeping, which would
                    # either mutate the abandoned lease through the lenient
                    # N=0 disposition fence or mask this signal behind an
                    # AuditIntegrityError when another member remains.
                    self._pending_branch_losses.clear()
                    exc.add_note("worker membership lost during row processing; in-flight token result was abandoned")
                    raise
                except Exception as processing_exc:
                    try:
                        self._scheduler.mark_failed(
                            work_item_id=claimed.work_item_id,
                            now=self._clock.now_utc(),
                            expected_lease_owner=claimed_lease_owner,
                            branch_loss=self.take_claim_branch_loss(claimed.token_id),
                            worker_id=self._disposition_fence_worker_id(),
                        )
                    except RunWorkerEvictedError as evicted_exc:
                        # The membership fence refused the failure bookkeeping:
                        # this worker was evicted mid-processing, so the peer
                        # reap path owns the item now. Propagate the eviction
                        # signal (followers exit on it) instead of wrapping it
                        # as an audit-integrity crash.
                        evicted_exc.add_note(f"processing exception {type(processing_exc).__name__} was superseded by the eviction refusal")
                        raise
                    except Exception as scheduler_exc:
                        raise AuditIntegrityError(
                            f"Scheduler failed to mark work_item_id={claimed.work_item_id!r} failed after original "
                            f"processing exception {type(processing_exc).__name__}: {processing_exc}. "
                            f"The scheduler failure write raised {type(scheduler_exc).__name__}: {scheduler_exc}."
                        ) from scheduler_exc
                    raise
            finally:
                self._active_claim_work_item_id = None
                self._last_heartbeat_at = None

            if result is not None and is_buffered_scheduler_result(result):
                for child_item in child_items:
                    self.enqueue_work_item(child_item, pending_items)
                self._mark_claimed_scheduler_work_blocked(
                    claimed,
                    item,
                    now=self._clock.now_utc(),
                    queue_key=None,
                    barrier_key=self.barrier_key_for_live_hold(claimed.token_id),
                )
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)
                # §E.2: ALWAYS take another iteration — the next iteration's
                # journal-first intake adopts the row just marked BLOCKED (and
                # fires any count/condition trigger it satisfies) before the
                # drain may exit.
                continue

            if result is None and not child_items:
                self._mark_claimed_scheduler_work_blocked(claimed, item, now=self._clock.now_utc())
                # §E.2: ALWAYS take another iteration (see the buffered arm).
                continue

            if (sink_bound_result := scheduler_sink_bound_result_for_claimed_token(result, claimed.token_id)) is not None:
                row_payload_json = self._scheduler.serialize_row_payload(sink_bound_result.token.row_data)
                sink_name = require_scheduler_sink_name(sink_bound_result)
                outcome = require_scheduler_outcome(sink_bound_result).value
                path = sink_bound_result.path.value
                error_hash = scheduler_error_hash(sink_bound_result)
                error_message = scheduler_error_message(sink_bound_result)
                pending_sink_now = self._clock.now_utc()
                branch_loss = self.take_claim_branch_loss(claimed.token_id)
                worker_id = self._disposition_fence_worker_id()
                if child_items:
                    _, scheduled_children = self._scheduler.mark_pending_sink_with_ready_children(
                        work_item_id=claimed.work_item_id,
                        emitted_ready=tuple(self._work_codec.ready_emission(child_item) for child_item in child_items),
                        row_payload_json=row_payload_json,
                        sink_name=sink_name,
                        outcome=outcome,
                        path=path,
                        error_hash=error_hash,
                        error_message=error_message,
                        now=pending_sink_now,
                        expected_lease_owner=claimed_lease_owner,
                        branch_loss=branch_loss,
                        worker_id=worker_id,
                    )
                    self._retain_scheduled_children(child_items, scheduled_children, pending_items)
                else:
                    self._scheduler.mark_pending_sink(
                        work_item_id=claimed.work_item_id,
                        row_payload_json=row_payload_json,
                        sink_name=sink_name,
                        outcome=outcome,
                        path=path,
                        error_hash=error_hash,
                        error_message=error_message,
                        now=pending_sink_now,
                        expected_lease_owner=claimed_lease_owner,
                        branch_loss=branch_loss,
                        worker_id=worker_id,
                    )
                result = with_scheduler_pending_sink_handoff(result, claimed.token_id)
            elif scheduler_result_failed_claimed_token(result, claimed.token_id):
                failed_now = self._clock.now_utc()
                branch_loss = self.take_claim_branch_loss(claimed.token_id)
                worker_id = self._disposition_fence_worker_id()
                if child_items:
                    _, scheduled_children = self._scheduler.mark_failed_with_ready_children(
                        work_item_id=claimed.work_item_id,
                        emitted_ready=tuple(self._work_codec.ready_emission(child_item) for child_item in child_items),
                        now=failed_now,
                        expected_lease_owner=claimed_lease_owner,
                        branch_loss=branch_loss,
                        worker_id=worker_id,
                    )
                    self._retain_scheduled_children(child_items, scheduled_children, pending_items)
                else:
                    self._scheduler.mark_failed(
                        work_item_id=claimed.work_item_id,
                        now=failed_now,
                        expected_lease_owner=claimed_lease_owner,
                        branch_loss=branch_loss,
                        worker_id=worker_id,
                    )
            else:
                terminal_now = self._clock.now_utc()
                branch_loss = self.take_claim_branch_loss(claimed.token_id)
                worker_id = self._disposition_fence_worker_id()
                if child_items:
                    _, scheduled_children = self._scheduler.mark_terminal_with_ready_children(
                        work_item_id=claimed.work_item_id,
                        emitted_ready=tuple(self._work_codec.ready_emission(child_item) for child_item in child_items),
                        now=terminal_now,
                        expected_lease_owner=claimed_lease_owner,
                        branch_loss=branch_loss,
                        worker_id=worker_id,
                    )
                    self._retain_scheduled_children(child_items, scheduled_children, pending_items)
                else:
                    self._scheduler.mark_terminal(
                        work_item_id=claimed.work_item_id,
                        now=terminal_now,
                        expected_lease_owner=claimed_lease_owner,
                        branch_loss=branch_loss,
                        worker_id=worker_id,
                    )

            if result is not None:
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)

            if not recover_pending_sinks and not pending_items:
                break

        return results

    def _drain_preexisting_pending_sinks(self, results: list[RowResult]) -> None:
        """Re-emit PENDING_SINK rows left durable by a prior crashed worker.

        Called once at the top of a recovery drain (``recover_pending_sinks=True``).
        Each claim transitions a PENDING_SINK row to LEASED under this worker's
        lease_owner and appends a reconstructed RowResult to ``results``;
        terminalization happens later via the sink-write callback
        (``mark_sink_bound_scheduler_terminal``).

        ``recover_expired_leases`` is run on every iteration so that a prior
        worker's expired sink-bound lease (recovered as PENDING_SINK by
        scheduler_repository.recover_expired_leases) is picked up in the same
        drain pass. Iteration is bounded by ``MAX_WORK_QUEUE_ITERATIONS``
        because real fault-injected runs can produce hundreds of stranded
        pending-sinks; the cap prevents an unbounded loop if some other code
        path keeps recreating PENDING_SINK rows behind us.

        Only the resume entry point (``drain_scheduled_work``) calls into this
        path. Inside the main claim_ready loop, PENDING_SINK rows produced by
        ``mark_pending_sink`` MUST NOT be re-claimed here — see
        ``drain_claims`` docstring.
        """
        coordination_token = self._processor._require_coordination_token()
        iterations = 0
        while True:
            iterations += 1
            if iterations > MAX_WORK_QUEUE_ITERATIONS:
                raise RuntimeError(
                    f"Pending-sink recovery exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. "
                    "Possible infinite loop in PENDING_SINK recovery."
                )

            now = self._clock.now_utc()
            self._scheduler.recover_expired_leases(
                now=now,
                coordination_token=coordination_token,
            )
            repaired = self._scheduler.terminalize_pending_sinks_with_terminal_outcomes(
                run_id=self._run_id,
                now=now,
                caller_owner=self._scheduler_lease_owner,
                coordination_token=coordination_token,
            )
            if repaired:
                continue
            pending_sink = self._scheduler.claim_pending_sink(
                run_id=self._run_id,
                lease_owner=self._scheduler_lease_owner,
                lease_seconds=self._scheduler_lease_seconds,
                now=now,
            )
            if pending_sink is None:
                return
            results.append(self.row_result_from_pending_sink(pending_sink))

    def row_result_from_pending_sink(self, scheduled: TokenWorkItem) -> RowResult:
        """Rebuild a sink-bound row result without re-running its producer node."""
        if scheduled.pending_sink_name is None or scheduled.pending_outcome is None or scheduled.pending_path is None:
            raise AuditIntegrityError(f"Scheduler pending sink work_item_id={scheduled.work_item_id!r} is missing sink outcome metadata.")
        # Attempt-offset derivation: if the original run already opened a SINK
        # node_state for this token (the sink write itself crashed after
        # opening attempt 0), the re-driven sink write must run at the bumped
        # attempt or its node_state insert collides with audited history.
        # Scoped to the sink step — the only step a pending-sink re-drive
        # writes; producer-node attempts must not inflate the offset.
        max_attempts = self._barrier_restore_reads.get_max_node_state_attempts(
            self._run_id,
            [scheduled.token_id],
            step_index=self._processor.resolve_sink_step(),
        )
        if scheduled.token_id in max_attempts:
            attempt_offset = max_attempts[scheduled.token_id] + 1
        else:
            # No sink node_state exists yet: pending sinks are routinely parked
            # before their sink write ever opens attempt 0, so absence is the
            # normal first-attempt case, not corruption. The provenance guard
            # below still rejects offset > 0 without a resume checkpoint.
            attempt_offset = 0
        if attempt_offset > 0 and self._resume_checkpoint_id is None:
            raise AuditIntegrityError(
                f"Scheduler pending sink token {scheduled.token_id!r} (run {self._run_id!r}) already has "
                f"node_states up to attempt {attempt_offset - 1} but this processor has no resume checkpoint "
                "provenance; re-driving without a resume_checkpoint_id would write an unattributed retry attempt."
            )
        token = TokenInfo(
            row_id=scheduled.row_id,
            token_id=scheduled.token_id,
            row_data=self._scheduler.deserialize_row_payload(scheduled.row_payload_json),
            branch_name=scheduled.branch_name,
            fork_group_id=scheduled.fork_group_id,
            join_group_id=scheduled.join_group_id,
            expand_group_id=scheduled.expand_group_id,
            resume_attempt_offset=attempt_offset,
            resume_checkpoint_id=self._resume_checkpoint_id if attempt_offset > 0 else None,
        )
        is_on_error_routed = scheduled.pending_path == TerminalPath.ON_ERROR_ROUTED.value
        if is_on_error_routed and not scheduled.pending_error_hash:
            # The parking disposition always persists the originating error
            # hash for routed failures, so its absence is audit corruption —
            # refuse to replay with a recomputed (synthetic) hash
            # (filigree elspeth-d74d19f901).
            raise AuditIntegrityError(
                f"Scheduler pending sink work_item_id={scheduled.work_item_id!r} is ON_ERROR_ROUTED but carries no "
                "pending_error_hash; the replayed outcome cannot preserve the originally-audited error hash."
            )
        return RowResult(
            token=token,
            final_data=token.row_data,
            outcome=TerminalOutcome(scheduled.pending_outcome),
            path=TerminalPath(scheduled.pending_path),
            sink_name=scheduled.pending_sink_name,
            error=FailureInfo(exception_type="ResumedPendingSink", message=scheduled.pending_error_message or "")
            if is_on_error_routed
            else None,
            scheduler_pending_sink=True,
            authoritative_error_hash=scheduled.pending_error_hash if is_on_error_routed else None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dispositions and fence identities
    # ─────────────────────────────────────────────────────────────────────────

    def _mark_claimed_scheduler_work_blocked(
        self,
        claimed: TokenWorkItem,
        item: WorkItem,
        *,
        now: datetime,
        queue_key: str | None = None,
        barrier_key: str | None = None,
    ) -> None:
        """Persist BLOCKED state only when resume has a durable release key."""
        queue_key = self._processor._queue_key_for_blocked_item(item) if queue_key is None and barrier_key is None else queue_key
        barrier_key = self._processor._barrier_key_for_blocked_item(item) if queue_key is None and barrier_key is None else barrier_key
        if queue_key is None and barrier_key is None:
            raise OrchestrationInvariantError(
                f"Work item {claimed.work_item_id!r} (token={item.token.token_id!r}, node={item.current_node_id!r}) "
                "produced no result and no children, but has no queue or barrier key; cannot be unblocked. "
                "This is a processor bug."
            )
        self._scheduler.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key=queue_key,
            barrier_key=barrier_key,
            now=now,
            expected_lease_owner=self._claimed_scheduler_lease_owner(claimed),
            worker_id=self._disposition_fence_worker_id(),
        )

    def _disposition_fence_worker_id(self) -> str | None:
        """Membership-fence identity for drain disposition writes.

        ADR-030 §G parity (filigree elspeth-ba7b2cc25d): registered workers
        thread their identity so an evicted worker's dispositions are refused
        at the scheduler; unregistered (N=0 / legacy / test-fixture) builds
        pass None and stay unfenced — the same gating ``enqueue_ready``'s
        slice-5 fence uses.
        """
        return self._scheduler_lease_owner if self._scheduler_lease_owner_registered else None

    @staticmethod
    def _claimed_scheduler_lease_owner(claimed: TokenWorkItem) -> str:
        """Return the proven lease owner for a claimed scheduler item."""
        if claimed.lease_owner is None:
            raise AuditIntegrityError(
                f"Scheduler claimed work_item_id={claimed.work_item_id!r} without a lease_owner; "
                "cannot perform an owner-fenced state transition."
            )
        return claimed.lease_owner

    def take_claim_branch_loss(self, claimed_token_id: str) -> BranchLossSpec | None:
        """Take the staged §E.5 loss record for the claim being disposed.

        A single claim loses at most ONE branch (the claimed token reaches
        exactly one lossy terminal arm); the record rides the claim's own
        ``mark_failed`` / ``mark_pending_sink`` / ``mark_terminal``
        transaction. More than one staged record, or a record for a different
        token, is a processor bug.
        """
        if not self._pending_branch_losses:
            return None
        if len(self._pending_branch_losses) > 1:
            staged = [(spec.token_id, spec.branch_name) for spec in self._pending_branch_losses]
            raise OrchestrationInvariantError(
                f"Claim disposition for token {claimed_token_id!r} found {len(self._pending_branch_losses)} staged "
                f"branch-loss records ({staged!r}); one claim loses at most one branch. Processor bug."
            )
        spec = self._pending_branch_losses.pop()
        if spec.token_id != claimed_token_id:
            raise OrchestrationInvariantError(
                f"Claim disposition for token {claimed_token_id!r} found a staged branch-loss record for "
                f"token {spec.token_id!r} (branch {spec.branch_name!r}); the loss must ride its own token's "
                "disposition. Processor bug."
            )
        return spec

    # ─────────────────────────────────────────────────────────────────────────
    # Active-claim heartbeat and live-hold keys
    # ─────────────────────────────────────────────────────────────────────────

    def heartbeat_active_claim(self) -> None:
        """Refresh the active scheduler lease if heartbeat interval has elapsed.

        Called (via the processor's ``_heartbeat_active_claim`` delegate) from
        ``_process_single_token`` on every node-iteration boundary
        (ADR-026 RC6 multi-worker, filigree elspeth-ddde8144b6). The actual
        DB write fires at most once per ``scheduler_heartbeat_seconds`` so
        fast plugin chains do not incur a write per node.

        No-op when no claim is active or the interval has not yet elapsed.

        Raises:
            SchedulerLeaseLostError: the lease was reaped or reassigned by a
                peer between the claim and this heartbeat. The caller (the
                drain loop) catches this specifically and skips both the
                in-flight result emission and the terminal ``mark_*`` write —
                issuing either would CAS-fail and cascade into a Tier-1
                AuditIntegrityError, which is the exact failure mode this
                primitive exists to eliminate.
            RunWorkerEvictedError: the registered lease owner is no longer an
                active run member. The existing eviction path propagates this
                clean-abandon signal without a scheduler disposition mutation.

        **Single-plugin-call limitation.** This heartbeat fires *between*
        plugin calls, not *during* a single plugin call. If one plugin call
        exceeds ``scheduler_lease_seconds`` on its own, the lease still
        expires while that call is in-flight. The operator must size
        ``scheduler_lease_seconds`` to bracket the longest expected
        single-plugin call. Sub-call-level protection (option b watchdog or
        thread-based heartbeat) is a separate concern and not in scope for
        the ticket's option (a).
        """
        if self._active_claim_work_item_id is None:
            return
        now = self._clock.now_utc()
        if self._last_heartbeat_at is not None and (now - self._last_heartbeat_at).total_seconds() < self._scheduler_heartbeat_seconds:
            return
        self._scheduler.heartbeat_lease(
            run_id=self._run_id,
            work_item_id=self._active_claim_work_item_id,
            lease_owner=self._scheduler_lease_owner,
            lease_seconds=self._scheduler_lease_seconds,
            now=now,
            # Explicit boundary: registered production workers require the
            # strict active-membership EXISTS predicate. Legacy/N=0 processors
            # select the unfenced compatibility arm deliberately; registry
            # emptiness inside the repository never chooses that arm.
            membership_fenced=self._scheduler_lease_owner_registered,
        )
        self._last_heartbeat_at = now

    def barrier_key_for_live_hold(self, token_id: str) -> str:
        """Resolve the barrier that owns a token about to be marked BLOCKED.

        §E.2: the historical derivation read the in-claim BUFFERED outcome's
        batch_id, which no longer exists at block time — the producer
        (``_process_batch_aggregation_node`` / ``_maybe_coalesce_token``)
        stashed the barrier_key alongside the live token instead.
        """
        try:
            hold = self._live_barrier_holds[token_id]
        except KeyError:
            raise AuditIntegrityError(
                f"Buffered scheduler result for token {token_id!r} has no live barrier hold stash; "
                "cannot persist a durable release barrier. Processor bug."
            ) from None
        return hold.barrier_key

    # ─────────────────────────────────────────────────────────────────────────
    # READY work-item persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _retain_scheduled_children(
        self,
        child_items: list[WorkItem],
        scheduled_children: tuple[TokenWorkItem, ...],
        pending_items: dict[str, WorkItem],
    ) -> None:
        """Retain live payloads for children made durable by an atomic disposition."""
        for child_item, scheduled in zip(child_items, scheduled_children, strict=True):
            if scheduled.status is TokenWorkStatus.READY or (
                scheduled.status is TokenWorkStatus.LEASED and scheduled.lease_owner == self._scheduler_lease_owner
            ):
                pending_items[scheduled.work_item_id] = child_item

    def enqueue_work_item(
        self,
        item: WorkItem,
        pending_items: dict[str, WorkItem],
        *,
        claim_immediately: bool = False,
    ) -> TokenWorkItem:
        """Persist a READY scheduler item and retain the live token payload."""
        available_at = self._clock.now_utc()
        fields = self._work_codec.ready_fields(item)
        if claim_immediately:
            enqueue_claimed = (
                self._scheduler.enqueue_ready_claimed
                if self._scheduler_lease_owner_registered
                else self._scheduler.enqueue_ready_claimed_legacy_unfenced
            )
            scheduled = enqueue_claimed(
                run_id=self._run_id,
                token_id=fields.token_id,
                row_id=fields.row_id,
                node_id=fields.node_id,
                step_index=fields.step_index,
                ingest_sequence=fields.ingest_sequence,
                row_payload_json=fields.row_payload_json,
                available_at=available_at,
                queue_key=fields.queue_key,
                barrier_key=fields.barrier_key,
                on_success_sink=fields.on_success_sink,
                branch_name=fields.branch_name,
                fork_group_id=fields.fork_group_id,
                join_group_id=fields.join_group_id,
                expand_group_id=fields.expand_group_id,
                coalesce_node_id=fields.coalesce_node_id,
                coalesce_name=fields.coalesce_name,
                lease_owner=self._scheduler_lease_owner,
                lease_seconds=self._scheduler_lease_seconds,
                now=available_at,
            )
        else:
            scheduled = self._scheduler.enqueue_ready(
                run_id=self._run_id,
                token_id=fields.token_id,
                row_id=fields.row_id,
                node_id=fields.node_id,
                step_index=fields.step_index,
                ingest_sequence=fields.ingest_sequence,
                row_payload_json=fields.row_payload_json,
                available_at=available_at,
                queue_key=fields.queue_key,
                barrier_key=fields.barrier_key,
                on_success_sink=fields.on_success_sink,
                branch_name=fields.branch_name,
                fork_group_id=fields.fork_group_id,
                join_group_id=fields.join_group_id,
                expand_group_id=fields.expand_group_id,
                coalesce_node_id=fields.coalesce_node_id,
                coalesce_name=fields.coalesce_name,
                # Membership fence (ADR-030 §G, slice 5): thread the registered
                # worker identity so an evicted RowProcessor cannot enqueue READY
                # items that no active worker will claim. The fence is active only
                # when scheduler_lease_owner was explicitly registered in run_workers
                # (production multi-worker path: leaders via processor_factory.py, followers
                # via follower.py). Legacy / single-worker / test-fixture builds
                # pass scheduler_lease_owner=None → auto-generate an unregistered
                # identity → _scheduler_lease_owner_registered=False → fence skipped.
                worker_id=self._scheduler_lease_owner if self._scheduler_lease_owner_registered else None,
            )
        if scheduled.status is TokenWorkStatus.READY or (
            scheduled.status is TokenWorkStatus.LEASED and scheduled.lease_owner == self._scheduler_lease_owner
        ):
            pending_items[scheduled.work_item_id] = item
        return scheduled
