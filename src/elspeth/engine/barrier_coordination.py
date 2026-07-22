"""Barrier subsystem: journal-first intake and resume restore coordinators.

Extracted from RowProcessor (elspeth-e76a186916). Barrier adoption and
restore used to be choreography spread across the processor, the scheduler
repository, and the aggregation/coalesce executors, with the crash-window
ordering (open batch membership -> fenced adoption -> feed executor memory
-> evaluate trigger) preserved only by caller convention and docstring
prose. The two coordinators own that ordering behind one boundary:

- ``BarrierIntakeCoordinator`` — the ADR-030 §E.2/§E.3/§E.3a/§E.5
  journal-first intake pass: adopt intake-pending BLOCKED arrivals, feed
  executor memory with backdated accept timing, replay durable branch
  losses, and evaluate aggregation triggers from the same intake step as
  the triggering arrival's adoption. Each adopted row resolves to a typed
  ``BarrierIntakeDisposition`` (held / terminal / pending-sink /
  ready-continuation / flush-fired).
- ``BarrierRecoveryCoordinator`` — the F1 resume restore: rebuild
  aggregation buffers and coalesce pendings from journal BLOCKED rows +
  audit tables, reconciling the §E.3a/§E.4 crash windows before any
  executor mutation runs.

Both coordinators take their environmental lookups (repositories,
executors, navigator, clock) and the processor-owned continuation seams
(flush execution, coalesce fire completion, telemetry emission) as
injected collaborators; RowProcessor delegates through thin methods so its
public protocol is unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from elspeth.contracts import RowResult, TokenInfo, TransformProtocol
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars, CoalescePendingScalars
from elspeth.contracts.enums import BatchStatus, TerminalOutcome, TerminalPath, TriggerType
from elspeth.contracts.errors import AuditIntegrityError, OrchestrationInvariantError
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.results import FailureInfo
from elspeth.contracts.scheduler import BatchMembershipSpec, BufferedOutcomeSpec, TokenWorkItem
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.landscape.scheduler_repository import token_from_journal_item
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.work_items import WorkItem, WorkItemFactory

if TYPE_CHECKING:
    from elspeth.contracts import Batch
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.config import AggregationSettings
    from elspeth.core.landscape.data_flow_repository import DataFlowRepository
    from elspeth.core.landscape.execution_repository import ExecutionRepository
    from elspeth.core.landscape.scheduler import BarrierRestoreReadModel
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome
    from elspeth.engine.dag_navigator import DAGNavigator
    from elspeth.engine.executors import AggregationExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _LiveBarrierHold:
    """In-memory companion of one durable BLOCKED barrier hold (ADR-030 §E.2).

    Stashed by the processor at the moment a claimed token is about to block
    at a barrier (aggregation buffering / coalesce hold) and consumed by the
    next drain iteration's journal-first intake: the LIVE token preserves the
    exact post-transform payload and resume provenance the old in-claim accept
    used (N=1 parity). Inherited rows with no stash entry (leader takeover)
    fall back to journal rehydration with audit-derived attempt offsets —
    the same semantics as the restore path.
    """

    token: TokenInfo
    barrier_key: str


@dataclass(frozen=True, slots=True)
class BarrierJournalRestoreContext:
    """Resume inputs for the journal-based barrier restore (F1 design D3).

    Built by the resume path (``ResumeCoordinator``) and handed to
    ``RowProcessor.__init__``; its presence IS the resume signal — a normal
    run passes ``None`` and no restore sweep runs.

    Attributes:
        resume_checkpoint_id: Checkpoint id stamped on every journal-restored
            token (resume provenance).
        barrier_scalars: Underivable scalar barrier metadata from the
            checkpoint row (trigger latches / lost_branches). ``None`` means
            the checkpoint carried no scalars — every node restores with
            unlatched / no-losses defaults. The scalars snapshot is
            NON-transactional vs the journal (D3 staleness model): an absent
            entry always means not-fired / re-derivable, never corruption.
        batch_id_remap: old->retry batch_id mapping returned by
            ``handle_incomplete_batches`` — BUFFERED token_outcomes still
            reference the dead original batch ids, so the restored in-progress
            batch id must be read through this remap.
    """

    resume_checkpoint_id: str
    barrier_scalars: BarrierScalars | None
    batch_id_remap: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.resume_checkpoint_id:
            raise ValueError("BarrierJournalRestoreContext.resume_checkpoint_id must not be empty")
        object.__setattr__(self, "batch_id_remap", deep_freeze(self.batch_id_remap))


@dataclass(frozen=True, slots=True)
class _AggregationRestorePlan:
    """Derived restore inputs for one aggregation node (journal restore).

    Built during the derivation phase of
    ``BarrierRecoveryCoordinator.restore_from_journal`` so every audit read
    completes (and can raise) before any executor mutation runs.
    """

    node_id: NodeID
    items: Sequence[TokenWorkItem]
    member_order: Sequence[str]
    batch_id: str | None
    accepted_count_total: int
    completed_flush_count: int
    scalars: AggregationNodeScalars

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", deep_freeze(self.items))
        object.__setattr__(self, "member_order", deep_freeze(self.member_order))


class BarrierIntakeDispositionKind(StrEnum):
    """Typed disposition taxonomy for one adopted barrier arrival."""

    HELD = "held"
    TERMINAL = "terminal"
    PENDING_SINK = "pending_sink"
    READY_CONTINUATION = "ready_continuation"
    FLUSH_FIRED = "flush_fired"


@dataclass(frozen=True, slots=True)
class BarrierIntakeDisposition:
    """One adopted arrival's resolution at the intake boundary.

    ``results`` and ``child_items`` carry the arrival's outputs in the exact
    order the pre-extraction choreography appended them, so flattening a
    disposition sequence reproduces the historical result ordering.
    """

    kind: BarrierIntakeDispositionKind
    results: tuple[RowResult, ...] = ()
    child_items: tuple[WorkItem, ...] = ()


@dataclass(frozen=True, slots=True)
class BarrierIntakePassOutcome:
    """All dispositions produced by one §E.2 intake pass, in intake order."""

    dispositions: tuple[BarrierIntakeDisposition, ...]

    @property
    def results(self) -> list[RowResult]:
        return [result for disposition in self.dispositions for result in disposition.results]

    @property
    def child_items(self) -> list[WorkItem]:
        return [item for disposition in self.dispositions for item in disposition.child_items]


class BarrierIntakeCoordinator:
    """Journal-first barrier intake (ADR-030 §E.2/§E.3/§E.3a/§E.5, slice 3).

    Owns the ordered adoption sequence the executors' docstrings used to
    delegate to caller convention: batch membership opens BEFORE the fenced
    adoption verb, executor memory is fed ONLY on the adopted=True arm with
    backdated accept timing, and aggregation triggers are evaluated from the
    SAME intake step as the triggering arrival's adoption.
    """

    def __init__(
        self,
        *,
        run_id: str,
        scheduler: TokenSchedulerRepository,
        data_flow: DataFlowRepository,
        execution: ExecutionRepository,
        barrier_restore_reads: BarrierRestoreReadModel | ExecutionRepository,
        aggregation_executor: AggregationExecutor,
        coalesce_executor: CoalesceExecutor | None,
        nav: DAGNavigator,
        work_items: WorkItemFactory,
        clock: Clock,
        aggregation_settings: Mapping[NodeID, AggregationSettings],
        coalesce_node_ids: Mapping[CoalesceName, NodeID],
        coordination_token: CoordinationToken | None,
        scheduler_lease_owner: str,
        live_barrier_holds: dict[str, _LiveBarrierHold],
        resume_checkpoint_id: str | None,
        flush_batch: Callable[[NodeID, TransformProtocol, PluginContext, TriggerType], tuple[tuple[RowResult, ...], list[WorkItem]]],
        complete_coalesce_fire: Callable[..., None],
        terminal_coalesce_row_result: Callable[..., RowResult],
        emit_token_completed: Callable[..., None],
        mark_coalesce_consumed_terminal: Callable[..., None],
    ) -> None:
        self._run_id = run_id
        self._scheduler = scheduler
        self._data_flow = data_flow
        self._execution = execution
        self._barrier_restore_reads = barrier_restore_reads
        self._aggregation_executor = aggregation_executor
        self._coalesce_executor = coalesce_executor
        self._nav = nav
        self._work_items = work_items
        self._clock = clock
        self._aggregation_settings = aggregation_settings
        self._coalesce_node_ids = coalesce_node_ids
        self._coordination_token = coordination_token
        self._scheduler_lease_owner = scheduler_lease_owner
        self._live_barrier_holds = live_barrier_holds
        self._resume_checkpoint_id = resume_checkpoint_id
        self._flush_batch = flush_batch
        self._complete_coalesce_fire = complete_coalesce_fire
        self._terminal_coalesce_row_result = terminal_coalesce_row_result
        self._emit_token_completed = emit_token_completed
        self._mark_coalesce_consumed_terminal = mark_coalesce_consumed_terminal

    def _require_coordination_token(self) -> CoordinationToken:
        """The leader fencing token, REQUIRED for the slice-3 adoption verbs."""
        if self._coordination_token is None:
            raise OrchestrationInvariantError(
                "Journal-first barrier intake requires the leader coordination token (ADR-030 §E.2): "
                "adopt_blocked_barrier_item / adopt_coalesce_branch_losses are fenced verbs with no "
                "unfenced arm. Construct RowProcessor with coordination_token — the orchestrator "
                "binds it at begin_run (epoch 1) or at the resume takeover CAS."
            )
        return self._coordination_token

    def _backdated_accept_monotonic(self, row: TokenWorkItem) -> float:
        """Convert a row's durable ``barrier_blocked_at`` onto the monotonic scale.

        §E.2 backdated accept timing — the EXACT clamped wall->monotonic
        transform the journal restore uses (coalesce restore_from_journal /
        aggregation elapsed_age derivation): trigger latches and coalesce
        arrival anchors are pure functions of durable state + config, hence
        invariant under leader takeover (§H 476).
        """
        if row.barrier_blocked_at is None:
            raise AuditIntegrityError(
                f"BLOCKED journal row for token {row.token_id!r} (run {self._run_id!r}) has NULL "
                "barrier_blocked_at — the backdated accept instant cannot be derived; journal "
                "corruption (mark_blocked stamps every hold)."
            )
        now_wall = self._clock.now_utc()
        return self._clock.monotonic() - max(0.0, (now_wall - row.barrier_blocked_at).total_seconds())

    def _token_for_intake(self, row: TokenWorkItem) -> TokenInfo:
        """Resolve the TokenInfo to feed executor memory for one adopted row.

        Live stash first (N=1 parity: the exact post-transform token the old
        in-claim accept used, with its original resume provenance). Without a
        live stash, the durable row is still authoritative: fresh leaders use
        offset zero for normal follower handoffs, while resume leaders use the
        audit-derived offset stamped with checkpoint provenance.
        """
        hold = self._live_barrier_holds.pop(row.token_id, None)
        if hold is not None:
            return hold.token
        if self._resume_checkpoint_id is None:
            return token_from_journal_item(row, attempt_offset=0, resume_checkpoint_id=None)
        max_attempts = self._barrier_restore_reads.get_max_node_state_attempts(self._run_id, [row.token_id])
        return token_from_journal_item(
            row,
            attempt_offset=max_attempts.get(row.token_id, -1) + 1,
            resume_checkpoint_id=self._resume_checkpoint_id,
        )

    def run_intake_pass(self, ctx: PluginContext) -> BarrierIntakePassOutcome:
        """One §E.2 intake pass: adopt arrivals, replay losses, fire triggers.

        Runs at the top of every drain iteration (and from the orchestrator's
        EOF loop via ``RowProcessor.run_barrier_intake``). Steps, in design
        order:

        1. arrival intake — every intake-pending BLOCKED barrier row
           (``barrier_adopted_epoch IS NULL``) is adopted via the fenced
           backdated adoption verb and fed into executor memory; coalesce
           adoption runs the executor accept, surfacing merge fires, group
           failures and late-arrival releases (§E.3a) here;
        2. per-adoption aggregation trigger evaluation — count/condition
           triggers fire from the SAME intake step as the triggering
           arrival's adoption (the §E.2 replacement for the deleted in-claim
           flush arm; batch composition is preserved because the check runs
           after EACH adoption, exactly like the old accept-then-check);
        3. branch-loss replay (§E.5) — unadopted durable losses are marked
           (journal-first) and replayed through ``notify_branch_lost``
           before the next trigger evaluation; at N=1 this is a structural
           no-op (record-then-notify already ran in-claim).

        Returns the typed dispositions in intake order; flattening their
        results/child_items reproduces the pre-extraction append ordering.
        """
        dispositions: list[BarrierIntakeDisposition] = []
        if not self._aggregation_settings and self._coalesce_executor is None:
            return BarrierIntakePassOutcome(dispositions=())

        # ADR-030 §E.2 intake scan shape: only intake-pending rows
        # (barrier_adopted_epoch IS NULL).  The SQL predicate avoids
        # materializing adopted rows on every drain iteration — without it a
        # filling count-N batch costs O(N²/2) full-row hydrations (finding 2).
        pending_rows = self._scheduler.list_pending_blocked_barrier_items(run_id=self._run_id)
        if pending_rows:
            coalesce_keys = {str(name) for name in self._coalesce_node_ids}
            aggregation_keys = {str(node_id) for node_id in self._aggregation_settings}
            for row in pending_rows:
                if row.barrier_key in aggregation_keys:
                    disposition = self._adopt_aggregation_row(row, ctx)
                elif row.barrier_key in coalesce_keys:
                    disposition = self._adopt_coalesce_row(row)
                else:
                    raise AuditIntegrityError(
                        f"Intake-pending BLOCKED row for token {row.token_id!r} (run {self._run_id!r}) carries "
                        f"orphan barrier_key {row.barrier_key!r}: not a configured coalesce "
                        f"({sorted(coalesce_keys)}) or aggregation node ({sorted(aggregation_keys)})."
                    )
                if disposition is not None:
                    dispositions.append(disposition)

        dispositions.extend(self._replay_branch_losses())
        return BarrierIntakePassOutcome(dispositions=tuple(dispositions))

    def _adopt_aggregation_row(self, row: TokenWorkItem, ctx: PluginContext) -> BarrierIntakeDisposition | None:
        """Adopt one aggregation barrier row, then evaluate the node's trigger.

        Ordering by construction (the invariant the executors' "caller
        obligations" prose used to delegate to callers): batch membership
        opens BEFORE the fenced adoption verb, and executor memory is fed
        ONLY on the adopted=True arm.
        """
        if row.barrier_key is None:  # pragma: no cover - excluded by the query contract
            raise AuditIntegrityError(f"Intake aggregation row {row.work_item_id!r} has no barrier_key.")
        node_id = NodeID(row.barrier_key)
        coordination_token = self._require_coordination_token()
        # Resolve the token BEFORE the fenced verb so an invalid journal row is
        # refused with ZERO durable mutation. Valid follower handoffs can be
        # rebuilt from the durable row even without a live stash entry.
        token = self._token_for_intake(row)
        batch_id, ordinal = self._aggregation_executor.open_batch_membership(node_id)
        adoption = self._scheduler.adopt_blocked_barrier_item(
            run_id=self._run_id,
            work_item_id=row.work_item_id,
            token_id=row.token_id,
            barrier_key=row.barrier_key,
            membership=BatchMembershipSpec(batch_id=batch_id, ordinal=ordinal),
            buffered_outcome=BufferedOutcomeSpec(batch_id=batch_id),
            now=self._clock.now_utc(),
            coordination_token=coordination_token,
        )
        if not adoption.adopted:
            # Idempotent success-SKIP: already adopted (a racing duplicate of
            # this leader's own pass). MUST NOT re-feed memory (§C.4 row 6a).
            return None
        self._aggregation_executor.accept_adopted_row(node_id, token, accept_time=self._backdated_accept_monotonic(row))

        # Step 2: per-adoption trigger evaluation — the §E.2 home of the old
        # in-claim count/condition flush decision (accept-then-check), so
        # batch composition is byte-identical to the in-claim era.
        should_flush, trigger_type = self._aggregation_executor.check_flush_status(node_id)
        if not should_flush:
            return BarrierIntakeDisposition(kind=BarrierIntakeDispositionKind.HELD)
        transform = self._nav.resolve_plugin_for_node(node_id)
        if not isinstance(transform, TransformProtocol) or not transform.is_batch_aware:
            raise OrchestrationInvariantError(
                f"Aggregation node {node_id!r} fired a {trigger_type} trigger at intake but resolves to "
                f"{type(transform).__name__!r}, not a batch-aware transform. DAG/config inconsistency."
            )
        flush_results, flush_child_items = self._flush_batch(
            node_id,
            transform,
            ctx,
            trigger_type if trigger_type is not None else TriggerType.COUNT,
        )
        return BarrierIntakeDisposition(
            kind=BarrierIntakeDispositionKind.FLUSH_FIRED,
            results=tuple(flush_results),
            child_items=tuple(flush_child_items),
        )

    def _adopt_coalesce_row(self, row: TokenWorkItem) -> BarrierIntakeDisposition | None:
        """Adopt one coalesce barrier row and run the intake-time accept."""
        if row.barrier_key is None:  # pragma: no cover - excluded by the query contract
            raise AuditIntegrityError(f"Intake coalesce row {row.work_item_id!r} has no barrier_key.")
        if self._coalesce_executor is None:  # pragma: no cover - partition guarantees a coalesce key
            raise OrchestrationInvariantError(f"Intake coalesce row for {row.barrier_key!r} but no CoalesceExecutor is configured.")
        coalesce_name = CoalesceName(row.barrier_key)
        coordination_token = self._require_coordination_token()
        # Resolve the token BEFORE the fenced verb (refusal-before-mutation
        # for invalid journal rows — see the aggregation arm).
        token = self._token_for_intake(row)
        adoption = self._scheduler.adopt_blocked_barrier_item(
            run_id=self._run_id,
            work_item_id=row.work_item_id,
            token_id=row.token_id,
            barrier_key=row.barrier_key,
            membership=None,
            buffered_outcome=None,
            now=self._clock.now_utc(),
            coordination_token=coordination_token,
        )
        if not adoption.adopted:
            return None
        outcome = self._coalesce_executor.accept(
            token=token,
            coalesce_name=str(coalesce_name),
            arrival_time=self._backdated_accept_monotonic(row),
        )

        if outcome.held:
            return BarrierIntakeDisposition(kind=BarrierIntakeDispositionKind.HELD)

        if outcome.late_arrival:
            # §E.3a: the group already completed — release THIS row alone in
            # the same drain iteration, with forensic late-arrival context.
            # The executor already recorded the FAILED node_state + FAILURE
            # outcome (outcomes_recorded=True on the late arm).
            released = self._scheduler.mark_blocked_barrier_terminal(
                run_id=self._run_id,
                barrier_key=str(coalesce_name),
                token_ids=(token.token_id,),
                now=self._clock.now_utc(),
                coordination_token=coordination_token,
                release_context={
                    "late_arrival": True,
                    "reason": outcome.failure_reason,
                    "released_by": self._scheduler_lease_owner,
                    "scope_row_id": row.row_id,
                },
            )
            if released != 1:
                raise AuditIntegrityError(
                    f"Late-arrival release for token {token.token_id!r} at coalesce {coalesce_name!r} "
                    f"(run {self._run_id!r}) terminalized {released} rows; expected exactly one."
                )
            self._emit_token_completed(token, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)
            return BarrierIntakeDisposition(
                kind=BarrierIntakeDispositionKind.TERMINAL,
                results=(
                    RowResult(
                        token=token,
                        final_data=token.row_data,
                        outcome=TerminalOutcome.FAILURE,
                        path=TerminalPath.UNROUTED,
                        error=FailureInfo(exception_type="CoalesceFailure", message=outcome.failure_reason or "late_arrival_after_merge"),
                    ),
                ),
            )

        if outcome.merged_token is not None:
            return self._fire_coalesce_merge(coalesce_name, outcome, scope_row_id=row.row_id)

        if outcome.failure_reason:
            # Group failure completed by this arrival: every consumed branch
            # (this one included) holds a BLOCKED row — release them all.
            self._mark_coalesce_consumed_terminal(
                coalesce_name=coalesce_name,
                consumed_tokens=tuple(outcome.consumed_tokens),
            )
            error_msg = outcome.failure_reason
            # Bug 9z8 fix: only record if CoalesceExecutor didn't already record.
            if not outcome.outcomes_recorded:
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error_hash=compute_error_hash(error_msg),
                )
            # Emit TokenCompleted telemetry AFTER Landscape recording. Only
            # the arriving token surfaces a RowResult (the held siblings'
            # outcomes were recorded by the executor) — the pre-§E.2 shape.
            self._emit_token_completed(token, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)
            return BarrierIntakeDisposition(
                kind=BarrierIntakeDispositionKind.TERMINAL,
                results=(
                    RowResult(
                        token=token,
                        final_data=token.row_data,
                        outcome=TerminalOutcome.FAILURE,
                        path=TerminalPath.UNROUTED,
                        error=FailureInfo(exception_type="CoalesceFailure", message=error_msg),
                    ),
                ),
            )

        raise OrchestrationInvariantError(
            f"CoalesceOutcome for token {token.token_id} in coalesce '{coalesce_name}' is in invalid state: "
            f"held={outcome.held}, merged_token={outcome.merged_token is not None}, "
            f"failure_reason={outcome.failure_reason!r}"
        )

    def _fire_coalesce_merge(
        self,
        coalesce_name: CoalesceName,
        outcome: CoalesceOutcome,
        *,
        scope_row_id: str,
    ) -> BarrierIntakeDisposition:
        """Complete an intake-time coalesce merge fire (terminal or not).

        Non-terminal: mirrors ``complete_coalesce_merge``'s shape — the merged
        child's READY continuation is inserted atomically with the consumption
        (F1/D6) and the same WorkItem is handed back for the caller's
        idempotent enqueue.

        Terminal: the COALESCED sink-bound result is emitted as a fresh
        PENDING_SINK row in the SAME atomic completion — the merged output is
        journal-durable the moment its inputs are consumed (the pre-§E.2
        in-claim ride to ``mark_pending_sink`` left it memory-only between the
        consumption and the claim disposition).
        """
        if outcome.merged_token is None:  # pragma: no cover - caller checks
            raise OrchestrationInvariantError("merged_token is None in _fire_coalesce_merge")
        coalesce_node_id = self._coalesce_node_ids[coalesce_name]
        if self._nav.resolve_next_node(coalesce_node_id) is None:
            terminal_result = self._terminal_coalesce_row_result(
                outcome.merged_token,
                coalesce_name,
                context=f"intake coalesce fire for token '{outcome.merged_token.token_id}'",
            )
            self._complete_coalesce_fire(
                coalesce_name=coalesce_name,
                consumed_tokens=tuple(outcome.consumed_tokens),
                scope_row_id=scope_row_id,
                merged_sink_result=terminal_result,
            )
            return BarrierIntakeDisposition(
                kind=BarrierIntakeDispositionKind.PENDING_SINK,
                results=(replace(terminal_result, scheduler_pending_sink=True),),
            )
        merged_item = self._work_items.create(
            token=outcome.merged_token,
            current_node_id=coalesce_node_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )
        self._complete_coalesce_fire(
            coalesce_name=coalesce_name,
            consumed_tokens=tuple(outcome.consumed_tokens),
            scope_row_id=scope_row_id,
            merged_item=merged_item,
        )
        return BarrierIntakeDisposition(
            kind=BarrierIntakeDispositionKind.READY_CONTINUATION,
            child_items=(merged_item,),
        )

    def _replay_branch_losses(self) -> list[BarrierIntakeDisposition]:
        """§E.5 loss intake: mark unadopted durable losses, replay into memory.

        Journal-first: the fenced cursor mark commits BEFORE the in-memory
        replay — a crash between mark and replay loses nothing because the
        takeover restore derives lost_branches from the FULL loss table. At
        N=1 the replay arm is structurally idle: the producer already
        notified in-claim (record-then-notify), so ``has_recorded_branch_loss``
        (or the executor's completed-keys check) dedups every row.
        """
        dispositions: list[BarrierIntakeDisposition] = []
        if self._coalesce_executor is None:
            return dispositions
        losses = self._scheduler.list_unadopted_coalesce_branch_losses(run_id=self._run_id)
        if not losses:
            return dispositions
        coordination_token = self._require_coordination_token()
        self._scheduler.adopt_coalesce_branch_losses(
            run_id=self._run_id,
            loss_ids=[loss.loss_id for loss in losses],
            now=self._clock.now_utc(),
            coordination_token=coordination_token,
        )
        for loss in losses:
            if self._coalesce_executor.has_recorded_branch_loss(loss.coalesce_name, loss.row_id, loss.branch_name):
                continue
            outcome = self._coalesce_executor.notify_branch_lost(
                coalesce_name=loss.coalesce_name,
                row_id=loss.row_id,
                lost_branch=loss.branch_name,
                reason=loss.reason,
            )
            if outcome is None:
                continue
            coalesce_name = CoalesceName(loss.coalesce_name)
            if outcome.merged_token is not None:
                dispositions.append(self._fire_coalesce_merge(coalesce_name, outcome, scope_row_id=loss.row_id))
                continue
            if outcome.failure_reason:
                self._mark_coalesce_consumed_terminal(
                    coalesce_name=coalesce_name,
                    consumed_tokens=tuple(outcome.consumed_tokens),
                )
                # Replayed must-fail (§E.5: a must-fail group fails within one
                # drain iteration of the loss becoming visible): mirror the
                # branch-loss notification failure arm — RowResults for the
                # held siblings the failure consumed.
                failure_results: list[RowResult] = []
                for consumed_token in outcome.consumed_tokens:
                    self._emit_token_completed(consumed_token, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)
                    failure_results.append(
                        RowResult(
                            token=consumed_token,
                            final_data=consumed_token.row_data,
                            outcome=TerminalOutcome.FAILURE,
                            path=TerminalPath.UNROUTED,
                            error=FailureInfo(exception_type="CoalesceFailure", message=outcome.failure_reason),
                        )
                    )
                dispositions.append(
                    BarrierIntakeDisposition(
                        kind=BarrierIntakeDispositionKind.TERMINAL,
                        results=tuple(failure_results),
                    )
                )
                continue
            raise OrchestrationInvariantError(
                f"Replayed branch loss {loss.loss_id!r} ({loss.coalesce_name!r}/{loss.row_id!r}/{loss.branch_name!r}) "
                f"produced an invalid CoalesceOutcome: held={outcome.held}, merged=None, failure_reason=None."
            )
        return dispositions


class BarrierRecoveryCoordinator:
    """F1 resume restore: rebuild barrier state from journal + audit tables."""

    def __init__(
        self,
        *,
        run_id: str,
        scheduler: TokenSchedulerRepository,
        barrier_restore_reads: BarrierRestoreReadModel,
        execution: ExecutionRepository,
        aggregation_executor: AggregationExecutor,
        coalesce_executor: CoalesceExecutor | None,
        clock: Clock,
        aggregation_settings: Mapping[NodeID, AggregationSettings],
        coalesce_node_ids: Mapping[CoalesceName, NodeID],
        coordination_token: CoordinationToken,
        scheduler_lease_owner: str,
    ) -> None:
        self._run_id = run_id
        self._scheduler = scheduler
        self._barrier_restore_reads = barrier_restore_reads
        self._execution = execution
        self._aggregation_executor = aggregation_executor
        self._coalesce_executor = coalesce_executor
        self._clock = clock
        self._aggregation_settings = aggregation_settings
        self._coalesce_node_ids = coalesce_node_ids
        self._coordination_token = coordination_token
        self._scheduler_lease_owner = scheduler_lease_owner

    def restore_from_journal(self, restore: BarrierJournalRestoreContext) -> None:
        """Rebuild aggregation buffers and coalesce pendings from journal BLOCKED rows.

        F1 resume path. The journal (token_work_items BLOCKED rows with a
        non-NULL barrier_key) is authoritative for buffered/held token
        payloads; counters, batch membership, hold state ids and attempt
        offsets derive from audit tables; the checkpoint contributes only
        the underivable scalars (``restore.barrier_scalars``).

        Discipline: ALL derivations complete (and raise, if they must) before
        any executor restore call mutates state. The coalesce restore is a
        single all-or-nothing call (its contract — a second call would discard
        the first); aggregation restores run per node afterwards, each
        internally validate-before-mutate.

        BLOCKED rows manufactured by the pre-epoch-20 blob-materialization
        restore path lacked ``barrier_blocked_at``; that writer is deleted and
        the epoch-20 delete-the-DB policy retired its rows, so the executors'
        NULL-means-corruption assert is honest — no live DB carries them.

        Raises:
            AuditIntegrityError: On any journal/audit disagreement (orphan
                barrier_key, missing/foreign batch ids, membership mismatch,
                NULL barrier_blocked_at, missing hold state ...).
        """
        now = self._clock.now_utc()
        # ADR-030 §E.4 belt: one run-wide duplicate-acceptance sweep at restore
        # entry. token_outcomes has NO non-terminal uniqueness — the adoption
        # CAS is the structural guard; >1 live BUFFERED rows for a token means
        # a deposed leader's unfenced intake wrote a second acceptance.
        duplicate_acceptances = self._barrier_restore_reads.find_duplicate_live_buffered_acceptances(self._run_id)
        if duplicate_acceptances:
            details = ", ".join(f"{token_id} ({count} live BUFFERED)" for token_id, count in duplicate_acceptances)
            raise AuditIntegrityError(
                f"Barrier journal restore for run {self._run_id!r} (resume checkpoint "
                f"{restore.resume_checkpoint_id!r}) found duplicate live BUFFERED acceptances: {details}. "
                "A deposed leader's unfenced intake wrote a second acceptance — refusing silent latest-wins."
            )
        items = self._scheduler.list_blocked_barrier_items(run_id=self._run_id)
        scalars = restore.barrier_scalars if restore.barrier_scalars is not None else BarrierScalars(aggregation={}, coalesce={})

        # ---- Partition (design D1) ----------------------------------------
        # A BLOCKED row's barrier KIND is decided by its barrier_key ONLY:
        # barrier_key == coalesce_name        -> coalesce barrier
        # barrier_key == str(aggregation node_id) -> aggregation barrier
        # NEVER partition on node_id (a BLOCKED row's node_id is the
        # enqueue-time cursor, not the barrier owner) and NEVER on
        # coalesce_name (aggregation rows may carry non-NULL coalesce_name
        # LINEAGE for tokens that will coalesce after the flush).
        coalesce_keys: set[str] = {str(name) for name in self._coalesce_node_ids}
        aggregation_keys: set[str] = {str(node_id) for node_id in self._aggregation_settings}
        ambiguous = coalesce_keys & aggregation_keys
        if ambiguous:
            raise OrchestrationInvariantError(
                f"Barrier-key namespace collision between coalesce names and aggregation node ids: {sorted(ambiguous)}. "
                "The journal restore partition cannot disambiguate BLOCKED rows for these keys."
            )

        agg_items_by_node: dict[NodeID, list[TokenWorkItem]] = {}
        coalesce_items: list[TokenWorkItem] = []
        intake_pending_count = 0
        for item in items:
            if item.barrier_key is None:  # pragma: no cover - excluded by the query contract
                raise AuditIntegrityError(
                    f"list_blocked_barrier_items returned a row without barrier_key "
                    f"(work_item_id={item.work_item_id!r}, run {self._run_id!r})."
                )
            if item.barrier_key not in coalesce_keys and item.barrier_key not in aggregation_keys:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} (run {self._run_id!r}, resume "
                    f"checkpoint {restore.resume_checkpoint_id!r}) carries orphan barrier_key "
                    f"{item.barrier_key!r}: not a configured coalesce ({sorted(coalesce_keys)}) "
                    f"or aggregation node ({sorted(aggregation_keys)}). The journal references a "
                    "barrier this pipeline no longer has — refusing to resume."
                )
            if item.barrier_adopted_epoch is None:
                # ADR-030 §E.2/§E.4: an intake-pending row was deposited but
                # never adopted (crash between mark_blocked and adoption, or a
                # mid-adoption rollback). It has NO batch_members/BUFFERED
                # rows and NO executor memory to restore — the legitimate
                # disposition is the next drain iteration's journal-first
                # intake, which adopts it under THIS leader's epoch.
                intake_pending_count += 1
                continue
            if item.barrier_key in coalesce_keys:
                coalesce_items.append(item)
            else:
                agg_items_by_node.setdefault(NodeID(item.barrier_key), []).append(item)
        if intake_pending_count:
            logger.info(
                "barrier journal restore: %d intake-pending BLOCKED row(s) left for the journal-first intake (run %s)",
                intake_pending_count,
                self._run_id,
            )
        # Adopted rows only from here down: derivations (attempt offsets,
        # batch ids, hold state ids) cover restored memory, not intake-pending
        # rows.
        items = [*coalesce_items, *(item for node_items in agg_items_by_node.values() for item in node_items)]
        if coalesce_items and self._coalesce_executor is None:
            raise OrchestrationInvariantError(
                f"Journal has {len(coalesce_items)} BLOCKED coalesce rows but no CoalesceExecutor is configured."
            )

        # ---- Audit derivations (no mutation yet) ---------------------------
        # Attempt offsets: max node_states attempt per journal token, + 1.
        # Derived here with ONE focused query rather than plumbed from
        # recovery's incomplete_by_row map: that map's exclusion set reads
        # journal BLOCKED rows (so the resume loop does not re-drive blocked
        # tokens), which excludes exactly the tokens this restore needs
        # offsets for.
        token_ids = [item.token_id for item in items]
        max_attempts = self._barrier_restore_reads.get_max_node_state_attempts(self._run_id, token_ids) if token_ids else {}
        attempt_offsets: dict[str, int] = {
            token_id: (max_attempts[token_id] if token_id in max_attempts else -1) + 1 for token_id in token_ids
        }

        # Per-node batch metadata for every configured aggregation node.
        agg_plans: list[_AggregationRestorePlan] = []
        if self._aggregation_settings:
            members_by_batch: dict[str, list[str]] = {}
            for member in self._execution.get_all_batch_members_for_run(self._run_id):
                members_by_batch.setdefault(member.batch_id, []).append(member.token_id)
            batches_by_node: dict[str, list[Batch]] = {}
            for batch in self._execution.get_batches(self._run_id):
                batches_by_node.setdefault(str(batch.aggregation_node_id), []).append(batch)

            for node_id in sorted(self._aggregation_settings, key=str):
                node_items = agg_items_by_node[node_id] if node_id in agg_items_by_node else []
                # Reconciliation join (configured nodes against audit batches): a
                # configured node with no batches yet (batches are created
                # lazily on first row arrival) legitimately has zero rows, so
                # absence is an empty bucket, not Tier-1 corruption.
                node_batches = batches_by_node[str(node_id)] if str(node_id) in batches_by_node else []
                # COUNT(DISTINCT token_id), not raw COUNT: retry_batch COPIES
                # members into the retry batch, and handle_incomplete_batches
                # runs BEFORE this derivation on resume — a raw count would
                # double-count every member of a retried batch.
                accepted_count_total = len(
                    {
                        member_token
                        for node_batch in node_batches
                        for member_token in (members_by_batch[node_batch.batch_id] if node_batch.batch_id in members_by_batch else ())
                    }
                )
                completed_flush_count = sum(1 for node_batch in node_batches if node_batch.status is BatchStatus.COMPLETED)
                node_scalars = (
                    scalars.aggregation[str(node_id)] if str(node_id) in scalars.aggregation else AggregationNodeScalars(None, None)
                )
                # ---- ADR-030 §E.3a aggregation reconcile (elspeth-55546a6fd6) ---
                # A FAILED out-of-claim flush records terminal FAILURE/UNROUTED
                # token_outcomes for every buffered token (_handle_flush_error)
                # and THEN releases their BLOCKED scheduler rows in a SEPARATE
                # transaction (_mark_buffered_scheduler_work_terminal). A crash
                # between the two strands durable BLOCKED rows whose tokens are
                # already terminally failed: they carry NO live BUFFERED outcome,
                # so _derive_restored_batch_id below would refuse loudly and
                # brick EVERY resume attempt. Mirror the coalesce §E.3a holdless
                # path: the tokens are done — journal-release their orphaned
                # BLOCKED rows here (under this leader's coordination token) and
                # drop them from the restore set so the deriver sees only live
                # tokens. A fully-reconciled node then falls through to the
                # counter-only branch below ("flushes all FAILED" — exactly the
                # state that branch already anticipates).
                #
                # Scoped to (FAILURE, UNROUTED): the success-path BATCH_CONSUMED
                # crash residual (elspeth-3977d8ab60) still owes a sink output
                # and is NOT swept here — it keeps hitting the loud refusal.
                if node_items:
                    failed_terminal_ids = self._barrier_restore_reads.find_failed_unrouted_terminal_token_ids(
                        self._run_id, [item.token_id for item in node_items]
                    )
                    if failed_terminal_ids:
                        reconciled = [item for item in node_items if item.token_id in failed_terminal_ids]
                        released = self._scheduler.mark_blocked_barrier_terminal(
                            run_id=self._run_id,
                            barrier_key=str(node_id),
                            token_ids=tuple(item.token_id for item in reconciled),
                            now=now,
                            coordination_token=self._coordination_token,
                            release_context={
                                "reason": "failed_flush_crash_reconcile",
                                "released_by": self._scheduler_lease_owner,
                                "restore_reconcile": True,
                            },
                        )
                        if released != len(reconciled):
                            raise AuditIntegrityError(
                                f"Restore §E.3a aggregation reconcile: FAILED-flush release at aggregation node "
                                f"{node_id!r} (run {self._run_id!r}, resume checkpoint {restore.resume_checkpoint_id!r}) "
                                f"terminalized {released} rows; expected exactly {len(reconciled)} orphaned "
                                "terminally-failed BLOCKED row(s)."
                            )
                        logger.info(
                            "barrier journal restore: §E.3a aggregation reconcile released %d orphaned BLOCKED row(s) "
                            "with terminal FAILURE/UNROUTED outcomes at node %s (run %s)",
                            len(reconciled),
                            node_id,
                            self._run_id,
                        )
                        node_items = [item for item in node_items if item.token_id not in failed_terminal_ids]
                if node_items:
                    batch_id = self._derive_restored_batch_id(node_id, node_items, restore)
                    agg_plans.append(
                        _AggregationRestorePlan(
                            node_id=node_id,
                            items=node_items,
                            member_order=(members_by_batch[batch_id] if batch_id in members_by_batch else []),
                            batch_id=batch_id,
                            accepted_count_total=accepted_count_total,
                            completed_flush_count=completed_flush_count,
                            scalars=node_scalars,
                        )
                    )
                elif completed_flush_count > 0 or accepted_count_total > 0 or str(node_id) in scalars.aggregation:
                    # Counter-only node: nothing buffered, but the audit trail
                    # shows prior activity — completed flushes, accepted rows
                    # (a node whose flushes all FAILED has accepted > 0 with
                    # zero COMPLETED batches and an empty buffer), or a stale
                    # scalars entry. Restore the derived counters so post-flush
                    # pagination metadata survives the resume.
                    agg_plans.append(
                        _AggregationRestorePlan(
                            node_id=node_id,
                            items=[],
                            member_order=[],
                            batch_id=None,
                            accepted_count_total=accepted_count_total,
                            completed_flush_count=completed_flush_count,
                            scalars=node_scalars,
                        )
                    )

        # Coalesce hold state ids: the OPEN node_state written at accept()
        # time at the coalesce node (the executor calls it the PENDING hold).
        coalesce_state_ids: Mapping[str, str] = {}
        if coalesce_items:
            coalesce_state_ids = self._barrier_restore_reads.get_open_node_state_ids(
                self._run_id,
                node_ids=[str(node_id) for node_id in self._coalesce_node_ids.values()],
                token_ids=[item.token_id for item in coalesce_items],
            )

        # ---- ADR-030 §E.3a/§E.4 crash-window reconcile (findings 1 & 3) -----
        # Adopted coalesce rows with no OPEN state_id are in a crash window:
        # the adoption CAS committed (barrier_adopted_epoch non-NULL) but
        # accept() never wrote the PENDING hold node_state (the leader died
        # between steps 1 and 2 of the coalesce intake adoption).
        #
        # Two sub-cases, identified by whether the row's (coalesce_name, row_id)
        # key is Landscape-completed:
        #
        # a. Key completed (late-arrival crash §E.3a): the group already
        #    resolved; journal-release the row here at restore exactly like the
        #    live §E.3a path (mark_blocked_barrier_terminal with late_arrival
        #    context), using the new leader's coordination token.
        #
        # b. Key NOT completed (normal adoption crash §E.3): accept() never ran
        #    — re-run it after restore_from_journal populates executor state,
        #    writing the missing hold node_state under a fresh attempt offset.
        #    The row was already adopted, so the intake IS-NULL filter won't
        #    pick it up; this post-restore accept is the only recovery path.
        #
        # This replaces the old hard-refusal in restore_from_journal's
        # state_ids check, which fired on both reachable crash states.
        coalesce_holdless_items: list[TokenWorkItem] = []
        if coalesce_items:
            holdless = [item for item in coalesce_items if item.token_id not in coalesce_state_ids]
            if holdless:
                # Resolve the Landscape completed set once for all holdless rows.
                node_id_to_coalesce_name: dict[str, str] = {str(nid): str(name) for name, nid in self._coalesce_node_ids.items()}
                completed_pairs = self._barrier_restore_reads.get_completed_row_ids_for_nodes(
                    self._run_id,
                    frozenset(node_id_to_coalesce_name.keys()),
                )
                completed_keys_set: frozenset[tuple[str, str]] = frozenset(
                    (node_id_to_coalesce_name[node_id_str], row_id)
                    for node_id_str, row_id in completed_pairs
                    if node_id_str in node_id_to_coalesce_name
                )
                for item in holdless:
                    coalesce_name_str = item.coalesce_name or item.barrier_key
                    key = (str(coalesce_name_str), item.row_id)
                    if key in completed_keys_set:
                        # §E.3a at restore: adopted-but-unreleased late row
                        # against a completed key — journal-release it now.
                        released = self._scheduler.mark_blocked_barrier_terminal(
                            run_id=self._run_id,
                            barrier_key=str(coalesce_name_str),
                            token_ids=(item.token_id,),
                            now=now,
                            coordination_token=self._coordination_token,
                            release_context={
                                "late_arrival": True,
                                "reason": "late_arrival_after_merge",
                                "released_by": self._scheduler_lease_owner,
                                "scope_row_id": item.row_id,
                                "restore_reconcile": True,
                            },
                        )
                        if released != 1:
                            raise AuditIntegrityError(
                                f"Restore §E.3a reconcile: late-arrival release for token {item.token_id!r} "
                                f"at coalesce {coalesce_name_str!r} (run {self._run_id!r}) terminalized "
                                f"{released} rows; expected exactly one."
                            )
                        logger.info(
                            "barrier journal restore: §E.3a reconcile released adopted-holdless late-arrival "
                            "token %s at coalesce %s/%s (run %s)",
                            item.token_id,
                            coalesce_name_str,
                            item.row_id,
                            self._run_id,
                        )
                    else:
                        # Normal adoption crash: accept() never ran — collect for
                        # post-restore accept (after restore_from_journal populates
                        # completed_keys so the executor can do late-arrival detection).
                        coalesce_holdless_items.append(item)
                # Remove ALL holdless items from the restore list; reconciled ones
                # are journal-released above, deferred ones handled post-restore.
                coalesce_items = [item for item in coalesce_items if item.token_id in coalesce_state_ids]

        # §E.5/§E.4: the durable coalesce_branch_losses ledger is the restore
        # source of truth for lost_branches across takeover; the D3 checkpoint
        # scalar is retained as a cross-check only (union, table wins on a
        # reason disagreement — the first durable record may already have
        # fired a must-fail policy).
        effective_coalesce_scalars: dict[tuple[str, str], CoalescePendingScalars] = dict(scalars.coalesce)
        if self._coalesce_executor is not None:
            for loss in self._scheduler.list_coalesce_branch_losses(run_id=self._run_id):
                key = (loss.coalesce_name, loss.row_id)
                existing = effective_coalesce_scalars[key] if key in effective_coalesce_scalars else None
                lost_branches = dict(existing.lost_branches) if existing is not None else {}
                checkpoint_reason = lost_branches[loss.branch_name] if loss.branch_name in lost_branches else None
                if checkpoint_reason is not None and checkpoint_reason != loss.reason:
                    logger.warning(
                        "coalesce restore: checkpoint lost-branch reason %r for %s/%s/%s disagrees with the durable "
                        "loss ledger %r; the ledger wins",
                        checkpoint_reason,
                        loss.coalesce_name,
                        loss.row_id,
                        loss.branch_name,
                        loss.reason,
                    )
                lost_branches[loss.branch_name] = loss.reason
                effective_coalesce_scalars[key] = CoalescePendingScalars(lost_branches=lost_branches)

        # ---- Mutate ---------------------------------------------------------
        # Coalesce first: ONE call for the whole executor (a second call would
        # discard this one — see CoalesceExecutor.restore_from_journal's caller
        # obligations). Called whenever a coalesce executor exists so completed
        # keys are reconstructed from the Landscape even with nothing pending,
        # and stale lost-branch scalars are dropped-with-log.
        if self._coalesce_executor is not None:
            # ---- §E.3 crash-window recovery: reset holdless non-completed rows ---
            # Adopted BLOCKED coalesce rows with no OPEN state_id whose key is NOT
            # completed are in the crash window between adopt_blocked_barrier_item
            # commit and CoalesceExecutor.accept() — accept() never wrote the hold.
            # Resetting barrier_adopted_epoch to NULL re-classifies them as
            # intake-pending: the first drain iteration's journal-first intake adopts
            # them afresh and runs the full accept + trigger path (merge, failure,
            # late-arrival), which the restore phase cannot safely produce (accept()
            # may fire a merge whose RowResult the __init__ path cannot commit to the
            # journal).  This reset is safe because the takeover CAS has already
            # committed — the old leader cannot concurrently re-adopt these rows.
            if coalesce_holdless_items:
                reset_count = self._scheduler.reset_adoption_marker_to_pending(
                    work_item_ids=[item.work_item_id for item in coalesce_holdless_items],
                    run_id=self._run_id,
                )
                if reset_count != len(coalesce_holdless_items):
                    logger.warning(
                        "barrier journal restore: §E.3 crash-window reset expected %d rows but "
                        "reset %d (run %s); the intake will re-classify any missed rows",
                        len(coalesce_holdless_items),
                        reset_count,
                        self._run_id,
                    )
                else:
                    logger.info(
                        "barrier journal restore: §E.3 crash-window reset %d holdless-non-completed "
                        "BLOCKED coalesce row(s) to intake-pending (run %s)",
                        reset_count,
                        self._run_id,
                    )
            self._coalesce_executor.restore_from_journal(
                items=coalesce_items,
                scalars=effective_coalesce_scalars,
                state_ids=coalesce_state_ids,
                attempt_offsets=attempt_offsets,
                resume_checkpoint_id=restore.resume_checkpoint_id,
                now=now,
            )

        for plan in agg_plans:
            self._aggregation_executor.restore_from_journal(
                node_id=plan.node_id,
                items=plan.items,
                member_order=plan.member_order,
                batch_id=plan.batch_id,
                accepted_count_total=plan.accepted_count_total,
                completed_flush_count=plan.completed_flush_count,
                scalars=plan.scalars,
                attempt_offsets=attempt_offsets,
                resume_checkpoint_id=restore.resume_checkpoint_id,
                now=now,
            )

    def _derive_restored_batch_id(
        self,
        node_id: NodeID,
        node_items: Sequence[TokenWorkItem],
        restore: BarrierJournalRestoreContext,
    ) -> str:
        """Resolve the in-progress batch id for one aggregation node's BLOCKED rows.

        Source of truth: each buffered token's BUFFERED token_outcome carries
        the batch_id it was accepted into (written by the fenced adoption
        verb since ADR-030 §E.2), read through the
        ``handle_incomplete_batches`` old->retry remap because the audit
        outcomes still reference the dead original batch when a crash
        interrupted a flush.

        Raises:
            AuditIntegrityError: If any token lacks a BUFFERED outcome with a
                batch_id, the group's tokens disagree on the (remapped)
                batch_id, the batch row is missing, or the batch belongs to a
                different aggregation node.
        """
        batch_id: str | None = None
        first_token_id: str | None = None
        for item in node_items:
            live_buffered = self._barrier_restore_reads.list_live_buffered_outcomes(TokenRef(token_id=item.token_id, run_id=self._run_id))
            if len(live_buffered) > 1:
                # ADR-030 §C.4 row 6a / §E.4: token_outcomes has no non-terminal
                # uniqueness; >1 live BUFFERED rows means a deposed leader's
                # unfenced intake wrote a second acceptance. Refuse loudly —
                # never the historical silent latest-wins.
                duplicates = "; ".join(
                    f"outcome_id={o.outcome_id!r} batch_id={o.batch_id!r} "
                    f"recorded_at={o.recorded_at.isoformat()} context={o.context_json!r}"
                    for o in live_buffered
                )
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at aggregation node {node_id!r} "
                    f"(run {self._run_id!r}, resume checkpoint {restore.resume_checkpoint_id!r}) has "
                    f"{len(live_buffered)} live BUFFERED token_outcomes — duplicate acceptances; a deposed "
                    "leader's unfenced intake wrote a second acceptance; refusing silent latest-wins. "
                    f"{duplicates}"
                )
            outcome = live_buffered[0] if live_buffered else None
            if outcome is None or outcome.batch_id is None:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at aggregation node {node_id!r} "
                    f"(run {self._run_id!r}, resume checkpoint {restore.resume_checkpoint_id!r}) has no "
                    f"matching BUFFERED token_outcome with a batch_id (got {outcome!r}) — the journal "
                    "and the audit trail disagree about this token being buffered."
                )
            resolved = restore.batch_id_remap.get(outcome.batch_id, outcome.batch_id)
            if batch_id is None:
                batch_id = resolved
                first_token_id = item.token_id
            elif resolved != batch_id:
                raise AuditIntegrityError(
                    f"BLOCKED journal rows at aggregation node {node_id!r} (run {self._run_id!r}, resume "
                    f"checkpoint {restore.resume_checkpoint_id!r}) split across batches: token "
                    f"{first_token_id!r} resolves batch_id={batch_id!r} but token {item.token_id!r} "
                    f"resolves batch_id={resolved!r}. One node has exactly one in-progress batch."
                )
        if batch_id is None:  # pragma: no cover - callers pass non-empty node_items
            raise AuditIntegrityError(f"_derive_restored_batch_id called with no journal rows for node {node_id!r}.")
        batch = self._execution.get_batch(batch_id)
        if batch is None:
            raise AuditIntegrityError(
                f"Restored batch_id {batch_id!r} for aggregation node {node_id!r} (run {self._run_id!r}) "
                "has no batches row — audit data corruption."
            )
        if str(batch.aggregation_node_id) != str(node_id):
            raise AuditIntegrityError(
                f"Restored batch_id {batch_id!r} belongs to aggregation node "
                f"{batch.aggregation_node_id!r}, but the journal BLOCKED rows carry barrier_key "
                f"{str(node_id)!r} (run {self._run_id!r}) — journal/audit disagreement."
            )
        return batch_id
