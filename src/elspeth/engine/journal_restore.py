"""Journal restore boundary for barrier executors (F1 resume path).

The crash-resume hydration logic for both stateful barriers lives here, out
of the live executors: each ``*JournalRestorer`` validates journal BLOCKED
rows against audit-derived inputs (state ids, attempt offsets, batch
membership, checkpoint scalars) and returns a frozen, already-validated
state object. The executor's ``restore_from_journal`` stays a thin facade —
it builds the restorer, applies the returned state, and keeps only live-path
responsibilities (state replacement, late-arrival point lookup).

Restore-specific invariants (validation order, corruption messages, staleness
handling) therefore evolve in this module without touching runtime barrier
behavior, and vice versa.

Both restorers share one design shape:

* validate-before-anything: every journal/audit disagreement raises
  ``AuditIntegrityError`` before any state object is built;
* token payloads rehydrate through ``token_from_journal_item`` (the journal
  row is authoritative for payload and lineage);
* pending/trigger age derives from the absolute ``barrier_blocked_at`` stamp
  of the OLDEST row against the caller-supplied wall clock, clamped at 0
  against backward wall-clock steps.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from elspeth.contracts import TokenInfo
from elspeth.contracts.barrier_scalars import AggregationNodeScalars, CoalescePendingScalars
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.scheduler import TokenWorkItem
from elspeth.contracts.types import NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.scheduler_repository import token_from_journal_item

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock

slog = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Coalesce restore state (frozen — the executor applies, never re-validates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RestoredCoalesceBranch:
    """One rehydrated arrived branch within a restored pending coalesce."""

    branch_name: str
    token: TokenInfo
    arrival_time: float  # Monotonic timestamp (blocked-at offsets preserved)
    state_id: str  # Landscape node_state ID for the PENDING hold


@dataclass(frozen=True, slots=True)
class RestoredPendingCoalesce:
    """One validated pending coalesce key ready to apply to the executor.

    ``branches`` preserves journal grouping order (insertion order of the
    BLOCKED rows) so the executor's rebuilt branch dict iterates identically
    to the pre-extraction restore.
    """

    key: tuple[str, str]  # (coalesce_name, row_id)
    branches: tuple[RestoredCoalesceBranch, ...]
    first_arrival: float  # Monotonic anchor of the OLDEST branch
    lost_branches: Mapping[str, str]

    def __post_init__(self) -> None:
        freeze_fields(self, "lost_branches")


@dataclass(frozen=True, slots=True)
class RestoredCoalesceState:
    """Whole-executor restored coalesce state (single-shot apply)."""

    pending: tuple[RestoredPendingCoalesce, ...]
    completed_keys: tuple[tuple[str, str], ...]
    token_count: int


class CoalesceJournalRestorer:
    """Validates and hydrates coalesce restore inputs into a frozen state object.

    Owns the restore-side half of ``CoalesceExecutor.restore_from_journal``:
    journal validation, token rehydration, scalar-only (zero-arrival
    loss-record) handling, and completed-key reconstruction from the
    Landscape. The executor keeps state replacement and the late-arrival
    Landscape point lookup.
    """

    def __init__(
        self,
        *,
        settings: Mapping[str, CoalesceSettings],
        node_ids: Mapping[str, NodeID],
        execution: ExecutionRepository,
        run_id: str,
        clock: "Clock",
    ) -> None:
        """Initialize restorer.

        Args:
            settings: Registered coalesce configurations, keyed by name.
            node_ids: Registered coalesce node ids, keyed by coalesce name —
                used to reconstruct completed keys from the Landscape.
            execution: Execution repository (Landscape audit reads only).
            run_id: Run identifier for error context and audit queries.
            clock: Clock supplying the monotonic scale restored arrival
                anchors are expressed on.
        """
        self._settings = settings
        self._node_ids = node_ids
        self._execution = execution
        self._run_id = run_id
        self._clock = clock

    def restore(
        self,
        *,
        items: Sequence[TokenWorkItem],
        scalars: Mapping[tuple[str, str], CoalescePendingScalars],
        state_ids: Mapping[str, str],
        attempt_offsets: Mapping[str, int],
        resume_checkpoint_id: str,
        now: datetime,
    ) -> RestoredCoalesceState:
        """Validate journal rows and build the restored coalesce state.

        Argument semantics are documented on the facade
        (``CoalesceExecutor.restore_from_journal``), which forwards verbatim.

        Raises:
            AuditIntegrityError: On any journal/audit disagreement — NULL
                barrier_blocked_at, missing branch_name or coalesce_name,
                unknown coalesce, duplicate journal rows, duplicate branch
                claims, missing attempt offset, missing state_id.
        """
        # Validate and group ALL items before building any state — if
        # validation fails, the executor's in-memory state must remain intact
        # for error recovery (same discipline as the old blob restore).
        grouped: dict[tuple[str, str], dict[str, TokenWorkItem]] = {}
        blocked_at_by_token: dict[str, datetime] = {}
        for item in items:
            if not item.coalesce_name:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} (run {self._run_id!r}, "
                    f"resume checkpoint {resume_checkpoint_id!r}) has no coalesce_name — "
                    "coalesce barrier rows always carry the coalesce cursor; journal corruption."
                )
            if item.coalesce_name not in self._settings:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} (run {self._run_id!r}, "
                    f"resume checkpoint {resume_checkpoint_id!r}) references unknown coalesce "
                    f"'{item.coalesce_name}'. Configured coalesces: {sorted(self._settings)}"
                )
            if item.barrier_blocked_at is None:
                # Every post-epoch-20 BLOCKED row is stamped by mark_blocked.
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at coalesce "
                    f"{item.coalesce_name!r} (run {self._run_id!r}, resume checkpoint "
                    f"{resume_checkpoint_id!r}) has NULL barrier_blocked_at — journal "
                    "corruption (every BLOCKED row is stamped at mark_blocked time)."
                )
            if not item.branch_name:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at coalesce "
                    f"{item.coalesce_name!r} (run {self._run_id!r}, resume checkpoint "
                    f"{resume_checkpoint_id!r}) has no branch_name — only forked branch "
                    "tokens block at a coalesce barrier; journal corruption."
                )
            if item.branch_name not in self._settings[item.coalesce_name].branches:
                # The live accept() path rejects unknown branches; restore must
                # apply the same allowlist (elspeth-a840cb774a) — a rogue branch
                # inflates quorum/best_effort arrival counts while contributing
                # no merge data.
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at coalesce "
                    f"{item.coalesce_name!r} (run {self._run_id!r}, resume checkpoint "
                    f"{resume_checkpoint_id!r}) claims branch '{item.branch_name}' which is "
                    f"not in the configured branches {sorted(self._settings[item.coalesce_name].branches)} — "
                    "journal corruption."
                )
            if item.token_id in blocked_at_by_token:
                raise AuditIntegrityError(
                    f"Duplicate BLOCKED journal rows for token {item.token_id!r} at "
                    f"coalesce {item.coalesce_name!r} (run {self._run_id!r}, resume "
                    f"checkpoint {resume_checkpoint_id!r}) — journal corruption."
                )
            if item.token_id not in attempt_offsets:
                raise AuditIntegrityError(
                    f"No entry in attempt_offsets for journal token {item.token_id!r} at "
                    f"coalesce {item.coalesce_name!r} (run {self._run_id!r}, resume "
                    f"checkpoint {resume_checkpoint_id!r}) — audit-derived offsets must "
                    "cover every BLOCKED journal row."
                )
            if item.token_id not in state_ids:
                # The PENDING node_state hold is written at accept() time, before
                # the journal row blocks; a BLOCKED row with no hold means the
                # journal and the audit trail disagree — corruption, not a default.
                raise AuditIntegrityError(
                    f"No entry in state_ids for journal token {item.token_id!r} at "
                    f"coalesce {item.coalesce_name!r} (run {self._run_id!r}, resume "
                    f"checkpoint {resume_checkpoint_id!r}) — every BLOCKED coalesce row "
                    "holds a PENDING node_state in the audit trail; a missing hold is "
                    "an audit inconsistency."
                )

            key = (item.coalesce_name, item.row_id)
            if key not in grouped:
                grouped[key] = {}
            branch_items = grouped[key]
            if item.branch_name in branch_items:
                raise AuditIntegrityError(
                    f"BLOCKED journal rows for tokens "
                    f"{branch_items[item.branch_name].token_id!r} and {item.token_id!r} "
                    f"both claim branch '{item.branch_name}' at coalesce "
                    f"{item.coalesce_name!r} for row {item.row_id!r} (run {self._run_id!r}, "
                    f"resume checkpoint {resume_checkpoint_id!r}) — accept() crashes on a "
                    "duplicate arrival, so this is journal corruption."
                )
            branch_items[item.branch_name] = item
            blocked_at_by_token[item.token_id] = item.barrier_blocked_at

        monotonic_now = self._clock.monotonic()
        restored_pending: dict[tuple[str, str], RestoredPendingCoalesce] = {}
        for key, branch_items in grouped.items():
            min_blocked_at = min(blocked_at_by_token[it.token_id] for it in branch_items.values())
            # Pending age derives from the absolute blocked-at stamp of the
            # OLDEST branch (first arrival of this pending key), clamped at 0:
            # a wall-clock backward step must not put first_arrival in the
            # monotonic future.
            first_arrival = monotonic_now - max(0.0, (now - min_blocked_at).total_seconds())
            branches: list[RestoredCoalesceBranch] = []
            for branch_name, branch_item in branch_items.items():
                token = token_from_journal_item(
                    branch_item,
                    attempt_offset=attempt_offsets[branch_item.token_id],
                    resume_checkpoint_id=resume_checkpoint_id,
                )
                branches.append(
                    RestoredCoalesceBranch(
                        branch_name=branch_name,
                        token=token,
                        arrival_time=first_arrival + (blocked_at_by_token[branch_item.token_id] - min_blocked_at).total_seconds(),
                        state_id=state_ids[branch_item.token_id],
                    )
                )

            key_scalars = scalars[key] if key in scalars else None
            lost_branches = dict(key_scalars.lost_branches) if key_scalars is not None else {}
            allowed_branches = self._settings[key[0]].branches
            unknown_lost = set(lost_branches) - set(allowed_branches)
            if unknown_lost:
                raise AuditIntegrityError(
                    f"Checkpoint lost_branches for coalesce {key[0]!r} row {key[1]!r} "
                    f"(run {self._run_id!r}, resume checkpoint {resume_checkpoint_id!r}) "
                    f"name branches {sorted(unknown_lost)} outside the configured branches "
                    f"{sorted(allowed_branches)} — checkpoint corruption."
                )
            arrived_and_lost = {b.branch_name for b in branches} & set(lost_branches)
            if arrived_and_lost:
                # Mirrors the live notify_branch_lost invariant: a branch
                # cannot both arrive and be lost for the same pending key.
                raise AuditIntegrityError(
                    f"Branches {sorted(arrived_and_lost)} at coalesce {key[0]!r} row {key[1]!r} "
                    f"(run {self._run_id!r}, resume checkpoint {resume_checkpoint_id!r}) "
                    "are recorded as both arrived and lost — journal/checkpoint corruption."
                )
            restored_pending[key] = RestoredPendingCoalesce(
                key=key,
                branches=tuple(branches),
                first_arrival=first_arrival,
                lost_branches=lost_branches,
            )

        # Reconstruct completed keys from Landscape (source of truth). The
        # executor seeds them into its bounded FIFO cache at apply time.
        # Queried BEFORE the executor mutates anything: a Landscape error
        # mid-restore must not leave the executor cleared-but-unrestored.
        completed_keys = self._reconstruct_completed_keys_from_landscape()
        completed_key_set = set(completed_keys)

        # Scalar-only entries have no arrived branch payloads in the journal.
        # If the Landscape says the key completed, the scalar is an older
        # checkpoint image and must be dropped. Otherwise, a non-empty
        # lost_branches scalar is the durable image of a zero-arrival pending
        # key: restore it so a later surviving branch accounts against the
        # recorded loss instead of forming a fresh, loss-free pending key.
        for scalar_key in scalars.keys() - grouped.keys():
            coalesce_name, row_id = scalar_key
            key_scalars = scalars[scalar_key]
            lost_branches = dict(key_scalars.lost_branches)
            if coalesce_name in self._settings and lost_branches and scalar_key not in completed_key_set:
                unknown_lost = set(lost_branches) - set(self._settings[coalesce_name].branches)
                if unknown_lost:
                    # An unknown BRANCH inside a configured, non-completed
                    # coalesce's scalars is corruption, not staleness — only
                    # unknown-coalesce / completed / empty keys drop-and-log.
                    raise AuditIntegrityError(
                        f"Checkpoint lost_branches for coalesce {coalesce_name!r} row {row_id!r} "
                        f"(run {self._run_id!r}, resume checkpoint {resume_checkpoint_id!r}) "
                        f"name branches {sorted(unknown_lost)} outside the configured branches "
                        f"{sorted(self._settings[coalesce_name].branches)} — checkpoint corruption."
                    )
                restored_pending[scalar_key] = RestoredPendingCoalesce(
                    key=scalar_key,
                    branches=(),
                    first_arrival=monotonic_now,
                    lost_branches=lost_branches,
                )
                slog.info(
                    "coalesce_journal_restored_loss_only_scalars",
                    coalesce_name=coalesce_name,
                    row_id=row_id,
                    run_id=self._run_id,
                    resume_checkpoint_id=resume_checkpoint_id,
                    lost_branches=lost_branches,
                )
                continue
            slog.info(
                "coalesce_journal_restore_dropped_stale_scalars",
                coalesce_name=coalesce_name,
                row_id=row_id,
                run_id=self._run_id,
                resume_checkpoint_id=resume_checkpoint_id,
                lost_branches=lost_branches,
            )

        return RestoredCoalesceState(
            pending=tuple(restored_pending.values()),
            completed_keys=tuple(completed_keys),
            token_count=len(blocked_at_by_token),
        )

    def _reconstruct_completed_keys_from_landscape(self) -> list[tuple[str, str]]:
        """Read completed coalesce keys from the Landscape audit trail.

        Queries node_states for completed entries at coalesce node IDs, joined
        with tokens to get row_ids. Maps node_id → coalesce_name via the
        reverse of the registered node_ids.

        This is the restore seeding path: the Landscape records all completed
        coalesces, but the executor keeps only a bounded FIFO performance
        cache. Late arrivals for evicted keys are rediscovered through an exact
        Landscape point lookup (which stays on the executor).
        """
        if not self._node_ids:
            return []

        # Build reverse map: node_id → coalesce_name
        node_id_to_name: dict[str, str] = {str(nid): name for name, nid in self._node_ids.items()}

        completed_pairs = self._execution.get_completed_row_ids_for_nodes(
            run_id=self._run_id,
            node_ids=frozenset(node_id_to_name.keys()),
        )

        return [(node_id_to_name[node_id_str], row_id) for node_id_str, row_id in completed_pairs if node_id_str in node_id_to_name]


# ---------------------------------------------------------------------------
# Aggregation restore state (frozen — the executor applies, never re-validates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RestoredTriggerLatch:
    """Validated trigger-evaluator restore arguments for an in-progress batch.

    Present only when the node has buffered journal rows; a counter-only
    node restores with no latch (the executor resets its trigger instead —
    see ``AggregationJournalRestorer.restore`` for why stale latches are
    dropped rather than replanted).
    """

    batch_count: int
    elapsed_age_seconds: float
    count_fire_offset: float | None
    condition_fire_offset: float | None


@dataclass(frozen=True, slots=True)
class RestoredAggregationState:
    """One node's validated aggregation restore state, ready to apply.

    ``tokens`` is in batch_members.ordinal order (the authoritative accept
    order); buffers derive from it at apply time.
    """

    node_id: NodeID
    tokens: tuple[TokenInfo, ...]
    batch_id: str | None
    accepted_count_total: int
    completed_flush_count: int
    trigger_latch: RestoredTriggerLatch | None

    @property
    def elapsed_age_seconds(self) -> float:
        """Restored batch age for observability (0.0 for a counter-only node)."""
        return self.trigger_latch.elapsed_age_seconds if self.trigger_latch is not None else 0.0


class AggregationJournalRestorer:
    """Validates and hydrates one aggregation node's restore inputs.

    Owns the restore-side half of ``AggregationExecutor.restore_from_journal``:
    journal validation, journal-vs-batch_members reconciliation, batch/counter
    sanity checks, token rehydration in member order, and the trigger-latch
    staleness decision. The executor keeps applying the returned state to its
    node buffers and trigger evaluator.
    """

    def __init__(self, *, run_id: str) -> None:
        """Initialize restorer.

        Args:
            run_id: Run identifier for error context.
        """
        self._run_id = run_id

    def restore(
        self,
        *,
        node_id: NodeID,
        items: Sequence[TokenWorkItem],
        member_order: Sequence[str],
        batch_id: str | None,
        accepted_count_total: int,
        completed_flush_count: int,
        scalars: AggregationNodeScalars,
        attempt_offsets: Mapping[str, int],
        resume_checkpoint_id: str,
        now: datetime,
    ) -> RestoredAggregationState:
        """Validate journal rows and build one node's restored state.

        Argument semantics are documented on the facade
        (``AggregationExecutor.restore_from_journal``), which forwards
        verbatim after resolving the node.

        Raises:
            AuditIntegrityError: On any journal/audit disagreement — NULL
                barrier_blocked_at, duplicate journal rows, membership
                mismatch, duplicate member_order entries, missing attempt
                offset, batch_id/items inconsistency, impossible counters.
        """
        tokens_by_id: dict[str, TokenInfo] = {}
        oldest_blocked_at: datetime | None = None
        for item in items:
            if item.barrier_blocked_at is None:
                # Every post-epoch-20 BLOCKED row is stamped by mark_blocked.
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at aggregation node "
                    f"{node_id!r} (run {self._run_id!r}, resume checkpoint "
                    f"{resume_checkpoint_id!r}) has NULL barrier_blocked_at — journal "
                    "corruption (every BLOCKED row is stamped at mark_blocked time)."
                )
            if item.token_id in tokens_by_id:
                raise AuditIntegrityError(
                    f"Duplicate BLOCKED journal rows for token {item.token_id!r} at "
                    f"aggregation node {node_id!r} (run {self._run_id!r}, resume "
                    f"checkpoint {resume_checkpoint_id!r}) — journal corruption."
                )
            try:
                attempt_offset = attempt_offsets[item.token_id]
            except KeyError:
                raise AuditIntegrityError(
                    f"No entry in attempt_offsets for journal token {item.token_id!r} at "
                    f"aggregation node {node_id!r} (run {self._run_id!r}, resume "
                    f"checkpoint {resume_checkpoint_id!r}) — audit-derived offsets must "
                    "cover every BLOCKED journal row."
                ) from None

            tokens_by_id[item.token_id] = token_from_journal_item(
                item,
                attempt_offset=attempt_offset,
                resume_checkpoint_id=resume_checkpoint_id,
            )
            if oldest_blocked_at is None or item.barrier_blocked_at < oldest_blocked_at:
                oldest_blocked_at = item.barrier_blocked_at

        self._reconcile_journal_batch_members(
            node_id=node_id,
            journal_token_ids=tokens_by_id.keys(),
            member_order=member_order,
        )

        # batch_id/items must agree: buffered journal rows imply an in-progress
        # batch; a batch_id with zero BLOCKED rows means batch membership
        # advanced past the journal (or vice versa) — corruption either way.
        if items and batch_id is None:
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r} (run {self._run_id!r}, resume checkpoint "
                f"{resume_checkpoint_id!r}) has {len(items)} BLOCKED journal rows but no "
                "batch_id — buffered tokens always belong to an in-progress batch."
            )
        if not items and batch_id is not None:
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r} (run {self._run_id!r}, resume checkpoint "
                f"{resume_checkpoint_id!r}) has batch_id {batch_id!r} but no BLOCKED "
                "journal rows — an in-progress batch must have blocked members."
            )

        # Counter sanity: the cumulative accept counter covers every currently
        # buffered row, so accepted_count_total < len(items) (or any negative
        # counter) is impossible audit state. Restoring it would silently emit
        # row_start <= 0 in the next flush's pagination metadata.
        if completed_flush_count < 0 or accepted_count_total < len(items):
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r} (run {self._run_id!r}, resume checkpoint "
                f"{resume_checkpoint_id!r}): audit-derived counters are impossible "
                f"(accepted_count_total={accepted_count_total}, "
                f"completed_flush_count={completed_flush_count}, buffered={len(items)}). "
                "accepted_count_total must cover every buffered row and counters must "
                "be non-negative."
            )

        ordered_tokens = tuple(tokens_by_id[token_id] for token_id in member_order)

        # Trigger age derives from the absolute blocked-at stamp of the OLDEST
        # buffered row (first accept of the in-progress batch), clamped at 0
        # against clock skew.
        trigger_latch: RestoredTriggerLatch | None
        if oldest_blocked_at is not None:
            trigger_latch = RestoredTriggerLatch(
                batch_count=len(ordered_tokens),
                elapsed_age_seconds=max(0.0, (now - oldest_blocked_at).total_seconds()),
                count_fire_offset=scalars.count_fire_offset,
                condition_fire_offset=scalars.condition_fire_offset,
            )
        else:
            # Counter-only node: no in-progress batch, and trigger latches are
            # batch-scoped — so any non-None scalars are STALE (the checkpoint
            # predates the journal: crash after a flush terminalized the
            # BLOCKED rows but before the next checkpoint — a legitimate
            # window under D3's staleness model, so rejecting would refuse
            # valid resumes). Drop them (logged) and have the executor leave
            # the trigger fully unlatched via reset(): restoring a latch here
            # would plant a phantom first-accept anchor at restore time that
            # survives into the NEXT genuine batch (record_accept min-rewinds
            # first_accept_time but never clears it, so a phantom anchor
            # lingers whenever the genuine arrivals come later) → wrong
            # timeout age and, with latched offsets, a pre-fired
            # count/condition latch.
            trigger_latch = None
            if scalars.count_fire_offset is not None or scalars.condition_fire_offset is not None:
                slog.info(
                    "aggregation_journal_restore_dropped_stale_scalars",
                    node_id=str(node_id),
                    run_id=self._run_id,
                    resume_checkpoint_id=resume_checkpoint_id,
                    count_fire_offset=scalars.count_fire_offset,
                    condition_fire_offset=scalars.condition_fire_offset,
                )

        return RestoredAggregationState(
            node_id=node_id,
            tokens=ordered_tokens,
            batch_id=batch_id,
            accepted_count_total=accepted_count_total,
            completed_flush_count=completed_flush_count,
            trigger_latch=trigger_latch,
        )

    def _reconcile_journal_batch_members(
        self,
        *,
        node_id: NodeID,
        journal_token_ids: Iterable[str],
        member_order: Sequence[str],
    ) -> None:
        """Ensure journal BLOCKED rows and persisted batch_members agree as SETS.

        This is the F1 descendant of the old checkpoint-vs-batch_members
        reconcile. It degenerates to set equality because ``member_order`` IS
        the batch_members.ordinal ordering (derived by the caller) — comparing
        ordered tuples would compare batch_members against itself, proving
        nothing. The real cross-check left is membership: a token with a
        BLOCKED journal row but no batch_members row (or vice versa) means the
        journal and the audit trail disagree about the in-progress batch.
        """
        journal_set = set(journal_token_ids)
        member_set = set(member_order)
        if len(member_set) != len(member_order):
            duplicated = sorted(token_id for token_id, count in Counter(member_order).items() if count > 1)
            raise AuditIntegrityError(
                f"Duplicate token ids in batch_members order for aggregation node "
                f"{node_id!r} (run {self._run_id!r}): {duplicated!r} — audit trail corruption."
            )
        if journal_set != member_set:
            missing_from_journal = sorted(member_set - journal_set)
            missing_from_members = sorted(journal_set - member_set)
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r} (run {self._run_id!r}): journal BLOCKED "
                f"rows and persisted batch_members disagree about batch membership. "
                f"In batch_members but not journal: {missing_from_journal!r}; "
                f"in journal but not batch_members: {missing_from_members!r}. "
                "Cannot safely resume this batch."
            )
