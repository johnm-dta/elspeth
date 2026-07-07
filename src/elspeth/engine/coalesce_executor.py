"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.contracts import TokenInfo
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.barrier_scalars import CoalescePendingScalars
from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry, CoalesceMetadata
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    AuditIntegrityError,
    CoalesceCollisionError,
    CoalesceFailureReason,
    ContractMergeError,
    ExecutionError,
    OrchestrationInvariantError,
)
from elspeth.contracts.scheduler import TokenWorkItem
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.contracts.union_merge import merge_union_contracts
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.coalesce_policy import CoalesceAction, CoalesceEvent, decide_coalesce
from elspeth.engine.journal_restore import CoalesceJournalRestorer
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.core.landscape.scheduler import BarrierRestoreReadModel
    from elspeth.engine.clock import Clock
    from elspeth.engine.tokens import TokenManager

slog = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CoalesceOutcome:
    """Result of a coalesce accept operation.

    Attributes:
        held: True if token is being held waiting for more branches
        merged_token: The merged token if merge is complete, None if held
        consumed_tokens: Tokens that were merged (marked COALESCED)
        coalesce_metadata: Audit metadata about the merge (branches, policy, etc.)
        failure_reason: Reason for failure if merge failed (timeout, missing branches)
        coalesce_name: Name of the coalesce point that produced this outcome
        outcomes_recorded: True if terminal outcomes were already recorded by executor.
            When True, caller MUST NOT record outcomes again (Bug 9z8 fix).
        late_arrival: True when this failure outcome is the late-arrival arm —
            the token arrived after its group already merged/failed (ADR-030
            §E.3a): the journal-first intake releases the token's BLOCKED row
            via ``mark_blocked_barrier_terminal`` with a ``late_arrival``
            release context instead of the standard group-failure consumption.
    """

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: tuple[TokenInfo, ...] = ()
    coalesce_metadata: CoalesceMetadata | None = None
    failure_reason: str | None = None
    coalesce_name: str | None = None
    outcomes_recorded: bool = False
    late_arrival: bool = False

    def __post_init__(self) -> None:
        # Validate mutual exclusivity of states
        if self.held:
            if self.merged_token is not None:
                raise OrchestrationInvariantError("CoalesceOutcome: held=True but merged_token is set — mutually exclusive states")
            if self.failure_reason is not None:
                raise OrchestrationInvariantError("CoalesceOutcome: held=True but failure_reason is set — mutually exclusive states")
        if self.merged_token is not None and self.failure_reason is not None:
            raise OrchestrationInvariantError("CoalesceOutcome: both merged_token and failure_reason are set — mutually exclusive states")


@dataclass(frozen=True, slots=True)
class _BranchEntry:
    """Per-branch state within a pending coalesce.

    Groups token, arrival time, and audit state_id that were previously
    scattered across three parallel dicts.  Frozen to prevent mutation
    after construction — a new entry is created per branch arrival.
    """

    token: TokenInfo
    arrival_time: float  # Monotonic timestamp of arrival
    state_id: str  # Landscape node_state ID for pending hold


@dataclass
class _PendingCoalesce:
    """Tracks pending tokens for a single row_id at a coalesce point."""

    branches: dict[str, _BranchEntry]  # branch_name -> entry
    first_arrival: float  # For timeout calculation
    lost_branches: dict[str, str] = field(default_factory=dict)  # branch_name -> loss reason


def _resolve_first_wins(
    merged: dict[str, Any],
    field_origins: dict[str, str],
    collision_values: dict[str, list[tuple[str, Any]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Rewrite merged + origins so collisions resolve to the first branch's value.

    Non-colliding fields pass through unchanged. Only the ``collision_values``
    entries are consulted — the first entry in each list is the first branch
    in ``settings.branches`` order that produced that field.
    """
    new_merged = dict(merged)
    new_origins = dict(field_origins)
    for collision_field, entries in collision_values.items():
        if len(entries) < 2:
            raise OrchestrationInvariantError(
                f"_resolve_first_wins: collision_values[{collision_field!r}] has "
                f"{len(entries)} entries; expected >=2. This indicates a bug in "
                "_merge_data collision seeding."
            )
        first_branch, first_value = entries[0]
        new_merged[collision_field] = first_value
        new_origins[collision_field] = first_branch
    return new_merged, new_origins


_MergeDataFn = Callable[
    [CoalesceSettings, dict[str, _BranchEntry]],
    tuple[
        dict[str, Any],
        dict[str, list[str]],
        dict[str, str],
        dict[str, list[tuple[str, Any]]],
    ],
]
_MergeWithOriginalNamesFn = Callable[
    [SchemaContract, dict[str, SchemaContract], dict[str, str]],
    SchemaContract,
]


@dataclass(frozen=True, slots=True)
class CoalesceMergePlan:
    """Pure merge plan produced before token or audit side effects."""

    merged_data: PipelineRow
    consumed_tokens: tuple[TokenInfo, ...]
    metadata: CoalesceMetadata


def _lost_branch_expected_fields(
    branch_expected_fields: Mapping[str, tuple[str, ...]] | None,
    lost_branches: Mapping[str, str],
) -> dict[str, tuple[str, ...]] | None:
    if branch_expected_fields is None:
        return None
    if not lost_branches:
        return None

    result: dict[str, tuple[str, ...]] = {}
    for branch_name in lost_branches:
        if branch_name in branch_expected_fields:
            result[branch_name] = tuple(branch_expected_fields[branch_name])
    return result if result else None


def _merge_with_original_names(
    precomputed: SchemaContract,
    branch_contracts: dict[str, SchemaContract],
    field_origins: dict[str, str],
) -> SchemaContract:
    """Merge precomputed schema semantics with original names from branch contracts."""
    # Build lookup of (normalized_name, branch_name) -> original_name from all branches
    # so we can retrieve original_name from the winning branch per field.
    branch_original_names: dict[tuple[str, str], str] = {}
    for branch_name, contract in branch_contracts.items():
        for fc in contract.fields:
            branch_original_names[(fc.normalized_name, branch_name)] = fc.original_name

    # Fields not contributed by any arrived branch fall back to first-seen
    # original names from arrived contracts, then normalized names.
    fallback_original_names: dict[str, str] = {}
    for contract in branch_contracts.values():
        for fc in contract.fields:
            if fc.normalized_name not in fallback_original_names:
                fallback_original_names[fc.normalized_name] = fc.original_name

    merged_fields: list[FieldContract] = []
    for fc in precomputed.fields:
        if fc.normalized_name in field_origins:
            winning_branch = field_origins[fc.normalized_name]
            original_name = branch_original_names[(fc.normalized_name, winning_branch)]
        else:
            original_name = fallback_original_names.get(fc.normalized_name, fc.normalized_name)
        merged_fields.append(
            FieldContract(
                normalized_name=fc.normalized_name,
                original_name=original_name,
                python_type=fc.python_type,
                required=fc.required,
                source=fc.source,
                nullable=fc.nullable,
            )
        )

    return SchemaContract(
        mode=precomputed.mode,
        fields=tuple(merged_fields),
        locked=precomputed.locked,
    )


def _merge_data(
    settings: CoalesceSettings,
    branches: dict[str, _BranchEntry],
) -> tuple[
    dict[str, Any],
    dict[str, list[str]],
    dict[str, str],
    dict[str, list[tuple[str, Any]]],
]:
    """Merge row data from arrived tokens based on strategy."""
    if settings.merge == "union":
        # Combine all fields from all branches.
        # On name collision, the last branch in settings.branches wins by default
        # (union_collision_policy="last_wins"). field_origins and collision_values
        # are always recorded so auditors can reconstruct lineage and inspect
        # overwritten values. Policy enforcement happens in build_coalesce_merge().
        merged: dict[str, Any] = {}
        field_origins: dict[str, str] = {}
        collisions: dict[str, list[str]] = {}
        collision_values: dict[str, list[tuple[str, Any]]] = {}
        for branch_name in settings.branches:
            if branch_name in branches:
                branch_data = branches[branch_name].token.row_data.to_dict()
                for merge_field, value in branch_data.items():
                    if merge_field in field_origins:
                        if merge_field not in collisions:
                            prior_branch = field_origins[merge_field]
                            prior_value = merged[merge_field]
                            collisions[merge_field] = [prior_branch]
                            collision_values[merge_field] = [(prior_branch, prior_value)]
                        collisions[merge_field].append(branch_name)
                        collision_values[merge_field].append((branch_name, value))
                    field_origins[merge_field] = branch_name
                    merged[merge_field] = value
        return merged, collisions, field_origins, collision_values

    if settings.merge == "nested":
        return (
            {branch_name: branches[branch_name].token.row_data.to_dict() for branch_name in settings.branches if branch_name in branches},
            {},
            {},
            {},
        )

    if settings.merge == "select":
        if settings.select_branch is None:
            raise RuntimeError(
                f"select_branch is None for select merge strategy at coalesce '{settings.name}'. This indicates a config validation bug."
            )
        if settings.select_branch not in branches:
            raise RuntimeError(
                f"select_branch '{settings.select_branch}' not in arrived branches {list(branches.keys())}. "
                f"This indicates a bug in _execute_merge validation (Bug 2ho fix should have caught this)."
            )
        return branches[settings.select_branch].token.row_data.to_dict(), {}, {}, {}

    raise RuntimeError(f"Unknown merge strategy: {settings.merge!r}")


def build_coalesce_merge(
    *,
    settings: CoalesceSettings,
    pending: _PendingCoalesce,
    coalesce_name: str,
    now: float,
    output_schema: SchemaContract | None,
    branch_expected_fields: Mapping[str, tuple[str, ...]] | None,
    merge_data: _MergeDataFn | None = None,
    merge_with_original_names: _MergeWithOriginalNamesFn | None = None,
) -> CoalesceMergePlan:
    """Build the pure merge plan; callers apply token and audit side effects."""
    merge_data_fn = merge_data or _merge_data
    merge_with_original_names_fn = merge_with_original_names or _merge_with_original_names

    merged_data_dict, union_collisions, field_origins, collision_values = merge_data_fn(settings, pending.branches)

    branch_contracts: dict[str, SchemaContract] = {}
    for branch_name, entry in pending.branches.items():
        contract = entry.token.row_data.contract
        if contract is None:
            raise OrchestrationInvariantError(
                f"Token {entry.token.token_id} on branch '{branch_name}' has no contract. "
                f"Cannot coalesce without contracts on all parents. "
                f"This indicates a bug in fork or transform execution."
            )
        branch_contracts[branch_name] = contract
    contracts: list[SchemaContract] = list(branch_contracts.values())

    if settings.merge == "union":
        all_branches_observed = all(c.mode == "OBSERVED" for c in contracts)

        if output_schema is not None:
            precomputed = output_schema
            use_precomputed = precomputed.mode != "OBSERVED"
        else:
            if not all_branches_observed:
                raise OrchestrationInvariantError(
                    f"Coalesce '{settings.name}' has typed branch contracts but no "
                    f"output_schema for union merge. The DAG builder must provide "
                    f"output_schema via register_coalesce(). "
                    f"If this is a test, use TestCoalesceExecutor from conftest."
                )
            precomputed = None
            use_precomputed = False

        if use_precomputed:
            assert precomputed is not None
            merged_contract = merge_with_original_names_fn(precomputed, branch_contracts, field_origins)
        else:
            merged_contract = merge_union_contracts(
                branch_contracts,
                require_all=settings.has_all_branch_semantics,
                collision_policy=settings.union_collision_policy,
                branch_order=tuple(settings.branches),
                coalesce_id=settings.name,
            )

    elif settings.merge == "nested":
        branch_fields = tuple(
            FieldContract(
                original_name=branch_name,
                normalized_name=branch_name,
                python_type=object,
                required=branch_name in pending.branches,
                source="declared",
            )
            for branch_name in settings.branches
        )
        merged_contract = SchemaContract(
            fields=branch_fields,
            mode="FIXED",
            locked=True,
        )

    elif settings.merge == "select":
        if settings.select_branch is None:
            raise RuntimeError(
                f"select_branch is None for select merge strategy at coalesce '{settings.name}'. This indicates a config validation bug."
            )
        selected_entry = pending.branches[settings.select_branch]
        selected_contract = selected_entry.token.row_data.contract
        if selected_contract is None:
            raise OrchestrationInvariantError(
                f"Token {selected_entry.token.token_id} on branch '{settings.select_branch}' has no contract. "
                f"Cannot coalesce without contracts on all parents. "
                f"This indicates a bug in fork or transform execution."
            )
        merged_contract = selected_contract

    else:
        raise RuntimeError(f"Unknown merge strategy: {settings.merge}")

    coalesce_metadata = CoalesceMetadata.for_merge(
        policy=CoalescePolicy(settings.policy),
        merge_strategy=MergeStrategy(settings.merge),
        expected_branches=tuple(settings.branches),
        branches_arrived=tuple(pending.branches.keys()),
        branches_lost=pending.lost_branches,
        lost_branch_expected_fields=_lost_branch_expected_fields(branch_expected_fields, pending.lost_branches),
        arrival_order=[
            ArrivalOrderEntry(
                branch=branch,
                arrival_offset_ms=(entry.arrival_time - pending.first_arrival) * 1000,
            )
            for branch, entry in sorted(pending.branches.items(), key=lambda x: x[1].arrival_time)
        ],
        wait_duration_ms=(now - pending.first_arrival) * 1000,
    )

    if settings.merge == "union":
        coalesce_metadata = CoalesceMetadata.with_union_result(
            coalesce_metadata,
            field_origins=field_origins,
            collisions=union_collisions if union_collisions else None,
            collision_values=collision_values if collision_values else None,
        )

        if union_collisions:
            if settings.union_collision_policy == "fail":
                raise CoalesceCollisionError(
                    f"union merge collisions in coalesce '{coalesce_name}': {sorted(union_collisions)}",
                    metadata=coalesce_metadata,
                )
            if settings.union_collision_policy == "first_wins":
                merged_data_dict, first_wins_origins = _resolve_first_wins(merged_data_dict, field_origins, collision_values)
                if use_precomputed:
                    assert precomputed is not None
                    merged_contract = merge_with_original_names_fn(precomputed, branch_contracts, first_wins_origins)
                coalesce_metadata = replace(
                    coalesce_metadata,
                    union_field_origins=first_wins_origins,
                )

    return CoalesceMergePlan(
        merged_data=PipelineRow(merged_data_dict, merged_contract),
        consumed_tokens=tuple(e.token for e in pending.branches.values()),
        metadata=coalesce_metadata,
    )


class CoalesceExecutor:
    """Executes coalesce operations with audit recording.

    Maintains state for pending coalesce operations:
    - Tracks which tokens have arrived for each row_id
    - Evaluates merge conditions based on policy
    - Merges row data according to strategy
    - Records audit trail via ExecutionRepository

    Example:
        executor = CoalesceExecutor(execution, span_factory, token_manager, run_id, step_resolver, data_flow=data_flow)

        # Configure coalesce point
        executor.register_coalesce(settings, node_id)

        # Accept tokens as they arrive
        for token in arriving_tokens:
            outcome = executor.accept(token, "coalesce_name")
            if outcome.merged_token:
                # Merged token continues through pipeline
                work_queue.append(outcome.merged_token)
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        span_factory: SpanFactory,
        token_manager: "TokenManager",
        run_id: str,
        step_resolver: StepResolver,
        data_flow: DataFlowRepository,
        clock: "Clock | None" = None,
        max_completed_keys: int = 10000,
        barrier_restore_reads: "BarrierRestoreReadModel | None" = None,
    ) -> None:
        """Initialize executor.

        Args:
            execution: Execution repository for audit trail
            span_factory: Span factory for tracing
            token_manager: TokenManager for creating merged tokens
            run_id: Run identifier for audit context
            step_resolver: Resolves NodeID to 1-indexed audit step position.
                           Injected at construction to eliminate step_in_pipeline
                           threading through public method signatures.
            data_flow: Data flow repository for token outcome recording.
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
            max_completed_keys: Maximum late-arrival completion keys retained in memory.
        """
        if max_completed_keys <= 0:
            raise OrchestrationInvariantError(f"max_completed_keys must be > 0, got {max_completed_keys}")
        if barrier_restore_reads is None:
            raise OrchestrationInvariantError("barrier_restore_reads is required for coalesce restore/late-arrival reads")

        self._execution = execution
        self._barrier_restore_reads = barrier_restore_reads
        self._data_flow = data_flow
        self._spans = span_factory
        self._token_manager = token_manager
        self._run_id = run_id
        self._step_resolver = step_resolver
        self._clock = clock if clock is not None else DEFAULT_CLOCK

        # Coalesce configuration: name -> settings
        self._settings: dict[str, CoalesceSettings] = {}
        # Node IDs: coalesce_name -> node_id
        self._node_ids: dict[str, NodeID] = {}
        # Branch schemas: coalesce_name -> branch_name -> guaranteed fields tuple
        # Used to record expected fields when a branch is lost (diverted to error sink).
        # This enables audit queries like "what fields were expected from lost branch X?"
        # NOTE: Populated by register_coalesce(), which the orchestrator calls BEFORE
        # restore_from_journal(). Branch schemas come from fresh graph data each run,
        # not from resume state — the journal stores token payloads and the checkpoint
        # row stores only lost-branch scalars.
        self._branch_expected_fields: dict[str, dict[str, tuple[str, ...]] | None] = {}
        # Pre-computed output schemas: coalesce_name -> SchemaContract
        # Used to ensure runtime contracts match DAG-computed schemas (P2 fix).
        # When populated, _execute_merge() uses this instead of runtime merge().
        self._output_schemas: dict[str, SchemaContract | None] = {}
        # Pending tokens: (coalesce_name, row_id) -> _PendingCoalesce
        self._pending: dict[tuple[str, str], _PendingCoalesce] = {}
        # Completed coalesces: tracks keys that have already merged/failed
        # Used to detect late arrivals after merge and reject them gracefully
        # Uses OrderedDict as bounded FIFO set to prevent unbounded memory growth
        # (values are None, we only care about key presence and insertion order)
        self._completed_keys: OrderedDict[tuple[str, str], None] = OrderedDict()
        # Maximum completed keys to retain (prevents OOM in long-running pipelines).
        # Configurable to match source cardinality and memory budget.
        self._max_completed_keys: int = max_completed_keys

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: NodeID,
        branch_schemas: dict[str, tuple[str, ...]] | None = None,
        output_schema: SchemaContract | None = None,
    ) -> None:
        """Register a coalesce point.

        Args:
            settings: Coalesce configuration
            node_id: Node ID assigned by orchestrator
            branch_schemas: Optional mapping of branch name to tuple of guaranteed
                field names. Used to record expected fields when a branch is lost.
                Keys are branch names from settings.branches; values are guaranteed
                fields from that branch's producer schema. May be None for pipelines
                using observed-mode schemas where no fields are declared.
            output_schema: Optional pre-computed output schema from DAG builder.
                When provided (and typed), used directly in union merge so the
                runtime contract matches the build-time computation exactly;
                all-OBSERVED unions merge branch contracts at runtime with
                merge_union_contracts(), which shares the same core algorithm.
        """
        self._settings[settings.name] = settings
        self._node_ids[settings.name] = node_id
        self._branch_expected_fields[settings.name] = branch_schemas
        self._output_schemas[settings.name] = output_schema

    def get_registered_names(self) -> list[str]:
        """Get names of all registered coalesce points.

        Used by processor for timeout checking loop.

        Returns:
            List of registered coalesce names
        """
        return list(self._settings.keys())

    def get_barrier_scalars(self) -> dict[tuple[str, str], CoalescePendingScalars]:
        """Return the underivable lost-branch scalars for the checkpoint row.

        F1 design D3: the checkpoint persists ONLY scalar barrier metadata —
        arrived-branch token payloads live in journal BLOCKED rows and state
        ids derive from audit tables at restore time. The only underivable
        coalesce state is the lost_branches record per pending key (loss
        notifications are in-memory events with no journal row of their own).

        Emission choice: only keys with a non-empty lost_branches record are
        emitted. The checkpoint writer serializes None when no scalars exist
        (``BarrierScalars.has_state``), and restore treats a missing entry as
        empty lost_branches — so emitting loss-free pending keys would add
        bytes without information. This mirrors aggregation's only-latched
        emission (Task 2.1).

        Returns:
            Mapping of (coalesce_name, row_id) -> CoalescePendingScalars for
            pending keys with recorded losses only.
        """
        scalars: dict[tuple[str, str], CoalescePendingScalars] = {}
        for key, pending in self._pending.items():
            if pending.lost_branches:
                scalars[key] = CoalescePendingScalars(lost_branches=dict(pending.lost_branches))
        return scalars

    def restore_from_journal(
        self,
        *,
        items: Sequence[TokenWorkItem],
        scalars: Mapping[tuple[str, str], CoalescePendingScalars],
        state_ids: Mapping[str, str],
        attempt_offsets: Mapping[str, int],
        resume_checkpoint_id: str,
        now: datetime,
    ) -> None:
        """Rebuild pending coalesces from journal BLOCKED rows (F1 resume path).

        Replaces the checkpoint-blob restore: the journal (token_work_items
        BLOCKED rows) is authoritative for arrived-branch token payloads; the
        caller (processor, Task 3.1) partitions journal items by barrier kind
        and derives state ids / attempt offsets from audit tables. Items group
        by ``(coalesce_name, row_id)``; per group, ``first_arrival`` anchors to
        the OLDEST ``barrier_blocked_at`` expressed on the executor's monotonic
        clock (clamped at 0 against wall-clock skew), and each branch's
        arrival_time preserves the absolute blocked-at offsets.

        Validation, token rehydration, scalar-only handling, and completed-key
        reconstruction from the Landscape (source of truth) live in
        ``CoalesceJournalRestorer`` (the restore/hydration boundary); this
        method applies the returned frozen state wholesale. Late-arrival
        point lookups stay on the executor (``_check_landscape_for_completion``).

        Caller obligations (Task 3.1):

        * **Call exactly once per resume.** Unlike the per-node aggregation
          twin, this method restores the WHOLE executor in one call — it ends
          by replacing ``_pending`` and ``_completed_keys`` wholesale, so a
          second call (e.g. looped per barrier key) discards prior restored
          state. Pass ALL coalesce journal items in one ``items`` sequence.
        * **Do not re-notify restored losses.** ``lost_branches`` restored
          from ``scalars`` are already recorded — re-deriving losses from
          audit and calling ``notify_branch_lost`` for a key that already
          carries them raises ``OrchestrationInvariantError`` (duplicate
          loss). For zero-arrival loss-only keys restored from scalars, replay
          dedup consults ``has_recorded_branch_loss`` before notifying again.

        Args:
            items: BLOCKED journal rows for coalesce barriers (all keys).
            scalars: Per-pending-key lost_branches records from the checkpoint
                row. A key with journal items but no scalars entry restores
                with empty lost_branches (the writer only emits keys with
                recorded losses — D3). A scalars entry whose key has NO
                journal items but non-empty lost_branches is a zero-arrival
                pending key unless the Landscape says the key already
                completed. Empty, unknown, or completed scalar-only entries
                are stale and are dropped with a log line rather than rejected.
            state_ids: Per-token node_state hold id — the PENDING node_states
                at the coalesce node, one per restored token (written at
                accept() time, derived from audit tables). Every journal
                token must be covered — a BLOCKED row whose hold is absent
                is an audit inconsistency.
            attempt_offsets: Per-token resume attempt offset (max_attempt + 1).
            resume_checkpoint_id: Checkpoint id stamped on restored tokens
                (resume provenance).
            now: Current wall-clock time (tz-aware) — pending age derives from
                ``now - min(barrier_blocked_at)``, not from an offset blob.

        Raises:
            AuditIntegrityError: On any journal/audit disagreement — NULL
                barrier_blocked_at, missing branch_name or coalesce_name,
                unknown coalesce, duplicate journal rows, duplicate branch
                claims, missing attempt offset, missing state_id.
        """
        restored = CoalesceJournalRestorer(
            settings=self._settings,
            node_ids=self._node_ids,
            barrier_restore_reads=self._barrier_restore_reads,
            run_id=self._run_id,
            clock=self._clock,
        ).restore(
            items=items,
            scalars=scalars,
            state_ids=state_ids,
            attempt_offsets=attempt_offsets,
            resume_checkpoint_id=resume_checkpoint_id,
            now=now,
        )

        # Apply the validated state wholesale — the restorer has already
        # raised on any journal/audit disagreement (validate-before-mutate:
        # a failed restore leaves the executor's in-memory state intact).
        self._pending.clear()
        self._completed_keys.clear()
        for completed_key in restored.completed_keys:
            self._mark_completed(completed_key)
        for group in restored.pending:
            self._pending[group.key] = _PendingCoalesce(
                branches={
                    branch.branch_name: _BranchEntry(
                        token=branch.token,
                        arrival_time=branch.arrival_time,
                        state_id=branch.state_id,
                    )
                    for branch in group.branches
                },
                first_arrival=group.first_arrival,
                lost_branches=dict(group.lost_branches),
            )

        slog.info(
            "coalesce_journal_restored",
            pending_keys=len(restored.pending),
            token_count=restored.token_count,
            run_id=self._run_id,
            resume_checkpoint_id=resume_checkpoint_id,
        )

    def _check_landscape_for_completion(self, coalesce_name: str, row_id: str) -> bool:
        """Check the Landscape for whether a coalesce key has completed.

        Cache-miss fallback for late-arrival detection. When the FIFO cache
        (self._completed_keys) doesn't contain a key, this queries the exact
        Landscape row before allowing a new pending entry. If the Landscape
        shows the coalesce completed, the key is added to the cache and
        the token is treated as a late arrival.

        This eliminates the FIFO eviction window: evicted keys are
        rediscovered from the Landscape on the next lookup.

        Args:
            coalesce_name: Coalesce point name
            row_id: Source row ID

        Returns:
            True if the Landscape shows this coalesce already completed
        """
        if coalesce_name not in self._node_ids:
            return False
        node_id = self._node_ids[coalesce_name]

        if self._barrier_restore_reads.has_completed_row_for_node(run_id=self._run_id, node_id=str(node_id), row_id=row_id):
            self._mark_completed((coalesce_name, row_id))
            return True
        return False

    def _mark_completed(self, key: tuple[str, str]) -> None:
        """Mark a coalesce key as completed with bounded memory.

        Uses FIFO eviction to prevent unbounded memory growth in long-running
        pipelines. When max capacity is exceeded, oldest entries are removed.
        Late arrivals after eviction are caught by the Landscape fallback in
        accept() — the FIFO is a performance cache, not a correctness mechanism.

        Args:
            key: (coalesce_name, row_id) tuple to mark as completed
        """
        self._completed_keys[key] = None
        # Evict oldest entries if over capacity.
        # Eviction is harmless: Landscape fallback in accept() catches
        # late arrivals for evicted keys.
        while len(self._completed_keys) > self._max_completed_keys:
            self._completed_keys.popitem(last=False)

    def accept(
        self,
        token: TokenInfo,
        coalesce_name: str,
        *,
        arrival_time: float | None = None,
    ) -> CoalesceOutcome:
        """Accept a token at a coalesce point.

        If merge conditions are met, returns the merged token.
        Otherwise, holds the token and returns held=True.

        ``arrival_time`` (ADR-030 §E.2 backdated accept timing): an explicit
        monotonic anchor for this arrival — the journal-first intake passes
        the row's durable ``barrier_blocked_at`` converted onto this clock's
        monotonic scale, so ``first_arrival`` (the timeout anchor) and the
        branch's ``arrival_time`` are invariant under leader takeover (§H
        pinned doctrine). ``None`` preserves the live-clock anchor.

        Step position is resolved internally via the injected StepResolver
        from the coalesce point's registered node_id.

        Resume state (attempt offset and checkpoint provenance) is read from the
        token itself (token.resume_attempt_offset, token.resume_checkpoint_id).
        The node_state for the arriving branch token is written under the arriving
        token's token_id — the offset must come from that token, not from a
        WorkItem-level scalar, so that each token in a multi-branch resume carries
        its own (possibly distinct) offset.

        Args:
            token: Token arriving at coalesce point (must have branch_name);
                token.resume_attempt_offset and token.resume_checkpoint_id carry
                the resume state for this specific arriving token.
            coalesce_name: Name of the coalesce configuration

        Returns:
            CoalesceOutcome indicating whether token was held or merged

        Raises:
            OrchestrationInvariantError: If coalesce_name not registered, token has no
                branch_name, or branch is not in the expected set
        """
        if coalesce_name not in self._settings:
            raise OrchestrationInvariantError(f"Coalesce '{coalesce_name}' not registered")

        if token.branch_name is None:
            raise OrchestrationInvariantError(f"Token {token.token_id} has no branch_name - only forked tokens can be coalesced")

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]
        step = self._step_resolver(node_id)

        # Validate branch is expected
        if token.branch_name not in settings.branches:
            raise OrchestrationInvariantError(
                f"Token branch '{token.branch_name}' not in expected branches for coalesce '{coalesce_name}': {settings.branches}"
            )

        # Get or create pending state for this row
        key = (coalesce_name, token.row_id)
        now = arrival_time if arrival_time is not None else self._clock.monotonic()

        # Check if this coalesce already completed (late arrival).
        # Two-level lookup: FIFO cache first, then Landscape fallback.
        # The FIFO is a performance optimization; the Landscape is the
        # source of truth. Evicted FIFO entries are rediscovered from
        # the Landscape, eliminating the eviction window.
        if key in self._completed_keys or self._check_landscape_for_completion(coalesce_name, token.row_id):
            # Late arrival after merge/failure already happened
            # Record failure audit trail for this late token
            failure_reason = "late_arrival_after_merge"
            error_hash = compute_error_hash(failure_reason)
            state = self._execution.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                run_id=self._run_id,
                step_index=step,
                input_data=token.row_data.to_dict(),  # Recorder expects dict
                attempt=token.resume_attempt_offset,
                resume_checkpoint_id=token.resume_checkpoint_id,
            )
            error = CoalesceFailureReason(
                failure_reason=failure_reason,
                expected_branches=tuple(settings.branches),
                branches_arrived=(),  # Late arrival — merge already happened
                merge_policy=settings.merge,
            )
            self._execution.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                error=error,
                duration_ms=0,
            )
            if self._data_flow is None:
                raise OrchestrationInvariantError(
                    "CoalesceExecutor.data_flow is None but token outcome recording requires DataFlowRepository"
                )
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=error_hash,
            )

            # Return failure outcome
            return CoalesceOutcome(
                held=False,
                failure_reason=failure_reason,
                consumed_tokens=(token,),
                coalesce_metadata=CoalesceMetadata.for_late_arrival(
                    policy=CoalescePolicy(settings.policy),
                    reason="Siblings already merged/failed, this token arrived too late",
                ),
                coalesce_name=coalesce_name,
                outcomes_recorded=True,
                late_arrival=True,
            )

        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                branches={},
                first_arrival=now,
            )

        pending = self._pending[key]
        if now < pending.first_arrival:
            # min-anchor: §H doctrine pins the timeout anchor to the OLDEST
            # member's durable arrival; a backdated adoption arriving out of
            # blocked-at order must still rewind the anchor.
            pending.first_arrival = now

        # Detect duplicate arrivals (indicates bug in upstream code)
        # Per "Plugin Ownership" principle: bugs in our code should crash, not hide
        if token.branch_name in pending.branches:
            existing = pending.branches[token.branch_name]
            raise OrchestrationInvariantError(
                f"Duplicate arrival for branch '{token.branch_name}' at coalesce '{coalesce_name}'. "
                f"Existing token: {existing.token.token_id}, new token: {token.token_id}. "
                f"This indicates a bug in fork, retry, or checkpoint/resume logic."
            )

        # Record pending node state for audit trail FIRST,
        # then store entry atomically (all per-branch state in one assignment)
        state = self._execution.begin_node_state(
            token_id=token.token_id,
            node_id=node_id,
            run_id=self._run_id,
            step_index=step,
            input_data=token.row_data.to_dict(),  # Recorder expects dict
            attempt=token.resume_attempt_offset,
            resume_checkpoint_id=token.resume_checkpoint_id,
        )
        pending.branches[token.branch_name] = _BranchEntry(
            token=token,
            arrival_time=now,
            state_id=state.state_id,
        )

        # Check if merge conditions are met
        if self._should_merge(settings, pending):
            return self._execute_merge(
                settings=settings,
                node_id=node_id,
                pending=pending,
                step=step,
                key=key,
                coalesce_name=coalesce_name,
            )

        # Hold token - audit trail already recorded above
        return CoalesceOutcome(held=True, coalesce_name=coalesce_name)

    def _should_merge(
        self,
        settings: CoalesceSettings,
        pending: _PendingCoalesce,
    ) -> bool:
        """Check if merge conditions are met based on policy.

        Thin delegate over :func:`decide_coalesce` (ARRIVAL event) — the
        policy matrix lives in ``elspeth.engine.coalesce_policy``. ARRIVAL
        never yields FAIL, so a boolean is a faithful projection of the
        decision.
        """
        decision = decide_coalesce(
            settings,
            CoalesceEvent.ARRIVAL,
            arrived_count=len(pending.branches),
            lost_branches=pending.lost_branches,
        )
        return decision.action is CoalesceAction.MERGE

    def _get_lost_branch_expected_fields(
        self,
        coalesce_name: str,
        lost_branches: dict[str, str],
    ) -> dict[str, tuple[str, ...]] | None:
        """Look up expected fields for lost branches.

        Returns a mapping of branch name to the tuple of guaranteed field names
        that branch would have contributed, or None in cases where field
        information is unavailable.

        Semantics of None:
        - Branch schemas were not registered for this coalesce point (observed-mode)
        - No branches were lost (nothing to report)
        - Lost branches exist but none had registered schemas

        An empty dict is never returned — it collapses to None for simpler
        downstream handling (callers only need to check ``is not None``).

        Args:
            coalesce_name: Name of the coalesce configuration
            lost_branches: Mapping of branch name to loss reason

        Returns:
            Mapping of lost branch name to its expected fields, or None if
            field information is unavailable (see semantics above).
        """
        if not lost_branches:
            return None

        branch_fields = self._registered_branch_expected_fields(coalesce_name)
        if branch_fields is None:
            return None
        result: dict[str, tuple[str, ...]] = {}
        for branch_name in lost_branches:
            if branch_name in branch_fields:
                result[branch_name] = branch_fields[branch_name]
            # If branch_name not in branch_fields, the branch used observed-mode
            # schema with no declared fields — omit from result rather than crash.
        return result if result else None

    def _registered_output_schema(self, coalesce_name: str) -> SchemaContract | None:
        try:
            return self._output_schemas[coalesce_name]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Missing output schema registry entry for registered coalesce '{coalesce_name}'. "
                "This indicates register_coalesce() state corruption."
            ) from exc

    def _registered_branch_expected_fields(self, coalesce_name: str) -> dict[str, tuple[str, ...]] | None:
        try:
            return self._branch_expected_fields[coalesce_name]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Missing branch expected fields registry entry for registered coalesce '{coalesce_name}'. "
                "This indicates register_coalesce() state corruption."
            ) from exc

    def _fail_pending(
        self,
        settings: CoalesceSettings,
        key: tuple[str, str],
        step: int,
        failure_reason: str,
        *,
        is_timeout: bool = False,
        select_branch: str | None = None,
        metadata: CoalesceMetadata | None = None,
    ) -> CoalesceOutcome:
        """Fail all arrived tokens in a pending coalesce and clean up.

        Shared helper used by check_timeouts(), flush_pending(),
        _evaluate_after_loss(), and _execute_merge() (select_branch_not_arrived)
        to avoid duplicating failure recording logic.

        Args:
            settings: Coalesce settings for metadata
            key: (coalesce_name, row_id) tuple
            step: Resolved audit step index for the coalesce node
            failure_reason: Machine-readable failure reason string
            is_timeout: Whether this failure was triggered by a timeout.
                Callers set this explicitly rather than inferring from the
                failure_reason string.
            select_branch: Target branch for select merge failures (passed through
                to CoalesceFailureReason).
            metadata: Pre-built CoalesceMetadata. When provided, used instead of
                the default CoalesceMetadata.for_failure() construction.

        Returns:
            CoalesceOutcome with failure_reason set and outcomes_recorded=True
        """
        coalesce_name = key[0]
        pending = self._pending[key]
        consumed_tokens = tuple(e.token for e in pending.branches.values())
        error_hash = compute_error_hash(failure_reason)
        now = self._clock.monotonic()

        # Complete pending node states with failure
        error = CoalesceFailureReason(
            failure_reason=failure_reason,
            expected_branches=tuple(settings.branches),
            branches_arrived=tuple(pending.branches.keys()),
            merge_policy=settings.merge,
            timeout_ms=int(settings.timeout_seconds * 1000) if is_timeout and settings.timeout_seconds is not None else None,
            select_branch=select_branch,
        )
        for _branch_name, entry in pending.branches.items():
            self._execution.complete_node_state(
                state_id=entry.state_id,
                status=NodeStateStatus.FAILED,
                error=error,
                duration_ms=(now - entry.arrival_time) * 1000,
            )
            if self._data_flow is None:
                raise OrchestrationInvariantError(
                    "CoalesceExecutor.data_flow is None but token outcome recording requires DataFlowRepository"
                )
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=entry.token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=error_hash,
            )

        del self._pending[key]
        self._mark_completed(key)

        if metadata is None:
            metadata = CoalesceMetadata.for_failure(
                policy=CoalescePolicy(settings.policy),
                expected_branches=tuple(settings.branches),
                branches_arrived=tuple(pending.branches.keys()),
                branches_lost=pending.lost_branches,
                lost_branch_expected_fields=self._get_lost_branch_expected_fields(coalesce_name, pending.lost_branches),
                quorum_required=settings.quorum_count,
                timeout_seconds=settings.timeout_seconds,
            )

        return CoalesceOutcome(
            held=False,
            failure_reason=failure_reason,
            consumed_tokens=consumed_tokens,
            coalesce_metadata=metadata,
            coalesce_name=coalesce_name,
            outcomes_recorded=True,
        )

    def _execute_merge(
        self,
        settings: CoalesceSettings,
        node_id: NodeID,
        pending: _PendingCoalesce,
        step: int,
        key: tuple[str, str],
        coalesce_name: str,
    ) -> CoalesceOutcome:
        """Execute the merge and create merged token."""
        now = self._clock.monotonic()

        # ─────────────────────────────────────────────────────────────────────
        # Defensive check: crash if any token has no contract
        # Per CLAUDE.md: "Bad data in the audit trail = crash immediately"
        # A token with None contract is a bug in upstream code (fork/transform).
        # ─────────────────────────────────────────────────────────────────────
        for branch, entry in pending.branches.items():
            if entry.token.row_data.contract is None:
                raise OrchestrationInvariantError(
                    f"Token {entry.token.token_id} on branch '{branch}' has no contract. "
                    f"Cannot coalesce without contracts on all parents. "
                    f"This indicates a bug in fork or transform execution."
                )

        # Validate select_branch is present for select merge strategy
        # (Bug 2ho fix: reject instead of silent fallback)
        if settings.merge == "select" and settings.select_branch not in pending.branches:
            # CoalesceSettings model validator ensures select_branch is non-None for merge="select"
            assert settings.select_branch is not None
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason="select_branch_not_arrived",
                select_branch=settings.select_branch,
                metadata=CoalesceMetadata.for_select_not_arrived(
                    policy=CoalescePolicy(settings.policy),
                    merge_strategy=MergeStrategy(settings.merge),
                    select_branch=settings.select_branch,
                    branches_arrived=tuple(pending.branches.keys()),
                ),
            )

        completed_state_ids: set[str] = set()
        # Captured so the failure cleanup handler can persist collision provenance
        # (union_field_origins, redacted union_field_collision_values) to the audit trail
        # when CoalesceCollisionError is raised under union_collision_policy=fail,
        # or when any other exception happens after metadata was built. Stays None
        # for early failures (e.g., contract merge) where no metadata exists yet.
        metadata_for_audit: CoalesceMetadata | None = None
        try:
            try:
                plan = build_coalesce_merge(
                    settings=settings,
                    pending=pending,
                    coalesce_name=coalesce_name,
                    now=now,
                    output_schema=self._registered_output_schema(settings.name),
                    branch_expected_fields=self._registered_branch_expected_fields(coalesce_name),
                    merge_data=self._merge_data,
                    merge_with_original_names=self._merge_with_original_names,
                )
            except ContractMergeError as e:
                # Type conflict between branches -- fail this row gracefully.
                return self._fail_pending(
                    settings=settings,
                    key=key,
                    step=step,
                    failure_reason=f"contract_type_conflict: {e}",
                )
            coalesce_metadata = plan.metadata
            metadata_for_audit = coalesce_metadata
            merged_data = plan.merged_data
            consumed_tokens = plan.consumed_tokens

            # Create merged token via TokenManager
            merged_token = self._token_manager.coalesce_tokens(
                parents=list(consumed_tokens),
                merged_data=merged_data,
                node_id=node_id,
                run_id=self._run_id,
            )

            # Complete pending node states for consumed tokens
            # (These states were created as "pending" when tokens were held in accept())
            for _branch_name, entry in pending.branches.items():
                # Complete it now that merge is happening
                # Bug l4h fix: include coalesce metadata in context_after for audit trail
                self._execution.complete_node_state(
                    state_id=entry.state_id,
                    status=NodeStateStatus.COMPLETED,
                    output_data={"merged_into": merged_token.token_id},
                    duration_ms=(now - entry.arrival_time) * 1000,
                    context_after=coalesce_metadata,
                )
                completed_state_ids.add(entry.state_id)

                # Record terminal token outcome (COALESCED)
                if self._data_flow is None:
                    raise OrchestrationInvariantError(
                        "CoalesceExecutor.data_flow is None but token outcome recording requires DataFlowRepository"
                    )
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=entry.token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.COALESCED,
                    sink_name=None,
                    join_group_id=merged_token.join_group_id,
                )

            # NOTE: The merged token does NOT get COALESCED recorded here.
            # - Consumed tokens: COALESCED (terminal) - they've been absorbed into the merge
            # - Merged token: its node_state status here is COMPLETED (set above at
            #   complete_node_state); its TERMINAL token_outcome is recorded later by
            #   the sink path, with path=COALESCED and sink_name SET (the merged token
            #   that reaches a sink), or path=COALESCED again if it is consumed by an
            #   outer coalesce (nested scenario, recorded by that outer executor here
            #   with sink_name=None). So a non-nested coalesce yields exactly ONE
            #   (SUCCESS, COALESCED) record with sink_name set — the discriminator
            #   derive_resume_terminal_status_from_audit uses to count rows_coalesced
            #   without double-counting the sink_name=None consumed inputs.
            # Recording COALESCED for merged token here would break nested coalesces where
            # the inner merge result becomes a consumed token in the outer merge.

            # Clean up pending state and mark as completed
            del self._pending[key]
            self._mark_completed(key)  # Track completion to reject late arrivals (bounded)

            return CoalesceOutcome(
                held=False,
                merged_token=merged_token,
                consumed_tokens=consumed_tokens,
                coalesce_metadata=coalesce_metadata,
                coalesce_name=coalesce_name,
            )
        except AuditIntegrityError:
            # If the audit database is already compromised, don't write more
            # records to it — leaving node states as pending is more honest
            # than writing FAILED to an untrustworthy database. Re-raise without
            # recording any further FAILED states to the untrustworthy DB.
            raise
        except Exception as merge_exc:
            if metadata_for_audit is None and isinstance(merge_exc, CoalesceCollisionError):
                metadata_for_audit = merge_exc.metadata
            # Generate error_hash once for all branches (consistent audit trail).
            error_hash = compute_error_hash(str(merge_exc), exception_type=type(merge_exc).__name__)

            for _branch, entry in pending.branches.items():
                # Skip branches already completed in the happy path —
                # overwriting COMPLETED with FAILED would corrupt the audit trail.
                # (Happy path already recorded COALESCED outcome for these.)
                if entry.state_id in completed_state_ids:
                    continue
                # Pass metadata_for_audit so union_collision_policy=fail's
                # collision provenance (field_origins + value fingerprints) reaches
                # the Landscape audit trail via context_after. None is acceptable for
                # early failures (e.g., contract merge) where no metadata exists.
                self._execution.complete_node_state(
                    state_id=entry.state_id,
                    status=NodeStateStatus.FAILED,
                    output_data={},
                    duration_ms=0.0,
                    error=ExecutionError(
                        exception=str(merge_exc),
                        exception_type=type(merge_exc).__name__,
                        phase="coalesce_merge_cleanup",
                    ),
                    context_after=metadata_for_audit,
                )
                # Record terminal FAILED outcome for consumed token.
                # Without this, recovery treats the row as incomplete and
                # lineage resolution can't find a terminal token.
                if self._data_flow is None:
                    raise OrchestrationInvariantError(
                        "CoalesceExecutor.data_flow is None but token outcome recording requires DataFlowRepository"
                    ) from merge_exc
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=entry.token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error_hash=error_hash,
                )

            # Clean up pending state only after every cleanup audit write succeeds.
            del self._pending[key]
            self._mark_completed(key)

            raise

    def _merge_with_original_names(
        self,
        precomputed: SchemaContract,
        branch_contracts: dict[str, SchemaContract],
        field_origins: dict[str, str],
    ) -> SchemaContract:
        """Compatibility adapter for tests that patch executor internals."""
        return _merge_with_original_names(precomputed, branch_contracts, field_origins)

    def _merge_data(
        self,
        settings: CoalesceSettings,
        branches: dict[str, _BranchEntry],
    ) -> tuple[
        dict[str, Any],
        dict[str, list[str]],
        dict[str, str],
        dict[str, list[tuple[str, Any]]],
    ]:
        """Compatibility adapter for tests that patch executor internals."""
        return _merge_data(settings, branches)

    def _resolve_pending(
        self,
        settings: CoalesceSettings,
        node_id: NodeID,
        pending: _PendingCoalesce,
        step: int,
        key: tuple[str, str],
        coalesce_name: str,
        *,
        is_timeout: bool = False,
    ) -> CoalesceOutcome:
        """Resolve a pending coalesce by dispatching on policy.

        Shared helper for check_timeouts() and flush_pending(). Decides whether
        to merge (enough branches arrived) or fail (not enough) via
        :func:`decide_coalesce` (TIMEOUT/FLUSH events) — the policy matrix
        lives in ``elspeth.engine.coalesce_policy``. TIMEOUT/FLUSH never
        yield WAIT (resolution is forced), so the WAIT arm below is an
        exhaustiveness guard, not a reachable path.

        Args:
            settings: Coalesce settings for this point
            node_id: DAG node ID for audit recording
            pending: The pending coalesce state
            step: Resolved audit step index
            key: (coalesce_name, row_id) tuple
            coalesce_name: Name of the coalesce configuration
            is_timeout: True when triggered by timeout (affects failure reasons
                and is_timeout flag on _fail_pending)
        """
        event = CoalesceEvent.TIMEOUT if is_timeout else CoalesceEvent.FLUSH
        decision = decide_coalesce(
            settings,
            event,
            arrived_count=len(pending.branches),
            lost_branches=pending.lost_branches,
            row_id=key[1],
        )
        if decision.action is CoalesceAction.MERGE:
            return self._execute_merge(
                settings=settings,
                node_id=node_id,
                pending=pending,
                step=step,
                key=key,
                coalesce_name=coalesce_name,
            )
        if decision.action is CoalesceAction.FAIL:
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason=decision.require_failure_reason(),
                is_timeout=is_timeout,
            )
        raise RuntimeError("unreachable: decide_coalesce never returns WAIT for TIMEOUT/FLUSH")

    def check_timeouts(
        self,
        coalesce_name: str,
    ) -> list[CoalesceOutcome]:
        """Check for timed-out pending coalesces and merge them.

        For best_effort policy, merges whatever has arrived when timeout expires.
        For quorum policy with timeout, merges if quorum met when timeout expires.
        For first policy, only a zero-arrival loss-created pending entry can
        time out; that fails cleanly rather than tripping the arrival invariant.

        Step position is resolved internally via the injected StepResolver
        from the coalesce point's registered node_id.

        Args:
            coalesce_name: Name of the coalesce configuration

        Returns:
            List of CoalesceOutcomes for any merges triggered by timeout
        """
        if coalesce_name not in self._settings:
            raise OrchestrationInvariantError(f"Coalesce '{coalesce_name}' not registered")

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]
        step = self._step_resolver(node_id)

        if settings.timeout_seconds is None:
            return []

        now = self._clock.monotonic()
        results: list[CoalesceOutcome] = []
        keys_to_process: list[tuple[str, str]] = []

        # Find timed-out entries
        for key, pending in self._pending.items():
            if key[0] != coalesce_name:
                continue

            elapsed = now - pending.first_arrival
            if elapsed >= settings.timeout_seconds:
                keys_to_process.append(key)

        # Process timed-out entries
        for key in keys_to_process:
            results.append(
                self._resolve_pending(
                    settings=settings,
                    node_id=node_id,
                    pending=self._pending[key],
                    step=step,
                    key=key,
                    coalesce_name=coalesce_name,
                    is_timeout=True,
                )
            )

        return results

    def flush_pending(self) -> list[CoalesceOutcome]:
        """Flush all pending coalesces (called at end-of-source or shutdown).

        For best_effort policy: merges whatever arrived.
        For quorum policy: merges if quorum met, returns failure otherwise.
        For require_all policy: returns failure (never partial merge).
        For first policy: arrived tokens should never be pending (merges
        immediately); a zero-arrival entry created by branch-loss accounting
        fails cleanly.

        Step positions are resolved internally via the injected StepResolver
        from each coalesce point's registered node_id.

        Returns:
            List of CoalesceOutcomes for all pending coalesces
        """
        results: list[CoalesceOutcome] = []
        keys_to_process = list(self._pending.keys())

        for key in keys_to_process:
            coalesce_name, _row_id = key
            settings = self._settings[coalesce_name]
            node_id = self._node_ids[coalesce_name]
            pending = self._pending[key]
            step = self._step_resolver(node_id)

            # Flush-specific invariant: zero branches AND zero lost branches
            # means the pending entry should never have been created.
            # (Timeout path can't hit this because accept() creates the entry on arrival.)
            if settings.policy == "best_effort" and len(pending.branches) == 0 and not pending.lost_branches:
                raise OrchestrationInvariantError(
                    f"Pending coalesce entry for {coalesce_name!r} (row {_row_id}) "
                    f"has zero branches and zero lost branches — "
                    f"this is a coalesce state invariant violation"
                )

            results.append(
                self._resolve_pending(
                    settings=settings,
                    node_id=node_id,
                    pending=pending,
                    step=step,
                    key=key,
                    coalesce_name=coalesce_name,
                )
            )

        # Clear completed keys to prevent unbounded memory growth
        # After flush, no more tokens will arrive (source ended), so late-arrival
        # detection is no longer needed. This prevents O(rows) memory accumulation
        # in long-running pipelines.
        self._completed_keys.clear()

        return results

    def has_recorded_branch_loss(self, coalesce_name: str, row_id: str, branch_name: str) -> bool:
        """Whether this branch loss is already in executor memory (§E.5 replay dedup).

        The journal-first loss replay (per-iteration intake / takeover
        restore) consults this BEFORE ``notify_branch_lost``: at N=1 the
        producer already notified in-claim (record-then-notify in the same
        drain step), and re-notifying a recorded loss is a Tier-1 duplicate.
        A completed/unknown key returns False — ``notify_branch_lost``'s own
        completed-keys check makes that replay a no-op.
        """
        pending = self._pending.get((coalesce_name, row_id))
        if pending is None:
            return False
        return branch_name in pending.lost_branches

    def notify_branch_lost(
        self,
        coalesce_name: str,
        row_id: str,
        lost_branch: str,
        reason: str,
    ) -> CoalesceOutcome | None:
        """Notify that a branch was error-routed and will never arrive.

        Called by the processor when a forked token is diverted to an error sink
        before reaching this coalesce point. Adjusts the expected branch count
        and re-evaluates merge conditions.

        Threading: CoalesceExecutor is single-threaded — called from the
        processor's synchronous work queue loop. The processor processes one
        work item at a time, so there is no concurrency within a single row's
        fork/coalesce lifecycle.

        Step position is resolved internally via the injected StepResolver
        from the coalesce point's registered node_id.

        Args:
            coalesce_name: Name of the coalesce configuration
            row_id: Source row ID (correlates forked tokens)
            lost_branch: Name of the branch that was error-routed
            reason: Machine-readable reason for the loss

        Returns:
            CoalesceOutcome if merge/failure triggered, None if still waiting.
        """
        if coalesce_name not in self._settings:
            raise OrchestrationInvariantError(f"Coalesce '{coalesce_name}' not registered")

        key = (coalesce_name, row_id)

        # Already completed (race with normal merge) — ignore.
        # Two-level lookup: FIFO cache then Landscape fallback.
        if key in self._completed_keys or self._check_landscape_for_completion(coalesce_name, row_id):
            return None

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]
        step = self._step_resolver(node_id)

        # Validate branch is expected
        if lost_branch not in settings.branches:
            raise OrchestrationInvariantError(
                f"Lost branch '{lost_branch}' not in expected branches for coalesce '{coalesce_name}': {settings.branches}"
            )

        # No pending entry yet — branch lost before ANY branch arrived.
        # Create a minimal pending entry with the loss recorded.
        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                branches={},
                first_arrival=self._clock.monotonic(),
                lost_branches={lost_branch: reason},
            )
            return self._evaluate_after_loss(settings, key, step)

        pending = self._pending[key]

        # Validate branch hasn't already arrived (would be a processor bug)
        if lost_branch in pending.branches:
            raise OrchestrationInvariantError(
                f"Branch '{lost_branch}' already arrived at coalesce '{coalesce_name}' "
                f"but was reported as lost. This indicates a bug in the processor — "
                f"a token cannot both arrive and be error-routed."
            )

        # Validate branch hasn't already been marked lost (would be a processor bug)
        if lost_branch in pending.lost_branches:
            raise OrchestrationInvariantError(
                f"Branch '{lost_branch}' already marked lost at coalesce '{coalesce_name}'. "
                f"Duplicate loss notification indicates a processor bug."
            )

        # Record the loss and re-evaluate
        pending.lost_branches[lost_branch] = reason
        return self._evaluate_after_loss(settings, key, step)

    def _evaluate_after_loss(
        self,
        settings: CoalesceSettings,
        key: tuple[str, str],
        step: int,
    ) -> CoalesceOutcome | None:
        """Re-evaluate merge conditions after a branch loss notification.

        Thin delegate over :func:`decide_coalesce` (LOSS event) — the policy
        matrix (require_all fails on ANY lost branch; quorum fails when
        impossible, merges when already met; best_effort merges when all
        branches are accounted for; first fails only when every branch is
        lost before any arrival) lives in ``elspeth.engine.coalesce_policy``.

        Args:
            settings: Coalesce settings for the affected point
            key: (coalesce_name, row_id) tuple
            step: Resolved audit step index for the coalesce node

        Returns:
            CoalesceOutcome if merge/failure triggered, None if still waiting.
        """
        pending = self._pending[key]
        decision = decide_coalesce(
            settings,
            CoalesceEvent.LOSS,
            arrived_count=len(pending.branches),
            lost_branches=pending.lost_branches,
            row_id=key[1],
        )
        if decision.action is CoalesceAction.MERGE:
            node_id = self._node_ids[settings.name]
            return self._execute_merge(settings, node_id, pending, step, key, settings.name)
        if decision.action is CoalesceAction.FAIL:
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason=decision.require_failure_reason(),
            )
        return None  # WAIT — still waiting for remaining branches
