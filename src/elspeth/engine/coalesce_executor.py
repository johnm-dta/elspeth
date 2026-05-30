"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from elspeth.contracts import TokenInfo
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.coalesce_checkpoint import (
    CoalesceCheckpointState,
    CoalescePendingCheckpoint,
    CoalesceTokenCheckpoint,
)
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
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock
    from elspeth.engine.tokens import TokenManager

COALESCE_CHECKPOINT_VERSION = "1.0"


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
    """

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: tuple[TokenInfo, ...] = ()
    coalesce_metadata: CoalesceMetadata | None = None
    failure_reason: str | None = None
    coalesce_name: str | None = None
    outcomes_recorded: bool = False

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


class CoalesceExecutor:
    """Executes coalesce operations with audit recording.

    Maintains state for pending coalesce operations:
    - Tracks which tokens have arrived for each row_id
    - Evaluates merge conditions based on policy
    - Merges row data according to strategy
    - Records audit trail via ExecutionRepository

    Example:
        executor = CoalesceExecutor(execution, span_factory, token_manager, run_id, step_resolver)

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
        clock: "Clock | None" = None,
        max_completed_keys: int = 10000,
        data_flow: DataFlowRepository | None = None,
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
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
            max_completed_keys: Maximum late-arrival completion keys retained in memory.
            data_flow: Data flow repository for token outcome recording.
                       Optional for backwards compatibility with tests.
        """
        if max_completed_keys <= 0:
            raise OrchestrationInvariantError(f"max_completed_keys must be > 0, got {max_completed_keys}")

        self._execution = execution
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
        # restore_from_checkpoint(). Branch schemas come from fresh graph data each run,
        # not from checkpoint — the checkpoint stores only pending tokens and lost branches.
        self._branch_expected_fields: dict[str, dict[str, tuple[str, ...]]] = {}
        # Pre-computed output schemas: coalesce_name -> SchemaContract
        # Used to ensure runtime contracts match DAG-computed schemas (P2 fix).
        # When populated, _execute_merge() uses this instead of runtime merge().
        self._output_schemas: dict[str, SchemaContract] = {}
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
                When provided, used directly in union merge instead of runtime
                SchemaContract.merge() to ensure build/runtime contract alignment.
        """
        self._settings[settings.name] = settings
        self._node_ids[settings.name] = node_id
        if branch_schemas is not None:
            self._branch_expected_fields[settings.name] = branch_schemas
        if output_schema is not None:
            self._output_schemas[settings.name] = output_schema

    def get_registered_names(self) -> list[str]:
        """Get names of all registered coalesce points.

        Used by processor for timeout checking loop.

        Returns:
            List of registered coalesce names
        """
        return list(self._settings.keys())

    def get_checkpoint_state(self) -> CoalesceCheckpointState:
        """Return checkpoint state for pending coalesces."""

        pending_entries: list[CoalescePendingCheckpoint] = []
        for (coalesce_name, row_id), pending in self._pending.items():
            branch_entries = {
                branch_name: CoalesceTokenCheckpoint(
                    token_id=entry.token.token_id,
                    row_id=entry.token.row_id,
                    branch_name=branch_name,
                    fork_group_id=entry.token.fork_group_id,
                    join_group_id=entry.token.join_group_id,
                    expand_group_id=entry.token.expand_group_id,
                    row_data=entry.token.row_data.to_dict(),
                    contract=entry.token.row_data.contract.to_checkpoint_format(),
                    state_id=entry.state_id,
                    arrival_offset_seconds=entry.arrival_time - pending.first_arrival,
                )
                for branch_name, entry in pending.branches.items()
            }
            pending_entries.append(
                CoalescePendingCheckpoint(
                    coalesce_name=coalesce_name,
                    row_id=row_id,
                    elapsed_age_seconds=self._clock.monotonic() - pending.first_arrival,
                    branches=branch_entries,
                    lost_branches=dict(pending.lost_branches),
                )
            )

        # completed_keys is no longer persisted in checkpoints — it's
        # reconstructed from the Landscape on restore (Phase 1 of
        # elspeth-cc36c8eaef). Empty tuple preserves schema compatibility
        # with existing checkpoint parsers / version validation.
        checkpoint = CoalesceCheckpointState(
            version=COALESCE_CHECKPOINT_VERSION,
            pending=tuple(pending_entries),
            completed_keys=(),
        )

        return checkpoint

    def restore_from_checkpoint(self, state: CoalesceCheckpointState) -> None:
        """Restore pending coalesces from checkpoint.

        Completed keys are reconstructed from the Landscape (source of truth)
        rather than from checkpoint data. This eliminates the FIFO eviction gap:
        the Landscape query returns ALL completed coalesces for this run, not
        just the last max_completed_keys entries.

        The checkpoint's completed_keys field is ignored (Phase 1 of
        elspeth-cc36c8eaef). Phase 3 will remove it from the checkpoint schema.
        """
        if state.version != COALESCE_CHECKPOINT_VERSION:
            raise AuditIntegrityError(
                f"Incompatible coalesce checkpoint version: {state.version!r}. Expected: {COALESCE_CHECKPOINT_VERSION!r}."
            )

        # Validate ALL entries before clearing state — if validation fails,
        # the executor's in-memory state must remain intact for error recovery.
        for pending_entry in state.pending:
            if pending_entry.coalesce_name not in self._settings:
                raise AuditIntegrityError(
                    f"Checkpoint references unknown coalesce '{pending_entry.coalesce_name}'. "
                    f"Configured coalesces: {sorted(self._settings)}"
                )

        now = self._clock.monotonic()
        self._pending.clear()
        self._completed_keys.clear()

        # Reconstruct completed keys from Landscape (source of truth).
        # This replaces checkpoint-based restoration, eliminating:
        # - FIFO eviction gap (Landscape has ALL completed, not just last 10K)
        # - Checkpoint-Landscape divergence risk
        self._reconstruct_completed_keys_from_landscape()

        for pending_entry in state.pending:
            first_arrival = now - pending_entry.elapsed_age_seconds
            branches: dict[str, _BranchEntry] = {}
            for branch_name, token_checkpoint in pending_entry.branches.items():
                restored_contract = SchemaContract.from_checkpoint(dict(token_checkpoint.contract))
                restored_row = PipelineRow(deep_thaw(token_checkpoint.row_data), restored_contract)
                token = TokenInfo(
                    row_id=token_checkpoint.row_id,
                    token_id=token_checkpoint.token_id,
                    row_data=restored_row,
                    branch_name=token_checkpoint.branch_name,
                    fork_group_id=token_checkpoint.fork_group_id,
                    join_group_id=token_checkpoint.join_group_id,
                    expand_group_id=token_checkpoint.expand_group_id,
                )
                branches[branch_name] = _BranchEntry(
                    token=token,
                    arrival_time=first_arrival + token_checkpoint.arrival_offset_seconds,
                    state_id=token_checkpoint.state_id,
                )

            self._pending[(pending_entry.coalesce_name, pending_entry.row_id)] = _PendingCoalesce(
                branches=branches,
                first_arrival=first_arrival,
                lost_branches=dict(pending_entry.lost_branches),
            )

    def _reconstruct_completed_keys_from_landscape(self) -> None:
        """Populate _completed_keys from the Landscape audit trail.

        Queries node_states for completed entries at coalesce node IDs, joined
        with tokens to get row_ids. Maps node_id → coalesce_name via the
        reverse of self._node_ids.

        This is the source-of-truth path: the Landscape records ALL completed
        coalesces (no FIFO eviction), so late-arrival detection works correctly
        even for pipelines with >max_completed_keys coalesced rows.
        """
        if not self._node_ids:
            return

        # Build reverse map: node_id → coalesce_name
        node_id_to_name: dict[str, str] = {str(nid): name for name, nid in self._node_ids.items()}

        completed_pairs = self._execution.get_completed_row_ids_for_nodes(
            run_id=self._run_id,
            node_ids=frozenset(node_id_to_name.keys()),
        )

        for node_id_str, row_id in completed_pairs:
            if node_id_str in node_id_to_name:
                self._completed_keys[(node_id_to_name[node_id_str], row_id)] = None

    def _check_landscape_for_completion(self, coalesce_name: str, row_id: str) -> bool:
        """Check the Landscape for whether a coalesce key has completed.

        Cache-miss fallback for late-arrival detection. When the FIFO cache
        (self._completed_keys) doesn't contain a key, this queries the
        Landscape before allowing a new pending entry. If the Landscape
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

        completed_pairs = self._execution.get_completed_row_ids_for_nodes(
            run_id=self._run_id,
            node_ids=frozenset({str(node_id)}),
        )

        # Check if any of the completed pairs match our row_id
        for _nid, completed_row_id in completed_pairs:
            if completed_row_id == row_id:
                # Cache hit: add to FIFO so subsequent lookups are fast
                self._completed_keys[(coalesce_name, row_id)] = None
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

    def _require_quorum_count(self, settings: CoalesceSettings) -> int:
        """Return quorum_count or crash if None — config validation should have caught this."""
        if settings.quorum_count is None:
            raise RuntimeError(
                f"quorum_count is None for quorum policy at coalesce '{settings.name}'. This indicates a config validation bug."
            )
        return settings.quorum_count

    def accept(
        self,
        token: TokenInfo,
        coalesce_name: str,
        resume_attempt_offset: int = 0,
        resume_checkpoint_id: str | None = None,
    ) -> CoalesceOutcome:
        """Accept a token at a coalesce point.

        If merge conditions are met, returns the merged token.
        Otherwise, holds the token and returns held=True.

        Step position is resolved internally via the injected StepResolver
        from the coalesce point's registered node_id.

        Args:
            token: Token arriving at coalesce point (must have branch_name)
            coalesce_name: Name of the coalesce configuration
            resume_attempt_offset: Added to attempt so re-driven node_states coexist
                with run-1 records under UniqueConstraint(token_id, node_id, attempt).
                Defaults to 0 (no-op for all normal processing).
            resume_checkpoint_id: Checkpoint ID stamped on re-driven node_states.
                Defaults to None (no-op for all normal processing).

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
        now = self._clock.monotonic()

        # Check if this coalesce already completed (late arrival).
        # Two-level lookup: FIFO cache first, then Landscape fallback.
        # The FIFO is a performance optimization; the Landscape is the
        # source of truth. Evicted FIFO entries are rediscovered from
        # the Landscape, eliminating the eviction window.
        if key in self._completed_keys or self._check_landscape_for_completion(coalesce_name, token.row_id):
            # Late arrival after merge/failure already happened
            # Record failure audit trail for this late token
            failure_reason = "late_arrival_after_merge"
            error_hash = hashlib.sha256(failure_reason.encode()).hexdigest()[:16]
            state = self._execution.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                run_id=self._run_id,
                step_index=step,
                input_data=token.row_data.to_dict(),  # Recorder expects dict
                attempt=resume_attempt_offset,
                resume_checkpoint_id=resume_checkpoint_id,
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
            )

        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                branches={},
                first_arrival=now,
            )

        pending = self._pending[key]

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
            attempt=resume_attempt_offset,
            resume_checkpoint_id=resume_checkpoint_id,
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
        """Check if merge conditions are met based on policy."""
        arrived_count = len(pending.branches)
        expected_count = len(settings.branches)

        if settings.policy == "require_all":
            return arrived_count == expected_count

        elif settings.policy == "first":
            return arrived_count >= 1

        elif settings.policy == "quorum":
            return arrived_count >= self._require_quorum_count(settings)

        elif settings.policy == "best_effort":
            # Merge on timeout (checked elsewhere) or when all branches accounted for.
            # Lost branches count as "accounted for" — they won't arrive but we know about them.
            accounted_count = arrived_count + len(pending.lost_branches)
            return accounted_count >= expected_count

        else:
            raise RuntimeError(f"Unknown coalesce policy: {settings.policy!r}")

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
        if coalesce_name not in self._branch_expected_fields:
            return None
        if not lost_branches:
            return None

        branch_fields = self._branch_expected_fields[coalesce_name]
        result: dict[str, tuple[str, ...]] = {}
        for branch_name in lost_branches:
            if branch_name in branch_fields:
                result[branch_name] = branch_fields[branch_name]
            # If branch_name not in branch_fields, the branch used observed-mode
            # schema with no declared fields — omit from result rather than crash.
        return result if result else None

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
        error_hash = hashlib.sha256(failure_reason.encode()).hexdigest()[:16]
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
        # (union_field_origins, union_field_collision_values) to the audit trail
        # when CoalesceCollisionError is raised under union_collision_policy=fail,
        # or when any other exception happens after metadata was built. Stays None
        # for early failures (e.g., contract merge) where no metadata exists yet.
        metadata_for_audit: CoalesceMetadata | None = None
        try:
            # ─────────────────────────────────────────────────────────────────────
            # Merge row data according to strategy (returns dict)
            # We do this FIRST so we can derive contract from actual data shape
            # ─────────────────────────────────────────────────────────────────────
            merged_data_dict, union_collisions, field_origins, collision_values = self._merge_data(settings, pending.branches)

            # ─────────────────────────────────────────────────────────────────────
            # Build contract based on merge strategy and actual data shape
            # ─────────────────────────────────────────────────────────────────────
            # Keyed by branch name so _merge_with_original_names can look up winning branch
            branch_contracts: dict[str, SchemaContract] = {
                branch_name: e.token.row_data.contract for branch_name, e in pending.branches.items()
            }
            contracts: list[SchemaContract] = list(branch_contracts.values())

            if settings.merge == "union":
                # For typed schemas, the DAG builder's merge_union_fields() computes
                # correct policy-aware semantics (OR for require_all, AND otherwise).
                # Runtime SchemaContract.merge() uses incorrect AND-only semantics,
                # so precomputed is REQUIRED for typed schemas.
                #
                # For OBSERVED schemas, precomputed is empty (fields=()) since types
                # are inferred at runtime. Skip precomputed entirely and merge branch
                # contracts directly. (P1 fix: skip precomputed for observed unions)
                all_branches_observed = all(c.mode == "OBSERVED" for c in contracts)

                if settings.name in self._output_schemas:
                    precomputed = self._output_schemas[settings.name]
                    # Use precomputed only for typed schemas (mode != OBSERVED)
                    use_precomputed = precomputed.mode != "OBSERVED"
                else:
                    # No precomputed registered. Only allowed for all-OBSERVED branches.
                    if not all_branches_observed:
                        raise OrchestrationInvariantError(
                            f"Coalesce '{settings.name}' has typed branch contracts but no "
                            f"output_schema for union merge. The DAG builder must provide "
                            f"output_schema via register_coalesce(). "
                            f"If this is a test, use TestCoalesceExecutor from conftest."
                        )
                    use_precomputed = False

                if use_precomputed:
                    # Typed schemas: use precomputed semantics (required/nullable/type) but
                    # preserve original_name from branch contracts. The precomputed contract
                    # only has normalized names (from config); branch contracts carry the
                    # original headers from the source.
                    # P2 fix: use field_origins to pick original_name from winning branch,
                    # not first-arrived branch.
                    merged_contract = self._merge_with_original_names(precomputed, branch_contracts, field_origins)
                else:
                    # OBSERVED or no precomputed: merge branch contracts directly.
                    # Type conflicts are still possible when types are inferred from data.
                    merged_contract = contracts[0]
                    for c in contracts[1:]:
                        try:
                            merged_contract = merged_contract.merge(c)
                        except ContractMergeError as e:
                            # Type conflict between branches — fail this row gracefully.
                            return self._fail_pending(
                                settings=settings,
                                key=key,
                                step=step,
                                failure_reason=f"contract_type_conflict: {e}",
                            )

            elif settings.merge == "nested":
                # Nested: Contract declares branch keys with object type
                # Data shape is {branch_a: {...}, branch_b: {...}} where each value
                # is the full row data from that branch as a plain dict.
                # We use object (the "any" type in VALID_FIELD_TYPES) because dict
                # is not a valid FieldContract type and the contract only needs to
                # declare that the field exists, not constrain its inner structure.
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
                # Select: Use selected branch's contract directly (data has only those fields)
                # Find the selected branch's contract
                if settings.select_branch is None:
                    raise RuntimeError(
                        f"select_branch is None for select merge strategy at coalesce '{settings.name}'. This indicates a config validation bug."
                    )

                # Get the token for the selected branch
                selected_entry = pending.branches[settings.select_branch]
                merged_contract = selected_entry.token.row_data.contract

            else:
                # Unreachable - config validation ensures merge is one of the above
                raise RuntimeError(f"Unknown merge strategy: {settings.merge}")

            # Build audit metadata BEFORE token creation so union_collision_policy
            # enforcement can attach the full collision record to failures, and so
            # first_wins resolution rewrites the merged data before the token is minted.
            # (Bug l4h fix retained: metadata is still finalized before node-state completion.)
            coalesce_metadata = CoalesceMetadata.for_merge(
                policy=CoalescePolicy(settings.policy),
                merge_strategy=MergeStrategy(settings.merge),
                expected_branches=tuple(settings.branches),
                branches_arrived=tuple(pending.branches.keys()),
                branches_lost=pending.lost_branches,
                lost_branch_expected_fields=self._get_lost_branch_expected_fields(coalesce_name, pending.lost_branches),
                arrival_order=[
                    ArrivalOrderEntry(
                        branch=branch,
                        arrival_offset_ms=(entry.arrival_time - pending.first_arrival) * 1000,
                    )
                    for branch, entry in sorted(pending.branches.items(), key=lambda x: x[1].arrival_time)
                ],
                wait_duration_ms=(now - pending.first_arrival) * 1000,
            )

            # Layer union-merge provenance onto metadata. field_origins is always
            # recorded; collisions/collision_values populate only when the union
            # merge produced overlapping fields.
            if settings.merge == "union":
                coalesce_metadata = CoalesceMetadata.with_union_result(
                    coalesce_metadata,
                    field_origins=field_origins,
                    collisions=union_collisions if union_collisions else None,
                    collision_values=collision_values if collision_values else None,
                )
                # Capture enriched metadata for the failure handler so a subsequent
                # CoalesceCollisionError (fail policy) can persist the full collision
                # record to complete_node_state(context_after=...).
                metadata_for_audit = coalesce_metadata

                # Apply union_collision_policy — only meaningful when collisions occurred.
                if union_collisions:
                    if settings.union_collision_policy == "fail":
                        # Metadata is captured; raise with it attached so the orchestrator's
                        # failure path can persist the full collision record to the audit trail.
                        raise CoalesceCollisionError(
                            f"union merge collisions in coalesce '{coalesce_name}': {sorted(union_collisions)}",
                            metadata=coalesce_metadata,
                        )
                    if settings.union_collision_policy == "first_wins":
                        merged_data_dict, first_wins_origins = _resolve_first_wins(merged_data_dict, field_origins, collision_values)
                        # Rebuild contract with first_wins origins so original_name
                        # mapping matches the winning branch, not last-wins default.
                        # (P2 fix: _merge_with_original_names was called with field_origins
                        # before _resolve_first_wins computed the correct origins.)
                        if use_precomputed:
                            merged_contract = self._merge_with_original_names(precomputed, branch_contracts, first_wins_origins)
                        # Rebuild metadata with the updated origins; collision_values
                        # unchanged (still records every contributing branch in order).
                        coalesce_metadata = replace(
                            coalesce_metadata,
                            union_field_origins=first_wins_origins,
                        )
                        metadata_for_audit = coalesce_metadata
                    # last_wins: no-op; merged_data_dict already reflects last-wins.
            else:
                # nested/select: capture the base metadata so any post-build failure
                # still propagates branches_arrived/arrival_order to the audit trail.
                metadata_for_audit = coalesce_metadata

            # Create PipelineRow with strategy-appropriate contract (after any
            # first_wins rewrite so the merged token reflects the resolved data).
            merged_data = PipelineRow(merged_data_dict, merged_contract)

            # Get list of consumed tokens
            consumed_tokens = tuple(e.token for e in pending.branches.values())

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
            # - Merged token: Will get COMPLETED when it reaches a sink, or COALESCED if
            #   consumed by an outer coalesce (nested coalesce scenario)
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
        except Exception as merge_exc:
            # If the audit database is already compromised, don't write more
            # records to it — leaving node states as pending is more honest
            # than writing FAILED to an untrustworthy database.
            if isinstance(merge_exc, AuditIntegrityError):
                raise

            # Generate error_hash once for all branches (consistent audit trail).
            error_hash = hashlib.sha256(str(merge_exc).encode()).hexdigest()[:16]

            for _branch, entry in pending.branches.items():
                # Skip branches already completed in the happy path —
                # overwriting COMPLETED with FAILED would corrupt the audit trail.
                # (Happy path already recorded COALESCED outcome for these.)
                if entry.state_id in completed_state_ids:
                    continue
                # Pass metadata_for_audit so union_collision_policy=fail's full
                # collision record (field_origins + collision_values) reaches the
                # Landscape audit trail via context_after. None is acceptable for
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
        """Merge precomputed schema semantics with original names from branch contracts.

        The precomputed contract has correct required/nullable/type semantics from
        build-time analysis, but only has normalized names (original_name == normalized_name).
        Branch contracts carry the actual original→normalized mapping from the source.

        This method creates a new contract that preserves:
        - Field definitions (type, required, nullable, source) from precomputed
        - original_name from the winning branch per field_origins (policy-aware)

        Args:
            precomputed: Contract from create_contract_from_config() with build-time semantics
            branch_contracts: Map of branch_name -> contract from branch tokens
            field_origins: Map of field_name -> winning branch_name (from _merge_data)

        Returns:
            Merged contract with preserved original names
        """
        # Build lookup of (normalized_name, branch_name) -> original_name from all branches
        # This allows us to retrieve original_name from the winning branch per field
        branch_original_names: dict[tuple[str, str], str] = {}
        for branch_name, contract in branch_contracts.items():
            for fc in contract.fields:
                branch_original_names[(fc.normalized_name, branch_name)] = fc.original_name

        # Fallback lookup for fields not in field_origins (e.g., non-colliding fields)
        # Uses first-seen semantics across all branches
        fallback_original_names: dict[str, str] = {}
        for contract in branch_contracts.values():
            for fc in contract.fields:
                if fc.normalized_name not in fallback_original_names:
                    fallback_original_names[fc.normalized_name] = fc.original_name

        # Rebuild precomputed fields with original names from winning branches
        merged_fields: list[FieldContract] = []
        for fc in precomputed.fields:
            # _merge_data populates field_origins[name] for every contributed field
            # under union semantics (not just collisions — it tracks the winning
            # branch for every field). branch_original_names is built from the same
            # branch contracts, so (normalized_name, winning_branch) MUST be present
            # whenever the field is in field_origins. Direct indexing — a missing
            # key would be an audit-load-bearing schema/contract integrity violation
            # (precomputed and branch contracts diverged) and must crash, not coerce.
            if fc.normalized_name in field_origins:
                winning_branch = field_origins[fc.normalized_name]
                original_name = branch_original_names[(fc.normalized_name, winning_branch)]
            else:
                # Precomputed-declared field not contributed by any branch
                # (e.g., nullable field with no data this row). fallback_original_names
                # is built from every branch contract's fields, so a precomputed field
                # absent here means precomputed and branch contracts disagree on
                # schema — crash on the integrity violation.
                original_name = fallback_original_names[fc.normalized_name]
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
        self,
        settings: CoalesceSettings,
        branches: dict[str, _BranchEntry],
    ) -> tuple[
        dict[str, Any],
        dict[str, list[str]],
        dict[str, str],
        dict[str, list[tuple[str, Any]]],
    ]:
        """Merge row data from arrived tokens based on strategy.

        Note: row_data is PipelineRow, so we use to_dict() to get raw dict.

        Returns:
            Tuple of ``(merged_data, union_collisions, field_origins, collision_values)``:

            * ``merged_data``: the merged dict to wrap in a PipelineRow.
            * ``union_collisions``: field name -> list of contributing branch names,
              populated only for union merges where field names overlap.
            * ``field_origins``: field name -> the branch that produced the winning
              value under ``last_wins`` semantics. Populated for every union merge;
              empty dict for nested/select.
            * ``collision_values``: field name -> ordered list of ``(branch, value)``
              entries for every contributing branch. Populated only for union
              merges that actually collided; empty dict otherwise.
        """
        if settings.merge == "union":
            # Combine all fields from all branches.
            # On name collision, the last branch in settings.branches wins by default
            # (union_collision_policy="last_wins"). field_origins and collision_values
            # are always recorded so auditors can reconstruct lineage and inspect
            # overwritten values. Policy enforcement happens at the call site.
            merged: dict[str, Any] = {}
            field_origins: dict[str, str] = {}
            collisions: dict[str, list[str]] = {}
            collision_values: dict[str, list[tuple[str, Any]]] = {}
            for branch_name in settings.branches:
                if branch_name in branches:
                    branch_data = branches[branch_name].token.row_data.to_dict()
                    for merge_field, value in branch_data.items():
                        if merge_field in field_origins:
                            # Collision: capture the prior branch's value before overwriting.
                            if merge_field not in collisions:
                                prior_branch = field_origins[merge_field]
                                prior_value = merged[merge_field]
                                # Seed with the prior entry now; the current branch's
                                # entry is appended immediately below. This guarantees
                                # collision_values[merge_field] always has >= 2 entries
                                # on first detection — an invariant that _resolve_first_wins
                                # relies on when it unconditionally indexes entries[0].
                                collisions[merge_field] = [prior_branch]
                                collision_values[merge_field] = [(prior_branch, prior_value)]
                            collisions[merge_field].append(branch_name)
                            collision_values[merge_field].append((branch_name, value))
                        field_origins[merge_field] = branch_name
                        merged[merge_field] = value
            return merged, collisions, field_origins, collision_values

        elif settings.merge == "nested":
            # Each branch as nested object (use to_dict() for serializable dict)
            return (
                {
                    branch_name: branches[branch_name].token.row_data.to_dict()
                    for branch_name in settings.branches
                    if branch_name in branches
                },
                {},
                {},
                {},
            )

        elif settings.merge == "select":
            # Take specific branch output
            # Note: _execute_merge validates select_branch presence before calling this method
            # If we get here without select_branch, it's a bug in our code
            if settings.select_branch is None:
                raise RuntimeError(
                    f"select_branch is None for select merge strategy at coalesce '{settings.name}'. This indicates a config validation bug."
                )
            if settings.select_branch not in branches:
                # This should be unreachable - _execute_merge catches this case first
                # Per "Plugin Ownership" principle: crash on our bugs, don't hide them
                raise RuntimeError(
                    f"select_branch '{settings.select_branch}' not in arrived branches {list(branches.keys())}. "
                    f"This indicates a bug in _execute_merge validation (Bug 2ho fix should have caught this)."
                )
            # Return copy of dict (to_dict() returns a copy already)
            return branches[settings.select_branch].token.row_data.to_dict(), {}, {}, {}

        else:
            raise RuntimeError(f"Unknown merge strategy: {settings.merge!r}")

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
        to merge (enough branches arrived) or fail (not enough) based on policy.

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
        if settings.policy == "best_effort":
            if len(pending.branches) > 0:
                return self._execute_merge(
                    settings=settings,
                    node_id=node_id,
                    pending=pending,
                    step=step,
                    key=key,
                    coalesce_name=coalesce_name,
                )
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason="best_effort_timeout_no_arrivals" if is_timeout else "all_branches_lost",
                is_timeout=is_timeout,
            )

        elif settings.policy == "quorum":
            if len(pending.branches) >= self._require_quorum_count(settings):
                return self._execute_merge(
                    settings=settings,
                    node_id=node_id,
                    pending=pending,
                    step=step,
                    key=key,
                    coalesce_name=coalesce_name,
                )
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason="quorum_not_met_at_timeout" if is_timeout else "quorum_not_met",
                is_timeout=is_timeout,
            )

        elif settings.policy == "require_all":
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason="incomplete_branches",
                is_timeout=is_timeout,
            )

        elif settings.policy == "first":
            raise RuntimeError(
                f"Invariant violation: 'first' policy should never have pending entries "
                f"at coalesce '{coalesce_name}', row_id='{key[1]}'. "
                f"'first' merges immediately on arrival — bug in accept()."
            )

        else:
            raise RuntimeError(f"Unknown coalesce policy: {settings.policy!r}")

    def check_timeouts(
        self,
        coalesce_name: str,
    ) -> list[CoalesceOutcome]:
        """Check for timed-out pending coalesces and merge them.

        For best_effort policy, merges whatever has arrived when timeout expires.
        For quorum policy with timeout, merges if quorum met when timeout expires.

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
        For first policy: should never have pending (merges immediately).

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

        Policy-specific consequences:
        - require_all: ANY lost branch = immediate failure
        - quorum: fail if quorum is now impossible, merge if already met
        - best_effort: merge immediately if all branches accounted for
        - first: no action (should have merged on first arrival)

        Args:
            settings: Coalesce settings for the affected point
            key: (coalesce_name, row_id) tuple
            step: Resolved audit step index for the coalesce node

        Returns:
            CoalesceOutcome if merge/failure triggered, None if still waiting.
        """
        pending = self._pending[key]
        arrived_count = len(pending.branches)
        total_branches = len(settings.branches)
        lost_count = len(pending.lost_branches)

        if settings.policy == "require_all":
            # require_all: ANY lost branch = immediate failure
            return self._fail_pending(
                settings,
                key,
                step,
                failure_reason=f"branch_lost:{','.join(sorted(pending.lost_branches.keys()))}",
            )

        elif settings.policy == "quorum":
            quorum = self._require_quorum_count(settings)
            # Check if quorum is now impossible
            max_possible = total_branches - lost_count
            if max_possible < quorum:
                return self._fail_pending(
                    settings,
                    key,
                    step,
                    failure_reason=f"quorum_impossible:need={quorum},max_possible={max_possible}",
                )
            # Check if arrived count already meets quorum
            if arrived_count >= quorum:
                node_id = self._node_ids[settings.name]
                return self._execute_merge(settings, node_id, pending, step, key, settings.name)
            return None  # Still waiting

        elif settings.policy == "best_effort":
            # All branches accounted for (arrived + lost)?
            if arrived_count + lost_count >= total_branches:
                if arrived_count > 0:
                    node_id = self._node_ids[settings.name]
                    return self._execute_merge(settings, node_id, pending, step, key, settings.name)
                return self._fail_pending(
                    settings,
                    key,
                    step,
                    failure_reason="all_branches_lost",
                )
            return None  # Still waiting for remaining branches

        elif settings.policy == "first":
            # first: should already have merged on first arrival
            # If no arrivals yet, nothing to merge
            return None

        else:
            raise RuntimeError(f"Unknown coalesce policy: {settings.policy!r}")
