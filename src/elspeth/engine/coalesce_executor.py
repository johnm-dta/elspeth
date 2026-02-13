"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import NodeStateStatus, RowOutcome
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock
    from elspeth.engine.tokens import TokenManager

slog = structlog.get_logger(__name__)


@dataclass
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
    consumed_tokens: list[TokenInfo] = field(default_factory=list)
    coalesce_metadata: dict[str, Any] | None = None
    failure_reason: str | None = None
    coalesce_name: str | None = None
    outcomes_recorded: bool = False


@dataclass
class _PendingCoalesce:
    """Tracks pending tokens for a single row_id at a coalesce point."""

    arrived: dict[str, TokenInfo]  # branch_name -> token
    arrival_times: dict[str, float]  # branch_name -> monotonic time
    first_arrival: float  # For timeout calculation
    pending_state_ids: dict[str, str]  # branch_name -> state_id (for completing pending states)
    lost_branches: dict[str, str] = field(default_factory=dict)  # branch_name -> loss reason


class CoalesceExecutor:
    """Executes coalesce operations with audit recording.

    Maintains state for pending coalesce operations:
    - Tracks which tokens have arrived for each row_id
    - Evaluates merge conditions based on policy
    - Merges row data according to strategy
    - Records audit trail via LandscapeRecorder

    Example:
        executor = CoalesceExecutor(recorder, span_factory, token_manager, run_id, step_resolver)

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
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        token_manager: "TokenManager",
        run_id: str,
        step_resolver: StepResolver,
        clock: "Clock | None" = None,
        max_completed_keys: int = 10000,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            token_manager: TokenManager for creating merged tokens
            run_id: Run identifier for audit context
            step_resolver: Resolves NodeID to 1-indexed audit step position.
                           Injected at construction to eliminate step_in_pipeline
                           threading through public method signatures.
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
            max_completed_keys: Maximum late-arrival completion keys retained in memory.
        """
        if max_completed_keys <= 0:
            raise OrchestrationInvariantError(f"max_completed_keys must be > 0, got {max_completed_keys}")

        self._recorder = recorder
        self._spans = span_factory
        self._token_manager = token_manager
        self._run_id = run_id
        self._step_resolver = step_resolver
        self._clock = clock if clock is not None else DEFAULT_CLOCK

        # Coalesce configuration: name -> settings
        self._settings: dict[str, CoalesceSettings] = {}
        # Node IDs: coalesce_name -> node_id
        self._node_ids: dict[str, NodeID] = {}
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
        # Temporary storage for union merge collision info (consumed by _execute_merge)
        self._last_union_collisions: dict[str, list[str]] = {}

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: NodeID,
    ) -> None:
        """Register a coalesce point.

        Args:
            settings: Coalesce configuration
            node_id: Node ID assigned by orchestrator
        """
        self._settings[settings.name] = settings
        self._node_ids[settings.name] = node_id

    def get_registered_names(self) -> list[str]:
        """Get names of all registered coalesce points.

        Used by processor for timeout checking loop.

        Returns:
            List of registered coalesce names
        """
        return list(self._settings.keys())

    def _mark_completed(self, key: tuple[str, str]) -> None:
        """Mark a coalesce key as completed with bounded memory.

        Uses FIFO eviction to prevent unbounded memory growth in long-running
        pipelines. When max capacity is exceeded, oldest entries are removed.
        Late arrivals after eviction will create new pending entries which
        timeout or flush correctly - acceptable trade-off vs OOM.

        Args:
            key: (coalesce_name, row_id) tuple to mark as completed
        """
        self._completed_keys[key] = None
        # Evict oldest entries if over capacity
        evicted_keys: list[tuple[str, str]] = []
        while len(self._completed_keys) > self._max_completed_keys:
            evicted_key, _ = self._completed_keys.popitem(last=False)
            evicted_keys.append(evicted_key)
        if evicted_keys:
            slog.warning(
                "coalesce_completed_keys_evicted",
                max_completed_keys=self._max_completed_keys,
                evicted_count=len(evicted_keys),
                oldest_evicted=evicted_keys[0],
                newest_evicted=evicted_keys[-1],
                retained_count=len(self._completed_keys),
            )

    def accept(
        self,
        token: TokenInfo,
        coalesce_name: str,
    ) -> CoalesceOutcome:
        """Accept a token at a coalesce point.

        If merge conditions are met, returns the merged token.
        Otherwise, holds the token and returns held=True.

        Step position is resolved internally via the injected StepResolver
        from the coalesce point's registered node_id.

        Args:
            token: Token arriving at coalesce point (must have branch_name)
            coalesce_name: Name of the coalesce configuration

        Returns:
            CoalesceOutcome indicating whether token was held or merged

        Raises:
            ValueError: If coalesce_name not registered or token has no branch_name
        """
        if coalesce_name not in self._settings:
            raise ValueError(f"Coalesce '{coalesce_name}' not registered")

        if token.branch_name is None:
            raise ValueError(f"Token {token.token_id} has no branch_name - only forked tokens can be coalesced")

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]
        step = self._step_resolver(node_id)

        # Validate branch is expected
        if token.branch_name not in settings.branches:
            raise ValueError(
                f"Token branch '{token.branch_name}' not in expected branches for coalesce '{coalesce_name}': {settings.branches}"
            )

        # Get or create pending state for this row
        key = (coalesce_name, token.row_id)
        now = self._clock.monotonic()

        # Check if this coalesce already completed (late arrival)
        if key in self._completed_keys:
            # Late arrival after merge/failure already happened
            # Record failure audit trail for this late token
            failure_reason = "late_arrival_after_merge"
            error_hash = hashlib.sha256(failure_reason.encode()).hexdigest()[:16]
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                run_id=self._run_id,
                step_index=step,
                input_data=token.row_data.to_dict(),  # Recorder expects dict
            )
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                error={"failure_reason": failure_reason},
                duration_ms=0,
            )
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=token.token_id,
                outcome=RowOutcome.FAILED,
                error_hash=error_hash,
            )

            # Return failure outcome
            return CoalesceOutcome(
                held=False,
                failure_reason=failure_reason,
                consumed_tokens=[token],
                coalesce_metadata={
                    "policy": settings.policy,
                    "reason": "Siblings already merged/failed, this token arrived too late",
                },
                coalesce_name=coalesce_name,
                outcomes_recorded=True,
            )

        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                arrived={},
                arrival_times={},
                first_arrival=now,
                pending_state_ids={},
            )

        pending = self._pending[key]

        # Detect duplicate arrivals (indicates bug in upstream code)
        # Per "Plugin Ownership" principle: bugs in our code should crash, not hide
        if token.branch_name in pending.arrived:
            existing = pending.arrived[token.branch_name]
            raise ValueError(
                f"Duplicate arrival for branch '{token.branch_name}' at coalesce '{coalesce_name}'. "
                f"Existing token: {existing.token_id}, new token: {token.token_id}. "
                f"This indicates a bug in fork, retry, or checkpoint/resume logic."
            )

        # Record arrival
        pending.arrived[token.branch_name] = token
        pending.arrival_times[token.branch_name] = now

        # Record pending node state for audit trail
        # This ensures held tokens are visible in explain() queries
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node_id,
            run_id=self._run_id,
            step_index=step,
            input_data=token.row_data.to_dict(),  # Recorder expects dict
        )
        pending.pending_state_ids[token.branch_name] = state.state_id

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
        arrived_count = len(pending.arrived)
        expected_count = len(settings.branches)

        if settings.policy == "require_all":
            return arrived_count == expected_count

        elif settings.policy == "first":
            return arrived_count >= 1

        elif settings.policy == "quorum":
            if settings.quorum_count is None:
                raise RuntimeError(
                    f"quorum_count is None for quorum policy at coalesce '{settings.name}'. This indicates a config validation bug."
                )
            return arrived_count >= settings.quorum_count

        # settings.policy == "best_effort":
        # Merge on timeout (checked elsewhere) or when all branches accounted for.
        # Lost branches count as "accounted for" — they won't arrive but we know about them.
        accounted_count = arrived_count + len(pending.lost_branches)
        return accounted_count >= expected_count

    def _fail_pending(
        self,
        settings: CoalesceSettings,
        key: tuple[str, str],
        step: int,
        failure_reason: str,
    ) -> CoalesceOutcome:
        """Fail all arrived tokens in a pending coalesce and clean up.

        Shared helper used by check_timeouts(), flush_pending(), and
        _evaluate_after_loss() to avoid duplicating failure recording logic.

        Args:
            settings: Coalesce settings for metadata
            key: (coalesce_name, row_id) tuple
            step: Resolved audit step index for the coalesce node
            failure_reason: Machine-readable failure reason string

        Returns:
            CoalesceOutcome with failure_reason set and outcomes_recorded=True
        """
        coalesce_name = key[0]
        pending = self._pending[key]
        consumed_tokens = list(pending.arrived.values())
        error_hash = hashlib.sha256(failure_reason.encode()).hexdigest()[:16]
        now = self._clock.monotonic()

        # Complete pending node states with failure
        for branch_name, token in pending.arrived.items():
            state_id = pending.pending_state_ids[branch_name]
            self._recorder.complete_node_state(
                state_id=state_id,
                status=NodeStateStatus.FAILED,
                error={"failure_reason": failure_reason},
                duration_ms=(now - pending.arrival_times[branch_name]) * 1000,
            )
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=token.token_id,
                outcome=RowOutcome.FAILED,
                error_hash=error_hash,
            )

        del self._pending[key]
        self._mark_completed(key)

        # Build metadata with policy-specific fields
        metadata: dict[str, Any] = {
            "policy": settings.policy,
            "expected_branches": settings.branches,
            "branches_arrived": list(pending.arrived.keys()),
        }
        if pending.lost_branches:
            metadata["branches_lost"] = pending.lost_branches
        if settings.quorum_count is not None:
            metadata["quorum_required"] = settings.quorum_count
        if settings.timeout_seconds is not None:
            metadata["timeout_seconds"] = settings.timeout_seconds

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
        for branch, token in pending.arrived.items():
            if token.row_data.contract is None:
                raise ValueError(
                    f"Token {token.token_id} on branch '{branch}' has no contract. "
                    f"Cannot coalesce without contracts on all parents. "
                    f"This indicates a bug in fork or transform execution."
                )

        # Validate select_branch is present for select merge strategy
        # (Bug 2ho fix: reject instead of silent fallback)
        if settings.merge == "select" and settings.select_branch not in pending.arrived:
            # select_branch not arrived - this is a failure, not a fallback
            consumed_tokens = list(pending.arrived.values())
            error_msg = "select_branch_not_arrived"
            error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

            for branch_name, token in pending.arrived.items():
                state_id = pending.pending_state_ids[branch_name]
                self._recorder.complete_node_state(
                    state_id=state_id,
                    status=NodeStateStatus.FAILED,
                    error={
                        "failure_reason": "select_branch_not_arrived",
                        "select_branch": settings.select_branch,
                        "branches_arrived": list(pending.arrived.keys()),
                    },
                    duration_ms=(now - pending.arrival_times[branch_name]) * 1000,
                )
                self._recorder.record_token_outcome(
                    run_id=self._run_id,
                    token_id=token.token_id,
                    outcome=RowOutcome.FAILED,
                    error_hash=error_hash,
                )

            del self._pending[key]
            self._mark_completed(key)
            return CoalesceOutcome(
                held=False,
                failure_reason="select_branch_not_arrived",
                consumed_tokens=consumed_tokens,
                coalesce_metadata={
                    "policy": settings.policy,
                    "merge_strategy": settings.merge,
                    "select_branch": settings.select_branch,
                    "branches_arrived": list(pending.arrived.keys()),
                },
                coalesce_name=coalesce_name,
                outcomes_recorded=True,  # Bug 9z8 fix: token outcomes already recorded above
            )

        # ─────────────────────────────────────────────────────────────────────
        # Merge row data according to strategy (returns dict)
        # We do this FIRST so we can derive contract from actual data shape
        # ─────────────────────────────────────────────────────────────────────
        merged_data_dict = self._merge_data(settings, pending.arrived)

        # ─────────────────────────────────────────────────────────────────────
        # Build contract based on merge strategy and actual data shape
        # ─────────────────────────────────────────────────────────────────────
        contracts: list[SchemaContract] = [t.row_data.contract for t in pending.arrived.values()]

        if settings.merge == "union":
            # Union: Merge all contracts (current behavior - correct)
            merged_contract = contracts[0]
            for c in contracts[1:]:
                try:
                    merged_contract = merged_contract.merge(c)
                except Exception as e:
                    # Contract merge failure is an orchestration invariant violation
                    # Contracts with conflicting types cannot be merged
                    slog.error(
                        "contract_merge_failed",
                        coalesce_name=coalesce_name,
                        branches=list(pending.arrived.keys()),
                        error=str(e),
                    )
                    raise OrchestrationInvariantError(
                        f"Contract merge failed at coalesce point '{coalesce_name}'. Branches: {list(pending.arrived.keys())}. Error: {e}"
                    ) from e

            slog.info(
                "contract_merge_success",
                coalesce_name=coalesce_name,
                merge_strategy="union",
                branch_count=len(pending.arrived),
                branches=list(pending.arrived.keys()),
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
                    required=branch_name in pending.arrived,
                    source="declared",
                )
                for branch_name in settings.branches
            )
            merged_contract = SchemaContract(
                fields=branch_fields,
                mode="FIXED",
                locked=True,
            )

            slog.info(
                "contract_created_for_nested_merge",
                coalesce_name=coalesce_name,
                merge_strategy="nested",
                branch_count=len(pending.arrived),
                branches=list(pending.arrived.keys()),
                output_keys=list(merged_data_dict.keys()),
            )

        elif settings.merge == "select":
            # Select: Use selected branch's contract directly (data has only those fields)
            # Find the selected branch's contract
            if settings.select_branch is None:
                raise RuntimeError(
                    f"select_branch is None for select merge strategy at coalesce '{settings.name}'. This indicates a config validation bug."
                )

            # Get the token for the selected branch
            selected_token = pending.arrived[settings.select_branch]
            merged_contract = selected_token.row_data.contract

            slog.info(
                "contract_from_selected_branch",
                coalesce_name=coalesce_name,
                merge_strategy="select",
                selected_branch=settings.select_branch,
                branches_arrived=list(pending.arrived.keys()),
            )

        else:
            # Unreachable - config validation ensures merge is one of the above
            raise RuntimeError(f"Unknown merge strategy: {settings.merge}")

        # Create PipelineRow with strategy-appropriate contract
        merged_data = PipelineRow(merged_data_dict, merged_contract)

        # Get list of consumed tokens
        consumed_tokens = list(pending.arrived.values())

        # Create merged token via TokenManager
        merged_token = self._token_manager.coalesce_tokens(
            parents=consumed_tokens,
            merged_data=merged_data,
            node_id=node_id,
        )

        # Build audit metadata BEFORE completing node states (Bug l4h fix)
        # This allows us to include it in context_after for each consumed token
        coalesce_metadata: dict[str, Any] = {
            "policy": settings.policy,
            "merge_strategy": settings.merge,
            "expected_branches": settings.branches,
            "branches_arrived": list(pending.arrived.keys()),
            "branches_lost": pending.lost_branches if pending.lost_branches else {},
            "arrival_order": [
                {
                    "branch": branch,
                    "arrival_offset_ms": (t - pending.first_arrival) * 1000,
                }
                for branch, t in sorted(pending.arrival_times.items(), key=lambda x: x[1])
            ],
            "wait_duration_ms": (now - pending.first_arrival) * 1000,
        }

        # Include union merge collision info in audit trail if present
        if self._last_union_collisions:
            coalesce_metadata["union_field_collisions"] = self._last_union_collisions
            self._last_union_collisions = {}

        # Complete pending node states for consumed tokens
        # (These states were created as "pending" when tokens were held in accept())
        for branch_name, token in pending.arrived.items():
            # Get the pending state_id that was created when token was held
            state_id = pending.pending_state_ids[branch_name]

            # Complete it now that merge is happening
            # Bug l4h fix: include coalesce_context in context_after for audit trail
            self._recorder.complete_node_state(
                state_id=state_id,
                status=NodeStateStatus.COMPLETED,
                output_data={"merged_into": merged_token.token_id},
                duration_ms=(now - pending.arrival_times[branch_name]) * 1000,
                context_after={"coalesce_context": coalesce_metadata},
            )

            # Record terminal token outcome (COALESCED)
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COALESCED,
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

    def _merge_data(
        self,
        settings: CoalesceSettings,
        arrived: dict[str, TokenInfo],
    ) -> dict[str, Any]:
        """Merge row data from arrived tokens based on strategy.

        Note: row_data is PipelineRow, so we use to_dict() to get raw dict.
        """
        if settings.merge == "union":
            # Combine all fields from all branches.
            # On name collision, the last branch in settings.branches wins.
            # Collisions are recorded in the audit trail (coalesce_metadata)
            # so that overwritten values are never silently lost.
            merged: dict[str, Any] = {}
            field_origins: dict[str, str] = {}  # field -> branch that set it
            collisions: dict[str, list[str]] = {}  # field -> [branch1, branch2, ...]
            for branch_name in settings.branches:
                if branch_name in arrived:
                    branch_data = arrived[branch_name].row_data.to_dict()
                    for field in branch_data:
                        if field in field_origins:
                            if field not in collisions:
                                collisions[field] = [field_origins[field]]
                            collisions[field].append(branch_name)
                        field_origins[field] = branch_name
                    merged.update(branch_data)
            if collisions:
                slog.warning(
                    "union_merge_field_collisions",
                    collisions=dict(collisions),
                    winner_branch={f: branches[-1] for f, branches in collisions.items()},
                )
            # Stash collisions for audit metadata (read by _execute_merge)
            self._last_union_collisions = collisions
            return merged

        elif settings.merge == "nested":
            # Each branch as nested object (use to_dict() for serializable dict)
            return {branch_name: arrived[branch_name].row_data.to_dict() for branch_name in settings.branches if branch_name in arrived}

        # settings.merge == "select":
        # Take specific branch output
        # Note: _execute_merge validates select_branch presence before calling this method
        # If we get here without select_branch, it's a bug in our code
        if settings.select_branch is None:
            raise RuntimeError(
                f"select_branch is None for select merge strategy at coalesce '{settings.name}'. This indicates a config validation bug."
            )
        if settings.select_branch not in arrived:
            # This should be unreachable - _execute_merge catches this case first
            # Per "Plugin Ownership" principle: crash on our bugs, don't hide them
            raise RuntimeError(
                f"select_branch '{settings.select_branch}' not in arrived branches {list(arrived.keys())}. "
                f"This indicates a bug in _execute_merge validation (Bug 2ho fix should have caught this)."
            )
        # Return copy of dict (to_dict() returns a copy already)
        return arrived[settings.select_branch].row_data.to_dict()

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
            raise ValueError(f"Coalesce '{coalesce_name}' not registered")

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
            pending = self._pending[key]

            # For best_effort, always merge on timeout if anything arrived
            if settings.policy == "best_effort" and len(pending.arrived) > 0:
                outcome = self._execute_merge(
                    settings=settings,
                    node_id=node_id,
                    pending=pending,
                    step=step,
                    key=key,
                    coalesce_name=coalesce_name,
                )
                results.append(outcome)

            # For quorum, merge on timeout only if quorum met
            elif settings.policy == "quorum":
                if settings.quorum_count is None:
                    raise RuntimeError(
                        f"quorum_count is None for quorum policy at coalesce '{settings.name}'. This indicates a config validation bug."
                    )
                if len(pending.arrived) >= settings.quorum_count:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step=step,
                        key=key,
                        coalesce_name=coalesce_name,
                    )
                    results.append(outcome)
                else:
                    # Quorum not met at timeout - record failure
                    results.append(
                        self._fail_pending(
                            settings,
                            key,
                            step,
                            failure_reason="quorum_not_met_at_timeout",
                        )
                    )

            # For require_all, timeout means incomplete - record failure
            # (Bug P1-2026-01-30 fix: require_all was missing from check_timeouts)
            elif settings.policy == "require_all":
                # require_all never does partial merge - timeout is always a failure
                results.append(
                    self._fail_pending(
                        settings,
                        key,
                        step,
                        failure_reason="incomplete_branches",
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

            if settings.policy == "best_effort":
                # Merge whatever arrived (or fail if nothing arrived)
                if len(pending.arrived) > 0:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step=step,
                        key=key,
                        coalesce_name=coalesce_name,
                    )
                    results.append(outcome)
                elif pending.lost_branches:
                    # All branches lost — no data to merge
                    results.append(
                        self._fail_pending(
                            settings,
                            key,
                            step,
                            failure_reason="all_branches_lost",
                        )
                    )

            elif settings.policy == "quorum":
                if settings.quorum_count is None:
                    raise RuntimeError(
                        f"quorum_count is None for quorum policy at coalesce '{settings.name}'. This indicates a config validation bug."
                    )
                if len(pending.arrived) >= settings.quorum_count:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step=step,
                        key=key,
                        coalesce_name=coalesce_name,
                    )
                    results.append(outcome)
                else:
                    # Quorum not met - record failure
                    results.append(
                        self._fail_pending(
                            settings,
                            key,
                            step,
                            failure_reason="quorum_not_met",
                        )
                    )

            elif settings.policy == "require_all":
                # require_all never does partial merge
                results.append(
                    self._fail_pending(
                        settings,
                        key,
                        step,
                        failure_reason="incomplete_branches",
                    )
                )

            elif settings.policy == "first":
                # first policy merges immediately on first arrival - pending entries indicate a bug
                raise RuntimeError(
                    f"Invariant violation: 'first' policy should never have pending entries. "
                    f"Found pending coalesce for row_id='{key[1]}' at '{coalesce_name}' with "
                    f"branches {list(pending.arrived.keys())}. This indicates a bug in accept() logic."
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
            raise ValueError(f"Coalesce '{coalesce_name}' not registered")

        key = (coalesce_name, row_id)

        # Already completed (race with normal merge) — ignore
        if key in self._completed_keys:
            return None

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]
        step = self._step_resolver(node_id)

        # Validate branch is expected
        if lost_branch not in settings.branches:
            raise ValueError(f"Lost branch '{lost_branch}' not in expected branches for coalesce '{coalesce_name}': {settings.branches}")

        # No pending entry yet — branch lost before ANY branch arrived.
        # Create a minimal pending entry with the loss recorded.
        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                arrived={},
                arrival_times={},
                first_arrival=self._clock.monotonic(),
                pending_state_ids={},
                lost_branches={lost_branch: reason},
            )
            return self._evaluate_after_loss(settings, key, step)

        pending = self._pending[key]

        # Validate branch hasn't already arrived (would be a processor bug)
        if lost_branch in pending.arrived:
            raise ValueError(
                f"Branch '{lost_branch}' already arrived at coalesce '{coalesce_name}' "
                f"but was reported as lost. This indicates a bug in the processor — "
                f"a token cannot both arrive and be error-routed."
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
        arrived_count = len(pending.arrived)
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
            if settings.quorum_count is None:
                raise RuntimeError(
                    f"quorum_count is None for quorum policy at coalesce '{settings.name}'. This indicates a config validation bug."
                )
            # Check if quorum is now impossible
            max_possible = total_branches - lost_count
            if max_possible < settings.quorum_count:
                return self._fail_pending(
                    settings,
                    key,
                    step,
                    failure_reason=f"quorum_impossible:need={settings.quorum_count},max_possible={max_possible}",
                )
            # Check if arrived count already meets quorum
            if arrived_count >= settings.quorum_count:
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

        # settings.policy == "first":
        # first: should already have merged on first arrival
        # If no arrivals yet, nothing to merge
        return None
