"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import NodeStateStatus, RowOutcome
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock
    from elspeth.engine.tokens import TokenManager


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


class CoalesceExecutor:
    """Executes coalesce operations with audit recording.

    Maintains state for pending coalesce operations:
    - Tracks which tokens have arrived for each row_id
    - Evaluates merge conditions based on policy
    - Merges row data according to strategy
    - Records audit trail via LandscapeRecorder

    Example:
        executor = CoalesceExecutor(recorder, span_factory, token_manager, run_id)

        # Configure coalesce point
        executor.register_coalesce(settings, node_id)

        # Accept tokens as they arrive
        for token in arriving_tokens:
            outcome = executor.accept(token, "coalesce_name", step_in_pipeline)
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
        clock: "Clock | None" = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            token_manager: TokenManager for creating merged tokens
            run_id: Run identifier for audit context
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
        """
        self._recorder = recorder
        self._spans = span_factory
        self._token_manager = token_manager
        self._run_id = run_id
        self._clock = clock if clock is not None else DEFAULT_CLOCK

        # Coalesce configuration: name -> settings
        self._settings: dict[str, CoalesceSettings] = {}
        # Node IDs: coalesce_name -> node_id
        self._node_ids: dict[str, str] = {}
        # Pending tokens: (coalesce_name, row_id) -> _PendingCoalesce
        self._pending: dict[tuple[str, str], _PendingCoalesce] = {}
        # Completed coalesces: tracks keys that have already merged/failed
        # Used to detect late arrivals after merge and reject them gracefully
        # Uses OrderedDict as bounded FIFO set to prevent unbounded memory growth
        # (values are None, we only care about key presence and insertion order)
        self._completed_keys: OrderedDict[tuple[str, str], None] = OrderedDict()
        # Maximum completed keys to retain (prevents OOM in long-running pipelines)
        # Late arrivals after eviction create new pending entries (timeout/flush correctly)
        self._max_completed_keys: int = 10000

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: str,
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
        while len(self._completed_keys) > self._max_completed_keys:
            self._completed_keys.popitem(last=False)

    def accept(
        self,
        token: TokenInfo,
        coalesce_name: str,
        step_in_pipeline: int,
    ) -> CoalesceOutcome:
        """Accept a token at a coalesce point.

        If merge conditions are met, returns the merged token.
        Otherwise, holds the token and returns held=True.

        Args:
            token: Token arriving at coalesce point (must have branch_name)
            coalesce_name: Name of the coalesce configuration
            step_in_pipeline: Current position in DAG

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
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                run_id=self._run_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                error={"failure_reason": "late_arrival_after_merge"},
                duration_ms=0,
            )

            # Return failure outcome
            return CoalesceOutcome(
                held=False,
                failure_reason="late_arrival_after_merge",
                consumed_tokens=[token],
                coalesce_metadata={
                    "policy": settings.policy,
                    "reason": "Siblings already merged/failed, this token arrived too late",
                },
                coalesce_name=coalesce_name,
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
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )
        pending.pending_state_ids[token.branch_name] = state.state_id

        # Check if merge conditions are met
        if self._should_merge(settings, pending):
            return self._execute_merge(
                settings=settings,
                node_id=node_id,
                pending=pending,
                step_in_pipeline=step_in_pipeline,
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
        # Only merge on timeout (checked elsewhere) or if all arrived
        return arrived_count == expected_count

    def _execute_merge(
        self,
        settings: CoalesceSettings,
        node_id: str,
        pending: _PendingCoalesce,
        step_in_pipeline: int,
        key: tuple[str, str],
        coalesce_name: str,
    ) -> CoalesceOutcome:
        """Execute the merge and create merged token."""
        now = self._clock.monotonic()

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

        # Merge row data according to strategy
        merged_data = self._merge_data(settings, pending.arrived)

        # Get list of consumed tokens
        consumed_tokens = list(pending.arrived.values())

        # Create merged token via TokenManager
        merged_token = self._token_manager.coalesce_tokens(
            parents=consumed_tokens,
            merged_data=merged_data,
            step_in_pipeline=step_in_pipeline,
        )

        # Build audit metadata BEFORE completing node states (Bug l4h fix)
        # This allows us to include it in context_after for each consumed token
        coalesce_metadata = {
            "policy": settings.policy,
            "merge_strategy": settings.merge,
            "expected_branches": settings.branches,
            "branches_arrived": list(pending.arrived.keys()),
            "arrival_order": [
                {
                    "branch": branch,
                    "arrival_offset_ms": (t - pending.first_arrival) * 1000,
                }
                for branch, t in sorted(pending.arrival_times.items(), key=lambda x: x[1])
            ],
            "wait_duration_ms": (now - pending.first_arrival) * 1000,
        }

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
        """Merge row data from arrived tokens based on strategy."""
        if settings.merge == "union":
            # Combine all fields (later branches override earlier)
            merged: dict[str, Any] = {}
            for branch_name in settings.branches:
                if branch_name in arrived:
                    merged.update(arrived[branch_name].row_data)
            return merged

        elif settings.merge == "nested":
            # Each branch as nested object
            return {branch_name: arrived[branch_name].row_data for branch_name in settings.branches if branch_name in arrived}

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
        return arrived[settings.select_branch].row_data.copy()

    def check_timeouts(
        self,
        coalesce_name: str,
        step_in_pipeline: int,
    ) -> list[CoalesceOutcome]:
        """Check for timed-out pending coalesces and merge them.

        For best_effort policy, merges whatever has arrived when timeout expires.
        For quorum policy with timeout, merges if quorum met when timeout expires.

        Args:
            coalesce_name: Name of the coalesce configuration
            step_in_pipeline: Current position in DAG

        Returns:
            List of CoalesceOutcomes for any merges triggered by timeout
        """
        if coalesce_name not in self._settings:
            raise ValueError(f"Coalesce '{coalesce_name}' not registered")

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]

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
                    step_in_pipeline=step_in_pipeline,
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
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                        coalesce_name=coalesce_name,
                    )
                    results.append(outcome)
                else:
                    # Quorum not met at timeout - record failure
                    # (Bug 6tb fix: mirrors flush_pending() failure handling)
                    consumed_tokens = list(pending.arrived.values())
                    error_msg = "quorum_not_met_at_timeout"
                    error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]
                    failure_time = self._clock.monotonic()

                    # Complete pending node states with failure
                    for branch_name, token in pending.arrived.items():
                        state_id = pending.pending_state_ids[branch_name]
                        self._recorder.complete_node_state(
                            state_id=state_id,
                            status=NodeStateStatus.FAILED,
                            error={"failure_reason": "quorum_not_met_at_timeout"},
                            duration_ms=(failure_time - pending.arrival_times[branch_name]) * 1000,
                        )
                        self._recorder.record_token_outcome(
                            run_id=self._run_id,
                            token_id=token.token_id,
                            outcome=RowOutcome.FAILED,
                            error_hash=error_hash,
                        )

                    del self._pending[key]
                    self._mark_completed(key)
                    results.append(
                        CoalesceOutcome(
                            held=False,
                            failure_reason="quorum_not_met_at_timeout",
                            consumed_tokens=consumed_tokens,
                            coalesce_metadata={
                                "policy": settings.policy,
                                "quorum_required": settings.quorum_count,
                                "branches_arrived": list(pending.arrived.keys()),
                                "timeout_seconds": settings.timeout_seconds,
                            },
                            coalesce_name=coalesce_name,
                            outcomes_recorded=True,  # Bug 9z8 fix: token outcomes recorded above
                        )
                    )

            # For require_all, timeout means incomplete - record failure
            # (Bug P1-2026-01-30 fix: require_all was missing from check_timeouts)
            elif settings.policy == "require_all":
                # require_all never does partial merge - timeout is always a failure
                consumed_tokens = list(pending.arrived.values())
                error_msg = "incomplete_branches"
                error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]
                failure_time = self._clock.monotonic()

                # Complete pending node states with failure
                for branch_name, token in pending.arrived.items():
                    state_id = pending.pending_state_ids[branch_name]
                    self._recorder.complete_node_state(
                        state_id=state_id,
                        status=NodeStateStatus.FAILED,
                        error={"failure_reason": "incomplete_branches"},
                        duration_ms=(failure_time - pending.arrival_times[branch_name]) * 1000,
                    )
                    self._recorder.record_token_outcome(
                        run_id=self._run_id,
                        token_id=token.token_id,
                        outcome=RowOutcome.FAILED,
                        error_hash=error_hash,
                    )

                del self._pending[key]
                self._mark_completed(key)
                results.append(
                    CoalesceOutcome(
                        held=False,
                        failure_reason="incomplete_branches",
                        consumed_tokens=consumed_tokens,
                        coalesce_metadata={
                            "policy": settings.policy,
                            "expected_branches": settings.branches,
                            "branches_arrived": list(pending.arrived.keys()),
                            "timeout_seconds": settings.timeout_seconds,
                        },
                        coalesce_name=coalesce_name,
                        outcomes_recorded=True,
                    )
                )

        return results

    def flush_pending(
        self,
        step_map: dict[str, int],
    ) -> list[CoalesceOutcome]:
        """Flush all pending coalesces (called at end-of-source or shutdown).

        For best_effort policy: merges whatever arrived.
        For quorum policy: merges if quorum met, returns failure otherwise.
        For require_all policy: returns failure (never partial merge).
        For first policy: should never have pending (merges immediately).

        Args:
            step_map: Map of coalesce_name -> step_in_pipeline for audit trail

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
            step_in_pipeline = step_map[coalesce_name]

            if settings.policy == "best_effort":
                # Always merge whatever arrived
                if len(pending.arrived) > 0:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                        coalesce_name=coalesce_name,
                    )
                    results.append(outcome)

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
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                        coalesce_name=coalesce_name,
                    )
                    results.append(outcome)
                else:
                    # Quorum not met - record failure
                    # MUST capture tokens before deleting pending state
                    consumed_tokens = list(pending.arrived.values())

                    # Compute error hash for failure reason
                    error_msg = "quorum_not_met"
                    error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

                    # Compute wait duration
                    now = self._clock.monotonic()

                    # Complete pending node states for consumed tokens (audit trail)
                    # (These states were created as "pending" when tokens were held in accept())
                    for branch_name, token in pending.arrived.items():
                        # Get the pending state_id that was created when token was held
                        state_id = pending.pending_state_ids[branch_name]

                        # Complete it now with failure status
                        self._recorder.complete_node_state(
                            state_id=state_id,
                            status=NodeStateStatus.FAILED,
                            error={"failure_reason": "quorum_not_met"},
                            duration_ms=(now - pending.arrival_times[branch_name]) * 1000,
                        )

                        # Record terminal token outcome (FAILED)
                        self._recorder.record_token_outcome(
                            run_id=self._run_id,
                            token_id=token.token_id,
                            outcome=RowOutcome.FAILED,
                            error_hash=error_hash,
                        )

                    del self._pending[key]
                    self._mark_completed(key)  # Track completion to reject late arrivals (bounded)
                    results.append(
                        CoalesceOutcome(
                            held=False,
                            failure_reason="quorum_not_met",
                            consumed_tokens=consumed_tokens,
                            coalesce_metadata={
                                "policy": settings.policy,
                                "quorum_required": settings.quorum_count,
                                "branches_arrived": list(pending.arrived.keys()),
                            },
                            coalesce_name=coalesce_name,
                            outcomes_recorded=True,  # Bug 9z8 fix: token outcomes recorded above
                        )
                    )

            elif settings.policy == "require_all":
                # require_all never does partial merge
                # MUST capture tokens before deleting pending state
                consumed_tokens = list(pending.arrived.values())

                # Compute error hash for failure reason
                error_msg = "incomplete_branches"
                error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

                # Compute wait duration
                now = self._clock.monotonic()

                # Complete pending node states for consumed tokens (audit trail)
                # (These states were created as "pending" when tokens were held in accept())
                for branch_name, token in pending.arrived.items():
                    # Get the pending state_id that was created when token was held
                    state_id = pending.pending_state_ids[branch_name]

                    # Complete it now with failure status
                    self._recorder.complete_node_state(
                        state_id=state_id,
                        status=NodeStateStatus.FAILED,
                        error={"failure_reason": "incomplete_branches"},
                        duration_ms=(now - pending.arrival_times[branch_name]) * 1000,
                    )

                    # Record terminal token outcome (FAILED)
                    self._recorder.record_token_outcome(
                        run_id=self._run_id,
                        token_id=token.token_id,
                        outcome=RowOutcome.FAILED,
                        error_hash=error_hash,
                    )

                del self._pending[key]
                self._mark_completed(key)  # Track completion to reject late arrivals (bounded)
                results.append(
                    CoalesceOutcome(
                        held=False,
                        failure_reason="incomplete_branches",
                        consumed_tokens=consumed_tokens,
                        coalesce_metadata={
                            "policy": settings.policy,
                            "expected_branches": settings.branches,
                            "branches_arrived": list(pending.arrived.keys()),
                        },
                        coalesce_name=coalesce_name,
                        outcomes_recorded=True,  # Bug 9z8 fix: token outcomes recorded above
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
