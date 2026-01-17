"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import TokenInfo
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
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
    """

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: list[TokenInfo] = field(default_factory=list)
    coalesce_metadata: dict[str, Any] | None = None
    failure_reason: str | None = None


@dataclass
class _PendingCoalesce:
    """Tracks pending tokens for a single row_id at a coalesce point."""

    arrived: dict[str, TokenInfo]  # branch_name -> token
    arrival_times: dict[str, float]  # branch_name -> monotonic time
    first_arrival: float  # For timeout calculation


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
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            token_manager: TokenManager for creating merged tokens
            run_id: Run identifier for audit context
        """
        self._recorder = recorder
        self._spans = span_factory
        self._token_manager = token_manager
        self._run_id = run_id

        # Coalesce configuration: name -> settings
        self._settings: dict[str, CoalesceSettings] = {}
        # Node IDs: coalesce_name -> node_id
        self._node_ids: dict[str, str] = {}
        # Pending tokens: (coalesce_name, row_id) -> _PendingCoalesce
        self._pending: dict[tuple[str, str], _PendingCoalesce] = {}

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
            raise ValueError(
                f"Token {token.token_id} has no branch_name - "
                "only forked tokens can be coalesced"
            )

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]

        # Validate branch is expected
        if token.branch_name not in settings.branches:
            raise ValueError(
                f"Token branch '{token.branch_name}' not in expected branches "
                f"for coalesce '{coalesce_name}': {settings.branches}"
            )

        # Get or create pending state for this row
        key = (coalesce_name, token.row_id)
        now = time.monotonic()

        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                arrived={},
                arrival_times={},
                first_arrival=now,
            )

        pending = self._pending[key]

        # Record arrival
        pending.arrived[token.branch_name] = token
        pending.arrival_times[token.branch_name] = now

        # Check if merge conditions are met
        if self._should_merge(settings, pending):
            return self._execute_merge(
                settings=settings,
                node_id=node_id,
                pending=pending,
                step_in_pipeline=step_in_pipeline,
                key=key,
            )

        # Hold token
        return CoalesceOutcome(held=True)

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
            assert settings.quorum_count is not None
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
    ) -> CoalesceOutcome:
        """Execute the merge and create merged token."""
        now = time.monotonic()

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

        # Record node states for consumed tokens
        for token in consumed_tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                output_data={"merged_into": merged_token.token_id},
                duration_ms=0,
            )

        # Build audit metadata
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
                for branch, t in sorted(
                    pending.arrival_times.items(), key=lambda x: x[1]
                )
            ],
            "wait_duration_ms": (now - pending.first_arrival) * 1000,
        }

        # Clean up pending state
        del self._pending[key]

        return CoalesceOutcome(
            held=False,
            merged_token=merged_token,
            consumed_tokens=consumed_tokens,
            coalesce_metadata=coalesce_metadata,
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
            return {
                branch_name: arrived[branch_name].row_data
                for branch_name in settings.branches
                if branch_name in arrived
            }

        # settings.merge == "select":
        # Take specific branch output
        assert settings.select_branch is not None
        if settings.select_branch in arrived:
            return arrived[settings.select_branch].row_data.copy()
        # Fallback to first arrived if select branch not present
        return next(iter(arrived.values())).row_data.copy()

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

        now = time.monotonic()
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
                )
                results.append(outcome)

            # For quorum, merge on timeout only if quorum met
            elif settings.policy == "quorum":
                assert settings.quorum_count is not None
                if len(pending.arrived) >= settings.quorum_count:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)

        return results

    def flush_pending(
        self,
        step_in_pipeline: int,
    ) -> list[CoalesceOutcome]:
        """Flush all pending coalesces (called at end-of-source or shutdown).

        For best_effort policy: merges whatever arrived.
        For quorum policy: merges if quorum met, returns failure otherwise.
        For require_all policy: returns failure (never partial merge).
        For first policy: should never have pending (merges immediately).

        Args:
            step_in_pipeline: Current position in DAG

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

            if settings.policy == "best_effort":
                # Always merge whatever arrived
                if len(pending.arrived) > 0:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)

            elif settings.policy == "quorum":
                assert settings.quorum_count is not None
                if len(pending.arrived) >= settings.quorum_count:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)
                else:
                    # Quorum not met - record failure
                    del self._pending[key]
                    results.append(
                        CoalesceOutcome(
                            held=False,
                            failure_reason="quorum_not_met",
                            coalesce_metadata={
                                "policy": settings.policy,
                                "quorum_required": settings.quorum_count,
                                "branches_arrived": list(pending.arrived.keys()),
                            },
                        )
                    )

            elif settings.policy == "require_all":
                # require_all never does partial merge
                del self._pending[key]
                results.append(
                    CoalesceOutcome(
                        held=False,
                        failure_reason="incomplete_branches",
                        coalesce_metadata={
                            "policy": settings.policy,
                            "expected_branches": settings.branches,
                            "branches_arrived": list(pending.arrived.keys()),
                        },
                    )
                )

            # first policy: should never have pending entries (merges immediately)

        return results
