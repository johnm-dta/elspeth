"""TokenTraversalEngine: the per-token DAG traversal state machine.

Extracted from ``RowProcessor`` (filigree elspeth-c49f33d6e4, component 4 — the
final slice of the god-class split). Owns ``process_single_token`` (the per-token
DAG traversal loop) and its transform / gate / terminal handler family.

Processor-owned collaborators (navigation, executors, coalesce, telemetry, audit
recording, the scheduler-lease heartbeat) are reached at CALL time through
``self._processor.<seam>`` — never captured at construction. Test-seam contract:
patching ``processor._process_single_token`` intercepts drain-driven work (the
SchedulerDrainHost seam resolves it on the processor at call time), but the
traversal loop dispatches its handler family on the ENGINE itself — patching
``processor._handle_transform_node`` / ``_handle_transform_error_status``
intercepts only DIRECT calls to those thin delegates, not loop-driven dispatch;
to intercept the loop, patch ``processor._token_traversal.<handler>`` instead.
The ``_Transform*`` / ``_Gate*`` discriminated-union outcome types live
here and are re-exported from ``processor.py`` for callers that import them there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from elspeth.contracts import RowResult, TokenInfo, TransformProtocol, TransformResult
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import RoutingKind, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import MaxRetriesExceeded, OrchestrationInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import FailureInfo
from elspeth.contracts.types import BranchName, CoalesceName, NodeID
from elspeth.core.config import GateSettings
from elspeth.engine._best_effort import best_effort
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.dag_navigator import WorkItem

if TYPE_CHECKING:
    from elspeth.engine.executors import GateOutcome
    from elspeth.engine.processor import RowProcessor

logger = logging.getLogger(__name__)


# --- Discriminated union types for _process_single_token extraction ---


@dataclass(frozen=True, slots=True)
class _TransformContinue:
    """Token should advance to the next node in the DAG."""

    updated_token: TokenInfo
    updated_sink: str


@dataclass(frozen=True, slots=True)
class _TransformTerminal:
    """Token has reached a terminal state (completed, failed, quarantined, etc.)."""

    result: RowResult | tuple[RowResult, ...]


type _TransformOutcome = _TransformContinue | _TransformTerminal


@dataclass(frozen=True, slots=True)
class _GateContinue:
    """Gate says advance to next node (or jump to a specific node)."""

    updated_token: TokenInfo
    updated_sink: str
    next_node_id: NodeID | None = None  # None = next structural node


@dataclass(frozen=True, slots=True)
class _GateTerminal:
    """Gate has routed, forked, or diverted the token to a terminal state."""

    result: RowResult | tuple[RowResult, ...]


type _GateOutcome = _GateContinue | _GateTerminal


class TokenTraversalEngine:
    """DAG token-traversal state machine, extracted from RowProcessor (c49 component 4).

    Reaches processor-owned seams at call time via ``self._processor`` so tests
    that patch those methods on the processor continue to intercept them.
    """

    def __init__(self, processor: RowProcessor) -> None:
        self._processor = processor

    def handle_transform_node(
        self,
        transform: TransformProtocol,
        current_token: TokenInfo,
        ctx: PluginContext,
        node_id: NodeID,
        child_items: list[WorkItem],
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
        current_on_success_sink: str,
        attempt_offset: int = 0,
    ) -> _TransformOutcome:
        """Handle a single transform node: execute with retry, route errors, handle multi-row.

        Args:
            transform: The transform plugin to execute.
            current_token: Token being processed through the DAG.
            ctx: Plugin context for the current run.
            node_id: Current DAG node ID (needed for deaggregation expand_token() and
                child work item creation via create_continuation_work_item()).
            child_items: Mutable list — deaggregation appends child work items here.
            coalesce_node_id: Coalesce barrier node for fork branches (or None).
            coalesce_name: Coalesce point name for fork branches (or None).
            current_on_success_sink: Current sink name, may be updated by transform.on_success.

        Resume state (attempt offset and checkpoint provenance) is carried on
        current_token.resume_attempt_offset and current_token.resume_checkpoint_id
        and flow through to execute_transform without explicit threading.

        Returns:
            _TransformContinue: Token should advance to next node (updated token + updated sink).
            _TransformTerminal: Token reached terminal state (FAILED, QUARANTINED, ROUTED, or EXPANDED).
        """
        # 1. Execute transform with retry
        try:
            transform_result, current_token, error_sink = self._processor._execute_transform_with_retry(
                transform=transform,
                token=current_token,
                ctx=ctx,
                attempt_offset=attempt_offset,
            )
            # Emit TransformCompleted telemetry AFTER Landscape recording succeeds
            # (Landscape recording happens inside _execute_transform_with_retry)
            self._processor._emit_transform_completed(
                token=current_token,
                transform=transform,
                transform_result=transform_result,
            )
        except MaxRetriesExceeded as e:
            # All retries exhausted - return FAILED outcome
            error_hash = compute_error_hash(str(e), exception_type=type(e).__name__)
            self._processor._data_flow.record_token_outcome(
                ref=TokenRef(token_id=current_token.token_id, run_id=self._processor._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=error_hash,
            )
            # Emit TokenCompleted telemetry AFTER Landscape recording
            self._processor._emit_token_completed(
                current_token,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )
            # Notify coalesce if this is a forked branch
            sibling_results = self._processor._notify_coalesce_of_lost_branch(
                current_token,
                f"max_retries_exceeded:{e}",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error=FailureInfo.from_max_retries_exceeded(e),
            )
            if sibling_results:
                return _TransformTerminal(result=(current_result, *sibling_results))
            return _TransformTerminal(result=current_result)

        # 2. Handle error status
        if transform_result.status == "error":
            return self.handle_transform_error_status(
                transform_result,
                current_token,
                error_sink,
                child_items,
            )

        # 3. Track on_success for sink routing at end of chain
        updated_sink = current_on_success_sink
        if transform.on_success is not None:
            updated_sink = transform.on_success

        # 4. Handle multi-row output (deaggregation)
        # NOTE: This is ONLY for non-aggregation transforms. Aggregation
        # transforms route through _process_batch_aggregation_node() above.
        if transform_result.is_multi_row:
            if transform_result.rows is None:
                raise OrchestrationInvariantError("is_multi_row guarantees rows is not None")
            if len(transform_result.rows) == 0:
                self._processor._record_dropped_by_filter_outcome(
                    token=current_token,
                    transform_name=transform.name,
                    node_id=node_id,
                    path_label="after success_empty()",
                )
                self._processor._emit_token_completed(
                    current_token,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.FILTER_DROPPED,
                )
                sibling_results = self._processor._notify_coalesce_of_lost_branch(
                    current_token,
                    "dropped_by_filter",
                    child_items,
                )
                current_result = RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.FILTER_DROPPED,
                )
                if sibling_results:
                    return _TransformTerminal(result=(current_result, *sibling_results))
                return _TransformTerminal(result=current_result)

            # Validate transform is allowed to create tokens
            if not transform.creates_tokens:
                raise RuntimeError(
                    f"Transform '{transform.name}' returned multi-row result "
                    f"but has creates_tokens=False. Either set creates_tokens=True "
                    f"or return single row via TransformResult.success(row). "
                    f"(Multi-row is allowed in aggregation passthrough mode.)"
                )

            # Deaggregation: create child tokens for each output row
            # NOTE: Parent EXPANDED outcome is recorded atomically in expand_token()
            # Contract consistency is enforced by TransformResult.success_multi()
            output_contract = transform_result.rows[0].contract
            child_tokens, _expand_group_id = self._processor._token_manager.expand_token(
                parent_token=current_token,
                expanded_rows=[r.to_dict() for r in transform_result.rows],
                output_contract=output_contract,
                node_id=node_id,
                run_id=self._processor._run_id,
            )

            # Queue each child for continued processing.
            # Pass updated_sink so terminal children inherit the
            # expanding transform's sink instead of defaulting to source_on_success.
            # Children born during a re-drive get fresh token_ids with no prior node_states,
            # so they use the default resume_attempt_offset=0 / resume_checkpoint_id=None.
            for child_token in child_tokens:
                child_coalesce_name = coalesce_name if coalesce_name is not None and child_token.branch_name is not None else None
                child_items.append(
                    self._processor._nav.create_continuation_work_item(
                        token=child_token,
                        current_node_id=node_id,
                        coalesce_name=child_coalesce_name,
                        on_success_sink=updated_sink,
                    )
                )

            # NOTE: Parent EXPANDED outcome is recorded atomically in expand_token()
            # to eliminate crash window between child creation and outcome recording.
            return _TransformTerminal(
                result=RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=TerminalOutcome.TRANSIENT,
                    path=TerminalPath.EXPAND_PARENT,
                )
            )

        # 5. Single row success — continue to next node
        # (current_token already updated by _execute_transform_with_retry)
        return _TransformContinue(updated_token=current_token, updated_sink=updated_sink)

    def handle_transform_error_status(
        self,
        transform_result: TransformResult,
        current_token: TokenInfo,
        error_sink: str | None,
        child_items: list[WorkItem],
    ) -> _TransformTerminal:
        """Handle transform error status: quarantine (discard) or route to error sink.

        Args:
            transform_result: The failed transform result.
            current_token: Token that failed processing.
            error_sink: "discard" for quarantine, or a sink name for error routing.
            child_items: Mutable list — coalesce notifications may append child work items.

        Returns:
            _TransformTerminal with QUARANTINED or ROUTED_ON_ERROR outcome.
        """
        if error_sink == "discard":
            # Intentionally discarded - QUARANTINED
            # The QUARANTINED path tolerates an "unknown_error" fallback for
            # historical reasons; do NOT extend that fallback to ROUTED_ON_ERROR
            # below — see the offensive guard in the routed branch.
            error_detail = str(transform_result.reason) if transform_result.reason else "unknown_error"
            quarantine_error_hash = compute_error_hash(error_detail)
            self._processor._data_flow.record_token_outcome(
                ref=TokenRef(token_id=current_token.token_id, run_id=self._processor._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
                error_hash=quarantine_error_hash,
            )
            # Emit TokenCompleted telemetry AFTER Landscape recording
            self._processor._emit_token_completed(
                current_token,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )
            # Notify coalesce if this is a forked branch
            sibling_results = self._processor._notify_coalesce_of_lost_branch(
                current_token,
                f"quarantined:{error_detail}",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )
            if sibling_results:
                return _TransformTerminal(result=(current_result, *sibling_results))
            return _TransformTerminal(result=current_result)

        # Routed to error sink — emit ROUTED_ON_ERROR (DIVERT semantics).
        # NOTE: Do NOT record the outcome here - the token hasn't been written yet.
        # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
        #
        # Offensive: refuse to fabricate Tier-1 audit data. If the upstream
        # transform did not provide a reason, that is a producer bug; crashing
        # here is correct because emitting `FailureInfo.message="unknown_error"`
        # would create a deterministic error_hash collision across unrelated
        # falsy-error failures and falsify the audit trail.
        if not transform_result.reason:
            raise OrchestrationInvariantError(
                "ROUTED_ON_ERROR requires transform_result.reason; refusing to "
                "fabricate FailureInfo.message='unknown_error' for audit hashing"
            )
        error_detail = str(transform_result.reason)

        sibling_results = self._processor._notify_coalesce_of_lost_branch(
            current_token,
            f"error_routed:{error_detail}",
            child_items,
        )
        # Capture the originating transform error so the audit trail records both
        # sink_name and error_hash on the ROUTED_ON_ERROR outcome (mirror of DIVERTED's
        # contract). The accumulator converts FailureInfo.message -> error_hash before
        # the pending-sink record is handed to SinkExecutor for durable recording.
        failure = FailureInfo(exception_type="TransformError", message=error_detail)
        current_result = RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.ON_ERROR_ROUTED,
            sink_name=error_sink,
            error=failure,
        )
        if sibling_results:
            return _TransformTerminal(result=(current_result, *sibling_results))
        return _TransformTerminal(result=current_result)

    def handle_gate_node(
        self,
        gate: GateSettings,
        current_token: TokenInfo,
        ctx: PluginContext,
        node_id: NodeID,
        child_items: list[WorkItem],
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
        current_on_success_sink: str,
    ) -> _GateOutcome:
        """Handle a gate node: evaluate, then fork/route/divert/continue.

        Args:
            gate: Gate configuration to evaluate.
            current_token: Token being processed through the DAG.
            ctx: Plugin context for the current run.
            node_id: Current DAG node ID (passed to gate executor and used for
                fork child work item creation).
            child_items: Mutable list — fork paths append child work items here.
            coalesce_node_id: Coalesce barrier node for fork branches (or None).
            coalesce_name: Coalesce point name for fork branches (or None).
            current_on_success_sink: Current sink name, carried forward or overridden by jumps.

        Returns:
            _GateTerminal: Gate routed to sink, forked to paths, or diverted (contains result + child_items populated).
            _GateContinue: Gate says continue — updated_token, updated_sink, and optional next_node_id for jumps.
        """
        # 1. Execute gate
        outcome = self._processor._gate_executor.execute_config_gate(
            gate_config=gate,
            node_id=node_id,
            token=current_token,
            ctx=ctx,
            token_manager=self._processor._token_manager,
        )
        current_token = outcome.updated_token

        # 2. Emit GateEvaluated telemetry AFTER Landscape recording succeeds
        # (Landscape recording happens inside execute_config_gate)
        self._processor._emit_gate_evaluated(
            token=current_token,
            gate_name=gate.name,
            gate_node_id=node_id,
            routing_mode=outcome.result.action.mode,
            destinations=self._processor._get_gate_destinations(outcome),
        )

        # 3. Check if gate routed to a sink
        if outcome.sink_name is not None:
            # NOTE: Do NOT record ROUTED outcome here - the token hasn't been written yet.
            # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
            # Notify coalesce if this is a forked branch
            sibling_results = self._processor._notify_coalesce_of_lost_branch(
                current_token,
                f"gate_routed_to_sink:{outcome.sink_name}",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
                sink_name=outcome.sink_name,
            )
            if sibling_results:
                return _GateTerminal(result=(current_result, *sibling_results))
            return _GateTerminal(result=current_result)

        if outcome.discarded:
            self._processor._record_gate_discarded_outcome(
                token=current_token,
                gate_name=gate.name,
                node_id=node_id,
            )
            with best_effort(
                "TokenCompleted telemetry after gate discard audit",
                run_id=self._processor._run_id,
                token_id=current_token.token_id,
                gate_node_id=node_id,
                gate_name=gate.name,
            ):
                self._processor._emit_token_completed(
                    current_token,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.GATE_DISCARDED,
                )
            sibling_results = self._processor._notify_coalesce_of_lost_branch(
                current_token,
                "gate_discarded",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_DISCARDED,
            )
            if sibling_results:
                return _GateTerminal(result=(current_result, *sibling_results))
            return _GateTerminal(result=current_result)

        # 4. Fork to paths
        if outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
            return self.handle_gate_fork(outcome, current_token, node_id, child_items)

        # 5. Jump to specific node
        if outcome.next_node_id is not None:
            # Validate jump target exists in the DAG (our data — crash on invariant violation).
            # Without this check, a nonexistent target silently passes the coalesce ordering
            # check below (both .get() calls return None → condition is False) and only fails
            # one iteration later with a less informative error from resolve_plugin_for_node().
            if outcome.next_node_id not in self._processor._node_step_map:
                raise OrchestrationInvariantError(
                    f"Gate at node '{node_id}' jumped token '{current_token.token_id}' to "
                    f"node '{outcome.next_node_id}' which is not in the DAG step map. "
                    f"Known nodes: {sorted(self._processor._node_step_map.keys())}"
                )

            updated_sink = current_on_success_sink
            resolved_sink = self._processor._nav.resolve_jump_target_sink(outcome.next_node_id)
            if resolved_sink is not None:
                updated_sink = resolved_sink

            # Re-validate coalesce ordering invariant after gate jump.
            # The initial check at entry only validates the starting node.
            # A gate jump can move the token past its coalesce node,
            # which would silently bypass join handling.
            #
            # IMPORTANT: Use outcome.next_node_id (not the caller's node_id param)
            # because we're validating the JUMP TARGET, not the current position.
            if coalesce_node_id is not None:
                jump_target_step = self._processor._node_step_map[outcome.next_node_id]
                coalesce_barrier_step = self._processor._node_step_map[coalesce_node_id]
                if jump_target_step > coalesce_barrier_step:
                    raise OrchestrationInvariantError(
                        f"Gate jump moved token '{current_token.token_id}' to node '{outcome.next_node_id}' "
                        f"(step {jump_target_step}) which is past its coalesce node '{coalesce_node_id}' "
                        f"(step {coalesce_barrier_step}). This would bypass join handling."
                    )

            return _GateContinue(
                updated_token=current_token,
                updated_sink=updated_sink,
                next_node_id=outcome.next_node_id,
            )

        # 6. CONTINUE: config gate says "proceed to next structural node."
        if outcome.result.action.kind != RoutingKind.CONTINUE:
            raise OrchestrationInvariantError(
                f"Unhandled config gate routing kind {outcome.result.action.kind!r} "
                f"for token {current_token.token_id} at node '{node_id}'. "
                f"Expected CONTINUE when no sink_name, fork, or next_node_id is set."
            )
        return _GateContinue(updated_token=current_token, updated_sink=current_on_success_sink)

    def handle_gate_fork(
        self,
        outcome: GateOutcome,
        current_token: TokenInfo,
        node_id: NodeID,
        child_items: list[WorkItem],
    ) -> _GateTerminal:
        """Handle fork-to-paths routing: build child work items for each fork branch.

        Iterates child tokens from the gate outcome, resolves coalesce info for each
        branch, and appends continuation or terminal work items to child_items.

        Args:
            outcome: Config gate outcome containing child tokens and routing info.
            current_token: Parent token being forked.
            node_id: Current gate node ID for continuation work items.
            child_items: Mutable list — fork paths append child work items here.

        Returns:
            _GateTerminal with FORKED outcome for the parent token.
        """
        for child_token in outcome.child_tokens:
            # Look up coalesce info for this branch
            cfg_branch_name = child_token.branch_name
            cfg_coalesce_name: CoalesceName | None = None

            if cfg_branch_name and BranchName(cfg_branch_name) in self._processor._branch_to_coalesce:
                cfg_coalesce_name = self._processor._branch_to_coalesce[BranchName(cfg_branch_name)]

            # See config gate fork handler above for routing logic.
            # Children born during a re-drive get fresh token_ids with no prior node_states,
            # so they use the default resume_attempt_offset=0 / resume_checkpoint_id=None.
            if cfg_coalesce_name is None and cfg_branch_name and BranchName(cfg_branch_name) in self._processor._branch_to_sink:
                child_items.append(
                    self._processor._nav.create_work_item(
                        token=child_token,
                        current_node_id=None,
                    )
                )
            else:
                child_items.append(
                    self._processor._nav.create_continuation_work_item(
                        token=child_token,
                        current_node_id=node_id,
                        coalesce_name=cfg_coalesce_name,
                    )
                )

        # NOTE: Parent FORKED outcome is now recorded atomically in fork_token()
        # to eliminate crash window between child creation and outcome recording.
        return _GateTerminal(
            result=RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.FORK_PARENT,
            )
        )

    def validate_coalesce_ordering(
        self,
        token: TokenInfo,
        current_node_id: NodeID | None,
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
    ) -> None:
        """Validate that tokens with coalesce metadata don't start downstream of their coalesce point.

        A malformed work item starting past the coalesce node would silently skip coalesce handling
        because _maybe_coalesce_token only triggers on exact node equality.

        Raises:
            OrchestrationInvariantError: If the token's starting node is downstream of its coalesce barrier.
        """
        if (
            coalesce_node_id is not None
            and current_node_id is not None
            and coalesce_name is not None
            and current_node_id != coalesce_node_id
            and current_node_id in self._processor._node_step_map
            and coalesce_node_id in self._processor._node_step_map
        ):
            current_step = self._processor._node_step_map[current_node_id]
            coalesce_step = self._processor._node_step_map[coalesce_node_id]
            if current_step > coalesce_step:
                raise OrchestrationInvariantError(
                    f"Token {token.token_id} started at node '{current_node_id}' (step {current_step}), "
                    f"which is downstream of coalesce '{coalesce_name}' (step {coalesce_step}). "
                    f"Work items with coalesce metadata must start at or before the coalesce point."
                )

    def handle_terminal_token(
        self,
        current_token: TokenInfo,
        current_on_success_sink: str,
    ) -> RowResult:
        """Handle a token that has traversed all nodes: resolve final sink, return result.

        Determines the effective sink from:
        1. branch_to_sink mapping (for fork branches routing directly to sinks)
        2. last_on_success_sink (inherited from transforms or source)

        If the token has a branch_name that maps to a direct sink via _branch_to_sink,
        that takes precedence. Otherwise, the accumulated on_success sink is used.

        Raises:
            OrchestrationInvariantError: If no effective sink can be determined (indicates
                a DAG construction or on_success configuration bug).

        Returns:
            RowResult with COMPLETED outcome and resolved sink_name.
        """
        # Determine sink name from explicit routing maps. Fork children
        # targeting direct sinks are resolved via _branch_to_sink (built from
        # DAG COPY edges at construction time). Non-fork tokens use the last
        # transform's on_success or the source's on_success.
        effective_sink = current_on_success_sink
        if current_token.branch_name is not None:
            branch = BranchName(current_token.branch_name)
            if branch in self._processor._branch_to_sink:
                effective_sink = self._processor._branch_to_sink[branch]

        if not effective_sink or not effective_sink.strip():
            raise OrchestrationInvariantError(
                f"No effective sink for token {current_token.token_id}: "
                f"last_on_success_sink={current_on_success_sink!r}, "
                f"branch_name={current_token.branch_name!r}. "
                f"This indicates a DAG construction or on_success configuration bug."
            )

        return RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name=effective_sink,
        )

    def process_single_token(
        self,
        token: TokenInfo,
        ctx: PluginContext,
        current_node_id: NodeID | None,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
        on_success_sink: str | None = None,
        attempt_offset: int = 0,
    ) -> tuple[RowResult | tuple[RowResult, ...] | None, list[WorkItem]]:
        """Process a single token through processing nodes starting at node_id.

        Args:
            token: Token to process; token.resume_attempt_offset and
                token.resume_checkpoint_id carry the resume state for this token
                and propagate automatically through all node_state writes.
            ctx: Plugin context
            current_node_id: Node ID to start processing from. None is valid only
                for terminal work items that already have explicit sink context
                (inherited on_success_sink or branch_to_sink mapping).
            coalesce_node_id: Node ID at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging
            on_success_sink: Inherited sink from parent (e.g. terminal deagg parent's on_success)
            attempt_offset: Starting audit attempt offset for lease-recovered work

        Returns:
            Tuple of (RowResult or list of RowResults or None if held for coalesce,
                      list of child WorkItems to queue)
            - Single RowResult for most operations
            - List of RowResults for passthrough aggregation mode
            - None for held coalesce tokens
        """
        current_token = token
        # MUTATION CONTRACT: child_items is passed by reference to _handle_transform_node(),
        # _handle_gate_node(), _notify_coalesce_of_lost_branch(), and _maybe_coalesce_token().
        # These methods append child WorkItems (fork paths, deaggregation, coalesce merges)
        # directly into this list. The caller returns child_items alongside the RowResult.
        # Do NOT replace with return-value-based patterns without updating all call sites.
        child_items: list[WorkItem] = []

        # current_node_id=None skips traversal loop entirely, so only allow it
        # when sink routing is explicit (inherited sink or branch->sink map).
        if current_node_id is None:
            has_branch_sink = (
                current_token.branch_name is not None and BranchName(current_token.branch_name) in self._processor._branch_to_sink
            )
            if on_success_sink is None and not has_branch_sink:
                raise OrchestrationInvariantError(
                    f"Token {token.token_id} has current_node_id=None without explicit terminal sink context. "
                    "Expected inherited on_success_sink or branch_to_sink mapping."
                )

        last_on_success_sink: str = on_success_sink if on_success_sink is not None else self._processor._source_on_success
        if coalesce_name is not None and current_node_id is not None:
            coalesce_node_id_for_name = self._processor._coalesce_node_ids[coalesce_name]
            if coalesce_node_id_for_name == current_node_id and self._processor._nav.resolve_next_node(current_node_id) is None:
                last_on_success_sink = self._processor._nav.resolve_coalesce_sink(
                    coalesce_name,
                    context=f"start of token processing for token '{token.token_id}'",
                )

        self.validate_coalesce_ordering(token, current_node_id, coalesce_node_id, coalesce_name)

        node_id: NodeID | None = current_node_id
        max_inner_iterations = len(self._processor._node_to_next) + 1
        inner_iterations = 0
        while node_id is not None:
            inner_iterations += 1
            if inner_iterations > max_inner_iterations:
                raise OrchestrationInvariantError(
                    f"Inner traversal exceeded {max_inner_iterations} iterations for token "
                    f"{token.token_id}. Possible cycle in node_to_next map."
                )
            # Refresh active scheduler lease (filigree elspeth-ddde8144b6).
            # No-op when no claim is active. Raises SchedulerLeaseLostError
            # when the lease was reaped by a peer — propagates up to
            # ``_drain_scheduler_claims`` which catches it specifically and
            # abandons this iteration cleanly.
            self._processor._heartbeat_active_claim()
            handled, result = self._processor._maybe_coalesce_token(
                current_token,
                current_node_id=node_id,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
                child_items=child_items,
            )
            if handled:
                return (result, child_items)

            next_node_id = self._processor._nav.resolve_next_node(node_id)
            plugin = self._processor._nav.resolve_plugin_for_node(node_id)
            if plugin is None:
                # Non-processing structural nodes (e.g. coalesce) are traversed but not executed.
                node_id = next_node_id
                continue

            # Type-safe plugin detection using protocols
            if isinstance(plugin, TransformProtocol):
                row_transform = plugin
                # Check if this is a batch-aware transform at an aggregation node
                transform_node_id = row_transform.node_id
                if (
                    row_transform.is_batch_aware
                    and transform_node_id is not None
                    and transform_node_id in self._processor._aggregation_settings
                ):
                    # Use engine buffering for aggregation
                    return self._processor._process_batch_aggregation_node(
                        transform=row_transform,
                        current_token=current_token,
                        ctx=ctx,
                        child_items=child_items,
                        coalesce_node_id=coalesce_node_id,
                        coalesce_name=coalesce_name,
                    )

                # ADR-030 §B (slice 5, follower aggregation barrier hand-off):
                # a follower has no AggregationSettings (trigger evaluation is
                # leader-only per §B.2).  If this batch-aware transform sits at
                # a known aggregation node, the follower must NOT execute it
                # row-wise — doing so produces wrong aggregate output and
                # bypasses the leader's barrier.  Return (None, []) so that
                # _drain_scheduler_claims hits the ``result is None and not
                # child_items`` arm (line 4241) and calls mark_blocked with the
                # aggregation barrier key.  The leader's next journal-intake
                # adopts the arrival and runs trigger evaluation.
                if (
                    row_transform.is_batch_aware
                    and transform_node_id is not None
                    and transform_node_id in self._processor._follower_barrier_node_ids
                ):
                    logger.debug(
                        "follower: aggregation barrier hold for token %r at node %r — marking blocked; leader adopts via journal-intake",
                        current_token.token_id,
                        transform_node_id,
                    )
                    return None, child_items

                # NOTE: child_items is mutated inside (deagg appends, coalesce notifications).
                transform_outcome = self.handle_transform_node(
                    row_transform,
                    current_token,
                    ctx,
                    node_id,
                    child_items,
                    coalesce_node_id,
                    coalesce_name,
                    last_on_success_sink,
                    attempt_offset,
                )
                if isinstance(transform_outcome, _TransformTerminal):
                    return transform_outcome.result, child_items
                current_token = transform_outcome.updated_token
                last_on_success_sink = transform_outcome.updated_sink
            elif isinstance(plugin, GateSettings):
                # NOTE: child_items is mutated inside (fork paths, coalesce notifications).
                gate_outcome = self.handle_gate_node(
                    plugin,
                    current_token,
                    ctx,
                    node_id,
                    child_items,
                    coalesce_node_id,
                    coalesce_name,
                    last_on_success_sink,
                )
                if isinstance(gate_outcome, _GateTerminal):
                    return gate_outcome.result, child_items
                current_token = gate_outcome.updated_token
                last_on_success_sink = gate_outcome.updated_sink
                if gate_outcome.next_node_id is not None:
                    node_id = gate_outcome.next_node_id
                    continue

            else:
                raise TypeError(f"Unknown transform type: {type(plugin).__name__}. Expected TransformProtocol or GateSettings.")

            node_id = next_node_id

        result = self.handle_terminal_token(current_token, last_on_success_sink)
        return result, child_items
