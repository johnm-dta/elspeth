# src/elspeth/engine/executors.py
"""Plugin executors that wrap plugin calls with audit recording.

Each executor handles a specific plugin type:
- TransformExecutor: Row transforms
- GateExecutor: Routing gates (Task 14)
- AggregationExecutor: Stateful aggregations (Task 15)
- SinkExecutor: Output sinks (Task 16)
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import (
    Artifact,
    BatchPendingError,
    ConfigGateReason,
    ExecutionError,
    NodeStateOpen,
    PendingOutcome,
    RoutingAction,
    RoutingSpec,
    TokenInfo,
)
from elspeth.contracts.enums import (
    BatchStatus,
    NodeStateStatus,
    RoutingKind,
    RoutingMode,
    TriggerType,
)
from elspeth.contracts.errors import OrchestrationInvariantError, PluginContractViolation
from elspeth.contracts.types import NodeID
from elspeth.core.canonical import stable_hash
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.core.operations import track_operation
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.expression_parser import ExpressionParser
from elspeth.engine.spans import SpanFactory
from elspeth.engine.triggers import TriggerEvaluator
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import (
    GateProtocol,
    SinkProtocol,
    TransformProtocol,
)
from elspeth.plugins.results import (
    GateResult,
    TransformResult,
)

if TYPE_CHECKING:
    from elspeth.engine.batch_adapter import SharedBatchAdapter
    from elspeth.engine.clock import Clock
    from elspeth.engine.tokens import TokenManager

__all__ = [
    "AggregationExecutor",
    "GateExecutor",
    "GateOutcome",
    "MissingEdgeError",
    "SinkExecutor",
    "TokenInfo",  # Re-exported from contracts for convenience
    "TransformExecutor",
    "TriggerType",  # Re-exported from contracts.enums for convenience
]

logger = logging.getLogger(__name__)


class MissingEdgeError(Exception):
    """Raised when routing refers to an unregistered edge.

    This is an audit integrity error - every routing decision must be
    traceable to a registered edge. Silent edge loss is unacceptable.
    """

    def __init__(self, node_id: NodeID, label: str) -> None:
        """Initialize with routing details.

        Args:
            node_id: Node that attempted routing
            label: Edge label that was not found
        """
        self.node_id = node_id
        self.label = label
        super().__init__(
            f"No edge registered from node {node_id} with label '{label}'. Audit trail would be incomplete - refusing to proceed."
        )


@dataclass
class GateOutcome:
    """Result of gate execution with routing information.

    Contains the gate result plus information about how the token
    should be routed and any child tokens created.
    """

    result: GateResult
    updated_token: TokenInfo
    child_tokens: list[TokenInfo] = field(default_factory=list)
    sink_name: str | None = None


class TransformExecutor:
    """Executes transforms with audit recording.

    Wraps transform.process() to:
    1. Record node state start
    2. Time the operation
    3. Populate audit fields in result
    4. Record node state completion
    5. Emit OpenTelemetry span

    Example:
        executor = TransformExecutor(recorder, span_factory)
        result, updated_token, error_sink = executor.execute_transform(
            transform=my_transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        max_workers: int | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            max_workers: Maximum concurrent workers (None = no limit)
        """
        self._recorder = recorder
        self._spans = span_factory
        self._max_workers = max_workers

    def _get_batch_adapter(self, transform: TransformProtocol) -> "SharedBatchAdapter":
        """Get or create shared batch adapter for transform.

        Creates adapter once per transform instance and stores it as an
        instance attribute for reuse across rows. This solves the deadlock
        where per-row adapters were created but only the first was connected.

        Args:
            transform: The batch-aware transform

        Returns:
            SharedBatchAdapter for this transform
        """
        from elspeth.engine.batch_adapter import SharedBatchAdapter

        if not hasattr(transform, "_executor_batch_adapter"):
            adapter = SharedBatchAdapter()
            transform._executor_batch_adapter = adapter  # type: ignore[attr-defined]

            # Connect output (one-time setup)
            # Use _pool_size stored by LLM transforms, default to 30
            # Cap to max_workers if configured (enforces global concurrency limit)
            max_pending = getattr(transform, "_pool_size", 30)
            if self._max_workers is not None:
                max_pending = min(max_pending, self._max_workers)
            transform.connect_output(output=adapter, max_pending=max_pending)  # type: ignore[attr-defined]
            transform._batch_initialized = True  # type: ignore[attr-defined]

        return transform._executor_batch_adapter  # type: ignore[attr-defined, return-value, no-any-return]

    def execute_transform(
        self,
        transform: TransformProtocol,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
        attempt: int = 0,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Execute a transform with full audit recording and error routing.

        This method handles a SINGLE ATTEMPT. Retry logic is the caller's
        responsibility (e.g., RetryManager wraps this for retryable transforms).
        Each attempt gets its own node_state record with attempt number tracked
        by the caller.

        Supports two execution modes:
        1. Synchronous: transform.process() returns TransformResult immediately
        2. Asynchronous (BatchTransformMixin): transform.accept() submits work,
           results flow through output port and are awaited synchronously

        Error Routing:
        - TransformResult.error() is a LEGITIMATE processing failure
        - Routes to configured sink via transform._on_error
        - RuntimeError if transform errors without on_error config
        - Exceptions are BUGS and propagate (not routed)

        Args:
            transform: Transform plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            attempt: Attempt number for retry tracking (0-indexed, default 0)

        Returns:
            Tuple of (TransformResult with audit fields, updated TokenInfo, error_sink)
            where error_sink is:
            - None if transform succeeded
            - "discard" if transform errored and _on_error == "discard"
            - The sink name if transform errored and _on_error is a sink name

        Raises:
            Exception: Re-raised from transform.process() after recording failure
            RuntimeError: Transform returned error but has no on_error configured
        """
        if transform.node_id is None:
            raise OrchestrationInvariantError(f"Transform '{transform.name}' executed without node_id - orchestrator bug")
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform.node_id,
            run_id=ctx.run_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
            attempt=attempt,
        )

        # Set state_id and node_id on context for external call recording
        # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
        ctx.state_id = state.state_id
        ctx.node_id = transform.node_id
        # Note: call_index allocation is handled by LandscapeRecorder.allocate_call_index()
        # which automatically starts at 0 for each new state_id

        # Detect batch transforms (those using BatchTransformMixin)
        # They have accept() method and process() raises NotImplementedError
        has_accept = hasattr(transform, "accept") and callable(getattr(transform, "accept", None))

        # Execute with timing and span
        # P2-2026-01-21: Pass token_id for accurate child token attribution in traces
        # P2-2026-01-21: Pass node_id for disambiguation when multiple plugin instances exist
        with self._spans.transform_span(
            transform.name,
            node_id=transform.node_id,
            input_hash=input_hash,
            token_id=token.token_id,
        ):
            start = time.perf_counter()
            try:
                if has_accept:
                    # Batch transform: use accept() with SharedBatchAdapter
                    # One adapter per transform, multiple waiters per adapter
                    adapter = self._get_batch_adapter(transform)

                    # Register waiter for THIS token AND attempt (before accept!)
                    # Using (token_id, state_id) ensures retry safety: if a timeout
                    # occurs and retry happens, the new attempt's waiter won't receive
                    # stale results from the previous attempt.
                    waiter = adapter.register(token.token_id, state.state_id)

                    # Set token on context for BatchTransformMixin
                    ctx.token = token

                    # Submit work - this returns immediately
                    transform.accept(token.row_data, ctx)  # type: ignore[attr-defined]

                    # Block until THIS row's result arrives.
                    #
                    # DESIGN DECISION: Sequential row processing
                    # The orchestrator processes rows one at a time, blocking here
                    # until each row completes. This is intentional:
                    # - Concurrency happens WITHIN each row (multi-query transforms
                    #   make 10+ LLM calls concurrently for a single row)
                    # - Across rows, processing is sequential for:
                    #   1. Simpler audit ordering (deterministic state progression)
                    #   2. Natural backpressure (no unbounded queue growth)
                    #   3. Single-threaded orchestrator (easier to reason about)
                    #
                    # For true cross-row parallelism, the orchestrator would need
                    # to be async/await or multi-threaded, which adds complexity.
                    #
                    # Timeout is derived from transform's batch_wait_timeout config
                    # (default 3600s = 1 hour) to allow for sustained rate limiting
                    # and AIMD backoff during capacity errors.
                    wait_timeout = getattr(transform, "_batch_wait_timeout", 3600.0)
                    result = waiter.wait(timeout=wait_timeout)
                else:
                    # Regular transform: synchronous process()
                    result = transform.process(token.row_data, ctx)

                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                # Record failure
                error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error,
                )

                # For TimeoutError on batch transforms, evict the buffer entry
                # to prevent FIFO blocking on retry attempts.
                #
                # The eviction flow:
                # 1. First attempt times out at waiter.wait()
                # 2. We call evict_submission() to remove buffer entry
                # 3. Retry attempt gets new sequence number and can proceed
                # 4. Original worker may still complete, but result is discarded
                if isinstance(e, TimeoutError) and has_accept:
                    # has_accept guarantees transform has evict_submission (batch protocol)
                    evict_fn = transform.evict_submission  # type: ignore[attr-defined]
                    if not callable(evict_fn):
                        raise TypeError(
                            f"Transform '{transform.name}' evict_submission must be callable, got {type(evict_fn).__name__}"
                        ) from None
                    try:
                        evict_fn(token.token_id, state.state_id)
                    except Exception as evict_err:
                        raise RuntimeError(f"Failed to evict timed-out submission for token {token.token_id}") from evict_err

                raise

        # Populate audit fields
        # Wrap stable_hash calls to convert canonicalization errors to PluginContractViolation.
        # stable_hash calls canonical_json which rejects NaN, Infinity, non-serializable types.
        # Per CLAUDE.md: plugin bugs must crash with clear error messages.
        result.input_hash = input_hash
        try:
            if result.row is not None:
                result.output_hash = stable_hash(result.row)
            elif result.rows is not None:
                result.output_hash = stable_hash(result.rows)
            else:
                result.output_hash = None
        except (TypeError, ValueError) as e:
            raise PluginContractViolation(
                f"Transform '{transform.name}' emitted non-canonical data: {e}. "
                f"Ensure output contains only JSON-serializable types. "
                f"Use None instead of NaN for missing values."
            ) from e
        result.duration_ms = duration_ms

        # Initialize error_sink - will be set if transform errors with on_error configured
        error_sink: str | None = None

        # Complete node state
        if result.status == "success":
            # TransformResult.success() or success_multi() always sets output data
            if not result.has_output_data:
                raise RuntimeError(f"Transform '{transform.name}' returned success but has no output data")

            # For single-row: output_data is the row
            # For multi-row: output_data is the rows list (engine handles expansion)
            output_data: dict[str, Any] | list[dict[str, Any]]
            if result.row is not None:
                output_data = result.row
            else:
                # result.rows is guaranteed non-None by has_output_data check above
                output_data = result.rows  # type: ignore[assignment]

            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                success_reason=result.success_reason,
                context_after=result.context_after,
            )
            # Update token with new row data, preserving all lineage metadata
            # For multi-row results, keep original row_data (engine will expand tokens later)
            new_data = result.row if result.row is not None else token.row_data
            updated_token = token.with_updated_data(new_data)
        else:
            # Transform returned error status (not exception)
            # This is a LEGITIMATE processing failure, not a bug
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=result.reason,
            )

            # Handle error routing - _on_error is part of TransformProtocol
            on_error = transform._on_error

            if on_error is None:
                raise RuntimeError(
                    f"Transform '{transform.name}' returned error but has no on_error "
                    f"configured. Either configure on_error or fix the transform to not "
                    f"return errors for this input. Error: {result.reason}"
                )

            # Set error_sink so caller knows where the error was routed
            error_sink = on_error

            # Record error event (always, even for discard - audit completeness)
            # Use node_id (unique DAG identifier), not name (plugin type)
            # Bug fix: P2-2026-01-19-transform-errors-ambiguous-transform-id
            #
            # result.reason MUST be set for error results - TransformResult.error() requires it.
            # If None, that's a bug in the transform (constructed error result without reason).
            assert result.reason is not None, (
                f"Transform '{transform.name}' returned error but reason is None. "
                'Use TransformResult.error({{"reason": "...", ...}}) to create error results.'
            )
            ctx.record_transform_error(
                token_id=token.token_id,
                transform_id=transform.node_id,
                row=token.row_data,
                error_details=result.reason,
                destination=on_error,
            )

            # Route to sink if not discarding
            if on_error != "discard":
                ctx.route_to_sink(
                    sink_name=on_error,
                    row=token.row_data,
                    metadata={"transform_error": result.reason},
                )

            updated_token = token

        return result, updated_token, error_sink


class GateExecutor:
    """Executes gates with audit recording and routing.

    Wraps gate.evaluate() to:
    1. Record node state start
    2. Time the operation
    3. Populate audit fields in result
    4. Record routing events
    5. Create child tokens for fork operations
    6. Record node state completion
    7. Emit OpenTelemetry span

    CRITICAL: Status is always "completed" for successful gate execution.
    Terminal state (ROUTED, FORKED) is DERIVED from routing_events/token_parents,
    NOT stored in node_states.status.

    Example:
        executor = GateExecutor(recorder, span_factory, edge_map)
        outcome = executor.execute_gate(
            gate=my_gate,
            token=token,
            ctx=ctx,
            step_in_pipeline=2,
            token_manager=manager,  # Required for fork_to_paths
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        edge_map: dict[tuple[NodeID, str], str] | None = None,
        route_resolution_map: dict[tuple[NodeID, str], str] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            edge_map: Maps (node_id, label) -> edge_id for routing
            route_resolution_map: Maps (node_id, label) -> "continue" | sink_name
        """
        self._recorder = recorder
        self._spans = span_factory
        self._edge_map = edge_map or {}
        self._route_resolution_map = route_resolution_map or {}

    def execute_gate(
        self,
        gate: GateProtocol,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
        token_manager: "TokenManager | None" = None,
    ) -> GateOutcome:
        """Execute a gate with full audit recording.

        Args:
            gate: Gate plugin to execute
            token: Current token with row data
            ctx: Plugin context (includes run_id for atomic fork outcome recording)
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            token_manager: TokenManager for fork operations (required for fork_to_paths)

        Returns:
            GateOutcome with result, updated token, and routing info

        Raises:
            MissingEdgeError: If routing refers to an unregistered edge
            Exception: Re-raised from gate.evaluate() after recording failure
        """
        if gate.node_id is None:
            raise OrchestrationInvariantError(f"Gate '{gate.name}' executed without node_id - orchestrator bug")
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            run_id=ctx.run_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        # BUG-RECORDER-01 fix: Set state_id on context for external call recording
        # Gates may need to make external calls (e.g., LLM API for routing decisions)
        ctx.state_id = state.state_id
        ctx.node_id = gate.node_id
        # Note: call_index allocation handled by LandscapeRecorder.allocate_call_index()

        # Execute with timing and span
        # P2-2026-01-21: Pass token_id for accurate child token attribution in traces
        # P2-2026-01-21: Pass node_id for disambiguation when multiple plugin instances exist
        with self._spans.gate_span(
            gate.name,
            node_id=gate.node_id,
            input_hash=input_hash,
            token_id=token.token_id,
        ):
            start = time.perf_counter()
            try:
                result = gate.evaluate(token.row_data, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                # Record failure
                error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error,
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row)
        result.duration_ms = duration_ms

        # Process routing based on action kind
        action = result.action
        child_tokens: list[TokenInfo] = []
        sink_name: str | None = None

        try:
            if action.kind == RoutingKind.CONTINUE:
                # Record explicit continue routing for audit completeness (AUD-002)
                # Preserve gate's reason and mode for full auditability
                self._record_routing(
                    state_id=state.state_id,
                    node_id=gate.node_id,
                    action=RoutingAction.route("continue", mode=action.mode, reason=action.reason),
                )

            elif action.kind == RoutingKind.ROUTE:
                # Gate returned a route label - resolve via routes config
                route_label = action.destinations[0]
                destination = self._route_resolution_map.get((NodeID(gate.node_id), route_label))

                if destination is None:
                    # Label not in routes config - this is a configuration error
                    raise MissingEdgeError(node_id=NodeID(gate.node_id), label=route_label)

                if destination == "continue":
                    # Route label resolves to "continue" - record routing event (AUD-002)
                    # Preserve gate's reason and mode for full auditability
                    self._record_routing(
                        state_id=state.state_id,
                        node_id=gate.node_id,
                        action=RoutingAction.route("continue", mode=action.mode, reason=action.reason),
                    )
                else:
                    # Route label resolves to a sink name
                    sink_name = destination
                    # Record routing event using the route label to find the edge
                    self._record_routing(
                        state_id=state.state_id,
                        node_id=gate.node_id,
                        action=action,
                    )

            elif action.kind == RoutingKind.FORK_TO_PATHS:
                if token_manager is None:
                    raise RuntimeError(
                        f"Gate {gate.node_id} returned fork_to_paths but no TokenManager provided. "
                        "Cannot create child tokens - audit integrity would be compromised."
                    )
                # Record routing events for all paths
                self._record_routing(
                    state_id=state.state_id,
                    node_id=gate.node_id,
                    action=action,
                )
                # Create child tokens (ATOMIC: also records parent FORKED outcome)
                child_tokens, _fork_group_id = token_manager.fork_token(
                    parent_token=token,
                    branches=list(action.destinations),
                    step_in_pipeline=step_in_pipeline,
                    run_id=ctx.run_id,
                    row_data=result.row,
                )

        except (MissingEdgeError, RuntimeError) as e:
            # Record failure before re-raising - ensures node_state is never left OPEN
            routing_error: ExecutionError = {
                "exception": str(e),
                "type": type(e).__name__,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=routing_error,
            )
            raise

        # Complete node state - always "completed" for successful execution
        # Terminal state is DERIVED from routing_events, not stored here
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data=result.row,
            duration_ms=duration_ms,
        )

        # Update token with new row data, preserving all lineage metadata
        updated_token = token.with_updated_data(result.row)

        return GateOutcome(
            result=result,
            updated_token=updated_token,
            child_tokens=child_tokens,
            sink_name=sink_name,
        )

    def execute_config_gate(
        self,
        gate_config: GateSettings,
        node_id: str,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
        token_manager: "TokenManager | None" = None,
    ) -> GateOutcome:
        """Execute a config-driven gate using ExpressionParser.

        Unlike execute_gate() which uses a GateProtocol plugin,
        this method evaluates the gate condition directly using
        the expression parser. The condition expression is evaluated
        against the token's row_data.

        Route Resolution:
        - If condition returns a string, it's used as the route label directly
        - If condition returns a boolean, it's converted to "true"/"false" label
        - The label is then looked up in gate_config.routes to get the destination

        Args:
            gate_config: Gate configuration with condition and routes
            node_id: Node ID assigned by orchestrator
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            token_manager: TokenManager for fork operations (required for fork destinations)

        Returns:
            GateOutcome with result, updated token, and routing info

        Raises:
            MissingEdgeError: If routing refers to an unregistered edge
            ValueError: If condition result doesn't match any route label
            RuntimeError: If fork destination without token_manager
        """
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node_id,
            run_id=ctx.run_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        # Create parser and evaluate condition
        # P2-2026-01-21: Pass token_id for accurate child token attribution in traces
        # P2-2026-01-21: Pass node_id for disambiguation when multiple config gates exist
        with self._spans.gate_span(
            gate_config.name,
            node_id=node_id,
            input_hash=input_hash,
            token_id=token.token_id,
        ):
            start = time.perf_counter()
            try:
                parser = ExpressionParser(gate_config.condition)
                eval_result = parser.evaluate(token.row_data)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                # Record failure
                error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error,
                )
                raise

        # Convert evaluation result to route label
        if isinstance(eval_result, bool):
            route_label = "true" if eval_result else "false"
        elif isinstance(eval_result, str):
            route_label = eval_result
        else:
            # Unexpected result type - convert to string
            route_label = str(eval_result)

        # Look up destination in routes config
        if route_label not in gate_config.routes:
            # Record failure before raising
            error = {
                "exception": f"Route label '{route_label}' not found in routes config",
                "type": "ValueError",
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=error,
            )
            raise ValueError(
                f"Gate '{gate_config.name}' condition returned '{route_label}' which is not in routes: {list(gate_config.routes.keys())}"
            )

        destination = gate_config.routes[route_label]

        # Build routing action and process based on destination
        child_tokens: list[TokenInfo] = []
        sink_name: str | None = None
        reason: ConfigGateReason = {"condition": gate_config.condition, "result": route_label}

        try:
            if destination == "continue":
                # Continue to next node - record routing event (AUD-002)
                # Use CONTINUE kind for GateResult, ROUTE for recording (matches edge label)
                action = RoutingAction.continue_(reason=reason)
                self._record_routing(
                    state_id=state.state_id,
                    node_id=node_id,
                    action=RoutingAction.route("continue", mode=RoutingMode.MOVE, reason=reason),
                )

            elif destination == "fork":
                # Fork to multiple paths - fork_to guaranteed by GateSettings.validate_fork_consistency()
                # (Pydantic validation ensures fork_to is not None when routes include "fork")
                # Local binding for type narrowing (mypy can't see Pydantic validation)
                fork_branches: list[str] = gate_config.fork_to  # type: ignore[assignment]

                if token_manager is None:
                    raise RuntimeError(
                        f"Gate {node_id} routes to fork but no TokenManager provided. "
                        "Cannot create child tokens - audit integrity would be compromised."
                    )

                action = RoutingAction.fork_to_paths(fork_branches, reason=reason)

                # Record routing events for all paths
                self._record_routing(
                    state_id=state.state_id,
                    node_id=node_id,
                    action=action,
                )

                # Create child tokens (ATOMIC: also records parent FORKED outcome)
                child_tokens, _fork_group_id = token_manager.fork_token(
                    parent_token=token,
                    branches=fork_branches,
                    step_in_pipeline=step_in_pipeline,
                    run_id=ctx.run_id,
                    row_data=token.row_data,
                )

            else:
                # Route to a named sink
                sink_name = destination
                action = RoutingAction.route(route_label, mode=RoutingMode.MOVE, reason=reason)

                # Record routing event
                self._record_routing(
                    state_id=state.state_id,
                    node_id=node_id,
                    action=action,
                )

        except (MissingEdgeError, RuntimeError) as e:
            # Record failure before re-raising - ensures node_state is never left OPEN
            routing_error: ExecutionError = {
                "exception": str(e),
                "type": type(e).__name__,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=routing_error,
            )
            raise

        # Create GateResult for audit fields
        result = GateResult(
            row=token.row_data,
            action=action,
        )
        result.input_hash = input_hash
        result.output_hash = stable_hash(token.row_data)
        result.duration_ms = duration_ms

        # Complete node state - always "completed" for successful execution
        # Terminal state is DERIVED from routing_events, not stored here
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data=token.row_data,
            duration_ms=duration_ms,
        )

        # Token row_data is unchanged (config gates don't modify data)
        # Use with_updated_data anyway to preserve all lineage metadata
        updated_token = token.with_updated_data(token.row_data)

        return GateOutcome(
            result=result,
            updated_token=updated_token,
            child_tokens=child_tokens,
            sink_name=sink_name,
        )

    def _record_routing(
        self,
        state_id: str,
        node_id: str,
        action: "RoutingAction",
    ) -> None:
        """Record routing events for a routing action.

        Raises:
            MissingEdgeError: If any destination has no registered edge.
        """
        typed_node_id = NodeID(node_id)
        if len(action.destinations) == 1:
            dest = action.destinations[0]
            edge_id = self._edge_map.get((typed_node_id, dest))
            if edge_id is None:
                raise MissingEdgeError(node_id=typed_node_id, label=dest)

            self._recorder.record_routing_event(
                state_id=state_id,
                edge_id=edge_id,
                mode=action.mode,
                reason=action.reason,
            )
        else:
            # Multiple destinations (fork)
            routes = []
            for dest in action.destinations:
                edge_id = self._edge_map.get((typed_node_id, dest))
                if edge_id is None:
                    raise MissingEdgeError(node_id=typed_node_id, label=dest)
                routes.append(RoutingSpec(edge_id=edge_id, mode=action.mode))

            self._recorder.record_routing_events(
                state_id=state_id,
                routes=routes,
                reason=action.reason,
            )


class AggregationExecutor:
    """Executes aggregations with batch tracking and audit recording.

    Manages the lifecycle of batches:
    1. Create batch on first accept (if _batch_id is None)
    2. Track batch members as rows are accepted
    3. Transition batch through states: draft -> executing -> completed/failed
    4. Reset _batch_id after flush for next batch

    CRITICAL: Terminal state CONSUMED_IN_BATCH is DERIVED from batch_members table,
    NOT stored in node_states.status (which is always "completed" for successful accepts).

    Example:
        executor = AggregationExecutor(recorder, span_factory, run_id)

        # Accept rows into batch
        result = executor.accept(aggregation, token, ctx, step_in_pipeline)
        # Engine uses TriggerEvaluator to decide when to flush (WP-06)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        *,
        aggregation_settings: dict[NodeID, AggregationSettings] | None = None,
        clock: "Clock | None" = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Run identifier for batch creation
            aggregation_settings: Map of node_id -> AggregationSettings for trigger evaluation
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._clock = clock if clock is not None else DEFAULT_CLOCK
        self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals
        self._batch_ids: dict[NodeID, str | None] = {}  # node_id -> current batch_id
        self._aggregation_settings: dict[NodeID, AggregationSettings] = aggregation_settings or {}
        self._trigger_evaluators: dict[NodeID, TriggerEvaluator] = {}
        self._restored_states: dict[NodeID, dict[str, Any]] = {}  # node_id -> state

        # Engine-owned row buffers (node_id -> list of row dicts)
        self._buffers: dict[NodeID, list[dict[str, Any]]] = {}
        # Token tracking for audit trail (node_id -> list of TokenInfo)
        self._buffer_tokens: dict[NodeID, list[TokenInfo]] = {}

        # Create trigger evaluators for each configured aggregation
        for node_id, settings in self._aggregation_settings.items():
            self._trigger_evaluators[node_id] = TriggerEvaluator(settings.trigger, clock=self._clock)
            self._buffers[node_id] = []
            self._buffer_tokens[node_id] = []

    def buffer_row(
        self,
        node_id: NodeID,
        token: TokenInfo,
    ) -> None:
        """Buffer a row for aggregation.

        The engine owns the buffer. When trigger fires, buffered rows
        are passed to a batch-aware Transform.

        Args:
            node_id: Aggregation node ID
            token: Token with row data to buffer
        """
        if node_id not in self._buffers:
            self._buffers[node_id] = []
            self._buffer_tokens[node_id] = []

        # Create batch on first row if needed
        if self._batch_ids.get(node_id) is None:
            batch = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=node_id,
            )
            self._batch_ids[node_id] = batch.batch_id
            self._member_counts[batch.batch_id] = 0

        batch_id = self._batch_ids[node_id]
        assert batch_id is not None  # We just created it if it was None

        # Buffer the row
        self._buffers[node_id].append(token.row_data)
        self._buffer_tokens[node_id].append(token)

        # Record batch membership for audit trail
        ordinal = self._member_counts[batch_id]
        self._recorder.add_batch_member(
            batch_id=batch_id,
            token_id=token.token_id,
            ordinal=ordinal,
        )
        self._member_counts[batch_id] = ordinal + 1

        # Update trigger evaluator
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is not None:
            evaluator.record_accept()

    def get_buffered_rows(self, node_id: NodeID) -> list[dict[str, Any]]:
        """Get currently buffered rows (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts
        """
        return list(self._buffers.get(node_id, []))

    def get_buffered_tokens(self, node_id: NodeID) -> list[TokenInfo]:
        """Get currently buffered tokens (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered TokenInfo objects
        """
        return list(self._buffer_tokens.get(node_id, []))

    def _get_buffered_data(self, node_id: NodeID) -> tuple[list[dict[str, Any]], list[TokenInfo]]:
        """Internal: Get buffered rows and tokens without clearing.

        IMPORTANT: This method does NOT record audit trail. Production code
        should use execute_flush() instead. This method is exposed for:
        - Testing buffer contents without triggering flush

        Args:
            node_id: Aggregation node ID

        Returns:
            Tuple of (buffered_rows, buffered_tokens)
        """
        rows = list(self._buffers.get(node_id, []))
        tokens = list(self._buffer_tokens.get(node_id, []))
        return rows, tokens

    def execute_flush(
        self,
        node_id: NodeID,
        transform: TransformProtocol,
        ctx: PluginContext,
        step_in_pipeline: int,
        trigger_type: TriggerType,
    ) -> tuple[TransformResult, list[TokenInfo], str]:
        """Execute a batch flush with full audit recording.

        This method:
        1. Transitions batch to "executing" with trigger reason
        2. Records node_state for the flush operation
        3. Executes the batch-aware transform
        4. Transitions batch to "completed" or "failed"
        5. Resets batch_id for next batch

        Args:
            node_id: Aggregation node ID
            transform: Batch-aware transform plugin
            ctx: Plugin context
            step_in_pipeline: Current position in DAG
            trigger_type: What triggered the flush (COUNT, TIMEOUT, END_OF_SOURCE, etc.)

        Returns:
            Tuple of (TransformResult with audit fields, list of consumed tokens, batch_id)

        Raises:
            Exception: Re-raised from transform.process() after recording failure
        """
        # Get batch_id - must exist if we're flushing
        batch_id = self._batch_ids.get(node_id)
        if batch_id is None:
            raise RuntimeError(f"No batch exists for node {node_id} - cannot flush")

        # Get buffered data
        buffered_rows = list(self._buffers.get(node_id, []))
        buffered_tokens = list(self._buffer_tokens.get(node_id, []))

        if not buffered_rows:
            raise RuntimeError(f"Cannot flush empty buffer for node {node_id}")

        # Defensive validation: buffer and tokens must be same length
        # This should never happen (checkpoint restore ensures they stay in sync)
        # but crashes explicitly if internal state is corrupted
        if len(buffered_rows) != len(buffered_tokens):
            raise RuntimeError(
                f"Internal state corruption in AggregationExecutor node '{node_id}': "
                f"buffer has {len(buffered_rows)} rows but tokens has {len(buffered_tokens)} entries. "
                f"These must always match. This indicates a bug in checkpoint "
                f"restore or buffer management."
            )

        # Use first token for node_state (represents the batch operation)
        representative_token = buffered_tokens[0]

        # Step 1: Transition batch to "executing"
        self._recorder.update_batch_status(
            batch_id=batch_id,
            status=BatchStatus.EXECUTING,
            trigger_type=trigger_type,
        )

        # Step 2: Begin node state for flush operation
        # Wrap batch rows in a dict for node_state recording
        batch_input: dict[str, Any] = {"batch_rows": buffered_rows}

        # Compute input hash AFTER wrapping (must match what begin_node_state records)
        # See: P2-2026-01-21-aggregation-input-hash-mismatch
        input_hash = stable_hash(batch_input)
        state = self._recorder.begin_node_state(
            token_id=representative_token.token_id,
            node_id=node_id,
            run_id=ctx.run_id,
            step_index=step_in_pipeline,
            input_data=batch_input,
            attempt=0,
        )

        # Set state_id and node_id on context for external call recording
        # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
        ctx.state_id = state.state_id
        ctx.node_id = node_id
        # Note: call_index allocation handled by LandscapeRecorder.allocate_call_index()

        # Step 3: Execute with timing and span
        # P2-2026-01-21: Use aggregation_span (not transform_span) for flush operations
        # This ensures spans are distinguishable from regular transforms and carry batch_id
        # P2-2026-01-21: Pass node_id for disambiguation when multiple aggregations exist
        # P3-2026-02-01: Pass input_hash for trace-to-audit correlation
        batch_token_ids = [t.token_id for t in buffered_tokens]
        with self._spans.aggregation_span(
            transform.name,
            node_id=node_id,
            input_hash=input_hash,
            batch_id=batch_id,
            token_ids=batch_token_ids,
        ):
            start = time.perf_counter()
            try:
                result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]
                duration_ms = (time.perf_counter() - start) * 1000
            except BatchPendingError:
                # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
                # The batch has been submitted but isn't complete yet.
                # Complete node_state with PENDING status and link batch for audit trail, then re-raise.
                duration_ms = (time.perf_counter() - start) * 1000

                # Close node_state with "pending" status - the submission succeeded
                # but the result isn't available yet. This prevents orphaned OPEN states.
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.PENDING,
                    duration_ms=duration_ms,
                )

                # Link batch to the aggregation state for traceability.
                # Keep status as "executing" but set aggregation_state_id.
                self._recorder.update_batch_status(
                    batch_id=batch_id,
                    status=BatchStatus.EXECUTING,
                    state_id=state.state_id,
                )

                # Re-raise for orchestrator to schedule retry.
                # The batch remains in "executing" status, checkpoint is preserved.
                raise
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000

                # Record failure in node_state
                error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error,
                )

                # Transition batch to failed
                self._recorder.complete_batch(
                    batch_id=batch_id,
                    status=BatchStatus.FAILED,
                    trigger_type=trigger_type,
                    state_id=state.state_id,
                )

                # Reset for next batch
                self._reset_batch_state(node_id)
                raise

        # Step 4: Populate audit fields on result
        # Wrap stable_hash calls to convert canonicalization errors to PluginContractViolation.
        # stable_hash calls canonical_json which rejects NaN, Infinity, non-serializable types.
        # Per CLAUDE.md: plugin bugs must crash with clear error messages.
        result.input_hash = input_hash
        try:
            if result.row is not None:
                result.output_hash = stable_hash(result.row)
            elif result.rows is not None:
                result.output_hash = stable_hash(result.rows)
            else:
                result.output_hash = None
        except (TypeError, ValueError) as e:
            raise PluginContractViolation(
                f"Aggregation transform '{transform.name}' emitted non-canonical data: {e}. "
                f"Ensure output contains only JSON-serializable types. "
                f"Use None instead of NaN for missing values."
            ) from e
        result.duration_ms = duration_ms

        # Step 5: Complete node state
        if result.status == "success":
            output_data: dict[str, Any] | list[dict[str, Any]]
            if result.row is not None:
                output_data = result.row
            elif result.rows is not None:
                output_data = result.rows
            else:
                # Contract violation: success status requires output data
                raise RuntimeError(
                    f"Aggregation transform '{transform.name}' returned success status but "
                    f"neither row nor rows contains data. Batch-aware transforms must return "
                    f"output via TransformResult.success(row) or TransformResult.success_multi(rows). "
                    f"This is a plugin bug."
                )

            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                success_reason=result.success_reason,
            )

            # Transition batch to completed
            self._recorder.complete_batch(
                batch_id=batch_id,
                status=BatchStatus.COMPLETED,
                trigger_type=trigger_type,
                state_id=state.state_id,
            )
        else:
            # Transform returned error status
            error_info: ExecutionError = {
                "exception": str(result.reason) if result.reason else "Transform returned error",
                "type": "TransformError",
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=error_info,
            )

            # Transition batch to failed
            self._recorder.complete_batch(
                batch_id=batch_id,
                status=BatchStatus.FAILED,
                trigger_type=trigger_type,
                state_id=state.state_id,
            )

        # Step 6: Save batch_id before reset (needed by caller for CONSUMED_IN_BATCH)
        # Note: batch_id was validated at the start of this method
        flushed_batch_id = batch_id

        # Reset for next batch and clear buffers
        self._reset_batch_state(node_id)
        self._buffers[node_id] = []
        self._buffer_tokens[node_id] = []

        # Reset trigger evaluator for next batch
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is not None:
            evaluator.reset()

        return result, buffered_tokens, flushed_batch_id

    def _reset_batch_state(self, node_id: NodeID) -> None:
        """Reset batch tracking state for next batch.

        Args:
            node_id: Aggregation node ID
        """
        batch_id = self._batch_ids.get(node_id)
        if batch_id is not None:
            del self._batch_ids[node_id]
            if batch_id in self._member_counts:
                del self._member_counts[batch_id]

    def get_buffer_count(self, node_id: NodeID) -> int:
        """Get the number of rows currently buffered for an aggregation.

        Args:
            node_id: Aggregation node ID

        Returns:
            Number of buffered rows, or 0 if no buffer exists
        """
        return len(self._buffers.get(node_id, []))

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state for persistence.

        Stores complete TokenInfo objects (not just IDs) to enable restoration
        without database queries. Validates size to prevent pathological growth.

        Returns:
            dict[str, Any]: Checkpoint state with format:
                {
                    "node_id_1": {
                        "tokens": [
                            {
                                "token_id": str,
                                "row_id": str,
                                "branch_name": str | None,
                                "fork_group_id": str | None,
                                "join_group_id": str | None,
                                "expand_group_id": str | None,
                                "row_data": dict[str, Any]
                            },
                            ...
                        ]
                    },
                    ...
                }

        Raises:
            RuntimeError: If checkpoint exceeds 10MB size limit
        """
        import json
        import logging

        logger = logging.getLogger(__name__)

        # Build checkpoint state from all buffers
        state: dict[str, Any] = {}
        for node_id, tokens in self._buffer_tokens.items():
            if not tokens:  # Only include non-empty buffers
                continue

            # Get trigger state for preservation (Bug #6 + P2-2026-02-01)
            evaluator = self._trigger_evaluators.get(node_id)
            elapsed_age_seconds = evaluator.get_age_seconds() if evaluator is not None else 0.0
            # P2-2026-02-01: Preserve fire time offsets for "first to fire wins" ordering
            count_fire_offset = evaluator.get_count_fire_offset() if evaluator is not None else None
            condition_fire_offset = evaluator.get_condition_fire_offset() if evaluator is not None else None

            if node_id not in self._batch_ids or self._batch_ids[node_id] is None:
                raise RuntimeError(
                    f"AggregationExecutor checkpoint missing batch_id for node {node_id}. "
                    "Buffered tokens exist without an active batch_id - internal state corruption."
                )

            batch_id = self._batch_ids[node_id]

            # Store full TokenInfo as dicts (not just IDs)
            # Include all lineage fields to preserve fork/join/expand metadata
            state[node_id] = {
                "tokens": [
                    {
                        "token_id": t.token_id,
                        "row_id": t.row_id,
                        "branch_name": t.branch_name,
                        "fork_group_id": t.fork_group_id,
                        "join_group_id": t.join_group_id,
                        "expand_group_id": t.expand_group_id,
                        "row_data": t.row_data,
                    }
                    for t in tokens
                ],
                "batch_id": batch_id,
                "elapsed_age_seconds": elapsed_age_seconds,  # Bug #6: Preserve timeout window
                # P2-2026-02-01: Preserve trigger fire time offsets
                "count_fire_offset": count_fire_offset,
                "condition_fire_offset": condition_fire_offset,
            }

        # Checkpoint format version
        # v1.0: Initial format with elapsed_age_seconds
        # v1.1: Added count_fire_offset/condition_fire_offset for trigger ordering (P2-2026-02-01)
        state["_version"] = "1.1"

        # Size validation (on serialized checkpoint)
        serialized = json.dumps(state)
        size_mb = len(serialized) / 1_000_000
        total_rows = sum(len(b) for b in self._buffer_tokens.values())

        if size_mb > 1:
            logger.warning(f"Large checkpoint: {size_mb:.1f}MB for {total_rows} buffered rows across {len(state)} nodes")

        if size_mb > 10:
            raise RuntimeError(
                f"Checkpoint size {size_mb:.1f}MB exceeds 10MB limit. "
                f"Buffer contains {total_rows} total rows across {len(state)} nodes. "
                f"Solutions: (1) Reduce aggregation count trigger to <5000 rows, "
                f"(2) Reduce row_data payload size, or (3) Implement checkpoint retention "
                f"policy (see P3-2026-01-21). See capacity planning in "
                f"docs/plans/2026-01-24-fix-aggregation-checkpoint-restore.md"
            )

        return state

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore executor state from checkpoint.

        Reconstructs full TokenInfo objects from checkpoint data, eliminating
        database queries during restoration. Expects format from get_checkpoint_state().

        Args:
            state: Checkpoint state with format:
                {
                    "_version": "1.0",
                    "node_id": {
                        "tokens": [{"token_id", "row_id", "branch_name", "row_data"}],
                        "batch_id": str
                    }
                }

        Raises:
            ValueError: If checkpoint format is invalid (per CLAUDE.md - our data, full trust)
        """
        # Validate checkpoint version (Bug #12 fix)
        CHECKPOINT_VERSION = "1.1"
        version = state.get("_version")

        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Incompatible checkpoint version: {version!r}. "
                f"Expected: {CHECKPOINT_VERSION!r}. "
                f"Cannot resume from incompatible checkpoint format. "
                f"This checkpoint may be from a different ELSPETH version."
            )

        for node_id_str, node_state in state.items():
            # Skip version metadata field
            if node_id_str == "_version":
                continue
            # Convert to typed NodeID for dictionary access
            node_id = NodeID(node_id_str)
            # Validate checkpoint format (OUR DATA - crash on mismatch, don't hide with .get())
            if "tokens" not in node_state:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: missing 'tokens' key. "
                    f"Found keys: {list(node_state.keys())}. "
                    f"Expected format: {{'tokens': [...], 'batch_id': str|None}}. "
                    f"This checkpoint may be from an incompatible ELSPETH version."
                )

            tokens_data = node_state["tokens"]

            # Validate tokens is a list
            if not isinstance(tokens_data, list):
                raise ValueError(f"Invalid checkpoint format for node {node_id}: 'tokens' must be a list, got {type(tokens_data).__name__}")

            # Reconstruct TokenInfo objects directly from checkpoint
            reconstructed_tokens = []
            for t in tokens_data:
                # Validate required fields (crash on missing - per CLAUDE.md)
                required_fields = {"token_id", "row_id", "row_data"}
                missing = required_fields - set(t.keys())
                if missing:
                    raise ValueError(
                        f"Checkpoint token missing required fields: {missing}. Required: {required_fields}. Found: {set(t.keys())}"
                    )

                # Reconstruct with explicit handling of optional fields
                # All lineage fields are optional per TokenInfo contract (default=None)
                reconstructed_tokens.append(
                    TokenInfo(
                        row_id=t["row_id"],
                        token_id=t["token_id"],
                        row_data=t["row_data"],
                        branch_name=t.get("branch_name"),
                        fork_group_id=t.get("fork_group_id"),
                        join_group_id=t.get("join_group_id"),
                        expand_group_id=t.get("expand_group_id"),
                    )
                )

            # Restore buffer state
            self._buffer_tokens[node_id] = reconstructed_tokens
            self._buffers[node_id] = [t.row_data for t in reconstructed_tokens]

            if "batch_id" not in node_state:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: missing 'batch_id' key. "
                    f"Found keys: {list(node_state.keys())}. "
                    "Checkpoint entries with tokens must include batch_id."
                )
            batch_id = node_state["batch_id"]
            if batch_id is None:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: 'batch_id' is None. "
                    "Checkpoint entries with tokens must include a batch_id."
                )
            self._batch_ids[node_id] = batch_id
            self._member_counts[batch_id] = len(reconstructed_tokens)

            # Restore trigger evaluator state (Bug #6 + P2-2026-02-01)
            if node_id in self._trigger_evaluators:
                evaluator = self._trigger_evaluators[node_id]

                # P2-2026-02-01: Use dedicated restore API that preserves fire time ordering
                # The old approach called record_accept() which set fire times to current time,
                # then rewound _first_accept_time, causing incorrect "first to fire wins" ordering.
                # NOTE: All fields are required in checkpoint format v1.1 - no backwards compat
                elapsed_seconds = node_state["elapsed_age_seconds"]
                count_fire_offset = node_state["count_fire_offset"]
                condition_fire_offset = node_state["condition_fire_offset"]

                evaluator.restore_from_checkpoint(
                    batch_count=len(reconstructed_tokens),
                    elapsed_age_seconds=elapsed_seconds,
                    count_fire_offset=count_fire_offset,
                    condition_fire_offset=condition_fire_offset,
                )

    def get_batch_id(self, node_id: NodeID) -> str | None:
        """Get current batch ID for an aggregation node.

        Primarily for testing - production code accesses this via checkpoint state.
        """
        return self._batch_ids.get(node_id)

    def should_flush(self, node_id: NodeID) -> bool:
        """Check if the aggregation should flush based on trigger config.

        Args:
            node_id: Aggregation node ID

        Returns:
            True if trigger condition is met, False otherwise
        """
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is None:
            return False
        return evaluator.should_trigger()

    def get_trigger_type(self, node_id: NodeID) -> "TriggerType | None":
        """Get the TriggerType for the trigger that fired.

        Args:
            node_id: Aggregation node ID

        Returns:
            TriggerType enum if a trigger fired, None otherwise
        """
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is None:
            return None
        return evaluator.get_trigger_type()

    def check_flush_status(self, node_id: NodeID) -> tuple[bool, "TriggerType | None"]:
        """Check flush status and get trigger type in a single operation.

        This is an optimized method that combines should_flush() and get_trigger_type()
        with a single dict lookup instead of two. Used in the hot path where
        timeout checks happen before every row is processed.

        Args:
            node_id: Aggregation node ID

        Returns:
            Tuple of (should_flush, trigger_type):
            - should_flush: True if trigger condition is met
            - trigger_type: The type of trigger that fired, or None
        """
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is None:
            return (False, None)

        should_flush = evaluator.should_trigger()
        trigger_type = evaluator.get_trigger_type() if should_flush else None
        return (should_flush, trigger_type)

    def restore_state(self, node_id: NodeID, state: dict[str, Any]) -> None:
        """Restore aggregation state from checkpoint.

        Called during recovery to restore plugin state. The state is stored
        for the aggregation plugin to access via get_restored_state().

        Args:
            node_id: Aggregation node ID
            state: Deserialized aggregation_state from checkpoint
        """
        self._restored_states[node_id] = state

    def get_restored_state(self, node_id: NodeID) -> dict[str, Any] | None:
        """Get restored state for an aggregation node.

        Used by aggregation plugins during recovery to restore their
        internal state from checkpoint.

        Args:
            node_id: Aggregation node ID

        Returns:
            Restored state dict, or None if no state was restored
        """
        return self._restored_states.get(node_id)

    def restore_batch(self, batch_id: str) -> None:
        """Restore a batch as the current in-progress batch.

        Called during recovery to resume a batch that was in progress
        when the crash occurred.

        Args:
            batch_id: The batch to restore as current

        Raises:
            ValueError: If batch not found
        """
        batch = self._recorder.get_batch(batch_id)
        if batch is None:
            raise ValueError(f"Batch not found: {batch_id}")

        node_id = NodeID(batch.aggregation_node_id)
        self._batch_ids[node_id] = batch_id

        # Restore member count from database
        members = self._recorder.get_batch_members(batch_id)
        self._member_counts[batch_id] = len(members)

    # NOTE: The old accept() and flush() methods that took AggregationProtocol
    # were DELETED in the aggregation structural cleanup.
    # Aggregation is now fully structural:
    # - Use buffer_row() to buffer rows
    # - Use should_flush() to check trigger
    # - Use execute_flush() to flush with full audit recording
    # - _get_buffered_data() is internal-only (for testing)


class SinkExecutor:
    """Executes sinks with artifact recording.

    Wraps sink.write() to:
    1. Create node_state for EACH token - this is how COMPLETED terminal state is derived
    2. Time the operation
    3. Record artifact produced by sink
    4. Complete all token states
    5. Emit OpenTelemetry span

    CRITICAL: Every token reaching a sink gets a node_state. This is the audit
    proof that the row reached its terminal state. The COMPLETED terminal state
    is DERIVED from having a completed node_state at a sink node.

    Example:
        executor = SinkExecutor(recorder, span_factory, run_id)
        artifact = executor.write(
            sink=my_sink,
            tokens=tokens_to_write,
            ctx=ctx,
            step_in_pipeline=5,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Run identifier for artifact registration
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id

    def write(
        self,
        sink: SinkProtocol,
        tokens: list[TokenInfo],
        ctx: PluginContext,
        step_in_pipeline: int,
        *,
        sink_name: str,
        pending_outcome: PendingOutcome,
        on_token_written: Callable[[TokenInfo], None] | None = None,
    ) -> Artifact | None:
        """Write tokens to sink with artifact recording.

        CRITICAL: Creates a node_state for EACH token written AND records
        token outcomes. Both records are created AFTER sink.flush()
        to ensure they only exist when data is durably written.

        This is the ONLY place terminal outcomes should be recorded for sink-bound
        tokens. Recording here (not in the orchestrator processing loop) ensures the
        token outcome contract is honored:
        - Invariant 3: "COMPLETED/ROUTED implies the token has a completed sink node_state"
        - Invariant 4: "Completed sink node_state implies a terminal token_outcome"

        Fix: P1-2026-01-31-quarantine-outcome-before-durability
        Uses PendingOutcome to carry error_hash for QUARANTINED outcomes through to
        recording, ensuring outcomes are only recorded after sink durability.

        Args:
            sink: Sink plugin to write to
            tokens: Tokens to write (may be empty)
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            sink_name: Name of the sink (for token_outcome recording)
            pending_outcome: PendingOutcome containing outcome and optional error_hash.
                    Required - all sink-bound tokens must have their outcome recorded.
            on_token_written: Optional callback called for each token after successful write.
                             Used for post-sink checkpointing.

        Returns:
            Artifact if tokens were written, None if empty

        Raises:
            Exception: Re-raised from sink.write() after recording failure
        """
        if not tokens:
            return None

        rows = [t.row_data for t in tokens]

        # Create node_state for EACH token - this is how we derive COMPLETED terminal state
        # Sink must have node_id assigned by orchestrator before execution
        if sink.node_id is None:
            raise OrchestrationInvariantError(f"Sink '{sink.name}' executed without node_id - orchestrator bug")
        sink_node_id: str = sink.node_id

        states: list[tuple[TokenInfo, NodeStateOpen]] = []
        for token in tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=sink_node_id,
                run_id=ctx.run_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            states.append((token, state))

        # CRITICAL: Clear state_id before entering operation context.
        # The ctx.state_id may still be set from the last transform that processed
        # these tokens. Sinks use operation_id for call attribution, and having both
        # state_id AND operation_id set would trigger the XOR constraint violation.
        ctx.state_id = None
        # Note: operation call_index is handled by LandscapeRecorder.allocate_operation_call_index()

        # Wrap sink I/O in operation for external call tracking
        # External calls during sink.write() are attributed to the operation (not token states)
        # The track_operation context manager sets ctx.operation_id automatically
        with track_operation(
            recorder=self._recorder,
            run_id=self._run_id,
            node_id=sink_node_id,
            operation_type="sink_write",
            ctx=ctx,
            input_data={"sink_plugin": sink.name, "row_count": len(tokens)},
        ) as handle:
            # Execute sink write with timing and span
            # P2-2026-01-21: Pass all token_ids being written for accurate attribution
            # P2-2026-01-21: Pass node_id for disambiguation when multiple sinks exist
            sink_token_ids = [t.token_id for t in tokens]
            with self._spans.sink_span(
                sink.name,
                node_id=sink_node_id,
                token_ids=sink_token_ids,
            ):
                start = time.perf_counter()
                try:
                    artifact_info = sink.write(rows, ctx)
                    duration_ms = (time.perf_counter() - start) * 1000
                except Exception as e:
                    duration_ms = (time.perf_counter() - start) * 1000
                    # Mark all token states as failed
                    error: ExecutionError = {
                        "exception": str(e),
                        "type": type(e).__name__,
                    }
                    for _, state in states:
                        self._recorder.complete_node_state(
                            state_id=state.state_id,
                            status=NodeStateStatus.FAILED,
                            duration_ms=duration_ms,
                            error=error,
                        )
                    raise

            # CRITICAL: Flush sink to ensure durability BEFORE checkpointing
            # If this fails, we want to crash - can't checkpoint non-durable data
            # But first we must complete node_states as FAILED to maintain audit integrity
            try:
                sink.flush()
            except Exception as e:
                # Flush failed - complete all node_states as FAILED before crashing
                # Without this, states remain OPEN permanently (audit integrity violation)
                flush_error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                    "phase": "flush",
                }
                flush_duration_ms = (time.perf_counter() - start) * 1000
                for _, state in states:
                    self._recorder.complete_node_state(
                        state_id=state.state_id,
                        status=NodeStateStatus.FAILED,
                        duration_ms=flush_duration_ms,
                        error=flush_error,
                    )
                raise

            # Set output data on operation handle for audit trail
            handle.output_data = {
                "artifact_path": artifact_info.path_or_uri,
                "content_hash": artifact_info.content_hash,
            }

        # Complete all token states - status=NodeStateStatus.COMPLETED means they reached terminal
        # Output is the row data that was written to the sink, plus artifact reference
        for token, state in states:
            sink_output = {
                "row": token.row_data,
                "artifact_path": artifact_info.path_or_uri,
                "content_hash": artifact_info.content_hash,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=sink_output,
                duration_ms=duration_ms,
            )

        # Register artifact (linked to first state for audit lineage)
        first_state = states[0][1]

        artifact = self._recorder.register_artifact(
            run_id=self._run_id,
            state_id=first_state.state_id,
            sink_node_id=sink_node_id,  # Already validated above
            artifact_type=artifact_info.artifact_type,
            path=artifact_info.path_or_uri,
            content_hash=artifact_info.content_hash,
            size_bytes=artifact_info.size_bytes,
        )

        # Record token outcomes AFTER sink durability is achieved
        # This is the ONLY correct place to record outcomes for sink-bound tokens - after:
        # 1. sink.write() succeeded
        # 2. sink.flush() succeeded (data is durable)
        # 3. node_states are marked COMPLETED
        # 4. artifact is registered
        # Recording here ensures Invariant 3: "COMPLETED/ROUTED implies completed sink node_state"
        #
        # Fix: P1-2026-01-31 - PendingOutcome carries error_hash for QUARANTINED outcomes
        # pending_outcome is REQUIRED - all sink-bound tokens must have outcomes recorded
        for token, _ in states:
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=token.token_id,
                outcome=pending_outcome.outcome,
                error_hash=pending_outcome.error_hash,
                sink_name=sink_name,
            )

        # Call checkpoint callback for each token after successful write + flush
        # CRITICAL: Sink write + flush are durable - we CANNOT roll them back.
        # If checkpoint creation fails, we log the error but don't raise.
        # The sink artifact exists, but no checkpoint record → resume will replay
        # these rows → duplicate writes (acceptable for RC-1, see Bug #10 docs).
        if on_token_written is not None:
            for token in tokens:
                try:
                    on_token_written(token)
                except Exception as e:
                    # Sink write is durable, can't undo. Log error and continue.
                    # Operator must manually clean up checkpoint inconsistency.
                    logger.error(
                        "Checkpoint failed after durable sink write for token %s. "
                        "Sink artifact exists but no checkpoint record created. "
                        "Resume will replay this row (duplicate write). "
                        "Manual cleanup may be required. Error: %s",
                        token.token_id,
                        e,
                        exc_info=True,
                    )
                    # Don't raise - we can't undo the sink write

        return artifact
