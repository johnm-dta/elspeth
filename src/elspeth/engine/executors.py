# src/elspeth/engine/executors.py
"""Plugin executors that wrap plugin calls with audit recording.

Each executor handles a specific plugin type:
- TransformExecutor: Row transforms
- GateExecutor: Routing gates (Task 14)
- AggregationExecutor: Stateful aggregations (Task 15)
- SinkExecutor: Output sinks (Task 16)
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import (
    Artifact,
    ExecutionError,
    NodeStateOpen,
    RoutingAction,
    RoutingSpec,
    TokenInfo,
)
from elspeth.contracts.enums import RoutingKind, RoutingMode, TriggerType
from elspeth.core.canonical import stable_hash
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.expression_parser import ExpressionParser
from elspeth.engine.spans import SpanFactory
from elspeth.engine.triggers import TriggerEvaluator
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import (
    AggregationProtocol,
    GateProtocol,
    SinkProtocol,
    TransformProtocol,
)
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    TransformResult,
)

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager


class MissingEdgeError(Exception):
    """Raised when routing refers to an unregistered edge.

    This is an audit integrity error - every routing decision must be
    traceable to a registered edge. Silent edge loss is unacceptable.
    """

    def __init__(self, node_id: str, label: str) -> None:
        """Initialize with routing details.

        Args:
            node_id: Node that attempted routing
            label: Edge label that was not found
        """
        self.node_id = node_id
        self.label = label
        super().__init__(
            f"No edge registered from node {node_id} with label '{label}'. "
            "Audit trail would be incomplete - refusing to proceed."
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
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
        """
        self._recorder = recorder
        self._spans = span_factory

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
        assert transform.node_id is not None, "node_id must be set by orchestrator"
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform.node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
            attempt=attempt,
        )

        # Execute with timing and span
        with self._spans.transform_span(transform.name, input_hash=input_hash):
            start = time.perf_counter()
            try:
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
                    status="failed",
                    duration_ms=duration_ms,
                    error=error,
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row) if result.row else None
        result.duration_ms = duration_ms

        # Initialize error_sink - will be set if transform errors with on_error configured
        error_sink: str | None = None

        # Complete node state
        if result.status == "success":
            # TransformResult.success() always sets row - this is a contract
            assert result.row is not None, "success status requires row data"
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                output_data=result.row,
                duration_ms=duration_ms,
            )
            # Update token with new row data
            updated_token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=result.row,
                branch_name=token.branch_name,
            )
        else:
            # Transform returned error status (not exception)
            # This is a LEGITIMATE processing failure, not a bug
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
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
            ctx.record_transform_error(
                token_id=token.token_id,
                transform_id=transform.name,
                row=token.row_data,
                error_details=result.reason or {},
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
        edge_map: dict[tuple[str, str], str] | None = None,
        route_resolution_map: dict[tuple[str, str], str] | None = None,
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
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            token_manager: TokenManager for fork operations (required for fork_to_paths)

        Returns:
            GateOutcome with result, updated token, and routing info

        Raises:
            MissingEdgeError: If routing refers to an unregistered edge
            Exception: Re-raised from gate.evaluate() after recording failure
        """
        assert gate.node_id is not None, "node_id must be set by orchestrator"
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        # Execute with timing and span
        with self._spans.gate_span(gate.name, input_hash=input_hash):
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
                    status="failed",
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

        if action.kind == RoutingKind.CONTINUE:
            # No routing event needed - just continue to next transform
            pass

        elif action.kind == RoutingKind.ROUTE:
            # Gate returned a route label - resolve via routes config
            route_label = action.destinations[0]
            destination = self._route_resolution_map.get((gate.node_id, route_label))

            if destination is None:
                # Label not in routes config - this is a configuration error
                raise MissingEdgeError(node_id=gate.node_id, label=route_label)

            if destination == "continue":
                # Route label resolves to "continue" - no routing event
                pass
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
            # Create child tokens
            child_tokens = token_manager.fork_token(
                parent_token=token,
                branches=list(action.destinations),
                step_in_pipeline=step_in_pipeline,
                row_data=result.row,
            )

        # Complete node state - always "completed" for successful execution
        # Terminal state is DERIVED from routing_events, not stored here
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data=result.row,
            duration_ms=duration_ms,
        )

        # Update token with new row data
        updated_token = TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=result.row,
            branch_name=token.branch_name,
        )

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
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        # Create parser and evaluate condition
        with self._spans.gate_span(gate_config.name, input_hash=input_hash):
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
                    status="failed",
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
                status="failed",
                duration_ms=duration_ms,
                error=error,
            )
            raise ValueError(
                f"Gate '{gate_config.name}' condition returned '{route_label}' "
                f"which is not in routes: {list(gate_config.routes.keys())}"
            )

        destination = gate_config.routes[route_label]

        # Build routing action and process based on destination
        child_tokens: list[TokenInfo] = []
        sink_name: str | None = None
        reason = {"condition": gate_config.condition, "result": route_label}

        if destination == "continue":
            # Continue to next node - no routing event needed
            action = RoutingAction.continue_(reason=reason)

        elif destination == "fork":
            # Fork to multiple paths - fork_to guaranteed by GateSettings.validate_fork_consistency()
            assert (
                gate_config.fork_to is not None
            )  # Pydantic validation guarantees this

            if token_manager is None:
                error = {
                    "exception": "fork requires TokenManager",
                    "type": "RuntimeError",
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error=error,
                )
                raise RuntimeError(
                    f"Gate {node_id} routes to fork but no TokenManager provided. "
                    "Cannot create child tokens - audit integrity would be compromised."
                )

            action = RoutingAction.fork_to_paths(gate_config.fork_to, reason=reason)

            # Record routing events for all paths
            self._record_routing(
                state_id=state.state_id,
                node_id=node_id,
                action=action,
            )

            # Create child tokens
            child_tokens = token_manager.fork_token(
                parent_token=token,
                branches=gate_config.fork_to,
                step_in_pipeline=step_in_pipeline,
                row_data=token.row_data,
            )

        else:
            # Route to a named sink
            sink_name = destination
            action = RoutingAction.route(
                route_label, mode=RoutingMode.MOVE, reason=reason
            )

            # Record routing event
            self._record_routing(
                state_id=state.state_id,
                node_id=node_id,
                action=action,
            )

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
            status="completed",
            output_data=token.row_data,
            duration_ms=duration_ms,
        )

        # Token row_data is unchanged (config gates don't modify data)
        updated_token = TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=token.row_data,
            branch_name=token.branch_name,
        )

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
        if len(action.destinations) == 1:
            dest = action.destinations[0]
            edge_id = self._edge_map.get((node_id, dest))
            if edge_id is None:
                raise MissingEdgeError(node_id=node_id, label=dest)

            self._recorder.record_routing_event(
                state_id=state_id,
                edge_id=edge_id,
                mode=action.mode,
                reason=dict(action.reason) if action.reason else None,
            )
        else:
            # Multiple destinations (fork)
            routes = []
            for dest in action.destinations:
                edge_id = self._edge_map.get((node_id, dest))
                if edge_id is None:
                    raise MissingEdgeError(node_id=node_id, label=dest)
                routes.append(RoutingSpec(edge_id=edge_id, mode=action.mode))

            self._recorder.record_routing_events(
                state_id=state_id,
                routes=routes,
                reason=dict(action.reason) if action.reason else None,
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
        aggregation_settings: dict[str, AggregationSettings] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Run identifier for batch creation
            aggregation_settings: Map of node_id -> AggregationSettings for trigger evaluation
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals
        self._batch_ids: dict[str, str | None] = {}  # node_id -> current batch_id
        self._aggregation_settings = aggregation_settings or {}
        self._trigger_evaluators: dict[str, TriggerEvaluator] = {}
        self._restored_states: dict[str, dict[str, Any]] = {}  # node_id -> state

        # Engine-owned row buffers (node_id -> list of row dicts)
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        # Token tracking for audit trail (node_id -> list of TokenInfo)
        self._buffer_tokens: dict[str, list[TokenInfo]] = {}

        # Create trigger evaluators for each configured aggregation
        for node_id, settings in self._aggregation_settings.items():
            self._trigger_evaluators[node_id] = TriggerEvaluator(settings.trigger)
            self._buffers[node_id] = []
            self._buffer_tokens[node_id] = []

    def buffer_row(
        self,
        node_id: str,
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

    def get_buffered_rows(self, node_id: str) -> list[dict[str, Any]]:
        """Get currently buffered rows (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts
        """
        return list(self._buffers.get(node_id, []))

    def get_buffered_tokens(self, node_id: str) -> list[TokenInfo]:
        """Get currently buffered tokens (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered TokenInfo objects
        """
        return list(self._buffer_tokens.get(node_id, []))

    def flush_buffer(self, node_id: str) -> list[dict[str, Any]]:
        """Get buffered rows and clear the buffer.

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts
        """
        rows = list(self._buffers.get(node_id, []))
        self._buffers[node_id] = []
        self._buffer_tokens[node_id] = []

        # Reset trigger evaluator for next batch
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is not None:
            evaluator.reset()

        # Clear batch ID for next batch
        self._batch_ids[node_id] = None

        return rows

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Get serializable state for checkpointing.

        Returns a dict that can be JSON-serialized and stored in
        checkpoint.aggregation_state_json. On recovery, pass this
        to restore_from_checkpoint().

        Returns:
            Dict mapping node_id -> buffer state (only non-empty buffers)
        """
        state: dict[str, Any] = {}
        for node_id in self._buffers:
            if self._buffers[node_id]:  # Only include non-empty buffers
                state[node_id] = {
                    "rows": list(self._buffers[node_id]),
                    "token_ids": [t.token_id for t in self._buffer_tokens[node_id]],
                    "batch_id": self._batch_ids.get(node_id),
                }
        return state

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore buffer state from checkpoint.

        Called during recovery to restore buffers from a previous
        run's checkpoint. Also restores trigger evaluator counts.

        Args:
            state: Dict from get_checkpoint_state() of previous run
        """
        for node_id, node_state in state.items():
            rows = node_state.get("rows", [])
            batch_id = node_state.get("batch_id")

            # Restore buffer (we don't store full TokenInfo, just rows)
            self._buffers[node_id] = list(rows)

            # Restore batch ID
            if batch_id:
                self._batch_ids[node_id] = batch_id

            # Restore trigger evaluator count
            evaluator = self._trigger_evaluators.get(node_id)
            if evaluator:
                for _ in range(len(rows)):
                    evaluator.record_accept()

            # Note: We don't restore full TokenInfo objects - only token_ids
            # are stored. The actual TokenInfo will be reconstructed if needed
            # from the tokens table. Clear the list for now.
            self._buffer_tokens[node_id] = []

    def get_batch_id(self, node_id: str) -> str | None:
        """Get current batch ID for an aggregation node.

        Primarily for testing - production code should use AcceptResult.batch_id.
        """
        return self._batch_ids.get(node_id)

    def should_flush(self, node_id: str) -> bool:
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

    def get_trigger_type(self, node_id: str) -> "TriggerType | None":
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

    def restore_state(self, node_id: str, state: dict[str, Any]) -> None:
        """Restore aggregation state from checkpoint.

        Called during recovery to restore plugin state. The state is stored
        for the aggregation plugin to access via get_restored_state().

        Args:
            node_id: Aggregation node ID
            state: Deserialized aggregation_state from checkpoint
        """
        self._restored_states[node_id] = state

    def get_restored_state(self, node_id: str) -> dict[str, Any] | None:
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

        node_id = batch.aggregation_node_id
        self._batch_ids[node_id] = batch_id

        # Restore member count from database
        members = self._recorder.get_batch_members(batch_id)
        self._member_counts[batch_id] = len(members)

    def accept(
        self,
        aggregation: AggregationProtocol,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> AcceptResult:
        """Accept a row into an aggregation batch.

        Creates batch on first accept (if no batch_id for this node).
        Records batch membership for accepted rows.

        Args:
            aggregation: Aggregation plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            AcceptResult with accepted flag and batch_id

        Raises:
            Exception: Re-raised from aggregation.accept() after recording failure
        """
        node_id = aggregation.node_id
        assert node_id is not None, "node_id must be set by orchestrator"

        # Create batch on first accept
        if self._batch_ids.get(node_id) is None:
            batch = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=node_id,
            )
            self._batch_ids[node_id] = batch.batch_id
            self._member_counts[batch.batch_id] = 0

        batch_id = self._batch_ids[node_id]
        assert batch_id is not None  # We just created it if it was None

        # Begin node state for this accept operation
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        start = time.perf_counter()
        try:
            result = aggregation.accept(token.row_data, ctx)
            duration_ms = (time.perf_counter() - start) * 1000

            if result.accepted:
                ordinal = self._member_counts[batch_id]
                self._recorder.add_batch_member(
                    batch_id=batch_id,
                    token_id=token.token_id,
                    ordinal=ordinal,
                )
                self._member_counts[batch_id] = ordinal + 1
                result.batch_id = batch_id

                # Update trigger evaluator if row was accepted
                evaluator = self._trigger_evaluators.get(node_id)
                if evaluator is not None:
                    evaluator.record_accept()

                # Output for accepted rows: the input row + batch membership metadata
                # This records exactly what was accepted and where it went
                accept_output = {
                    "row": token.row_data,
                    "batch_id": batch_id,
                    "ordinal": ordinal,
                }
            else:
                # Output for rejected rows: the input row + rejection indicator
                # This records what was rejected and why (no batch_id)
                accept_output = {
                    "row": token.row_data,
                    "accepted": False,
                }

            # Complete node state - always "completed" for successful accept
            # Terminal state CONSUMED_IN_BATCH is derived from batch_members
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed" if result.accepted else "rejected",
                output_data=accept_output,
                duration_ms=duration_ms,
            )
            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            # Record failure
            error: ExecutionError = {
                "exception": str(e),
                "type": type(e).__name__,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error=error,
            )
            raise

    def flush(
        self,
        aggregation: AggregationProtocol,
        ctx: PluginContext,
        trigger_reason: str,
        step_in_pipeline: int,
    ) -> list[dict[str, Any]]:
        """Flush an aggregation batch and return output rows.

        Transitions batch through: draft -> executing -> completed/failed.
        Resets batch_id to None for next batch.

        Args:
            aggregation: Aggregation plugin to flush
            ctx: Plugin context
            trigger_reason: Why the batch was triggered (e.g., "count_reached", "end_of_input")
            step_in_pipeline: Current position in DAG (for audit context)

        Returns:
            List of output rows from the aggregation

        Raises:
            ValueError: If no batch to flush (no batch_id for this node)
            Exception: Re-raised from aggregation.flush() after recording failure
        """
        node_id = aggregation.node_id
        assert node_id is not None, "node_id must be set by orchestrator"

        batch_id = self._batch_ids.get(node_id)
        if batch_id is None:
            raise ValueError(
                f"No batch to flush for aggregation {node_id}. "
                "Call accept() first to create a batch."
            )

        # Transition batch to executing
        self._recorder.update_batch_status(
            batch_id=batch_id,
            status="executing",
            trigger_reason=trigger_reason,
        )

        # Execute flush within aggregation span
        with self._spans.aggregation_span(aggregation.name, batch_id=batch_id):
            try:
                outputs = aggregation.flush(ctx)

                # Transition batch to completed
                self._recorder.update_batch_status(
                    batch_id=batch_id,
                    status="completed",
                )

                # Reset for next batch
                self._batch_ids[node_id] = None
                if batch_id in self._member_counts:
                    del self._member_counts[batch_id]

                # Reset trigger evaluator for next batch
                evaluator = self._trigger_evaluators.get(node_id)
                if evaluator is not None:
                    evaluator.reset()

                return outputs

            except Exception:
                # Transition batch to failed
                self._recorder.update_batch_status(
                    batch_id=batch_id,
                    status="failed",
                )
                raise


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
    ) -> Artifact | None:
        """Write tokens to sink with artifact recording.

        CRITICAL: Creates a node_state for EACH token written. This is how
        we derive the COMPLETED terminal state - every token that reaches
        a sink gets a completed node_state at the sink node.

        Args:
            sink: Sink plugin to write to
            tokens: Tokens to write (may be empty)
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

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
        assert sink.node_id is not None, "Sink node_id must be set before execution"
        sink_node_id: str = sink.node_id

        states: list[tuple[TokenInfo, NodeStateOpen]] = []
        for token in tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=sink_node_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            states.append((token, state))

        # Execute sink write with timing and span
        with self._spans.sink_span(sink.name):
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
                        status="failed",
                        duration_ms=duration_ms,
                        error=error,
                    )
                raise

        # Complete all token states - status="completed" means they reached terminal
        # Output is the row data that was written to the sink, plus artifact reference
        for token, state in states:
            sink_output = {
                "row": token.row_data,
                "artifact_path": artifact_info.path_or_uri,
                "content_hash": artifact_info.content_hash,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
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

        return artifact
