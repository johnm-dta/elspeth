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
from typing import TYPE_CHECKING, Any, Protocol

from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction, TransformResult

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


class TransformLike(Protocol):
    """Protocol for transform-like plugins."""

    name: str
    node_id: str

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row."""
        ...


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
        result, updated_token = executor.execute_transform(
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
        transform: TransformLike,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> tuple[TransformResult, TokenInfo]:
        """Execute a transform with full audit recording.

        This method handles a SINGLE ATTEMPT. Retry logic is the caller's
        responsibility (e.g., RetryManager wraps this for retryable transforms).
        Each attempt gets its own node_state record with attempt number tracked
        by the caller.

        Args:
            transform: Transform plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            Tuple of (TransformResult with audit fields, updated TokenInfo)

        Raises:
            Exception: Re-raised from transform.process() after recording failure
        """
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform.node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
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
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row) if result.row else None
        result.duration_ms = duration_ms

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
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error=result.reason,
            )
            updated_token = token

        return result, updated_token


class GateLike(Protocol):
    """Protocol for gate-like plugins."""

    name: str
    node_id: str

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate a row and decide routing."""
        ...


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
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            edge_map: Maps (node_id, label) -> edge_id for routing
        """
        self._recorder = recorder
        self._spans = span_factory
        self._edge_map = edge_map or {}

    def execute_gate(
        self,
        gate: GateLike,
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
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
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

        if action.kind == "continue":
            # No routing event needed - just continue to next transform
            pass

        elif action.kind == "route_to_sink":
            # Record routing event to sink
            self._record_routing(
                state_id=state.state_id,
                node_id=gate.node_id,
                action=action,
            )
            sink_name = action.destinations[0]

        elif action.kind == "fork_to_paths":
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
                routes.append({"edge_id": edge_id, "mode": action.mode})

            self._recorder.record_routing_events(
                state_id=state_id,
                routes=routes,
                reason=dict(action.reason) if action.reason else None,
            )
