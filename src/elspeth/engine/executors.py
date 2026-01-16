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

from elspeth.contracts import (
    Artifact,
    ExecutionError,
    NodeStateOpen,
    RoutingAction,
    RoutingSpec,
    TokenInfo,
)
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
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

        if action.kind == "continue":
            # No routing event needed - just continue to next transform
            pass

        elif action.kind == "route":
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
                routes.append(RoutingSpec(edge_id=edge_id, mode=action.mode))

            self._recorder.record_routing_events(
                state_id=state_id,
                routes=routes,
                reason=dict(action.reason) if action.reason else None,
            )


class AggregationLike(Protocol):
    """Protocol for aggregation-like plugins.

    Aggregations collect multiple rows into batches and flush them when
    triggered. The _batch_id is mutable state managed by AggregationExecutor.
    """

    name: str
    node_id: str
    _batch_id: str | None  # Mutable, managed by executor

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
        """Accept a row into the current batch.

        Returns AcceptResult indicating if row was accepted and if batch should trigger.
        """
        ...

    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        """Flush the current batch and return output rows."""
        ...


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
        if result.trigger:
            outputs = executor.flush(aggregation, ctx, "count_reached", step_in_pipeline)
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
            run_id: Run identifier for batch creation
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals

    def accept(
        self,
        aggregation: AggregationLike,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> AcceptResult:
        """Accept a row into an aggregation batch.

        Creates batch on first accept (if aggregation._batch_id is None).
        Records batch membership for accepted rows.

        Args:
            aggregation: Aggregation plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            AcceptResult with accepted flag, trigger flag, and batch_id

        Raises:
            Exception: Re-raised from aggregation.accept() after recording failure
        """
        # Create batch on first accept
        if aggregation._batch_id is None:
            batch = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=aggregation.node_id,
            )
            aggregation._batch_id = batch.batch_id
            self._member_counts[batch.batch_id] = 0

        # Begin node state for this accept operation
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=aggregation.node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        start = time.perf_counter()
        try:
            result = aggregation.accept(token.row_data, ctx)
            duration_ms = (time.perf_counter() - start) * 1000

            if result.accepted:
                ordinal = self._member_counts[aggregation._batch_id]
                self._recorder.add_batch_member(
                    batch_id=aggregation._batch_id,
                    token_id=token.token_id,
                    ordinal=ordinal,
                )
                self._member_counts[aggregation._batch_id] = ordinal + 1
                result.batch_id = aggregation._batch_id

                # Output for accepted rows: the input row + batch membership metadata
                # This records exactly what was accepted and where it went
                accept_output = {
                    "row": token.row_data,
                    "batch_id": aggregation._batch_id,
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
        aggregation: AggregationLike,
        ctx: PluginContext,
        trigger_reason: str,
        step_in_pipeline: int,
    ) -> list[dict[str, Any]]:
        """Flush an aggregation batch and return output rows.

        Transitions batch through: draft -> executing -> completed/failed.
        Resets aggregation._batch_id to None for next batch.

        Args:
            aggregation: Aggregation plugin to flush
            ctx: Plugin context
            trigger_reason: Why the batch was triggered (e.g., "count_reached", "end_of_input")
            step_in_pipeline: Current position in DAG (for audit context)

        Returns:
            List of output rows from the aggregation

        Raises:
            ValueError: If no batch to flush (aggregation._batch_id is None)
            Exception: Re-raised from aggregation.flush() after recording failure
        """
        batch_id = aggregation._batch_id
        if batch_id is None:
            raise ValueError(
                f"No batch to flush for aggregation {aggregation.node_id}. "
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
                aggregation._batch_id = None
                if batch_id in self._member_counts:
                    del self._member_counts[batch_id]

                return outputs

            except Exception:
                # Transition batch to failed
                self._recorder.update_batch_status(
                    batch_id=batch_id,
                    status="failed",
                )
                raise


class SinkLike(Protocol):
    """Protocol for sink-like plugins.

    This is an engine-internal adapter interface, not the same as Phase 2 SinkProtocol.
    Real SinkProtocol plugins write single rows. SinkAdapter (Phase 4) bridges between them.

    The write() method accepts a list of rows (batch write) and returns artifact info.
    """

    name: str
    node_id: str | None  # Set by orchestrator during registration

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write rows to sink.

        Args:
            rows: List of row data dictionaries
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with unified artifact info
        """
        ...


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
        sink: SinkLike,
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
        states: list[tuple[TokenInfo, NodeStateOpen]] = []
        for token in tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=sink.node_id,
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
            sink_node_id=sink.node_id,
            artifact_type=artifact_info.artifact_type,
            path=artifact_info.path_or_uri,
            content_hash=artifact_info.content_hash,
            size_bytes=artifact_info.size_bytes,
        )

        return artifact
