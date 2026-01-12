# src/elspeth/engine/executors.py
"""Plugin executors that wrap plugin calls with audit recording.

Each executor handles a specific plugin type:
- TransformExecutor: Row transforms
- GateExecutor: Routing gates (Task 14)
- AggregationExecutor: Stateful aggregations (Task 15)
- SinkExecutor: Output sinks (Task 16)
"""

import time
from typing import Any, Protocol

from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult


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
