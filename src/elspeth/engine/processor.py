# src/elspeth/engine/processor.py
"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation (plugin and config-driven)
- Aggregation handling
- Final outcome recording
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from elspeth.contracts import RowOutcome, RowResult, TokenInfo, TransformResult

if TYPE_CHECKING:
    from elspeth.engine.coalesce_executor import CoalesceExecutor

from elspeth.contracts.enums import RoutingKind
from elspeth.contracts.results import FailureInfo
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.retry import MaxRetriesExceeded, RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager
from elspeth.plugins.base import BaseAggregation, BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext

# Iteration guard to prevent infinite loops from bugs
MAX_WORK_QUEUE_ITERATIONS = 10_000


@dataclass
class _WorkItem:
    """Item in the work queue for DAG processing."""

    token: TokenInfo
    start_step: int  # Which step in transforms to start from (0-indexed)


class RowProcessor:
    """Processes rows through the transform pipeline.

    Handles:
    1. Creating initial tokens from source rows
    2. Executing transforms in sequence (including plugin gates)
    3. Executing config-driven gates (after transforms)
    4. Accepting rows into aggregations
    5. Recording final outcomes

    Pipeline order:
    - Plugin transforms/gates (from config.transforms)
    - Config-driven gates (from config.gates)
    - Output sink

    Example:
        processor = RowProcessor(
            recorder, span_factory, run_id, source_node_id,
            config_gates=[GateSettings(...)],
            config_gate_id_map={"gate_name": "node_id"},
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform1, transform2],
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        source_node_id: str,
        *,
        edge_map: dict[tuple[str, str], str] | None = None,
        route_resolution_map: dict[tuple[str, str], str] | None = None,
        config_gates: list[GateSettings] | None = None,
        config_gate_id_map: dict[str, str] | None = None,
        aggregation_settings: dict[str, AggregationSettings] | None = None,
        retry_manager: RetryManager | None = None,
        coalesce_executor: "CoalesceExecutor | None" = None,
    ) -> None:
        """Initialize processor.

        Args:
            recorder: Landscape recorder
            span_factory: Span factory for tracing
            run_id: Current run ID
            source_node_id: Source node ID
            edge_map: Map of (node_id, label) -> edge_id
            route_resolution_map: Map of (node_id, label) -> "continue" | sink_name
            config_gates: List of config-driven gate settings
            config_gate_id_map: Map of gate name -> node_id for config gates
            aggregation_settings: Map of node_id -> AggregationSettings for trigger evaluation
            retry_manager: Optional retry manager for transform execution
            coalesce_executor: Optional coalesce executor for fork/join operations
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id = source_node_id
        self._config_gates = config_gates or []
        self._config_gate_id_map = config_gate_id_map or {}
        self._retry_manager = retry_manager
        self._coalesce_executor = coalesce_executor

        self._token_manager = TokenManager(recorder)
        self._transform_executor = TransformExecutor(recorder, span_factory)
        self._gate_executor = GateExecutor(
            recorder, span_factory, edge_map, route_resolution_map
        )
        self._aggregation_executor = AggregationExecutor(
            recorder, span_factory, run_id, aggregation_settings=aggregation_settings
        )

    def _execute_transform_with_retry(
        self,
        transform: Any,
        token: TokenInfo,
        ctx: PluginContext,
        step: int,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Execute transform with optional retry for transient failures.

        Retry behavior:
        - If retry_manager is None: single attempt, no retry
        - If retry_manager is set: retry on transient exceptions

        Each attempt is recorded separately in the audit trail with attempt number.

        Note: TransformResult.error() is NOT retried - that's a processing error,
        not a transient failure. Only exceptions trigger retry.

        Args:
            transform: Transform to execute
            token: Current token
            ctx: Plugin context
            step: Pipeline step index

        Returns:
            Tuple of (TransformResult, updated TokenInfo, error_sink)
        """
        if self._retry_manager is None:
            # No retry configured - single attempt
            return self._transform_executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=step,
                attempt=0,
            )

        # Track attempt number for audit
        attempt_tracker = {"current": 0}

        def execute_attempt() -> tuple[TransformResult, TokenInfo, str | None]:
            attempt = attempt_tracker["current"]
            attempt_tracker["current"] += 1
            return self._transform_executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=step,
                attempt=attempt,
            )

        def is_retryable(e: BaseException) -> bool:
            # Retry transient errors (network, timeout, rate limit)
            # Don't retry programming errors (AttributeError, TypeError, etc.)
            return isinstance(e, ConnectionError | TimeoutError | OSError)

        return self._retry_manager.execute_with_retry(
            operation=execute_attempt,
            is_retryable=is_retryable,
        )

    def process_row(
        self,
        row_index: int,
        row_data: dict[str, Any],
        transforms: list[Any],
        ctx: PluginContext,
    ) -> list[RowResult]:
        """Process a row through all transforms.

        Uses a work queue to handle fork operations - when a fork creates
        child tokens, they are added to the queue and processed through
        the remaining transforms.

        Args:
            row_index: Position in source
            row_data: Initial row data
            transforms: List of transform plugins
            ctx: Plugin context

        Returns:
            List of RowResults, one per terminal token (parent + children)
        """
        # Create initial token
        token = self._token_manager.create_initial_token(
            run_id=self._run_id,
            source_node_id=self._source_node_id,
            row_index=row_index,
            row_data=row_data,
        )

        # Initialize work queue with initial token starting at step 0
        work_queue: deque[_WorkItem] = deque([_WorkItem(token=token, start_step=0)])
        results: list[RowResult] = []
        iterations = 0

        with self._spans.row_span(token.row_id, token.token_id):
            while work_queue:
                iterations += 1
                if iterations > MAX_WORK_QUEUE_ITERATIONS:
                    raise RuntimeError(
                        f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. "
                        "Possible infinite loop in pipeline."
                    )

                item = work_queue.popleft()
                result, child_items = self._process_single_token(
                    token=item.token,
                    transforms=transforms,
                    ctx=ctx,
                    start_step=item.start_step,
                )
                results.append(result)

                # Add any child tokens to the queue
                work_queue.extend(child_items)

        return results

    def _process_single_token(
        self,
        token: TokenInfo,
        transforms: list[Any],
        ctx: PluginContext,
        start_step: int,
    ) -> tuple[RowResult, list[_WorkItem]]:
        """Process a single token through transforms starting at given step.

        Args:
            token: Token to process
            transforms: List of transform plugins
            ctx: Plugin context
            start_step: Index in transforms to start from (0-indexed)

        Returns:
            Tuple of (RowResult for this token, list of child WorkItems to queue)
        """
        current_token = token
        child_items: list[_WorkItem] = []

        # Process transforms starting from start_step
        for step_offset, transform in enumerate(transforms[start_step:]):
            step = start_step + step_offset + 1  # 1-indexed for audit

            # Type-safe plugin detection using base classes
            if isinstance(transform, BaseGate):
                # Gate transform
                outcome = self._gate_executor.execute_gate(
                    gate=transform,
                    token=current_token,
                    ctx=ctx,
                    step_in_pipeline=step,
                    token_manager=self._token_manager,
                )
                current_token = outcome.updated_token

                # Check if gate routed to a sink (sink_name set by executor)
                if outcome.sink_name is not None:
                    return (
                        RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome=RowOutcome.ROUTED,
                            sink_name=outcome.sink_name,
                        ),
                        child_items,
                    )
                elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
                    # Parent becomes FORKED, children continue from NEXT step
                    next_step = start_step + step_offset + 1
                    for child_token in outcome.child_tokens:
                        child_items.append(
                            _WorkItem(token=child_token, start_step=next_step)
                        )

                    return (
                        RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome=RowOutcome.FORKED,
                        ),
                        child_items,
                    )

            elif isinstance(transform, BaseAggregation):
                # Aggregation transform
                self._aggregation_executor.accept(
                    aggregation=transform,
                    token=current_token,
                    ctx=ctx,
                    step_in_pipeline=step,
                )

                # Check if engine-controlled trigger condition is met (WP-06)
                node_id = transform.node_id
                assert node_id is not None, "node_id must be set by orchestrator"
                if self._aggregation_executor.should_flush(node_id):
                    trigger_type = self._aggregation_executor.get_trigger_type(node_id)
                    trigger_reason = trigger_type.value if trigger_type else "manual"
                    self._aggregation_executor.flush(
                        aggregation=transform,
                        ctx=ctx,
                        trigger_reason=trigger_reason,
                        step_in_pipeline=step,
                    )

                return (
                    RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome=RowOutcome.CONSUMED_IN_BATCH,
                    ),
                    child_items,
                )

            elif isinstance(transform, BaseTransform):
                # Regular transform (with optional retry)
                try:
                    result, current_token, error_sink = (
                        self._execute_transform_with_retry(
                            transform=transform,
                            token=current_token,
                            ctx=ctx,
                            step=step,
                        )
                    )
                except MaxRetriesExceeded as e:
                    # All retries exhausted - return FAILED outcome
                    return (
                        RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome=RowOutcome.FAILED,
                            error=FailureInfo.from_max_retries_exceeded(e),
                        ),
                        child_items,
                    )

                if result.status == "error":
                    # Determine outcome based on error routing
                    if error_sink == "discard":
                        # Intentionally discarded - QUARANTINED
                        return (
                            RowResult(
                                token=current_token,
                                final_data=current_token.row_data,
                                outcome=RowOutcome.QUARANTINED,
                            ),
                            child_items,
                        )
                    else:
                        # Routed to error sink
                        return (
                            RowResult(
                                token=current_token,
                                final_data=current_token.row_data,
                                outcome=RowOutcome.ROUTED,
                                sink_name=error_sink,
                            ),
                            child_items,
                        )

            else:
                raise TypeError(
                    f"Unknown transform type: {type(transform).__name__}. "
                    f"Expected BaseTransform, BaseGate, or BaseAggregation."
                )

        # Process config-driven gates (after all plugin transforms)
        # Step continues from where transforms left off
        config_gate_start_step = len(transforms) + 1
        for gate_idx, gate_config in enumerate(self._config_gates):
            step = config_gate_start_step + gate_idx

            # Get the node_id for this config gate
            node_id = self._config_gate_id_map[gate_config.name]

            outcome = self._gate_executor.execute_config_gate(
                gate_config=gate_config,
                node_id=node_id,
                token=current_token,
                ctx=ctx,
                step_in_pipeline=step,
                token_manager=self._token_manager,
            )
            current_token = outcome.updated_token

            # Check if gate routed to a sink
            if outcome.sink_name is not None:
                return (
                    RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome=RowOutcome.ROUTED,
                        sink_name=outcome.sink_name,
                    ),
                    child_items,
                )
            elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
                # Config gate fork - children continue from next config gate
                next_config_step = gate_idx + 1
                for child_token in outcome.child_tokens:
                    # Children start after ALL plugin transforms, at next config gate
                    child_items.append(
                        _WorkItem(
                            token=child_token,
                            start_step=len(transforms) + next_config_step,
                        )
                    )

                return (
                    RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome=RowOutcome.FORKED,
                    ),
                    child_items,
                )

        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.COMPLETED,
            ),
            child_items,
        )
