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
from elspeth.plugins.base import BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext

# Iteration guard to prevent infinite loops from bugs
MAX_WORK_QUEUE_ITERATIONS = 10_000


@dataclass
class _WorkItem:
    """Item in the work queue for DAG processing."""

    token: TokenInfo
    start_step: int  # Which step in transforms to start from (0-indexed)
    coalesce_at_step: int | None = None  # Step at which to coalesce (if any)
    coalesce_name: str | None = None  # Name of the coalesce point (if any)


class RowProcessor:
    """Processes rows through the transform pipeline.

    Handles:
    1. Creating initial tokens from source rows
    2. Executing transforms in sequence
    3. Executing config-driven gates (after transforms)
    4. Accepting rows into aggregations
    5. Recording final outcomes

    Pipeline order:
    - Transforms (from config.row_plugins)
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
        coalesce_node_ids: dict[str, str] | None = None,
        restored_aggregation_state: dict[str, dict[str, Any]] | None = None,
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
            coalesce_node_ids: Map of coalesce_name -> node_id for coalesce points
            restored_aggregation_state: Map of node_id -> state dict for crash recovery
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id = source_node_id
        self._config_gates = config_gates or []
        self._config_gate_id_map = config_gate_id_map or {}
        self._retry_manager = retry_manager
        self._coalesce_executor = coalesce_executor
        self._coalesce_node_ids = coalesce_node_ids or {}
        self._aggregation_settings = aggregation_settings or {}

        self._token_manager = TokenManager(recorder)
        self._transform_executor = TransformExecutor(recorder, span_factory)
        self._gate_executor = GateExecutor(
            recorder, span_factory, edge_map, route_resolution_map
        )
        self._aggregation_executor = AggregationExecutor(
            recorder, span_factory, run_id, aggregation_settings=aggregation_settings
        )

        # Restore aggregation state if provided (crash recovery)
        if restored_aggregation_state:
            for node_id, state in restored_aggregation_state.items():
                self._aggregation_executor.restore_state(node_id, state)

    @property
    def token_manager(self) -> TokenManager:
        """Expose token manager for orchestrator to create tokens for quarantined rows."""
        return self._token_manager

    def _process_batch_aggregation_node(
        self,
        transform: BaseTransform,
        current_token: TokenInfo,
        ctx: PluginContext,
        step: int,
        child_items: list[_WorkItem],
        total_steps: int,
    ) -> tuple[RowResult | list[RowResult], list[_WorkItem]]:
        """Process a row at an aggregation node using engine buffering.

        Engine buffers rows and calls transform.process(rows: list[dict])
        when the trigger fires.

        Args:
            transform: The batch-aware transform
            current_token: Current row token
            ctx: Plugin context
            step: Pipeline step number
            child_items: Work items to return with result
            total_steps: Total number of steps in the pipeline

        Returns:
            (RowResult or list[RowResult], child_items) tuple
            - Single RowResult for single/transform modes
            - List of RowResults for passthrough mode (one per buffered token)
        """
        node_id = transform.node_id
        assert node_id is not None

        # Get output_mode from aggregation settings
        # Caller guarantees node_id is in self._aggregation_settings (line 550 check)
        settings = self._aggregation_settings[node_id]
        output_mode = settings.output_mode

        # Buffer the row
        self._aggregation_executor.buffer_row(node_id, current_token)

        # Check if we should flush
        if self._aggregation_executor.should_flush(node_id):
            # Get buffered rows and tokens, then flush
            buffered_rows, buffered_tokens = self._aggregation_executor.flush_buffer(
                node_id
            )

            # Call transform with batch
            # Type ignore: batch-aware transforms accept list[dict] at runtime
            # but the protocol signature is dict[str, Any]. The transform's
            # is_batch_aware=True flag indicates it handles lists.
            result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]

            if result.status != "success":
                return (
                    RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome=RowOutcome.FAILED,
                        error=FailureInfo(
                            exception_type="TransformError",
                            message="Batch transform failed",
                        ),
                    ),
                    child_items,
                )

            # Handle output modes
            if output_mode == "single":
                # Single output: one aggregated result row
                final_data = result.row if result.row is not None else {}
                updated_token = TokenInfo(
                    row_id=current_token.row_id,
                    token_id=current_token.token_id,
                    row_data=final_data,
                    branch_name=current_token.branch_name,
                )
                return (
                    RowResult(
                        token=updated_token,
                        final_data=final_data,
                        outcome=RowOutcome.COMPLETED,
                    ),
                    child_items,
                )

            elif output_mode == "passthrough":
                # Passthrough: original tokens continue with enriched data
                # Validate result is multi-row
                if not result.is_multi_row:
                    raise ValueError(
                        f"Passthrough mode requires multi-row result, "
                        f"but transform '{transform.name}' returned single row. "
                        f"Use TransformResult.success_multi() for passthrough."
                    )

                # Validate row count matches
                assert result.rows is not None  # Guaranteed by is_multi_row
                if len(result.rows) != len(buffered_tokens):
                    raise ValueError(
                        f"Passthrough mode requires same number of output rows "
                        f"as input rows. Transform '{transform.name}' returned "
                        f"{len(result.rows)} rows but received {len(buffered_tokens)} input rows."
                    )

                # Build COMPLETED results for all buffered tokens with enriched data
                # Check if there are more transforms after this one
                more_transforms = step < total_steps

                if more_transforms:
                    # Queue enriched tokens as work items for remaining transforms
                    for token, enriched_data in zip(
                        buffered_tokens, result.rows, strict=True
                    ):
                        updated_token = TokenInfo(
                            row_id=token.row_id,
                            token_id=token.token_id,
                            row_data=enriched_data,
                            branch_name=token.branch_name,
                        )
                        child_items.append(
                            _WorkItem(
                                token=updated_token,
                                start_step=step,  # Continue from current step (0-indexed next)
                            )
                        )
                    # Return empty list - all results will come from child items
                    return ([], child_items)
                else:
                    # No more transforms - return COMPLETED for all tokens
                    results: list[RowResult] = []
                    for token, enriched_data in zip(
                        buffered_tokens, result.rows, strict=True
                    ):
                        updated_token = TokenInfo(
                            row_id=token.row_id,
                            token_id=token.token_id,
                            row_data=enriched_data,
                            branch_name=token.branch_name,
                        )
                        results.append(
                            RowResult(
                                token=updated_token,
                                final_data=enriched_data,
                                outcome=RowOutcome.COMPLETED,
                            )
                        )
                    return (results, child_items)

            elif output_mode == "transform":
                # Transform mode: N input rows -> M output rows with NEW tokens
                # Previously-buffered tokens already returned CONSUMED_IN_BATCH
                # when they were buffered (non-flushing path at bottom of method).
                # Only the triggering token (current_token) hasn't been returned yet.
                # New tokens are created for output rows via expand_token()

                # Get output rows - can be single or multi
                if result.is_multi_row:
                    assert result.rows is not None  # Guaranteed by is_multi_row
                    output_rows = result.rows
                else:
                    # Single row output is valid for transform mode
                    output_rows = [result.row] if result.row is not None else [{}]

                # Create new tokens via expand_token using triggering token as parent
                # This establishes audit trail linkage
                expanded_tokens = self._token_manager.expand_token(
                    parent_token=current_token,
                    expanded_rows=output_rows,
                    step_in_pipeline=step,
                )

                # The triggering token becomes CONSUMED_IN_BATCH
                triggering_result = RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=RowOutcome.CONSUMED_IN_BATCH,
                )

                # Check if there are more transforms after this one
                more_transforms = step < total_steps

                if more_transforms:
                    # Queue expanded tokens as work items for remaining transforms
                    for token in expanded_tokens:
                        child_items.append(
                            _WorkItem(
                                token=token,
                                start_step=step,  # Continue from current step (0-indexed next)
                            )
                        )
                    # Return triggering result - expanded tokens will produce results via work queue
                    return (triggering_result, child_items)
                else:
                    # No more transforms - return COMPLETED for expanded tokens
                    output_results: list[RowResult] = [triggering_result]
                    for token in expanded_tokens:
                        output_results.append(
                            RowResult(
                                token=token,
                                final_data=token.row_data,
                                outcome=RowOutcome.COMPLETED,
                            )
                        )
                    # Return triggering + completed results
                    return (output_results, child_items)

            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")

        # Not flushing yet - row is buffered
        # In passthrough mode: BUFFERED (non-terminal, will reappear)
        # In single/transform modes: CONSUMED_IN_BATCH (terminal)
        if output_mode == "passthrough":
            return (
                RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=RowOutcome.BUFFERED,
                ),
                child_items,
            )
        else:
            return (
                RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=RowOutcome.CONSUMED_IN_BATCH,
                ),
                child_items,
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
        *,
        coalesce_at_step: int | None = None,
        coalesce_name: str | None = None,
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
            coalesce_at_step: Step index at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging

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
        work_queue: deque[_WorkItem] = deque(
            [
                _WorkItem(
                    token=token,
                    start_step=0,
                    coalesce_at_step=coalesce_at_step,
                    coalesce_name=coalesce_name,
                )
            ]
        )
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
                    coalesce_at_step=item.coalesce_at_step,
                    coalesce_name=item.coalesce_name,
                )
                # Result can be:
                # - None for held coalesce tokens
                # - Single RowResult for most operations
                # - List of RowResults for passthrough aggregation mode
                if result is not None:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
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
        coalesce_at_step: int | None = None,
        coalesce_name: str | None = None,
    ) -> tuple[RowResult | list[RowResult] | None, list[_WorkItem]]:
        """Process a single token through transforms starting at given step.

        Args:
            token: Token to process
            transforms: List of transform plugins
            ctx: Plugin context
            start_step: Index in transforms to start from (0-indexed)
            coalesce_at_step: Step index at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging

        Returns:
            Tuple of (RowResult or list of RowResults or None if held for coalesce,
                      list of child WorkItems to queue)
            - Single RowResult for most operations
            - List of RowResults for passthrough aggregation mode
            - None for held coalesce tokens
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
                            _WorkItem(
                                token=child_token,
                                start_step=next_step,
                                coalesce_at_step=coalesce_at_step,
                                coalesce_name=coalesce_name,
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

            # NOTE: BaseAggregation branch was DELETED in aggregation structural cleanup.
            # Aggregation is now handled by batch-aware transforms (is_batch_aware=True).
            # The engine buffers rows and calls Transform.process(rows: list[dict]).

            elif isinstance(transform, BaseTransform):
                # Check if this is a batch-aware transform at an aggregation node
                node_id = transform.node_id
                if (
                    transform.is_batch_aware
                    and node_id is not None
                    and node_id in self._aggregation_settings
                ):
                    # Use engine buffering for aggregation
                    return self._process_batch_aggregation_node(
                        transform=transform,
                        current_token=current_token,
                        ctx=ctx,
                        step=step,
                        child_items=child_items,
                        total_steps=len(transforms),
                    )

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

                # Handle multi-row output (deaggregation)
                # NOTE: This is ONLY for non-aggregation transforms. Aggregation
                # transforms route through _process_batch_aggregation_node() above.
                if result.is_multi_row:
                    # Validate transform is allowed to create tokens
                    if not transform.creates_tokens:
                        raise RuntimeError(
                            f"Transform '{transform.name}' returned multi-row result "
                            f"but has creates_tokens=False. Either set creates_tokens=True "
                            f"or return single row via TransformResult.success(row). "
                            f"(Multi-row is allowed in aggregation passthrough mode.)"
                        )

                    # Deaggregation: create child tokens for each output row
                    # result.rows is guaranteed non-None when is_multi_row is True
                    child_tokens = self._token_manager.expand_token(
                        parent_token=current_token,
                        expanded_rows=result.rows,  # type: ignore[arg-type]
                        step_in_pipeline=step,
                    )

                    # Queue each child for continued processing
                    # Children start at next step (step_offset + 1 gives 0-indexed next)
                    next_step = start_step + step_offset + 1
                    for child_token in child_tokens:
                        child_items.append(
                            _WorkItem(
                                token=child_token,
                                start_step=next_step,
                                coalesce_at_step=coalesce_at_step,
                                coalesce_name=coalesce_name,
                            )
                        )

                    # Parent token is EXPANDED (terminal for parent)
                    return (
                        RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome=RowOutcome.EXPANDED,
                        ),
                        child_items,
                    )

                # Single row output (existing logic - current_token already updated
                # by _execute_transform_with_retry, continues to next transform)

            else:
                raise TypeError(
                    f"Unknown transform type: {type(transform).__name__}. "
                    f"Expected BaseTransform or BaseGate."
                )

        # Process config-driven gates (after all plugin transforms)
        # Step continues from where transforms left off
        config_gate_start_step = len(transforms) + 1

        # Calculate which config gate to start from based on start_step
        # If start_step > len(transforms), we skip some config gates
        # (e.g., fork children that already passed through earlier gates)
        config_gate_start_idx = max(0, start_step - len(transforms))
        for gate_idx, gate_config in enumerate(
            self._config_gates[config_gate_start_idx:], start=config_gate_start_idx
        ):
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
                    # Children start after ALL transforms, at next config gate
                    child_items.append(
                        _WorkItem(
                            token=child_token,
                            start_step=len(transforms) + next_config_step,
                            coalesce_at_step=coalesce_at_step,
                            coalesce_name=coalesce_name,
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

        # Check if this is a fork child that should be coalesced
        if (
            self._coalesce_executor is not None
            and current_token.branch_name is not None  # Is a fork child
            and coalesce_name is not None
            and coalesce_at_step is not None
        ):
            # Get the step we just completed
            completed_step = len(transforms) + len(self._config_gates)
            if completed_step >= coalesce_at_step:
                # Coalesce operation is at the next step after the last transform
                coalesce_step = completed_step + 1
                # Submit to coalesce executor
                coalesce_outcome = self._coalesce_executor.accept(
                    token=current_token,
                    coalesce_name=coalesce_name,
                    step_in_pipeline=coalesce_step,
                )

                if coalesce_outcome.held:
                    # Token is waiting for siblings - it's consumed by coalesce
                    # Return None - held tokens don't produce results until merged
                    return (None, child_items)

                if coalesce_outcome.merged_token is not None:
                    # All siblings arrived - return COALESCED with merged data
                    return (
                        RowResult(
                            token=coalesce_outcome.merged_token,
                            final_data=coalesce_outcome.merged_token.row_data,
                            outcome=RowOutcome.COALESCED,
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
