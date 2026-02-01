# src/elspeth/engine/processor.py
"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation (plugin and config-driven)
- Aggregation handling
- Final outcome recording
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from elspeth.contracts import RowOutcome, RowResult, TokenInfo, TransformResult
from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID

if TYPE_CHECKING:
    from elspeth.contracts.enums import RoutingMode
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.executors import GateOutcome
    from elspeth.telemetry import TelemetryManager

from elspeth.contracts.enums import OutputMode, RoutingKind, TriggerType
from elspeth.contracts.errors import OrchestrationInvariantError, TransformErrorReason
from elspeth.contracts.results import FailureInfo
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.retry import MaxRetriesExceeded, RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager
from elspeth.plugins.clients.llm import LLMClientError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol, TransformProtocol

# Iteration guard to prevent infinite loops from bugs
MAX_WORK_QUEUE_ITERATIONS = 10_000


@dataclass
class _WorkItem:
    """Item in the work queue for DAG processing."""

    token: TokenInfo
    start_step: int  # Which step in transforms to start from (0-indexed)
    coalesce_at_step: int | None = None  # Step at which to coalesce (if any)
    coalesce_name: CoalesceName | None = None  # Name of the coalesce point (if any)


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
        source_node_id: NodeID,
        *,
        edge_map: dict[tuple[NodeID, str], str] | None = None,
        route_resolution_map: dict[tuple[NodeID, str], str] | None = None,
        config_gates: list[GateSettings] | None = None,
        config_gate_id_map: dict[GateName, NodeID] | None = None,
        aggregation_settings: dict[NodeID, AggregationSettings] | None = None,
        retry_manager: RetryManager | None = None,
        coalesce_executor: CoalesceExecutor | None = None,
        coalesce_node_ids: dict[CoalesceName, NodeID] | None = None,
        branch_to_coalesce: dict[BranchName, CoalesceName] | None = None,
        coalesce_step_map: dict[CoalesceName, int] | None = None,
        restored_aggregation_state: dict[NodeID, dict[str, Any]] | None = None,
        payload_store: PayloadStore | None = None,
        clock: Clock | None = None,
        max_workers: int | None = None,
        telemetry_manager: TelemetryManager | None = None,
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
            branch_to_coalesce: Map of branch_name -> coalesce_name for fork/join routing
            coalesce_step_map: Map of coalesce_name -> step position in pipeline
            restored_aggregation_state: Map of node_id -> state dict for crash recovery
            payload_store: Optional PayloadStore for persisting source row payloads
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
            max_workers: Maximum concurrent workers for transform execution (None = no limit)
            telemetry_manager: Optional TelemetryManager for emitting telemetry events.
                               If None, telemetry emission is disabled.
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id: NodeID = source_node_id
        self._config_gates = config_gates or []
        self._config_gate_id_map: dict[GateName, NodeID] = config_gate_id_map or {}
        self._retry_manager = retry_manager
        self._coalesce_executor = coalesce_executor
        self._coalesce_node_ids: dict[CoalesceName, NodeID] = coalesce_node_ids or {}
        self._branch_to_coalesce: dict[BranchName, CoalesceName] = branch_to_coalesce or {}
        self._coalesce_step_map: dict[CoalesceName, int] = coalesce_step_map or {}
        self._aggregation_settings: dict[NodeID, AggregationSettings] = aggregation_settings or {}
        self._clock = clock if clock is not None else DEFAULT_CLOCK

        self._token_manager = TokenManager(recorder, payload_store=payload_store)
        self._transform_executor = TransformExecutor(recorder, span_factory, max_workers=max_workers)
        self._gate_executor = GateExecutor(recorder, span_factory, edge_map, route_resolution_map)
        self._aggregation_executor = AggregationExecutor(
            recorder, span_factory, run_id, aggregation_settings=aggregation_settings, clock=self._clock
        )
        self._telemetry_manager = telemetry_manager

        # Restore aggregation state if provided (crash recovery)
        if restored_aggregation_state:
            for node_id, state in restored_aggregation_state.items():
                self._aggregation_executor.restore_state(node_id, state)

    @property
    def token_manager(self) -> TokenManager:
        """Expose token manager for orchestrator to create tokens for quarantined rows."""
        return self._token_manager

    def _emit_telemetry(self, event: TelemetryEvent) -> None:
        """Emit telemetry event if manager is configured.

        Telemetry is emitted AFTER Landscape recording succeeds. Landscape is
        the legal record; telemetry is operational visibility.

        Args:
            event: The telemetry event to emit
        """
        if self._telemetry_manager is not None:
            self._telemetry_manager.handle_event(event)

    def _emit_transform_completed(
        self,
        token: TokenInfo,
        transform: TransformProtocol,
        transform_result: TransformResult,
    ) -> None:
        """Emit TransformCompleted telemetry event.

        Called AFTER Landscape recording succeeds in TransformExecutor.

        Args:
            token: Token that was processed
            transform: Transform that was executed
            transform_result: Result from the transform execution
        """
        if self._telemetry_manager is None:
            return

        from datetime import UTC, datetime

        from elspeth.contracts import TransformCompleted
        from elspeth.contracts.enums import NodeStateStatus

        status = NodeStateStatus.COMPLETED if transform_result.status == "success" else NodeStateStatus.FAILED

        self._emit_telemetry(
            TransformCompleted(
                timestamp=datetime.now(UTC),
                run_id=self._run_id,
                row_id=token.row_id,
                token_id=token.token_id,
                node_id=transform.node_id,  # type: ignore[arg-type]
                plugin_name=transform.name,
                status=status,
                duration_ms=transform_result.duration_ms or 0.0,
                input_hash=transform_result.input_hash,
                output_hash=transform_result.output_hash,
            )
        )

    def _emit_gate_evaluated(
        self,
        token: TokenInfo,
        gate_name: str,
        gate_node_id: str,
        routing_mode: RoutingMode,
        destinations: tuple[str, ...],
    ) -> None:
        """Emit GateEvaluated telemetry event.

        Called AFTER Landscape recording succeeds in GateExecutor.

        Args:
            token: Token that was routed
            gate_name: Name of the gate plugin
            gate_node_id: Node ID of the gate
            routing_mode: How routing was performed (move, copy)
            destinations: Destination node/sink names
        """
        if self._telemetry_manager is None:
            return

        from datetime import UTC, datetime

        from elspeth.contracts import GateEvaluated

        self._emit_telemetry(
            GateEvaluated(
                timestamp=datetime.now(UTC),
                run_id=self._run_id,
                row_id=token.row_id,
                token_id=token.token_id,
                node_id=gate_node_id,
                plugin_name=gate_name,
                routing_mode=routing_mode,
                destinations=destinations,
            )
        )

    def _emit_token_completed(
        self,
        token: TokenInfo,
        outcome: RowOutcome,
        sink_name: str | None = None,
    ) -> None:
        """Emit TokenCompleted telemetry event.

        Called AFTER Landscape recording succeeds (record_token_outcome).

        Args:
            token: Token that reached terminal state
            outcome: Terminal outcome (completed, routed, failed, etc.)
            sink_name: Destination sink if applicable
        """
        if self._telemetry_manager is None:
            return

        from datetime import UTC, datetime

        from elspeth.contracts import TokenCompleted

        self._emit_telemetry(
            TokenCompleted(
                timestamp=datetime.now(UTC),
                run_id=self._run_id,
                row_id=token.row_id,
                token_id=token.token_id,
                outcome=outcome,
                sink_name=sink_name,
            )
        )

    def _get_gate_destinations(self, outcome: GateOutcome) -> tuple[str, ...]:
        """Extract destination names from gate outcome for telemetry.

        Args:
            outcome: The gate outcome containing routing information

        Returns:
            Tuple of destination names (sink names or path names for forks)
        """
        if outcome.sink_name is not None:
            return (outcome.sink_name,)
        elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
            # For forks, return the branch names of child tokens
            return tuple(child.branch_name for child in outcome.child_tokens if child.branch_name)
        else:
            # Continue routing - destination is "continue"
            return ("continue",)

    # ─────────────────────────────────────────────────────────────────────────
    # Public facade for aggregation timeout checking
    # (Bug fix: P1-2026-01-22 - provides clean API for orchestrator timeout checks)
    # ─────────────────────────────────────────────────────────────────────────

    def check_aggregation_timeout(self, node_id: NodeID) -> tuple[bool, TriggerType | None]:
        """Check if an aggregation should flush due to timeout.

        This is a public facade for orchestrator to check timeout conditions
        without directly accessing private _aggregation_executor.

        Note: This method is called in the hot path (before every row is processed),
        so it uses the optimized check_flush_status() which does a single dict
        lookup instead of two separate calls.

        Args:
            node_id: The aggregation node ID to check

        Returns:
            Tuple of (should_flush, trigger_type):
            - should_flush: True if trigger condition is met
            - trigger_type: The type of trigger that fired (TIMEOUT, COUNT, etc.) or None
        """
        return self._aggregation_executor.check_flush_status(node_id)

    def get_aggregation_buffer_count(self, node_id: NodeID) -> int:
        """Get the number of rows buffered in an aggregation.

        Args:
            node_id: The aggregation node ID

        Returns:
            Number of rows currently buffered
        """
        return self._aggregation_executor.get_buffer_count(node_id)

    def process_token_from_step(
        self,
        token: TokenInfo,
        transforms: list[Any],
        ctx: PluginContext,
        start_step: int,
    ) -> list[RowResult]:
        """Process a token starting from a specific pipeline step.

        Used for continuing processing after timeout-triggered aggregation flushes.
        The token will be processed through remaining transforms starting at start_step.

        Args:
            token: The token to process
            transforms: List of transforms in the pipeline
            ctx: Plugin context
            start_step: The step index to start processing from

        Returns:
            List of RowResults from processing
        """
        work_queue: deque[_WorkItem] = deque(
            [
                _WorkItem(
                    token=token,
                    start_step=start_step,
                )
            ]
        )
        results: list[RowResult] = []
        iterations = 0

        with self._spans.row_span(token.row_id, token.token_id):
            while work_queue:
                iterations += 1
                if iterations > MAX_WORK_QUEUE_ITERATIONS:
                    raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")

                item = work_queue.popleft()
                result, child_items = self._process_single_token(
                    token=item.token,
                    transforms=transforms,
                    ctx=ctx,
                    start_step=item.start_step,
                    coalesce_at_step=item.coalesce_at_step,
                    coalesce_name=item.coalesce_name,
                )

                if result is not None:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)

                work_queue.extend(child_items)

        return results

    def flush_aggregation_timeout(
        self,
        node_id: NodeID,
        transform: TransformProtocol,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> tuple[TransformResult, list[TokenInfo], str | None]:
        """Execute a timeout-triggered flush for an aggregation.

        This is a public facade for orchestrator to flush timed-out aggregations
        without directly accessing private _aggregation_executor.

        Args:
            node_id: The aggregation node ID
            transform: The batch-aware transform to execute
            ctx: Plugin context
            step_in_pipeline: Position of this aggregation in the pipeline

        Returns:
            Tuple of (result, buffered_tokens, batch_id):
            - result: TransformResult from batch processing
            - buffered_tokens: List of tokens that were in the batch
            - batch_id: The batch ID (for audit trail)
        """
        return self._aggregation_executor.execute_flush(
            node_id=node_id,
            transform=transform,
            ctx=ctx,
            step_in_pipeline=step_in_pipeline,
            trigger_type=TriggerType.TIMEOUT,
        )

    def handle_timeout_flush(
        self,
        node_id: NodeID,
        transform: TransformProtocol,
        ctx: PluginContext,
        step: int,
        total_steps: int,
        trigger_type: TriggerType,
    ) -> tuple[list[RowResult], list[_WorkItem]]:
        """Handle an aggregation flush with proper output_mode semantics.

        This method mirrors the flush handling in _process_batch_aggregation_node but
        is designed for flushes that occur outside normal row processing:
        - TIMEOUT: Triggered between row arrivals when timeout expires
        - END_OF_SOURCE: Triggered at end of source to flush remaining buffers

        Handles all output_modes correctly:
        - passthrough: Routes all buffered tokens through remaining transforms
        - transform: Creates new tokens via expand_token (N→M output)

        Args:
            node_id: The aggregation node ID
            transform: The batch-aware transform to execute
            ctx: Plugin context
            step: Position of this aggregation in the pipeline (0-indexed).
                Converted to 1-indexed internally for audit recording.
            total_steps: Total number of transform steps in pipeline
            trigger_type: The trigger type (TIMEOUT or END_OF_SOURCE)

        Returns:
            Tuple of (results, work_items):
            - results: RowResults for completed tokens (terminal state)
            - work_items: _WorkItem list for tokens needing further processing
        """
        # Get aggregation settings for output_mode
        settings = self._aggregation_settings[node_id]
        output_mode = settings.output_mode

        # Convert 0-indexed step to 1-indexed for audit recording
        # Normal transform execution uses: step = start_step + step_offset + 1
        # Timeout/end-of-source paths must match this 1-indexed convention
        audit_step = step + 1

        # Execute flush with the specified trigger type
        result, buffered_tokens, _batch_id = self._aggregation_executor.execute_flush(
            node_id=node_id,
            transform=transform,
            ctx=ctx,
            step_in_pipeline=audit_step,
            trigger_type=trigger_type,
        )

        child_items: list[_WorkItem] = []
        results: list[RowResult] = []

        if result.status != "success":
            # Flush failed - handle based on output_mode
            #
            # CRITICAL: Token outcome recording depends on output_mode:
            # - passthrough: tokens have BUFFERED (non-terminal) → record FAILED
            # - transform: tokens have CONSUMED_IN_BATCH (terminal) → cannot record FAILED
            #
            # For transform mode, the batch failure is already recorded in the
            # batches table by execute_flush(). The CONSUMED_IN_BATCH outcome remains
            # semantically correct (tokens were consumed into a batch that failed).
            # Recording FAILED would violate the unique terminal outcome constraint.
            error_msg = "Batch transform failed during timeout flush"
            error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

            if output_mode == OutputMode.PASSTHROUGH:
                # Passthrough mode: tokens have BUFFERED outcome (non-terminal)
                # Record FAILED to give them a terminal outcome
                for token in buffered_tokens:
                    self._recorder.record_token_outcome(
                        run_id=self._run_id,
                        token_id=token.token_id,
                        outcome=RowOutcome.FAILED,
                        error_hash=error_hash,
                    )
                    # Emit TokenCompleted telemetry AFTER Landscape recording
                    self._emit_token_completed(token, RowOutcome.FAILED)
                    results.append(
                        RowResult(
                            token=token,
                            final_data=token.row_data,
                            outcome=RowOutcome.FAILED,
                            error=FailureInfo(
                                exception_type="TransformError",
                                message=error_msg,
                            ),
                        )
                    )
            else:
                # Single/transform mode: tokens already have CONSUMED_IN_BATCH (terminal)
                # DO NOT record FAILED - would violate unique terminal outcome constraint
                # Return FAILED results for count tracking, but no DB recording needed
                #
                # Bug P2-2026-02-01: Emit TokenCompleted for all buffered tokens.
                # TokenCompleted was deferred from buffer time to maintain ordering.
                # Even on failed flush, tokens have CONSUMED_IN_BATCH outcome (terminal).
                # Note: TransformCompleted is NOT emitted on error path (no successful processing).
                for token in buffered_tokens:
                    self._emit_token_completed(token, RowOutcome.CONSUMED_IN_BATCH)
                    results.append(
                        RowResult(
                            token=token,
                            final_data=token.row_data,
                            outcome=RowOutcome.FAILED,
                            error=FailureInfo(
                                exception_type="TransformError",
                                message=error_msg,
                            ),
                        )
                    )

            return (results, child_items)

        # SUCCESS PATH: Emit TransformCompleted telemetry for all buffered tokens
        # Each input token was processed by this aggregation transform as part of the batch.
        # Emitting per-token (rather than per-batch) maintains consistency with regular
        # transform telemetry and allows accurate token counting in observability dashboards.
        for token in buffered_tokens:
            self._emit_transform_completed(
                token=token,
                transform=transform,
                transform_result=result,
            )

        # Calculate if there are more transforms after this aggregation
        more_transforms = step < total_steps

        # Derive coalesce metadata from buffered tokens' branch_name
        # For timeout/end-of-source flushes, we need to preserve coalesce path
        # so tokens can still join at coalesce points after aggregation
        coalesce_at_step: int | None = None
        coalesce_name: CoalesceName | None = None
        if buffered_tokens:
            branch_name = buffered_tokens[0].branch_name
            if branch_name and BranchName(branch_name) in self._branch_to_coalesce:
                coalesce_name = self._branch_to_coalesce[BranchName(branch_name)]
                # Direct indexing: coalesce_step_map is internal data. If coalesce_name
                # exists in branch_to_coalesce but not in coalesce_step_map, that's a
                # bug in graph construction that must crash immediately (Tier 1 data).
                coalesce_at_step = self._coalesce_step_map[coalesce_name]

        if output_mode == OutputMode.PASSTHROUGH:
            # Passthrough: original tokens continue with enriched data
            if not result.is_multi_row:
                raise ValueError(
                    f"Passthrough mode requires multi-row result, "
                    f"but transform '{transform.name}' returned single row. "
                    f"Use TransformResult.success_multi() for passthrough."
                )

            if result.rows is None:
                raise RuntimeError("Multi-row result has rows=None")
            if len(result.rows) != len(buffered_tokens):
                raise ValueError(
                    f"Passthrough mode requires same number of output rows "
                    f"as input rows. Transform '{transform.name}' returned "
                    f"{len(result.rows)} rows but received {len(buffered_tokens)} input rows."
                )

            for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
                # Update token with enriched data, preserving all lineage metadata
                updated_token = token.with_updated_data(enriched_data)

                # Check if token needs to go to a coalesce point
                # This must happen EVEN if no more transforms - coalesce may be last step
                needs_coalesce = coalesce_at_step is not None and coalesce_name is not None and updated_token.branch_name is not None

                if more_transforms or needs_coalesce:
                    # Queue for remaining transforms with coalesce metadata
                    # NOTE: start_step is set to current 0-indexed position.
                    # Orchestrator adds +1 when calling process_token_from_step.
                    child_items.append(
                        _WorkItem(
                            token=updated_token,
                            start_step=step,  # 0-indexed current position
                            coalesce_at_step=coalesce_at_step,
                            coalesce_name=coalesce_name,
                        )
                    )
                else:
                    # No more transforms and no coalesce - return COMPLETED
                    results.append(
                        RowResult(
                            token=updated_token,
                            final_data=enriched_data,
                            outcome=RowOutcome.COMPLETED,
                        )
                    )

        elif output_mode == OutputMode.TRANSFORM:
            # Transform mode: N input rows -> M output rows with NEW tokens
            #
            # Bug P2-2026-02-01: Emit TokenCompleted for all buffered tokens AFTER
            # TransformCompleted (emitted above at line 556-561). TokenCompleted was
            # deferred from buffer time to maintain correct ordering.
            for token in buffered_tokens:
                self._emit_token_completed(token, RowOutcome.CONSUMED_IN_BATCH)

            # Get output rows
            if result.is_multi_row:
                if result.rows is None:
                    raise RuntimeError("Multi-row result has rows=None")
                output_rows = result.rows
            else:
                # Contract: batch-aware transforms in transform mode MUST return output data
                if result.row is None:
                    raise RuntimeError(
                        f"Aggregation transform '{transform.name}' returned None for result.row "
                        f"in 'transform' mode. Batch-aware transforms must return a row via "
                        f"TransformResult.success(row) or rows via TransformResult.success_multi(rows). "
                        f"This is a plugin bug."
                    )
                output_rows = [result.row]

            # Enforce expected_output_count if configured (plugin contract validation)
            if settings.expected_output_count is not None:
                actual_count = len(output_rows)
                if actual_count != settings.expected_output_count:
                    raise RuntimeError(
                        f"Aggregation '{settings.name}' produced {actual_count} output row(s), "
                        f"but expected_output_count={settings.expected_output_count}. "
                        f"This is a plugin contract violation."
                    )

            # Create new tokens via expand_token using first buffered token as parent
            # NOTE: Don't record EXPANDED - batch parents get CONSUMED_IN_BATCH separately
            if buffered_tokens:
                expanded_tokens, _expand_group_id = self._token_manager.expand_token(
                    parent_token=buffered_tokens[0],
                    expanded_rows=output_rows,
                    step_in_pipeline=audit_step,
                    run_id=self._run_id,
                    record_parent_outcome=False,
                )

                # Check if expanded tokens need to go to a coalesce point
                # This must happen EVEN if no more transforms - coalesce may be last step
                # Use first expanded token to check branch_name
                first_expanded_branch = expanded_tokens[0].branch_name if expanded_tokens else None
                needs_coalesce = coalesce_at_step is not None and coalesce_name is not None and first_expanded_branch is not None

                if more_transforms or needs_coalesce:
                    # Queue expanded tokens for remaining transforms with coalesce metadata
                    # NOTE: start_step is set to current 0-indexed position.
                    # Orchestrator adds +1 when calling process_token_from_step.
                    for token in expanded_tokens:
                        child_items.append(
                            _WorkItem(
                                token=token,
                                start_step=step,  # 0-indexed current position
                                coalesce_at_step=coalesce_at_step,
                                coalesce_name=coalesce_name,
                            )
                        )
                else:
                    # No more transforms and no coalesce - return COMPLETED
                    for token in expanded_tokens:
                        results.append(
                            RowResult(
                                token=token,
                                final_data=token.row_data,
                                outcome=RowOutcome.COMPLETED,
                            )
                        )

        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")

        return (results, child_items)

    def _process_batch_aggregation_node(
        self,
        transform: TransformProtocol,
        current_token: TokenInfo,
        ctx: PluginContext,
        step: int,
        child_items: list[_WorkItem],
        total_steps: int,
        coalesce_at_step: int | None = None,
        coalesce_name: CoalesceName | None = None,
    ) -> tuple[RowResult | list[RowResult], list[_WorkItem]]:
        """Process a row at an aggregation node using engine buffering.

        Engine buffers rows and calls transform.process(rows: list[dict])
        when the trigger fires.

        TEMPORAL DECOUPLING (Bug P2-2026-02-01):

        For transform-mode aggregation, there is intentional temporal decoupling
        between Landscape recording and telemetry emission:

        - **Landscape (audit trail)**: Records CONSUMED_IN_BATCH at buffer time.
          This is the source of truth - the token IS terminal when buffered.

        - **Telemetry (observability)**: Emits TokenCompleted at flush time.
          Deferred to maintain ordering invariant (TransformCompleted before
          TokenCompleted for each token).

        DO NOT assume "Landscape recording and telemetry emission happen together"
        for transform-mode aggregation. These two events have different timestamps.

        This decoupling is necessary because:
        1. Tokens become terminal (CONSUMED_IN_BATCH) when buffered
        2. But TransformCompleted can only fire when the batch actually processes
        3. Telemetry ordering requires TransformCompleted before TokenCompleted
        4. Therefore TokenCompleted must be deferred to flush time

        Args:
            transform: The batch-aware transform
            current_token: Current row token
            ctx: Plugin context
            step: Pipeline step number
            child_items: Work items to return with result
            total_steps: Total number of steps in the pipeline
            coalesce_at_step: Step index at which fork children should coalesce (optional)
            coalesce_name: Name of the coalesce point for merging (optional)

        Returns:
            (RowResult or list[RowResult], child_items) tuple
            - Single RowResult for transform mode (or list if N→M output)
            - List of RowResults for passthrough mode (one per buffered token)
        """
        raw_node_id = transform.node_id
        if raw_node_id is None:
            raise OrchestrationInvariantError("Node ID is None during edge resolution")
        node_id = NodeID(raw_node_id)

        # Get output_mode from aggregation settings
        # Caller guarantees node_id is in self._aggregation_settings (line 550 check)
        settings = self._aggregation_settings[node_id]
        output_mode = settings.output_mode

        # Buffer the row
        self._aggregation_executor.buffer_row(node_id, current_token)

        # Check if we should flush
        if self._aggregation_executor.should_flush(node_id):
            # Determine trigger type
            trigger_type = self._aggregation_executor.get_trigger_type(node_id)
            if trigger_type is None:
                trigger_type = TriggerType.COUNT  # Default if no evaluator

            # Execute flush with full audit recording
            result, buffered_tokens, batch_id = self._aggregation_executor.execute_flush(
                node_id=node_id,
                transform=transform,
                ctx=ctx,
                step_in_pipeline=step,
                trigger_type=trigger_type,
            )

            if result.status != "success":
                # Flush failed - handle based on output_mode
                #
                # CRITICAL: Token outcome recording depends on output_mode:
                # - passthrough: tokens have BUFFERED (non-terminal) → record FAILED
                # - transform: tokens have CONSUMED_IN_BATCH (terminal) → cannot record FAILED
                #
                # For transform mode, the batch failure is already recorded in the
                # batches table by execute_flush(). The CONSUMED_IN_BATCH outcome remains
                # semantically correct (tokens were consumed into a batch that failed).
                # Recording FAILED would violate the unique terminal outcome constraint.
                error_msg = "Batch transform failed"
                error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

                results: list[RowResult] = []
                if output_mode == OutputMode.PASSTHROUGH:
                    # Passthrough mode: tokens have BUFFERED outcome (non-terminal)
                    # Record FAILED for ALL buffered tokens to give them terminal outcome
                    for token in buffered_tokens:
                        self._recorder.record_token_outcome(
                            run_id=self._run_id,
                            token_id=token.token_id,
                            outcome=RowOutcome.FAILED,
                            error_hash=error_hash,
                        )
                        # Emit TokenCompleted telemetry AFTER Landscape recording
                        self._emit_token_completed(token, RowOutcome.FAILED)
                        results.append(
                            RowResult(
                                token=token,
                                final_data=token.row_data,
                                outcome=RowOutcome.FAILED,
                                error=FailureInfo(
                                    exception_type="TransformError",
                                    message=error_msg,
                                ),
                            )
                        )
                else:
                    # Single/transform mode: PREVIOUSLY buffered tokens have CONSUMED_IN_BATCH
                    # (recorded when they were buffered via the non-flushing return path in
                    # _process_transform_with_aggregation). However, the TRIGGERING token
                    # (current_token) went straight from buffer_row() to execute_flush(),
                    # skipping the non-flushing path. It needs CONSUMED_IN_BATCH.
                    #
                    # Record CONSUMED_IN_BATCH for triggering token only.
                    # DO NOT record FAILED - would violate unique terminal outcome constraint.
                    # Return FAILED results for count tracking, but no additional DB recording
                    # needed for previously buffered tokens.
                    self._recorder.record_token_outcome(
                        run_id=self._run_id,
                        token_id=current_token.token_id,
                        outcome=RowOutcome.CONSUMED_IN_BATCH,
                        batch_id=batch_id,
                    )

                    # Bug P2-2026-02-01: Emit TokenCompleted for ALL buffered tokens.
                    # TokenCompleted was deferred from buffer time to maintain ordering.
                    # Even on failed flush, tokens have CONSUMED_IN_BATCH outcome (terminal).
                    # Note: TransformCompleted is NOT emitted on error path (no successful processing).
                    # Note: buffered_tokens includes the triggering token (buffered before flush check).
                    for token in buffered_tokens:
                        self._emit_token_completed(token, RowOutcome.CONSUMED_IN_BATCH)

                    for token in buffered_tokens:
                        results.append(
                            RowResult(
                                token=token,
                                final_data=token.row_data,
                                outcome=RowOutcome.FAILED,
                                error=FailureInfo(
                                    exception_type="TransformError",
                                    message=error_msg,
                                ),
                            )
                        )
                return (results, child_items)

            # SUCCESS PATH: Emit TransformCompleted telemetry for all buffered tokens
            # Each input token was processed by this aggregation transform as part of the batch.
            # Emitting per-token (rather than per-batch) maintains consistency with regular
            # transform telemetry and allows accurate token counting in observability dashboards.
            #
            # NOTE: For transform mode, the triggering token (current_token) also needs
            # TransformCompleted, but it's emitted in the transform-mode block below AFTER
            # its Landscape recording and BEFORE its TokenCompleted (Bug P2-2026-02-01).
            for token in buffered_tokens:
                self._emit_transform_completed(
                    token=token,
                    transform=transform,
                    transform_result=result,
                )

            # Handle output modes
            if output_mode == OutputMode.PASSTHROUGH:
                # Passthrough: original tokens continue with enriched data
                # Validate result is multi-row
                if not result.is_multi_row:
                    raise ValueError(
                        f"Passthrough mode requires multi-row result, "
                        f"but transform '{transform.name}' returned single row. "
                        f"Use TransformResult.success_multi() for passthrough."
                    )

                # Validate row count matches
                if result.rows is None:
                    raise RuntimeError("Multi-row result has rows=None")
                if len(result.rows) != len(buffered_tokens):
                    raise ValueError(
                        f"Passthrough mode requires same number of output rows "
                        f"as input rows. Transform '{transform.name}' returned "
                        f"{len(result.rows)} rows but received {len(buffered_tokens)} input rows."
                    )

                # Build COMPLETED results for all buffered tokens with enriched data
                # Check if there are more transforms after this one
                more_transforms = step < total_steps

                # Check if tokens need to go to a coalesce point
                # This must happen EVEN if no more transforms - coalesce may be last step
                # Use first buffered token to check branch_name (all should have same branch)
                first_token_branch = buffered_tokens[0].branch_name if buffered_tokens else None
                needs_coalesce = coalesce_at_step is not None and coalesce_name is not None and first_token_branch is not None

                if more_transforms or needs_coalesce:
                    # Queue enriched tokens as work items for remaining transforms
                    # NOTE: `step` is 1-indexed (for audit) but happens to equal the
                    # 0-indexed position of the NEXT transform (since step = p + 1 where
                    # p is current 0-indexed position). So transforms[step:] gives remaining.
                    # If there's a coalesce point, skip directly there (matches fork behavior).
                    continuation_start = coalesce_at_step if coalesce_at_step is not None else step
                    for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
                        # Update token, preserving all lineage metadata
                        updated_token = token.with_updated_data(enriched_data)
                        child_items.append(
                            _WorkItem(
                                token=updated_token,
                                start_step=continuation_start,
                                coalesce_at_step=coalesce_at_step,
                                coalesce_name=coalesce_name,
                            )
                        )
                    # Return empty list - all results will come from child items
                    return ([], child_items)
                else:
                    # No more transforms and no coalesce - return COMPLETED for all tokens
                    passthrough_results: list[RowResult] = []
                    for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
                        # Update token, preserving all lineage metadata
                        updated_token = token.with_updated_data(enriched_data)
                        # COMPLETED outcome now recorded in orchestrator with sink_name (AUD-001)
                        passthrough_results.append(
                            RowResult(
                                token=updated_token,
                                final_data=enriched_data,
                                outcome=RowOutcome.COMPLETED,
                            )
                        )
                    return (passthrough_results, child_items)

            elif output_mode == OutputMode.TRANSFORM:
                # Transform mode: N input rows -> M output rows with NEW tokens
                # Previously-buffered tokens already returned CONSUMED_IN_BATCH
                # when they were buffered (non-flushing path at bottom of method).
                # Only the triggering token (current_token) hasn't been returned yet.
                # New tokens are created for output rows via expand_token()

                # Get output rows - can be single or multi
                if result.is_multi_row:
                    if result.rows is None:
                        raise RuntimeError("Multi-row result has rows=None")
                    output_rows = result.rows
                else:
                    # Contract: batch-aware transforms in transform mode MUST return output data
                    if result.row is None:
                        raise RuntimeError(
                            f"Aggregation transform '{transform.name}' returned None for result.row "
                            f"in 'transform' mode. Batch-aware transforms must return a row via "
                            f"TransformResult.success(row) or rows via TransformResult.success_multi(rows). "
                            f"This is a plugin bug."
                        )
                    output_rows = [result.row]

                # Enforce expected_output_count if configured (plugin contract validation)
                if settings.expected_output_count is not None:
                    actual_count = len(output_rows)
                    if actual_count != settings.expected_output_count:
                        raise RuntimeError(
                            f"Aggregation '{settings.name}' produced {actual_count} output row(s), "
                            f"but expected_output_count={settings.expected_output_count}. "
                            f"This is a plugin contract violation."
                        )

                # Create new tokens via expand_token using triggering token as parent
                # This establishes audit trail linkage
                # NOTE: Don't record EXPANDED - triggering token gets CONSUMED_IN_BATCH below
                expanded_tokens, _expand_group_id = self._token_manager.expand_token(
                    parent_token=current_token,
                    expanded_rows=output_rows,
                    step_in_pipeline=step,
                    run_id=self._run_id,
                    record_parent_outcome=False,
                )

                # The triggering token becomes CONSUMED_IN_BATCH
                # Note: batch_id comes from execute_flush() which captured it before reset
                self._recorder.record_token_outcome(
                    run_id=self._run_id,
                    token_id=current_token.token_id,
                    outcome=RowOutcome.CONSUMED_IN_BATCH,
                    batch_id=batch_id,
                )

                # Bug P2-2026-02-01: Emit TokenCompleted for ALL buffered tokens
                # TransformCompleted was already emitted for all tokens (including current_token)
                # in the loop at line 837-842 (buffered_tokens includes the triggering token
                # because buffer_row() adds it before should_flush() is checked).
                #
                # TokenCompleted was deferred from buffer time for non-triggering tokens to
                # maintain correct ordering (TransformCompleted before TokenCompleted).
                # Now emit TokenCompleted for all of them.
                for token in buffered_tokens:
                    self._emit_token_completed(token, RowOutcome.CONSUMED_IN_BATCH)

                triggering_result = RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=RowOutcome.CONSUMED_IN_BATCH,
                )

                # Check if there are more transforms after this one
                more_transforms = step < total_steps

                # Check if expanded tokens need to go to a coalesce point
                # This must happen EVEN if no more transforms - coalesce may be last step
                # Use first expanded token to check branch_name
                first_expanded_branch = expanded_tokens[0].branch_name if expanded_tokens else None
                needs_coalesce = coalesce_at_step is not None and coalesce_name is not None and first_expanded_branch is not None

                if more_transforms or needs_coalesce:
                    # Queue expanded tokens as work items for remaining transforms
                    # NOTE: `step` is 1-indexed (for audit) but happens to equal the
                    # 0-indexed position of the NEXT transform (since step = p + 1 where
                    # p is current 0-indexed position). So transforms[step:] gives remaining.
                    # If there's a coalesce point, skip directly there (matches fork behavior).
                    continuation_start = coalesce_at_step if coalesce_at_step is not None else step
                    for token in expanded_tokens:
                        child_items.append(
                            _WorkItem(
                                token=token,
                                start_step=continuation_start,
                                coalesce_at_step=coalesce_at_step,
                                coalesce_name=coalesce_name,
                            )
                        )
                    # Return triggering result - expanded tokens will produce results via work queue
                    return (triggering_result, child_items)
                else:
                    # No more transforms and no coalesce - return COMPLETED for expanded tokens
                    output_results: list[RowResult] = [triggering_result]
                    for token in expanded_tokens:
                        # COMPLETED outcome now recorded in orchestrator with sink_name (AUD-001)
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
        # In transform mode: CONSUMED_IN_BATCH (terminal)
        if output_mode == OutputMode.PASSTHROUGH:
            buf_batch_id = self._aggregation_executor.get_batch_id(node_id)
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=current_token.token_id,
                outcome=RowOutcome.BUFFERED,
                batch_id=buf_batch_id,
            )
            return (
                RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=RowOutcome.BUFFERED,
                ),
                child_items,
            )
        else:
            nf_batch_id = self._aggregation_executor.get_batch_id(node_id)
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=current_token.token_id,
                outcome=RowOutcome.CONSUMED_IN_BATCH,
                batch_id=nf_batch_id,
            )
            # NOTE: Do NOT emit TokenCompleted telemetry here!
            # Bug P2-2026-02-01: TokenCompleted must be deferred to flush time so that
            # TransformCompleted can be emitted first. The token IS terminal in Landscape
            # (CONSUMED_IN_BATCH recorded above), but telemetry ordering requires waiting
            # until the batch actually processes at flush time.
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
            # Must still catch retryable exceptions and convert to error results
            # to keep failures row-scoped (don't abort entire run)
            try:
                return self._transform_executor.execute_transform(
                    transform=transform,
                    token=token,
                    ctx=ctx,
                    step_in_pipeline=step,
                    attempt=0,
                )
            except LLMClientError as e:
                if e.retryable:
                    # Retryable error but no retry manager configured - convert to error result
                    # This keeps the failure row-scoped instead of aborting the run
                    #
                    # BUG FIX (P2-2026-01-27): Must validate on_error and record transform_error
                    # for audit trail completeness (same as TransformExecutor error handling)
                    on_error = transform._on_error
                    if on_error is None:
                        raise RuntimeError(
                            f"Transform '{transform.name}' raised retryable LLMClientError but has no "
                            f"on_error configured. Either configure on_error or enable retry. "
                            f"Error: {e}"
                        ) from e

                    error_details: TransformErrorReason = {"reason": "llm_retryable_error_no_retry", "error": str(e)}
                    ctx.record_transform_error(
                        token_id=token.token_id,
                        transform_id=transform.node_id,
                        row=token.row_data,
                        error_details=error_details,
                        destination=on_error,
                    )

                    return (
                        TransformResult.error(error_details, retryable=True),
                        token,
                        on_error,
                    )
                # Non-retryable errors re-raise (already handled by transform)
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                # Other retryable errors - convert to error result
                #
                # BUG FIX (P2-2026-01-27): Must validate on_error and record transform_error
                # for audit trail completeness (same as TransformExecutor error handling)
                on_error = transform._on_error
                if on_error is None:
                    raise RuntimeError(
                        f"Transform '{transform.name}' raised retryable {type(e).__name__} but has no "
                        f"on_error configured. Either configure on_error or enable retry. "
                        f"Error: {e}"
                    ) from e

                transient_error: TransformErrorReason = {"reason": "transient_error_no_retry", "error": str(e)}
                ctx.record_transform_error(
                    token_id=token.token_id,
                    transform_id=transform.node_id,
                    row=token.row_data,
                    error_details=transient_error,
                    destination=on_error,
                )

                return (
                    TransformResult.error(transient_error, retryable=True),
                    token,
                    on_error,
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
            #
            # LLMClientError has a retryable attribute:
            # - RateLimitError, NetworkError, ServerError: retryable=True
            # - ContentPolicyError, ContextLengthError: retryable=False
            if isinstance(e, LLMClientError):
                return e.retryable
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
        coalesce_name: CoalesceName | None = None,
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
                    raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")

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

    def process_existing_row(
        self,
        row_id: str,
        row_data: dict[str, Any],
        transforms: list[Any],
        ctx: PluginContext,
        *,
        coalesce_at_step: int | None = None,
        coalesce_name: CoalesceName | None = None,
    ) -> list[RowResult]:
        """Process an existing row (row already in database, create new token only).

        Used during resume when rows were created in the original run
        but need to be reprocessed. Unlike process_row(), this does NOT
        create a new row record - only a new token.

        Args:
            row_id: Existing row ID in the database
            row_data: Row data (retrieved from payload store)
            transforms: List of transform plugins
            ctx: Plugin context
            coalesce_at_step: Step index at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging

        Returns:
            List of RowResults, one per terminal token (parent + children)
        """
        # Create token for existing row (NOT a new row)
        token = self._token_manager.create_token_for_existing_row(
            row_id=row_id,
            row_data=row_data,
        )

        # Initialize work queue with token starting at step 0
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
                    raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")

                item = work_queue.popleft()
                result, child_items = self._process_single_token(
                    token=item.token,
                    transforms=transforms,
                    ctx=ctx,
                    start_step=item.start_step,
                    coalesce_at_step=item.coalesce_at_step,
                    coalesce_name=item.coalesce_name,
                )
                if result is not None:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)

                work_queue.extend(child_items)

        return results

    def process_token(
        self,
        token: TokenInfo,
        transforms: list[Any],
        ctx: PluginContext,
        *,
        start_step: int,
        coalesce_at_step: int | None = None,
        coalesce_name: CoalesceName | None = None,
    ) -> list[RowResult]:
        """Process an existing token through the pipeline starting at start_step.

        Used for mid-pipeline coalesce merges that must continue processing.
        """
        work_queue: deque[_WorkItem] = deque(
            [
                _WorkItem(
                    token=token,
                    start_step=start_step,
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
                    raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")

                item = work_queue.popleft()
                result, child_items = self._process_single_token(
                    token=item.token,
                    transforms=transforms,
                    ctx=ctx,
                    start_step=item.start_step,
                    coalesce_at_step=item.coalesce_at_step,
                    coalesce_name=item.coalesce_name,
                )
                if result is not None:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)

                work_queue.extend(child_items)

        return results

    def _maybe_coalesce_token(
        self,
        current_token: TokenInfo,
        *,
        coalesce_at_step: int | None,
        coalesce_name: CoalesceName | None,
        step_completed: int,
        total_steps: int,
        child_items: list[_WorkItem],
    ) -> tuple[bool, RowResult | None]:
        if (
            self._coalesce_executor is None
            or current_token.branch_name is None
            or coalesce_name is None
            or coalesce_at_step is None
            or step_completed < coalesce_at_step
        ):
            return False, None

        coalesce_outcome = self._coalesce_executor.accept(
            token=current_token,
            coalesce_name=coalesce_name,
            step_in_pipeline=coalesce_at_step,
        )

        if coalesce_outcome.held:
            return True, None

        if coalesce_outcome.merged_token is not None:
            if coalesce_at_step >= total_steps:
                return (
                    True,
                    RowResult(
                        token=coalesce_outcome.merged_token,
                        final_data=coalesce_outcome.merged_token.row_data,
                        outcome=RowOutcome.COALESCED,
                    ),
                )

            child_items.append(
                _WorkItem(
                    token=coalesce_outcome.merged_token,
                    start_step=coalesce_at_step,
                )
            )
            return True, None

        if coalesce_outcome.failure_reason:
            error_msg = coalesce_outcome.failure_reason
            error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

            # Bug 9z8 fix: Only record if CoalesceExecutor didn't already record
            if not coalesce_outcome.outcomes_recorded:
                self._recorder.record_token_outcome(
                    run_id=self._run_id,
                    token_id=current_token.token_id,
                    outcome=RowOutcome.FAILED,
                    error_hash=error_hash,
                )
            # Emit TokenCompleted telemetry AFTER Landscape recording
            self._emit_token_completed(current_token, RowOutcome.FAILED)

            return (
                True,
                RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=RowOutcome.FAILED,
                    error=FailureInfo(
                        exception_type="CoalesceFailure",
                        message=error_msg,
                    ),
                ),
            )

        return True, None

    def _process_single_token(
        self,
        token: TokenInfo,
        transforms: list[Any],
        ctx: PluginContext,
        start_step: int,
        coalesce_at_step: int | None = None,
        coalesce_name: CoalesceName | None = None,
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
        total_steps = len(transforms) + len(self._config_gates)

        handled, result = self._maybe_coalesce_token(
            current_token,
            coalesce_at_step=coalesce_at_step,
            coalesce_name=coalesce_name,
            step_completed=start_step,
            total_steps=total_steps,
            child_items=child_items,
        )
        if handled:
            return (result, child_items)

        # Process transforms starting from start_step
        for step_offset, transform in enumerate(transforms[start_step:]):
            step = start_step + step_offset + 1  # 1-indexed for audit

            # Type-safe plugin detection using protocols (supports protocol-only plugins)
            if isinstance(transform, GateProtocol):
                # Gate transform
                outcome = self._gate_executor.execute_gate(
                    gate=transform,
                    token=current_token,
                    ctx=ctx,
                    step_in_pipeline=step,
                    token_manager=self._token_manager,
                )
                current_token = outcome.updated_token

                # Emit GateEvaluated telemetry AFTER Landscape recording succeeds
                # (Landscape recording happens inside execute_gate)
                self._emit_gate_evaluated(
                    token=current_token,
                    gate_name=transform.name,
                    gate_node_id=transform.node_id,  # type: ignore[arg-type]
                    routing_mode=outcome.result.action.mode,
                    destinations=self._get_gate_destinations(outcome),
                )

                # Check if gate routed to a sink (sink_name set by executor)
                if outcome.sink_name is not None:
                    # NOTE: Do NOT record ROUTED outcome here - the token hasn't been written yet.
                    # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
                    # This prevents duplicate outcomes and ensures correct audit semantics:
                    # outcome is recorded at actual completion, not at routing decision time.
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
                        # Look up coalesce info for this branch
                        branch_name = child_token.branch_name
                        child_coalesce_name: CoalesceName | None = None
                        child_coalesce_step: int | None = None

                        if branch_name and BranchName(branch_name) in self._branch_to_coalesce:
                            child_coalesce_name = self._branch_to_coalesce[BranchName(branch_name)]
                            child_coalesce_step = self._coalesce_step_map[child_coalesce_name]

                        # Children skip directly to coalesce step (or next step if no coalesce)
                        child_items.append(
                            _WorkItem(
                                token=child_token,
                                start_step=child_coalesce_step if child_coalesce_step is not None else next_step,
                                coalesce_at_step=child_coalesce_step,
                                coalesce_name=child_coalesce_name,
                            )
                        )

                    # NOTE: Parent FORKED outcome is now recorded atomically in fork_token()
                    # to eliminate crash window between child creation and outcome recording.
                    return (
                        RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome=RowOutcome.FORKED,
                        ),
                        child_items,
                    )

                handled, result = self._maybe_coalesce_token(
                    current_token,
                    coalesce_at_step=coalesce_at_step,
                    coalesce_name=coalesce_name,
                    step_completed=step,
                    total_steps=total_steps,
                    child_items=child_items,
                )
                if handled:
                    return (result, child_items)

            # NOTE: BaseAggregation branch was DELETED in aggregation structural cleanup.
            # Aggregation is now handled by batch-aware transforms (is_batch_aware=True).
            # The engine buffers rows and calls Transform.process(rows: list[dict]).

            elif isinstance(transform, TransformProtocol):
                # Check if this is a batch-aware transform at an aggregation node
                node_id = transform.node_id
                if transform.is_batch_aware and node_id is not None and node_id in self._aggregation_settings:
                    # Use engine buffering for aggregation
                    return self._process_batch_aggregation_node(
                        transform=transform,
                        current_token=current_token,
                        ctx=ctx,
                        step=step,
                        child_items=child_items,
                        total_steps=total_steps,
                        coalesce_at_step=coalesce_at_step,
                        coalesce_name=coalesce_name,
                    )

                # Regular transform (with optional retry)
                try:
                    transform_result, current_token, error_sink = self._execute_transform_with_retry(
                        transform=transform,
                        token=current_token,
                        ctx=ctx,
                        step=step,
                    )
                    # Emit TransformCompleted telemetry AFTER Landscape recording succeeds
                    # (Landscape recording happens inside _execute_transform_with_retry)
                    self._emit_transform_completed(
                        token=current_token,
                        transform=transform,
                        transform_result=transform_result,
                    )
                except MaxRetriesExceeded as e:
                    # All retries exhausted - return FAILED outcome
                    error_hash = hashlib.sha256(str(e).encode()).hexdigest()[:16]
                    self._recorder.record_token_outcome(
                        run_id=self._run_id,
                        token_id=current_token.token_id,
                        outcome=RowOutcome.FAILED,
                        error_hash=error_hash,
                    )
                    # Emit TokenCompleted telemetry AFTER Landscape recording
                    self._emit_token_completed(current_token, RowOutcome.FAILED)
                    return (
                        RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome=RowOutcome.FAILED,
                            error=FailureInfo.from_max_retries_exceeded(e),
                        ),
                        child_items,
                    )

                if transform_result.status == "error":
                    # Determine outcome based on error routing
                    if error_sink == "discard":
                        # Intentionally discarded - QUARANTINED
                        error_detail = str(transform_result.reason) if transform_result.reason else "unknown_error"
                        quarantine_error_hash = hashlib.sha256(error_detail.encode()).hexdigest()[:16]
                        self._recorder.record_token_outcome(
                            run_id=self._run_id,
                            token_id=current_token.token_id,
                            outcome=RowOutcome.QUARANTINED,
                            error_hash=quarantine_error_hash,
                        )
                        # Emit TokenCompleted telemetry AFTER Landscape recording
                        self._emit_token_completed(current_token, RowOutcome.QUARANTINED)
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
                        # NOTE: Do NOT record ROUTED outcome here - the token hasn't been written yet.
                        # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
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
                if transform_result.is_multi_row:
                    # Validate transform is allowed to create tokens
                    if not transform.creates_tokens:
                        raise RuntimeError(
                            f"Transform '{transform.name}' returned multi-row result "
                            f"but has creates_tokens=False. Either set creates_tokens=True "
                            f"or return single row via TransformResult.success(row). "
                            f"(Multi-row is allowed in aggregation passthrough mode.)"
                        )

                    # Deaggregation: create child tokens for each output row
                    # transform_result.rows is guaranteed non-None when is_multi_row is True
                    # NOTE: Parent EXPANDED outcome is recorded atomically in expand_token()
                    child_tokens, _expand_group_id = self._token_manager.expand_token(
                        parent_token=current_token,
                        expanded_rows=transform_result.rows,  # type: ignore[arg-type]
                        step_in_pipeline=step,
                        run_id=self._run_id,
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

                    # NOTE: Parent EXPANDED outcome is recorded atomically in expand_token()
                    # to eliminate crash window between child creation and outcome recording.
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
                handled, result = self._maybe_coalesce_token(
                    current_token,
                    coalesce_at_step=coalesce_at_step,
                    coalesce_name=coalesce_name,
                    step_completed=step,
                    total_steps=total_steps,
                    child_items=child_items,
                )
                if handled:
                    return (result, child_items)

            else:
                raise TypeError(f"Unknown transform type: {type(transform).__name__}. Expected BaseTransform or BaseGate.")

        # Process config-driven gates (after all plugin transforms)
        # Step continues from where transforms left off
        config_gate_start_step = len(transforms) + 1

        # Calculate which config gate to start from based on start_step
        # If start_step > len(transforms), we skip some config gates
        # (e.g., fork children that already passed through earlier gates)
        config_gate_start_idx = max(0, start_step - len(transforms))
        for gate_idx, gate_config in enumerate(self._config_gates[config_gate_start_idx:], start=config_gate_start_idx):
            step = config_gate_start_step + gate_idx

            # Get the node_id for this config gate
            node_id = self._config_gate_id_map[GateName(gate_config.name)]

            outcome = self._gate_executor.execute_config_gate(
                gate_config=gate_config,
                node_id=node_id,
                token=current_token,
                ctx=ctx,
                step_in_pipeline=step,
                token_manager=self._token_manager,
            )
            current_token = outcome.updated_token

            # Emit GateEvaluated telemetry AFTER Landscape recording succeeds
            # (Landscape recording happens inside execute_config_gate)
            self._emit_gate_evaluated(
                token=current_token,
                gate_name=gate_config.name,
                gate_node_id=node_id,
                routing_mode=outcome.result.action.mode,
                destinations=self._get_gate_destinations(outcome),
            )

            # Check if gate routed to a sink
            if outcome.sink_name is not None:
                # NOTE: Do NOT record ROUTED outcome here - the token hasn't been written yet.
                # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
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
                    # Look up coalesce info for this branch
                    cfg_branch_name = child_token.branch_name
                    cfg_coalesce_name: CoalesceName | None = None
                    cfg_coalesce_step: int | None = None

                    if cfg_branch_name and BranchName(cfg_branch_name) in self._branch_to_coalesce:
                        cfg_coalesce_name = self._branch_to_coalesce[BranchName(cfg_branch_name)]
                        cfg_coalesce_step = self._coalesce_step_map[cfg_coalesce_name]

                    # Children skip directly to coalesce step (or next config gate if no coalesce)
                    child_items.append(
                        _WorkItem(
                            token=child_token,
                            start_step=cfg_coalesce_step if cfg_coalesce_step is not None else len(transforms) + next_config_step,
                            coalesce_at_step=cfg_coalesce_step,
                            coalesce_name=cfg_coalesce_name,
                        )
                    )

                # NOTE: Parent FORKED outcome is now recorded atomically in fork_token()
                # to eliminate crash window between child creation and outcome recording.
                return (
                    RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome=RowOutcome.FORKED,
                    ),
                    child_items,
                )

            handled, result = self._maybe_coalesce_token(
                current_token,
                coalesce_at_step=coalesce_at_step,
                coalesce_name=coalesce_name,
                step_completed=step,
                total_steps=total_steps,
                child_items=child_items,
            )
            if handled:
                return (result, child_items)

        # Final coalesce check after all transforms/gates
        # Use total_steps + 1 to match coalesce_at_step calculation (which is base_step + 1 + i)
        # This ensures step_completed >= coalesce_at_step when all processing is done
        final_step = total_steps + 1 if coalesce_at_step is not None else total_steps
        handled, result = self._maybe_coalesce_token(
            current_token,
            coalesce_at_step=coalesce_at_step,
            coalesce_name=coalesce_name,
            step_completed=final_step,
            total_steps=total_steps,
            child_items=child_items,
        )
        if handled:
            return (result, child_items)

        # COMPLETED outcome now recorded in orchestrator with sink_name (AUD-001)
        # Orchestrator knows branch→sink mapping, processor does not.
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.COMPLETED,
            ),
            child_items,
        )
