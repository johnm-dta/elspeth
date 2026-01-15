# src/elspeth/engine/processor.py
"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation
- Aggregation handling
- Final outcome recording
"""

from dataclasses import dataclass
from typing import Any

from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo, TokenManager
from elspeth.plugins.context import PluginContext


@dataclass
class RowResult:
    """Result of processing a row through the pipeline."""

    token: TokenInfo  # Preserve full token identity, not just IDs
    final_data: dict[str, Any]
    outcome: str  # completed, routed, forked, consumed, failed
    sink_name: str | None = None  # Set when outcome is "routed"

    @property
    def token_id(self) -> str:
        return self.token.token_id

    @property
    def row_id(self) -> str:
        return self.token.row_id


class RowProcessor:
    """Processes rows through the transform pipeline.

    Handles:
    1. Creating initial tokens from source rows
    2. Executing transforms in sequence
    3. Evaluating gates for routing decisions
    4. Accepting rows into aggregations
    5. Recording final outcomes

    Example:
        processor = RowProcessor(recorder, span_factory, run_id, source_node_id)

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
    ) -> None:
        """Initialize processor.

        Args:
            recorder: Landscape recorder
            span_factory: Span factory for tracing
            run_id: Current run ID
            source_node_id: Source node ID
            edge_map: Map of (node_id, label) -> edge_id
            route_resolution_map: Map of (node_id, label) -> "continue" | sink_name
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id = source_node_id

        self._token_manager = TokenManager(recorder)
        self._transform_executor = TransformExecutor(recorder, span_factory)
        self._gate_executor = GateExecutor(
            recorder, span_factory, edge_map, route_resolution_map
        )
        self._aggregation_executor = AggregationExecutor(recorder, span_factory, run_id)

    def process_row(
        self,
        row_index: int,
        row_data: dict[str, Any],
        transforms: list[Any],
        ctx: PluginContext,
    ) -> RowResult:
        """Process a row through all transforms.

        NOTE: This implementation handles LINEAR pipelines only. For DAG support
        (fork/join), this needs a work queue that processes child tokens from forks.
        Currently fork_to_paths returns "forked" and the caller must handle the children.

        Args:
            row_index: Position in source
            row_data: Initial row data
            transforms: List of transform plugins
            ctx: Plugin context

        Returns:
            RowResult with final outcome
        """
        # Create initial token
        token = self._token_manager.create_initial_token(
            run_id=self._run_id,
            source_node_id=self._source_node_id,
            row_index=row_index,
            row_data=row_data,
        )

        with self._spans.row_span(token.row_id, token.token_id):
            current_token = token

            for step, transform in enumerate(transforms, start=1):
                # Check transform type and execute accordingly
                if hasattr(transform, "evaluate"):
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
                        return RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome="routed",
                            sink_name=outcome.sink_name,
                        )
                    elif outcome.result.action.kind == "fork_to_paths":
                        # NOTE: For full DAG support, we'd push child_tokens to a work queue
                        # and continue processing them. For now, return "forked".
                        return RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome="forked",
                        )

                elif hasattr(transform, "accept"):
                    # Aggregation transform
                    accept_result = self._aggregation_executor.accept(
                        aggregation=transform,
                        token=current_token,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

                    if accept_result.trigger:
                        self._aggregation_executor.flush(
                            aggregation=transform,
                            ctx=ctx,
                            trigger_reason="threshold",
                            step_in_pipeline=step,
                        )

                    return RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome="consumed",
                    )

                else:
                    # Regular transform
                    result, current_token = self._transform_executor.execute_transform(
                        transform=transform,
                        token=current_token,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

                    if result.status == "error":
                        return RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome="failed",
                        )

            return RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome="completed",
            )
