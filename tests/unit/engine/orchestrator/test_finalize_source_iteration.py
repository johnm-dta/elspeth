# tests/unit/engine/orchestrator/test_finalize_source_iteration.py
"""Tests for _finalize_source_iteration context restoration.

Verifies that post-loop finalization restores BOTH ctx.node_id and
ctx.operation_id to source-scoped values, preventing audit misattribution.
"""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import MagicMock, patch

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import NodeID
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
    LoopContext,
)
from tests.fixtures.landscape import make_landscape_db


def _make_orchestrator() -> Orchestrator:
    """Create an Orchestrator with minimal dependencies."""
    return Orchestrator(make_landscape_db())


def _make_loop_ctx(ctx: PluginContext, *, coalesce_executor: MagicMock | None = None) -> LoopContext:
    """Build minimal LoopContext for finalization tests.

    Config has empty aggregation_settings and sinks, and coalesce_executor
    is None — so aggregation/coalesce flush branches are no-ops on both
    the interrupted and normal-exit paths. Processor is a MagicMock.
    """
    config = MagicMock()
    config.aggregation_settings = {}
    config.sinks = {}
    processor = MagicMock()
    processor.get_aggregation_buffer_count.return_value = 0
    return LoopContext(
        counters=ExecutionCounters(),
        pending_tokens={},
        processor=processor,
        ctx=ctx,
        config=config,
        agg_transform_lookup=MappingProxyType({}),
        coalesce_executor=coalesce_executor,
        coalesce_node_map=MappingProxyType({}),
    )


class TestFinalizeSourceIterationContext:
    """Verify _finalize_source_iteration restores both node_id and operation_id."""

    def test_node_id_restored_after_shutdown_interrupt(self) -> None:
        """ctx.node_id must be restored to source_id after shutdown break.

        Bug: _finalize_source_iteration restores ctx.operation_id but not
        ctx.node_id. After the iteration loop, ctx.node_id still points
        to whatever the last transform was. Post-loop audit operations
        are then misattributed to the wrong node.
        """
        source_id = NodeID("source-node-001")
        source_operation_id = "op-source-load-001"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform-residue-should-be-overwritten",
            operation_id=None,
        )

        orchestrator = _make_orchestrator()
        loop_ctx = _make_loop_ctx(ctx)

        orchestrator._finalize_source_iteration(
            loop_ctx,
            factory=MagicMock(),
            run_id="test-run",
            source_id=source_id,
            source_operation_id=source_operation_id,
            field_resolution_recorded=True,
            schema_contract_recorded=True,
            interrupted_by_shutdown=True,
            flush_end_of_input=False,
            active_source=MagicMock(),
        )

        assert ctx.node_id == source_id, (
            f"ctx.node_id should be '{source_id}' but was '{ctx.node_id}' — audit operations after source iteration would be misattributed"
        )
        assert ctx.operation_id == source_operation_id

    def test_node_id_restored_on_normal_exit(self) -> None:
        """ctx.node_id must also be restored on normal (non-interrupted) exit."""
        source_id = NodeID("source-node-002")
        source_operation_id = "op-source-load-002"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform-residue",
            operation_id=None,
        )

        orchestrator = _make_orchestrator()
        loop_ctx = _make_loop_ctx(ctx)

        orchestrator._finalize_source_iteration(
            loop_ctx,
            factory=MagicMock(),
            run_id="test-run",
            source_id=source_id,
            source_operation_id=source_operation_id,
            field_resolution_recorded=True,
            schema_contract_recorded=True,
            interrupted_by_shutdown=False,
            flush_end_of_input=True,
            active_source=MagicMock(),
        )

        assert ctx.node_id == source_id
        assert ctx.operation_id == source_operation_id

    def test_coalesce_flush_is_skipped_for_source_local_finalization(self) -> None:
        """Multi-source source completion is not global end-of-input."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform-residue",
            operation_id=None,
        )
        orchestrator = _make_orchestrator()
        coalesce_executor = MagicMock()
        loop_ctx = _make_loop_ctx(ctx, coalesce_executor=coalesce_executor)

        with patch("elspeth.engine.orchestrator.core.flush_coalesce_pending") as flush_coalesce:
            orchestrator._finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-orders"),
                source_operation_id="op-source-load-orders",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=False,
                active_source=MagicMock(),
            )

        flush_coalesce.assert_not_called()

    def test_coalesce_flush_runs_at_true_end_of_input(self) -> None:
        """Final source completion owns run-global coalesce flushing."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform-residue",
            operation_id=None,
        )
        orchestrator = _make_orchestrator()
        coalesce_executor = MagicMock()
        loop_ctx = _make_loop_ctx(ctx, coalesce_executor=coalesce_executor)

        with patch("elspeth.engine.orchestrator.core.flush_coalesce_pending") as flush_coalesce:
            orchestrator._finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-refunds"),
                source_operation_id="op-source-load-refunds",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=True,
                active_source=MagicMock(),
            )

        flush_coalesce.assert_called_once()

    def test_aggregation_flush_is_skipped_for_source_local_finalization(self) -> None:
        """Multi-source source completion must not flush shared aggregations."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform-residue",
            operation_id=None,
        )
        orchestrator = _make_orchestrator()
        loop_ctx = _make_loop_ctx(ctx)
        loop_ctx.config.aggregation_settings = {"aggregation_total_amounts": MagicMock()}

        with patch("elspeth.engine.orchestrator.core.flush_remaining_aggregation_buffers") as flush_aggregation:
            orchestrator._finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-orders"),
                source_operation_id="op-source-load-orders",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=False,
                active_source=MagicMock(),
            )

        flush_aggregation.assert_not_called()

    def test_aggregation_flush_runs_at_true_end_of_input(self) -> None:
        """Final source completion owns run-global aggregation flushing."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform-residue",
            operation_id=None,
        )
        orchestrator = _make_orchestrator()
        loop_ctx = _make_loop_ctx(ctx)
        loop_ctx.config.aggregation_settings = {"aggregation_total_amounts": MagicMock()}

        with patch("elspeth.engine.orchestrator.core.flush_remaining_aggregation_buffers") as flush_aggregation:
            orchestrator._finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-refunds"),
                source_operation_id="op-source-load-refunds",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=True,
                active_source=MagicMock(),
            )

        flush_aggregation.assert_called_once()
        loop_ctx.processor.get_aggregation_buffer_count.assert_called_once_with(NodeID("aggregation_total_amounts"))
