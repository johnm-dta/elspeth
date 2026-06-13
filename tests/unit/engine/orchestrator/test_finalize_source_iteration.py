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


def _make_active_source() -> MagicMock:
    """Active-source stand-in that satisfies _record_run_source_lifecycle.

    The exhausted, non-interrupted finalization path records source lifecycle,
    which probes ``get_field_resolution()`` (unpacked as a 2-tuple), serialises
    ``output_schema.model_json_schema()`` via ``json.dumps``, and hashes
    ``config`` via ``stable_hash``. A bare MagicMock yields un-unpackable /
    un-serialisable Mocks, so configure the minimum real-shaped returns. These
    tests assert on context restoration and flush gating — not on the recorded
    lifecycle — so empty/None evidence is sufficient and honest.
    """
    source = MagicMock()
    source.get_field_resolution.return_value = None
    source.output_schema.model_json_schema.return_value = {}
    source.config = {}
    return source


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
    # Slice 3 (ADR-030 §D): the EOF flush helper gates on journal quiescence,
    # runs a journal-first intake pass and loops until no BLOCKED holds remain.
    processor.count_unquiesced_scheduler_work.return_value = 0
    processor.run_barrier_intake.return_value = []
    processor.has_blocked_barrier_work.return_value = False
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

        orchestrator._source_driver.finalize_source_iteration(
            loop_ctx,
            factory=MagicMock(),
            run_id="test-run",
            source_id=source_id,
            active_source_name="source-node-001",
            source_operation_id=source_operation_id,
            field_resolution_recorded=True,
            schema_contract_recorded=True,
            source_exhausted=False,
            interrupted_by_shutdown=True,
            flush_end_of_input=False,
            active_source=_make_active_source(),
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

        orchestrator._source_driver.finalize_source_iteration(
            loop_ctx,
            factory=MagicMock(),
            run_id="test-run",
            source_id=source_id,
            active_source_name="source-node-002",
            source_operation_id=source_operation_id,
            field_resolution_recorded=True,
            schema_contract_recorded=True,
            source_exhausted=True,
            interrupted_by_shutdown=False,
            flush_end_of_input=True,
            active_source=_make_active_source(),
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

        # Slice 3 re-pin: the EOF flush moved behind run_end_of_input_barrier_flush
        # (orchestrator/aggregation.py), so the seam lives in that module now.
        with patch("elspeth.engine.orchestrator.aggregation.flush_coalesce_pending") as flush_coalesce:
            orchestrator._source_driver.finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-orders"),
                active_source_name="orders",
                source_operation_id="op-source-load-orders",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                source_exhausted=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=False,
                active_source=_make_active_source(),
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

        # Slice 3 re-pin: the EOF flush moved behind run_end_of_input_barrier_flush
        # (orchestrator/aggregation.py), so the seam lives in that module now.
        with patch("elspeth.engine.orchestrator.aggregation.flush_coalesce_pending") as flush_coalesce:
            orchestrator._source_driver.finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-refunds"),
                active_source_name="refunds",
                source_operation_id="op-source-load-refunds",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                source_exhausted=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=True,
                active_source=_make_active_source(),
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

        # Slice 3 re-pin: the EOF flush moved behind run_end_of_input_barrier_flush
        # (orchestrator/aggregation.py), so the seam lives in that module now.
        with patch("elspeth.engine.orchestrator.aggregation.flush_remaining_aggregation_buffers") as flush_aggregation:
            orchestrator._source_driver.finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-orders"),
                active_source_name="orders",
                source_operation_id="op-source-load-orders",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                source_exhausted=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=False,
                active_source=_make_active_source(),
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

        # Slice 3 re-pin: the EOF flush moved behind run_end_of_input_barrier_flush
        # (orchestrator/aggregation.py), so the seam lives in that module now.
        with patch("elspeth.engine.orchestrator.aggregation.flush_remaining_aggregation_buffers") as flush_aggregation:
            flush_aggregation.return_value = ExecutionCounters().to_flush_result()
            orchestrator._source_driver.finalize_source_iteration(
                loop_ctx,
                factory=MagicMock(),
                run_id="test-run",
                source_id=NodeID("source-refunds"),
                active_source_name="refunds",
                source_operation_id="op-source-load-refunds",
                field_resolution_recorded=True,
                schema_contract_recorded=True,
                source_exhausted=True,
                interrupted_by_shutdown=False,
                flush_end_of_input=True,
                active_source=_make_active_source(),
            )

        flush_aggregation.assert_called_once()
        loop_ctx.processor.get_aggregation_buffer_count.assert_called_once_with(NodeID("aggregation_total_amounts"))
