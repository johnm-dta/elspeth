# tests/integration/pipeline/test_aggregation_checkpoint_bug.py
"""
FIX VERIFICATION: elspeth-rapid-nsj
TITLE: Aggregation checkpoint state saved in production

This test VERIFIES the fix for elspeth-rapid-nsj:
- Orchestrator._maybe_checkpoint() now calls processor.get_aggregation_checkpoint_state()
- The aggregation state is passed to create_checkpoint()
- Checkpoints contain aggregation_state_json for crash recovery

The fix ensures buffered rows in aggregation nodes are not lost on crash.

Migrated from tests/integration/test_aggregation_checkpoint_bug_reproduction.py.
"""

from __future__ import annotations

from typing import Any

from elspeth.contracts import (
    Determinism,
    FieldContract,
    PipelineRow,
    RunStatus,
    SchemaContract,
)
from elspeth.core.config import (
    AggregationSettings,
    CheckpointSettings,
    ElspethSettings,
    SinkSettings,
    SourceSettings,
    TriggerConfig,
)
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import (
    CallbackSource,
    _TestSchema,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink


class BatchCollectorTransform(BaseTransform):
    """Batch-aware transform that collects rows until flush."""

    name = "batch_collector"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success: str | None = "output"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            # Batch mode - aggregate
            total = sum(r.get("value", 0) for r in row)
            output = {"id": row[0].get("id"), "value": total, "count": len(row)}

            # Provide contract for output (adds "count" field)
            contract = SchemaContract(
                mode="OBSERVED",
                fields=(
                    FieldContract(normalized_name="id", original_name="id", python_type=int, required=False, source="inferred"),
                    FieldContract(normalized_name="value", original_name="value", python_type=int, required=False, source="inferred"),
                    FieldContract(normalized_name="count", original_name="count", python_type=int, required=False, source="inferred"),
                ),
                locked=True,
            )

            return TransformResult.success(
                PipelineRow(output, contract),
                success_reason={"action": "batch"},
            )
        else:
            # Single row - passthrough
            return TransformResult.success(make_pipeline_row(dict(row)), success_reason={"action": "single"})


class TestAggregationCheckpointFixVerification:
    """
    FIX VERIFICATION: Verifies aggregation state IS saved to checkpoints.

    These tests verify the fix for elspeth-rapid-nsj where aggregation
    checkpoint state was never saved during normal pipeline execution.
    """

    def test_checkpoint_includes_aggregation_state(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """
        VERIFIES FIX: create_checkpoint() is called WITH aggregation_state

        Steps:
        1. Create pipeline with aggregation (count trigger = 10, high threshold)
        2. Process 5 rows (less than trigger, so rows stay buffered)
        3. Use mock to capture create_checkpoint() calls
        4. VERIFY: aggregation_state is passed to create_checkpoint()
        """
        # Source with 5 rows (less than count trigger of 10)
        callback_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},
                {"id": 2, "value": 200},
                {"id": 3, "value": 300},
                {"id": 4, "value": 400},
                {"id": 5, "value": 500},
            ],
            output_schema=_TestSchema,
            source_name="buffering_source",
            on_success="source_out",
        )
        source = as_source(callback_source)

        transform = as_transform(BatchCollectorTransform())
        collecting_sink = CollectSink(name="collecting_sink")
        sink = as_sink(collecting_sink)

        # Build graph
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=wire_transforms([transform], source_connection="source_out", final_sink="output"),
            sinks={"output": sink},
            aggregations={},
            gates=[],
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        # High count trigger so rows stay buffered
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_collector",
            input="source_out",
            on_error="discard",
            trigger=TriggerConfig(
                count=10,  # High count - won't trigger during 5 rows
                timeout_seconds=3600,  # 1 hour - won't trigger
            ),
            output_mode="transform",
        )

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={
                transform_node_id: agg_settings,
            },
            coalesce_settings=[],
        )

        # Enable checkpoint every row
        checkpoint_settings = CheckpointSettings(
            enabled=True,
            frequency="every_row",
        )

        settings = ElspethSettings(
            sources={"primary": SourceSettings(plugin="buffering_source", on_success="source_out", options={})},
            sinks={"output": SinkSettings(plugin="collecting_sink", on_write_failure="discard", options={})},
            transforms=[],
            gates=[],
            checkpoint=checkpoint_settings,
        )

        # Create checkpoint manager and config
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager(landscape_db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)

        # Track create_checkpoint calls
        checkpoint_calls: list[dict[str, Any]] = []
        original_create_checkpoint = checkpoint_mgr.create_checkpoint

        def capture_create_checkpoint(*args, **kwargs):
            """Capture the arguments passed to create_checkpoint."""
            checkpoint_calls.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "aggregation_state": kwargs.get("aggregation_state"),
                }
            )
            # Call the original
            return original_create_checkpoint(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = capture_create_checkpoint  # type: ignore[method-assign]

        # Run pipeline with checkpointing enabled
        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Verify run completed
        assert result.status == RunStatus.COMPLETED, f"Run failed: {result}"

        # Verify checkpoints were created
        assert len(checkpoint_calls) > 0, "No checkpoint calls captured"

        # FIX VERIFICATION:
        # All create_checkpoint calls should have aggregation_state
        # because Orchestrator._maybe_checkpoint() now passes aggregation state
        calls_with_agg_state = [call for call in checkpoint_calls if call["aggregation_state"] is not None]

        # FIX VERIFICATION: This assertion PASSES when fix is applied
        assert len(calls_with_agg_state) > 0, (
            f"FIX NOT WORKING: Found {len(calls_with_agg_state)} checkpoint calls WITH aggregation_state (expected > 0 after fix)."
        )

        # Verify the aggregation state has expected structure
        for call in calls_with_agg_state:
            agg_state = call["aggregation_state"]
            assert hasattr(agg_state, "version"), "Aggregation state should have version field"

    def test_aggregation_only_frequency_creates_checkpoints(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """
        FIX VERIFICATION: frequency=aggregation_only (0) must NOT be a no-op.

        Before the fix, _maybe_checkpoint skipped unconditionally when
        frequency == 0, leaving aggregation_only mode with zero resume
        points. Post-fix, frequency == 0 checkpoints on every post-sink
        callback (I/O reduction is inherent via aggregation cardinality).
        """
        # Source with 3 rows — count trigger at 2 forces one flush mid-stream
        callback_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},
                {"id": 2, "value": 200},
                {"id": 3, "value": 300},
            ],
            output_schema=_TestSchema,
            source_name="agg_only_source",
            on_success="source_out",
        )
        source = as_source(callback_source)

        transform = as_transform(BatchCollectorTransform())
        collecting_sink = CollectSink(name="collecting_sink")
        sink = as_sink(collecting_sink)

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=wire_transforms([transform], source_connection="source_out", final_sink="output"),
            sinks={"output": sink},
            aggregations={},
            gates=[],
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_collector",
            input="source_out",
            on_error="discard",
            trigger=TriggerConfig(
                count=2,
                timeout_seconds=3600,
            ),
            output_mode="transform",
        )

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={
                transform_node_id: agg_settings,
            },
            coalesce_settings=[],
        )

        # aggregation_only mode — this was the broken code path
        checkpoint_settings = CheckpointSettings(
            enabled=True,
            frequency="aggregation_only",
        )

        settings = ElspethSettings(
            sources={"primary": SourceSettings(plugin="agg_only_source", on_success="source_out", options={})},
            sinks={"output": SinkSettings(plugin="collecting_sink", on_write_failure="discard", options={})},
            transforms=[],
            gates=[],
            checkpoint=checkpoint_settings,
        )

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager(landscape_db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)
        assert checkpoint_config.frequency == 0, "aggregation_only should map to frequency=0"

        checkpoint_calls: list[dict[str, Any]] = []
        original_create_checkpoint = checkpoint_mgr.create_checkpoint

        def capture_create_checkpoint(*args, **kwargs):
            checkpoint_calls.append(kwargs)
            return original_create_checkpoint(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = capture_create_checkpoint  # type: ignore[method-assign]

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        # With count=2 and 3 rows: one flush at row 2, end-of-source flush for row 3.
        # Both produce tokens that reach sinks and trigger checkpoint_after_sink.
        assert len(checkpoint_calls) > 0, (
            "aggregation_only mode must create checkpoints — frequency=0 was previously a no-op in _maybe_checkpoint"
        )
