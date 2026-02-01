# tests/engine/test_processor_batch.py
"""Tests for batch-aware transforms and deaggregation in RowProcessor.

This module tests:
- Batch-aware transforms with aggregation configuration
- Buffer management and flush triggers
- Checkpoint restoration for aggregation state
- Deaggregation / multi-row output handling (expanding transforms)

Test plugins inherit from base classes (BaseTransform)
because the processor uses isinstance() for type-safe plugin detection.
"""

from typing import Any

from elspeth.contracts import NodeType
from elspeth.contracts.types import NodeID
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    RowOutcome,
    TransformResult,
)
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema


class TestProcessorBatchTransforms:
    """Tests for batch-aware transforms in RowProcessor."""

    def test_processor_buffers_rows_for_aggregation_node(self) -> None:
        """Processor buffers rows at aggregation nodes and flushes on trigger."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class SumTransform(BaseTransform):
            name = "sum"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    total = sum(r["value"] for r in rows)
                    return TransformResult.success({"total": total}, success_reason={"action": "test"})
                return TransformResult.success(rows, success_reason={"action": "test"})

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sum_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(sum_node.node_id): AggregationSettings(
                name="sum_batch",
                plugin="sum",
                trigger=TriggerConfig(count=3),
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            edge_map={},
            route_resolution_map={},
            aggregation_settings=aggregation_settings,
        )

        transform = SumTransform(sum_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 3 rows - should buffer first 2, flush on 3rd
        # Note: default output_mode is "transform", which means:
        # - All 3 original rows get CONSUMED_IN_BATCH
        # - The flush produces a NEW row with the aggregated result
        all_results = []
        for i in range(3):
            result_list = processor.process_row(
                row_index=i,
                row_data={"value": i + 1},  # 1, 2, 3
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(result_list)

        # In transform mode, all original rows are CONSUMED_IN_BATCH
        consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        assert len(consumed) == 3, f"Expected 3 consumed rows, got {len(consumed)}"

        # The flush produces a NEW aggregated row that gets COMPLETED
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 1, f"Expected 1 completed row, got {len(completed)}"
        assert completed[0].final_data == {"total": 6}

    def test_processor_batch_transform_without_aggregation_config(self) -> None:
        """Batch-aware transform without aggregation config uses single-row mode."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True  # But no aggregation config
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    # Batch mode - sum all values
                    return TransformResult.success({"value": sum(r["value"] for r in rows)}, success_reason={"action": "test"})
                # Single-row mode - double
                return TransformResult.success({"value": rows["value"] * 2}, success_reason={"action": "test"})

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # No aggregation_settings - so batch-aware transform uses single-row mode
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            edge_map={},
            route_resolution_map={},
            aggregation_settings={},  # Empty - not an aggregation node
        )

        transform = DoubleTransform(transform_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process row - should use single-row mode (double)
        result_list = processor.process_row(
            row_index=0,
            row_data={"value": 5},
            transforms=[transform],
            ctx=ctx,
        )

        result = result_list[0]
        assert result.outcome == RowOutcome.COMPLETED
        assert result.final_data == {"value": 10}  # Doubled, not summed

    def test_processor_buffers_restored_on_recovery(self) -> None:
        """Processor restores buffer state from checkpoint."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class SumTransform(BaseTransform):
            name = "sum"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    total = sum(r["value"] for r in rows)
                    return TransformResult.success({"total": total}, success_reason={"action": "test"})
                return TransformResult.success(rows, success_reason={"action": "test"})

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sum_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(sum_node.node_id): AggregationSettings(
                name="sum_batch",
                plugin="sum",
                trigger=TriggerConfig(count=3),  # Trigger at 3
            ),
        }

        # Create rows and tokens that will be referenced by the checkpoint
        row0 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            row_index=0,
            data={"value": 1},
        )
        token0 = recorder.create_token(row_id=row0.row_id)
        row1 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            row_index=1,
            data={"value": 2},
        )
        token1 = recorder.create_token(row_id=row1.row_id)

        # Create the batch that will be restored from checkpoint
        old_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=sum_node.node_id,
        )

        # Simulate restored checkpoint with 2 rows already buffered
        # Note: _version field required since Bug #12 checkpoint versioning fix
        # Note: elapsed_age_seconds required since Bug #6 timeout SLA preservation fix
        # Note: fire_offset fields required since P2-2026-02-01 trigger ordering fix
        restored_buffer_state = {
            "_version": "1.1",
            sum_node.node_id: {
                "tokens": [
                    {
                        "token_id": token0.token_id,
                        "row_id": row0.row_id,
                        "row_data": {"value": 1},
                        "branch_name": None,
                    },
                    {
                        "token_id": token1.token_id,
                        "row_id": row1.row_id,
                        "row_data": {"value": 2},
                        "branch_name": None,
                    },
                ],
                "batch_id": old_batch.batch_id,
                "elapsed_age_seconds": 0.0,  # Bug #6: timeout elapsed time
                "count_fire_offset": None,  # P2-2026-02-01: trigger ordering
                "condition_fire_offset": None,  # P2-2026-02-01: trigger ordering
            },
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            edge_map={},
            route_resolution_map={},
            aggregation_settings=aggregation_settings,
        )

        # Restore buffer state
        processor._aggregation_executor.restore_from_checkpoint(restored_buffer_state)

        transform = SumTransform(sum_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 1 more row - should trigger flush (2 restored + 1 new = 3)
        # Note: default output_mode is "transform", which means:
        # - The triggering row gets CONSUMED_IN_BATCH
        # - The flush produces a NEW row with the aggregated result
        result_list = processor.process_row(
            row_index=2,
            row_data={"value": 3},  # Third value
            transforms=[transform],
            ctx=ctx,
        )

        # In transform mode: triggering row is CONSUMED_IN_BATCH,
        # plus a NEW aggregated row that is COMPLETED
        consumed = [r for r in result_list if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in result_list if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 1, f"Expected 1 consumed, got {len(consumed)}"
        assert len(completed) == 1, f"Expected 1 completed, got {len(completed)}"
        assert completed[0].final_data == {"total": 6}  # 1 + 2 + 3


class TestProcessorDeaggregation:
    """Tests for deaggregation / multi-row output handling."""

    def test_processor_handles_expanding_transform(self) -> None:
        """Processor creates multiple RowResults for expanding transform."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        class ExpanderTransform(BaseTransform):
            name = "expander"
            creates_tokens = True  # This is a deaggregation transform
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                # Expand each row into 2 rows
                return TransformResult.success_multi(
                    [
                        {**row, "copy": 1},
                        {**row, "copy": 2},
                    ],
                    success_reason={"action": "test"},
                )

        # Setup real recorder
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="expander",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = ExpanderTransform(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process a row through the expanding transform
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform],
            ctx=ctx,
        )

        # Should get 3 results: 1 EXPANDED parent + 2 COMPLETED children
        assert len(results) == 3

        # Find the parent (EXPANDED) and children (COMPLETED)
        expanded = [r for r in results if r.outcome == RowOutcome.EXPANDED]
        completed = [r for r in results if r.outcome == RowOutcome.COMPLETED]

        assert len(expanded) == 1
        assert len(completed) == 2

        # Children should have different token_ids but same row_id
        assert completed[0].token.token_id != completed[1].token.token_id
        assert completed[0].token.row_id == completed[1].token.row_id

        # Children should have the expanded data
        child_copies = {r.final_data["copy"] for r in completed}
        assert child_copies == {1, 2}

    def test_processor_rejects_multi_row_without_creates_tokens(self) -> None:
        """Processor raises error if transform returns multi-row but creates_tokens=False."""
        import pytest

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        class BadTransform(BaseTransform):
            name = "bad"
            creates_tokens = False  # NOT allowed to create new tokens
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success_multi([row, row], success_reason={"action": "test"})  # But returns multi!

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="bad",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = BadTransform(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Should raise because creates_tokens=False but returns multi-row
        with pytest.raises(RuntimeError, match="creates_tokens=False"):
            processor.process_row(
                row_index=0,
                row_data={"value": 1},
                transforms=[transform],
                ctx=ctx,
            )

    def test_aggregation_transform_returns_none_raises_contract_error(self) -> None:
        """Aggregation transform returning None for result.row raises RuntimeError.

        This tests the contract enforcement added in P3-2026-01-28 bug fix.
        Batch-aware transforms MUST return a row via TransformResult.success(row).
        Defensive {} substitution is forbidden per CLAUDE.md's no-bug-hiding policy.
        """
        import pytest

        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class NoneReturningTransform(BaseTransform):
            """Transform that violates contract by returning success with None row."""

            name = "bad_transform"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                # Bug in plugin: creates TransformResult with row=None (contract violation)
                # This should NOT be masked by defensive {} substitution
                result = TransformResult(
                    status="success",
                    row=None,
                    reason=None,
                    rows=None,
                    success_reason={"action": "test"},  # Required, but row=None is still the bug
                )
                return result

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="bad_transform",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Test with single output_mode (default)
        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="bad_agg",
                plugin="bad_transform",
                trigger=TriggerConfig(count=2),
                output_mode="transform",
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            aggregation_settings=aggregation_settings,
        )

        transform = NoneReturningTransform(agg_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process first row - buffered (not flushed yet)
        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[transform],
            ctx=ctx,
        )
        assert results[0].outcome == RowOutcome.CONSUMED_IN_BATCH

        # Process second row - triggers flush, should raise contract error
        # The error message comes from execute_flush() in executors.py
        with pytest.raises(RuntimeError, match="neither row nor rows contains data"):
            processor.process_row(
                row_index=1,
                row_data={"value": 2},
                transforms=[transform],
                ctx=ctx,
            )

    def test_aggregation_transform_mode_returns_none_raises_contract_error(self) -> None:
        """Aggregation transform returning None in 'transform' mode raises RuntimeError.

        This tests the contract enforcement for output_mode="transform" (vs "single" above).
        Both modes require output data - this verifies the transform mode path.
        """
        import pytest

        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class NoneReturningTransform(BaseTransform):
            """Transform that violates contract by returning success with None row."""

            name = "bad_transform_multi"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                # Bug in plugin: creates TransformResult with row=None (contract violation)
                result = TransformResult(
                    status="success",
                    row=None,
                    reason=None,
                    rows=None,
                    success_reason={"action": "test"},  # Required, but row=None is still the bug
                )
                return result

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="bad_transform_multi",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Test with transform output_mode (creates new tokens)
        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="bad_agg_transform",
                plugin="bad_transform_multi",
                trigger=TriggerConfig(count=2),
                output_mode="transform",  # Different from "single" test above
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            aggregation_settings=aggregation_settings,
        )

        transform = NoneReturningTransform(agg_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process first row - buffered
        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[transform],
            ctx=ctx,
        )
        assert results[0].outcome == RowOutcome.CONSUMED_IN_BATCH

        # Process second row - triggers flush, should raise contract error
        with pytest.raises(RuntimeError, match="neither row nor rows contains data"):
            processor.process_row(
                row_index=1,
                row_data={"value": 2},
                transforms=[transform],
                ctx=ctx,
            )
