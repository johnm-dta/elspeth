# tests/engine/test_processor_modes.py
"""Tests for aggregation output modes in RowProcessor.

This module contains tests for passthrough and transform output modes
in batch-aware transforms (aggregations). These tests verify:

- Passthrough mode: preserves token identity, enriches rows in place
- Transform mode: creates new tokens, allows N->M row transformations

Extracted from test_processor.py to improve test organization.
"""

from typing import Any

from elspeth.contracts.enums import NodeType
from elspeth.contracts.types import NodeID
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema


class TestProcessorPassthroughMode:
    """Tests for passthrough output_mode in aggregation."""

    def test_aggregation_passthrough_mode(self) -> None:
        """Passthrough mode: BUFFERED while waiting, COMPLETED on flush with same tokens."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class PassthroughEnricher(BaseTransform):
            """Enriches each row in a batch with batch stats, returns same number of rows."""

            name = "passthrough_enricher"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = False  # Passthrough: same tokens, no new ones
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    # Batch mode: enrich each row with batch_size
                    batch_size = len(rows)
                    enriched = [{**row, "batch_size": batch_size, "enriched": True} for row in rows]
                    return TransformResult.success_multi(enriched, success_reason={"action": "test"})
                # Single row mode
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
        enricher_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="passthrough_enricher",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(enricher_node.node_id): AggregationSettings(
                name="batch_enrich",
                plugin="passthrough_enricher",
                trigger=TriggerConfig(count=3),
                output_mode="passthrough",  # KEY: passthrough mode
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

        transform = PassthroughEnricher(enricher_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Collect results for all 3 rows
        all_results = []
        buffered_token_ids = []

        for i in range(3):
            result_list = processor.process_row(
                row_index=i,
                row_data={"value": i + 1},  # 1, 2, 3
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(result_list)

            # Track buffered tokens (first 2 rows)
            if i < 2:
                assert len(result_list) == 1
                assert result_list[0].outcome == RowOutcome.BUFFERED
                buffered_token_ids.append(result_list[0].token.token_id)

        # After 3rd row, should have:
        # - 2 BUFFERED from first 2 rows
        # - 3 COMPLETED from flush (preserving original token_ids)
        buffered = [r for r in all_results if r.outcome == RowOutcome.BUFFERED]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        assert len(buffered) == 2, f"Expected 2 BUFFERED, got {len(buffered)}"
        assert len(completed) == 3, f"Expected 3 COMPLETED, got {len(completed)}"

        # CRITICAL: Passthrough preserves token_ids
        # The buffered tokens should reappear in completed results
        completed_token_ids = {r.token.token_id for r in completed}
        for token_id in buffered_token_ids:
            assert token_id in completed_token_ids, f"Buffered token {token_id} not found in completed results"

        # All completed rows should be enriched
        for result in completed:
            assert result.final_data["enriched"] is True
            assert result.final_data["batch_size"] == 3

        # Original values should be preserved
        values = {r.final_data["value"] for r in completed}
        assert values == {1, 2, 3}

    def test_aggregation_passthrough_validates_row_count(self) -> None:
        """Passthrough mode raises error if transform returns wrong row count."""
        import pytest

        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class BadPassthrough(BaseTransform):
            """Returns wrong number of rows in passthrough mode."""

            name = "bad_passthrough"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = False
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    # Wrong: returns fewer rows than input
                    return TransformResult.success_multi([rows[0]], success_reason={"action": "test"})
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
        bad_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="bad_passthrough",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(bad_node.node_id): AggregationSettings(
                name="bad_batch",
                plugin="bad_passthrough",
                trigger=TriggerConfig(count=3),
                output_mode="passthrough",
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

        transform = BadPassthrough(bad_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process first 2 rows (buffered)
        processor.process_row(row_index=0, row_data={"value": 1}, transforms=[transform], ctx=ctx)
        processor.process_row(row_index=1, row_data={"value": 2}, transforms=[transform], ctx=ctx)

        # 3rd row triggers flush - should fail because transform returns 1 row instead of 3
        with pytest.raises(ValueError, match="same number of output rows"):
            processor.process_row(row_index=2, row_data={"value": 3}, transforms=[transform], ctx=ctx)

    def test_aggregation_passthrough_continues_to_next_transform(self) -> None:
        """Passthrough mode rows continue through remaining transforms after flush."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class PassthroughEnricher(BaseTransform):
            """Enriches each row in a batch."""

            name = "enricher"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = False
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    enriched = [{**row, "batch_enriched": True} for row in rows]
                    return TransformResult.success_multi(enriched, success_reason={"action": "test"})
                return TransformResult.success(rows, success_reason={"action": "test"})

        class DoubleTransform(BaseTransform):
            """Doubles the value field."""

            name = "double"
            input_schema = _TestSchema
            output_schema = _TestSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "value": row["value"] * 2}, success_reason={"action": "test"})

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
        enricher_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        double_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(enricher_node.node_id): AggregationSettings(
                name="batch_enrich",
                plugin="enricher",
                trigger=TriggerConfig(count=2),
                output_mode="passthrough",
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

        enricher = PassthroughEnricher(enricher_node.node_id)
        doubler = DoubleTransform(double_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 2 rows through enricher (passthrough) then doubler
        all_results = []
        for i in range(2):
            result_list = processor.process_row(
                row_index=i,
                row_data={"value": i + 1},  # 1, 2
                transforms=[enricher, doubler],
                ctx=ctx,
            )
            all_results.extend(result_list)

        # First row buffered, second triggers flush
        # After flush, both rows go through doubler
        buffered = [r for r in all_results if r.outcome == RowOutcome.BUFFERED]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        assert len(buffered) == 1
        assert len(completed) == 2

        # Both completed rows should have batch_enriched AND doubled values
        for result in completed:
            assert result.final_data["batch_enriched"] is True

        # Values should be doubled: 1*2=2, 2*2=4
        values = {r.final_data["value"] for r in completed}
        assert values == {2, 4}


class TestProcessorTransformMode:
    """Tests for transform output_mode in aggregation."""

    def test_aggregation_transform_mode(self) -> None:
        """Transform mode returns M rows from N input rows with new tokens."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class GroupSplitter(BaseTransform):
            """Splits batch into groups, outputs one row per group."""

            name = "splitter"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = True  # Transform mode creates new tokens
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    # Group by 'category' and output one row per group
                    groups: dict[str, dict[str, Any]] = {}
                    for row in rows:
                        cat = row.get("category", "default")
                        if cat not in groups:
                            groups[cat] = {"category": cat, "count": 0, "total": 0}
                        groups[cat]["count"] += 1
                        groups[cat]["total"] += row.get("value", 0)
                    return TransformResult.success_multi(list(groups.values()), success_reason={"action": "test"})
                # Single row mode - not used in this test
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
        splitter_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(splitter_node.node_id): AggregationSettings(
                name="group_split",
                plugin="splitter",
                trigger=TriggerConfig(count=5),
                output_mode="transform",  # KEY: transform mode
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

        transform = GroupSplitter(splitter_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 5 rows with 2 categories (A and B)
        test_rows = [
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
            {"category": "A", "value": 30},
            {"category": "B", "value": 40},
            {"category": "A", "value": 50},
        ]

        all_results = []
        for i, row_data in enumerate(test_rows):
            results = processor.process_row(
                row_index=i,
                row_data=row_data,
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(results)

        # All 5 input rows get CONSUMED_IN_BATCH
        # The batch produces 2 COMPLETED outputs (one per category)
        consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 5, f"Expected 5 consumed, got {len(consumed)}"
        assert len(completed) == 2, f"Expected 2 completed, got {len(completed)}"

        # Verify group data
        categories = {r.final_data["category"] for r in completed}
        assert categories == {"A", "B"}

        # Verify counts and totals
        for result in completed:
            if result.final_data["category"] == "A":
                assert result.final_data["count"] == 3  # 3 A's
                assert result.final_data["total"] == 90  # 10 + 30 + 50
            else:
                assert result.final_data["count"] == 2  # 2 B's
                assert result.final_data["total"] == 60  # 20 + 40

        # Verify new token_ids created (not reusing input tokens)
        completed_tokens = {r.token.token_id for r in completed}
        consumed_tokens = {r.token.token_id for r in consumed}
        assert completed_tokens.isdisjoint(consumed_tokens), "Transform mode should create NEW tokens"

    def test_aggregation_transform_mode_single_row_output(self) -> None:
        """Transform mode with single row output still creates new token."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class BatchAggregator(BaseTransform):
            """Aggregates batch into a single summary row."""

            name = "aggregator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = True  # Transform mode
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    # Single aggregated output
                    total = sum(r.get("value", 0) for r in rows)
                    return TransformResult.success({"total": total, "count": len(rows)}, success_reason={"action": "test"})
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
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="aggregator",
                trigger=TriggerConfig(count=3),
                output_mode="transform",  # Transform mode with single output
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

        transform = BatchAggregator(agg_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 3 rows
        all_results = []
        for i in range(3):
            results = processor.process_row(
                row_index=i,
                row_data={"value": (i + 1) * 10},  # 10, 20, 30
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(results)

        consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 3, f"Expected 3 consumed, got {len(consumed)}"
        assert len(completed) == 1, f"Expected 1 completed, got {len(completed)}"

        # Verify aggregated data
        assert completed[0].final_data["total"] == 60  # 10 + 20 + 30
        assert completed[0].final_data["count"] == 3

        # Verify new token created
        completed_token = completed[0].token.token_id
        consumed_tokens = {r.token.token_id for r in consumed}
        assert completed_token not in consumed_tokens

    def test_aggregation_transform_mode_continues_to_next_transform(self) -> None:
        """Transform mode output rows continue through remaining transforms."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class GroupSplitter(BaseTransform):
            """Splits batch into groups."""

            name = "splitter"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = True
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    groups: dict[str, dict[str, Any]] = {}
                    for row in rows:
                        cat = row.get("category", "default")
                        if cat not in groups:
                            groups[cat] = {"category": cat, "count": 0}
                        groups[cat]["count"] += 1
                    return TransformResult.success_multi(list(groups.values()), success_reason={"action": "test"})
                return TransformResult.success(rows, success_reason={"action": "test"})

        class DoubleCount(BaseTransform):
            """Doubles the count field."""

            name = "doubler"
            input_schema = _TestSchema
            output_schema = _TestSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "count": row["count"] * 2, "doubled": True}, success_reason={"action": "test"})

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
        splitter_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        doubler_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="doubler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(splitter_node.node_id): AggregationSettings(
                name="group_split",
                plugin="splitter",
                trigger=TriggerConfig(count=3),
                output_mode="transform",
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

        splitter = GroupSplitter(splitter_node.node_id)
        doubler = DoubleCount(doubler_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 3 rows with 2 categories
        test_rows = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"},
        ]

        all_results = []
        for i, row_data in enumerate(test_rows):
            results = processor.process_row(
                row_index=i,
                row_data=row_data,
                transforms=[splitter, doubler],
                ctx=ctx,
            )
            all_results.extend(results)

        consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 3
        assert len(completed) == 2

        # Both outputs should have passed through doubler
        for result in completed:
            assert result.final_data["doubled"] is True

        # Counts should be doubled: A had count=2 -> 4, B had count=1 -> 2
        counts = {r.final_data["category"]: r.final_data["count"] for r in completed}
        assert counts["A"] == 4  # 2 * 2
        assert counts["B"] == 2  # 1 * 2


class TestProcessorSingleMode:
    """Tests for transform output_mode in aggregation (formerly 'single' mode).

    Transform mode: N input rows -> 1 aggregated output row as a NEW token.
    All original tokens get CONSUMED_IN_BATCH, aggregated row continues.

    Note: 'single' mode was removed - it had a bug where the triggering token
    was reused for the aggregated output, which caused problems with token
    identity. Transform mode creates a proper new token for the output.
    """

    def test_aggregation_single_mode_continues_to_next_transform(self) -> None:
        """Transform mode aggregated row continues through remaining transforms.

        This is the critical bug test - the old single mode was returning
        COMPLETED immediately without checking for downstream transforms.

        Pipeline: [SumTransform (transform mode)] -> [AddMarker]
        Expected: Aggregated row should have both 'total' and 'marker' fields.
        Bug (now fixed): Aggregated row only had 'total', skipping AddMarker entirely.
        """
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class SumTransform(BaseTransform):
            """Sums values in a batch, outputs single aggregated row."""

            name = "summer"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = False  # Single mode: triggering token continues
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    total = sum(r.get("value", 0) for r in rows)
                    return TransformResult.success({"total": total}, success_reason={"action": "test"})
                return TransformResult.success(rows, success_reason={"action": "test"})

        class AddMarker(BaseTransform):
            """Adds a marker field to prove it was executed."""

            name = "marker"
            input_schema = _TestSchema
            output_schema = _TestSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "marker": "DOWNSTREAM_EXECUTED"}, success_reason={"action": "test"})

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
        summer_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        marker_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="marker",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(summer_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                trigger=TriggerConfig(count=2),
                output_mode="transform",  # KEY: transform mode
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

        summer = SumTransform(summer_node.node_id)
        marker = AddMarker(marker_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 2 rows through summer (transform mode) then marker
        all_results = []
        for i in range(2):
            results = processor.process_row(
                row_index=i,
                row_data={"value": (i + 1) * 10},  # 10, 20
                transforms=[summer, marker],
                ctx=ctx,
            )
            all_results.extend(results)

        # Transform mode:
        # - First row: CONSUMED_IN_BATCH (buffered)
        # - Second row: CONSUMED_IN_BATCH (triggers flush, both consumed)
        # - Aggregated row: NEW token, continues through marker -> COMPLETED
        consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 2, f"Expected 2 consumed, got {len(consumed)}"
        assert len(completed) == 1, f"Expected 1 completed, got {len(completed)}"

        # CRITICAL: The aggregated row must have passed through AddMarker
        assert completed[0].final_data["total"] == 30, "Sum should be 10 + 20 = 30"
        assert "marker" in completed[0].final_data, (
            f"BUG: Aggregated row did not pass through downstream transform! Expected 'marker' field, got: {completed[0].final_data}"
        )
        assert completed[0].final_data["marker"] == "DOWNSTREAM_EXECUTED"

    def test_aggregation_transform_mode_no_downstream_completes_immediately(self) -> None:
        """Transform mode with no downstream transforms returns COMPLETED correctly."""
        from elspeth.contracts import Determinism
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        class SumTransform(BaseTransform):
            """Sums values in a batch."""

            name = "summer"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            creates_tokens = False
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0"

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, rows: list[dict[str, Any]] | dict[str, Any], ctx: PluginContext) -> TransformResult:
                if isinstance(rows, list):
                    total = sum(r.get("value", 0) for r in rows)
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
        summer_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(summer_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                trigger=TriggerConfig(count=2),
                output_mode="transform",
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

        summer = SumTransform(summer_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 2 rows through summer only (no downstream transforms)
        all_results = []
        for i in range(2):
            results = processor.process_row(
                row_index=i,
                row_data={"value": (i + 1) * 10},
                transforms=[summer],  # Only the aggregation, no downstream
                ctx=ctx,
            )
            all_results.extend(results)

        consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

        # Transform mode: both original rows are consumed, aggregated row completes
        assert len(consumed) == 2
        assert len(completed) == 1
        assert completed[0].final_data["total"] == 30
