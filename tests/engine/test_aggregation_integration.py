"""Integration tests for aggregation timeout behavior.

These tests verify that aggregation timeouts fire during active processing,
not just at end-of-source.

Bug reference: P1-2026-01-22-aggregation-timeout-idle-never-fires

Additional tests for timeout/end-of-source flush error handling:
- Bug: Duplicate terminal outcomes on flush errors in single/transform modes
- Bug: Routed/quarantined counts not tracked for timeout flushes
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from elspeth.contracts import (
    ArtifactDescriptor,
    RunStatus,
    SourceRow,
)
from elspeth.core.config import (
    AggregationSettings,
    ElspethSettings,
    SinkSettings,
    SourceSettings,
    TriggerConfig,
)
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import (
    CallbackSource,
    _TestSchema,
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)

# Note: ExecutionGraph.from_plugin_instances is used directly (imported locally)
# instead of build_production_graph, because we need to control node_id assignment


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Module-scoped in-memory database for aggregation integration tests."""
    return LandscapeDB.in_memory()


class TestAggregationTimeoutIntegration:
    """Test aggregation timeout fires during processing, not just at end-of-source.

    Bug: P1-2026-01-22-aggregation-timeout-idle-never-fires
    Root cause: should_flush() is only called after buffer_row(), so timeouts
    never fire during idle periods.
    """

    def test_aggregation_timeout_flushes_during_processing(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Aggregation should flush on timeout during processing, not wait for end-of-source.

        This test proves BUG P1-2026-01-22-aggregation-timeout-idle-never-fires:
        - Row 1 buffered at T=0 (timeout=0.1s)
        - MockClock advanced to 0.25s after row 1
        - Row 2 triggers timeout check which sees 0.25s elapsed
        - Expect: Row 1's batch flushes via timeout DURING processing of row 2
        - Actual (bug): Row 1's batch only flushes at end-of-source

        Uses MockClock for deterministic testing without time.sleep().
        """
        # Create mock clock starting at 0
        clock = MockClock(start=0.0)

        def advance_after_first_row(row_idx: int) -> None:
            """Advance clock after first row to trigger timeout before row 2."""
            if row_idx == 0:
                # Advance past the 0.1s timeout threshold
                clock.advance(0.25)

        # CallbackSource with clock advancement between rows
        callback_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Buffered at T=0
                {"id": 2, "value": 200},  # Arrives after clock advances to T=0.25
                {"id": 3, "value": 300},  # Will go in second batch
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_first_row,
            source_name="callback_source",
        )

        class BatchStatsTransform(BaseTransform):
            """Simple batch-aware transform that aggregates rows."""

            name = "batch_stats"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            # Note: node_id is assigned by graph construction, not pre-set

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                """Process single row or batch.

                Batch-aware transforms must check if input is a list (batch mode)
                or dict (single row mode) and dispatch accordingly.
                """
                if isinstance(row, list):
                    # Batch mode - aggregate the rows
                    rows = row
                    total = sum(r.get("value", 0) for r in rows)
                    return TransformResult.success(
                        {"id": rows[0].get("id"), "value": total, "count": len(rows)}, success_reason={"action": "test"}
                    )
                else:
                    # Single row mode - passthrough
                    return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectingSink(_TestSinkBase):
            """Sink that collects rows for verification."""

            name = "collecting_sink"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://test",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        # Build pipeline with short timeout
        source = as_source(callback_source)
        transform = as_transform(BatchStatsTransform())
        collecting_sink = CollectingSink()
        sink = as_sink(collecting_sink)

        # Build graph FIRST to get the assigned node_id for the transform
        # We pass an empty aggregations dict since we're using a batch-aware
        # transform in the transforms list, not a separate aggregation node
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},  # Empty - aggregation behavior comes from settings
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        # Get the graph-assigned node_id for the transform
        transform_id_map = graph.get_transform_id_map()
        assert 0 in transform_id_map, "Transform should have an assigned node_id"
        transform_node_id = transform_id_map[0]  # First (only) transform

        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            trigger=TriggerConfig(
                timeout_seconds=0.1,  # Short timeout - should fire during sleep
                count=100,  # High count - won't trigger by count
            ),
            output_mode="transform",
        )

        # Now create the config with the correct node_id in aggregation_settings
        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={
                transform_node_id: agg_settings,  # Use graph-assigned node_id
            },
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="callback_source", options={}),
            sinks={"output": SinkSettings(plugin="collecting_sink", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        # Inject MockClock into Orchestrator for deterministic timeout testing
        orchestrator = Orchestrator(db=landscape_db, clock=clock)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED, f"Run failed: {result}"

        # Verify we got results
        written_rows = collecting_sink.rows
        assert len(written_rows) >= 1, "Expected at least one batch to be flushed"

        # KEY ASSERTION: Timeout should cause MULTIPLE flushes
        # If timeout works during processing:
        #   - Row 1 buffered at T=0
        #   - Timeout of 0.1s should fire when row 2 arrives at T=0.25s
        #   - process_batch called with 1 row (row 1) -> count=1
        #   - Rows 2 and 3 go into a NEW batch, flushed at end-of-source -> count=2
        #   - Result: 2 output rows with different count values
        #
        # If bug exists (no timeout check during processing):
        #   - All 3 rows accumulate in ONE batch
        #   - Flushed at end-of-source
        #   - Result: 1 batch with count=3
        #
        # We can detect whether timeout worked by checking if we got multiple
        # output rows with different "count" values (each representing a separate batch)
        batch_counts = [r.get("count") for r in written_rows]

        # Timeout worked if we have more than one output row (multiple batches)
        assert len(written_rows) >= 2, (
            f"Aggregation timeout did not fire during processing! "
            f"Only {len(written_rows)} output row(s) with counts {batch_counts}. "
            f"Expected 2+ output rows (timeout batch + end-of-source batch). "
            f"Bug P1-2026-01-22: timeout only fires at end-of-source."
        )

        # Verify batch sizes: first batch should be row 1 (count=1),
        # second batch should be rows 2+3 (count=2)
        assert 1 in batch_counts, f"Expected a batch with count=1 (row 1 alone), got {batch_counts}"
        assert 2 in batch_counts, f"Expected a batch with count=2 (rows 2+3), got {batch_counts}"

        # If timeout didn't work, we'd see a single batch with count=3
        assert 3 not in batch_counts, (
            "Timeout did NOT fire! All 3 rows were in one batch (count=3). Bug P1-2026-01-22: timeout only fires at end-of-source."
        )

    def test_aggregation_timeout_loop_checks_all_nodes(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """The timeout check loop iterates over all aggregation nodes.

        QA Review requirement: Verify _check_aggregation_timeouts() loop
        correctly iterates over all aggregation node IDs.

        Note: This test verifies that when multiple aggregation nodes are
        registered, each one's timeout is checked independently. The existing
        test (test_aggregation_timeout_flushes_during_processing) already
        proves the timeout mechanism works for a single aggregation.

        The first aggregation node should flush multiple times due to timeout,
        proving the loop logic works correctly.

        Uses MockClock for deterministic testing without time.sleep().
        """
        batch_sizes: list[int] = []

        # Create mock clock starting at 0
        clock = MockClock(start=0.0)

        def advance_after_second_row(row_idx: int) -> None:
            """Advance clock after row 2 to trigger timeout before row 3."""
            if row_idx == 1:  # After second row (0-indexed)
                # Advance past the 0.1s timeout threshold
                clock.advance(0.2)

        # CallbackSource with clock advancement after row 2
        callback_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Row 1 at T=0
                {"id": 2, "value": 200},  # Row 2 at T=0, then clock advances to T=0.2
                {"id": 3, "value": 300},  # Row 3 triggers timeout check
                {"id": 4, "value": 400},  # Row 4 in second batch
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_second_row,
            source_name="callback_source_loop",
        )

        class AggTransform(BaseTransform):
            """Aggregation transform."""

            name = "agg_loop"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    batch_sizes.append(len(row))
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects all rows."""

            name = "collector_loop"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(callback_source)
        transform = as_transform(AggTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="agg_loop",
            plugin="agg_loop",
            trigger=TriggerConfig(timeout_seconds=0.1, count=100),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="callback_source_loop", options={}),
            sinks={"output": SinkSettings(plugin="collector_loop", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        # Inject MockClock into Orchestrator for deterministic timeout testing
        orchestrator = Orchestrator(db=landscape_db, clock=clock)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Aggregation should have flushed multiple times:
        # First flush: timeout at T=0.2 for first 2 rows
        # Second flush: end-of-source for last 2 rows
        assert len(batch_sizes) >= 2, (
            f"Aggregation should have multiple batches due to timeout. Got {len(batch_sizes)} batches: {batch_sizes}"
        )

        # First batch should be rows 1-2 (before clock advance)
        # Second batch should be rows 3-4 (after clock advance)
        assert batch_sizes == [2, 2], f"Expected [2, 2] batch pattern, got {batch_sizes}. Timeout should fire after clock advance."

    def test_timeout_fires_before_row_processing(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Timeout is checked BEFORE each row is processed, not after.

        QA Review requirement: Verify timeout and count triggers are handled
        correctly when both could fire.

        Architecture insight: Timeout is checked BEFORE buffering a new row.
        This means:
        - Timeout check happens BEFORE row 2 is added to the batch
        - If timeout fires, the current batch flushes FIRST
        - THEN row 2 starts a fresh batch
        - Count check happens AFTER row is buffered

        Setup:
        - count=2, timeout=0.1s
        - Row 1 at T=0
        - MockClock advances to T=0.15 (timeout expires)
        - Row 2 at T=0.15

        Expected (correct behavior):
        - Before processing row 2: timeout check fires (0.15s > 0.1s)
        - Row 1 flushed alone via timeout → batch size 1
        - Row 2 starts new batch, flushed at end-of-source → batch size 1
        - Total: 2 flushes with [1, 1]

        Uses MockClock for deterministic testing without time.sleep().
        """
        flush_count = 0
        batch_sizes: list[int] = []

        # Create mock clock starting at 0
        clock = MockClock(start=0.0)

        def advance_after_first_row(row_idx: int) -> None:
            """Advance clock after row 1 to trigger timeout before row 2."""
            if row_idx == 0:
                # Advance past the 0.1s timeout threshold
                clock.advance(0.15)

        # CallbackSource with clock advancement after row 1
        callback_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Row 1 at T=0, then clock advances to T=0.15
                {"id": 2, "value": 200},  # Row 2 arrives after timeout expired
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_first_row,
            source_name="timed_source",
        )

        class CountingAggTransform(BaseTransform):
            """Transform that counts flushes."""

            name = "counting_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                nonlocal flush_count
                if isinstance(row, list):
                    flush_count += 1
                    batch_sizes.append(len(row))
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class SimpleSink(_TestSinkBase):
            """Simple sink."""

            name = "simple_sink"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(callback_source)
        transform = as_transform(CountingAggTransform())
        sink = as_sink(SimpleSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        node_id = transform_id_map[0]

        # Both triggers configured: count=2 AND timeout=0.1s
        # But timeout fires FIRST (before row 2 is added)
        agg_settings = AggregationSettings(
            name="race_agg",
            plugin="counting_agg",
            trigger=TriggerConfig(timeout_seconds=0.1, count=2),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="timed_source", options={}),
            sinks={"output": SinkSettings(plugin="simple_sink", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        # Inject MockClock into Orchestrator for deterministic timeout testing
        orchestrator = Orchestrator(db=landscape_db, clock=clock)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # KEY ASSERTION: Two flushes should happen
        # 1. Timeout fires BEFORE row 2 is processed → row 1 alone → batch size 1
        # 2. End-of-source → row 2 alone → batch size 1
        assert flush_count == 2, (
            f"Expected 2 flushes (timeout + end-of-source), got {flush_count}. "
            f"Batch sizes: {batch_sizes}. "
            "Timeout should fire BEFORE row 2 is added to the batch."
        )

        # Each batch should have exactly 1 row
        assert batch_sizes == [1, 1], (
            f"Expected two batches of size [1, 1], got {batch_sizes}. Timeout should flush row 1 before row 2 is added."
        )


class TestEndOfSourceFlush:
    """Test end-of-source aggregation flush with all output_modes.

    These tests verify that _flush_remaining_aggregation_buffers correctly
    handles all output_mode semantics through the refactored handle_timeout_flush
    with TriggerType.END_OF_SOURCE.

    P2 Review requirement: QA requested 5 integration tests for END_OF_SOURCE:
    1. END_OF_SOURCE + single (tested below)
    2. END_OF_SOURCE + passthrough (tested below)
    3. END_OF_SOURCE + transform (tested below)
    4. END_OF_SOURCE + passthrough + downstream transform (tested below)
    5. END_OF_SOURCE + passthrough + downstream gate (tested below)
    """

    def test_end_of_source_single_mode(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """END_OF_SOURCE flush with single output_mode creates one aggregated token.

        Verifies that when the source completes with buffered rows, the
        _flush_remaining_aggregation_buffers method correctly:
        - Uses handle_timeout_flush with END_OF_SOURCE trigger
        - Creates a single output token with aggregated data
        """
        batch_data: list[dict[str, Any]] = []

        class FastSource(_TestSourceBase):
            """Source that completes immediately with rows still in buffer."""

            name = "fast_source_single"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                # Emit 3 rows, all will be buffered (count trigger is 100)
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})
                # Source completes - end-of-source flush should trigger

            def close(self) -> None:
                pass

        class SingleModeAgg(BaseTransform):
            """Aggregation that sums values and returns single row."""

            name = "single_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    batch_data.append({"rows": len(row), "total": sum(r.get("value", 0) for r in row)})
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_single"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(SingleModeAgg())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        agg_settings = AggregationSettings(
            name="test_single",
            plugin="single_agg",
            trigger=TriggerConfig(count=100),  # Won't trigger by count - needs END_OF_SOURCE
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_single", options={}),
            sinks={"output": SinkSettings(plugin="collector_single", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Single mode should produce exactly one output row
        assert len(sink.rows) == 1, f"Single mode should produce 1 row, got {len(sink.rows)}: {sink.rows}"

        # Verify aggregated data
        output = sink.rows[0]
        assert output["total"] == 600, f"Expected total=600 (100+200+300), got {output}"
        assert output["count"] == 3, f"Expected count=3, got {output}"

        # Verify batch was processed once with all 3 rows
        assert len(batch_data) == 1, f"Expected 1 batch, got {len(batch_data)}"
        assert batch_data[0]["rows"] == 3

    def test_end_of_source_passthrough_mode(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """END_OF_SOURCE flush with passthrough output_mode preserves all tokens.

        Verifies that passthrough mode:
        - Returns same number of output rows as input rows
        - Enriches each row with batch data
        - Maintains proper token count through end-of-source flush
        """
        batch_sizes: list[int] = []

        class FastSource(_TestSourceBase):
            """Source that completes immediately."""

            name = "fast_source_passthrough"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class PassthroughAgg(BaseTransform):
            """Aggregation that enriches rows while preserving them."""

            name = "passthrough_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    batch_sizes.append(len(row))
                    # Passthrough: return same number of rows, enriched
                    batch_total = sum(r.get("value", 0) for r in row)
                    enriched = [{**r, "batch_total": batch_total, "batch_size": len(row)} for r in row]
                    return TransformResult.success_multi(enriched, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_passthrough"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(PassthroughAgg())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        agg_settings = AggregationSettings(
            name="test_passthrough",
            plugin="passthrough_agg",
            trigger=TriggerConfig(count=100),
            output_mode="passthrough",  # KEY: passthrough mode
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_passthrough", options={}),
            sinks={"output": SinkSettings(plugin="collector_passthrough", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Passthrough mode should produce 3 output rows (one per input row)
        assert len(sink.rows) == 3, f"Passthrough mode should produce 3 rows, got {len(sink.rows)}"

        # All rows should be enriched with batch data
        for row in sink.rows:
            assert "batch_total" in row, f"Row should have batch_total: {row}"
            assert row["batch_total"] == 600  # 100+200+300
            assert row["batch_size"] == 3

        # Original values should be preserved
        values = {r["value"] for r in sink.rows}
        assert values == {100, 200, 300}

    def test_end_of_source_transform_mode(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """END_OF_SOURCE flush with transform output_mode creates new tokens.

        Verifies that transform mode (N→M):
        - Can produce different number of output rows than input
        - Creates new token IDs for outputs
        """
        batch_sizes: list[int] = []

        class FastSource(_TestSourceBase):
            """Source that completes immediately."""

            name = "fast_source_transform"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class TransformModeAgg(BaseTransform):
            """Aggregation that produces 2 output rows from N inputs (summary + detail)."""

            name = "transform_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    batch_sizes.append(len(row))
                    total = sum(r.get("value", 0) for r in row)
                    # Transform mode: N inputs → 2 outputs (summary row + count row)
                    return TransformResult.success_multi(
                        [
                            {"type": "summary", "total": total},
                            {"type": "count", "count": len(row)},
                        ],
                        success_reason={"action": "test"},
                    )
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_transform"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(TransformModeAgg())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        agg_settings = AggregationSettings(
            name="test_transform",
            plugin="transform_agg",
            trigger=TriggerConfig(count=100),
            output_mode="transform",  # KEY: transform mode (N→M)
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_transform", options={}),
            sinks={"output": SinkSettings(plugin="collector_transform", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Transform mode should produce 2 output rows (3 inputs → 2 outputs)
        assert len(sink.rows) == 2, f"Transform mode should produce 2 rows, got {len(sink.rows)}"

        # Check outputs
        types = {r["type"] for r in sink.rows}
        assert types == {"summary", "count"}

        summary = next(r for r in sink.rows if r["type"] == "summary")
        count_row = next(r for r in sink.rows if r["type"] == "count")

        assert summary["total"] == 600
        assert count_row["count"] == 3

    def test_end_of_source_passthrough_with_downstream_transform(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """END_OF_SOURCE flush with passthrough routes tokens through downstream transforms.

        Verifies that tokens flushed at end-of-source in passthrough mode
        correctly continue through remaining transforms in the pipeline.
        """

        class FastSource(_TestSourceBase):
            """Source that completes immediately."""

            name = "fast_source_downstream"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})

            def close(self) -> None:
                pass

        class PassthroughAgg(BaseTransform):
            """First transform: passthrough aggregation."""

            name = "passthrough_agg_ds"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Passthrough: enrich with batch_total
                    batch_total = sum(r.get("value", 0) for r in row)
                    enriched = [{**r, "batch_total": batch_total} for r in row]
                    return TransformResult.success_multi(enriched, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class DownstreamTransform(BaseTransform):
            """Second transform: adds 'processed' field."""

            name = "downstream_transform"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = False

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success({**row, "processed": True}, success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_downstream"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        agg_transform = as_transform(PassthroughAgg())
        downstream = as_transform(DownstreamTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[agg_transform, downstream],  # Two transforms in sequence
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        agg_node_id = transform_id_map[0]  # First transform (aggregation)

        agg_settings = AggregationSettings(
            name="test_passthrough_ds",
            plugin="passthrough_agg_ds",
            trigger=TriggerConfig(count=100),
            output_mode="passthrough",
        )

        config = PipelineConfig(
            source=source,
            transforms=[agg_transform, downstream],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={agg_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_downstream", options={}),
            sinks={"output": SinkSettings(plugin="collector_downstream", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Should have 2 output rows
        assert len(sink.rows) == 2, f"Expected 2 rows, got {len(sink.rows)}"

        # All rows should have batch_total from aggregation AND processed from downstream
        for row in sink.rows:
            assert "batch_total" in row, f"Row missing batch_total: {row}"
            assert row["batch_total"] == 300  # 100+200
            assert row.get("processed") is True, f"Row missing processed flag: {row}"

    def test_end_of_source_single_with_downstream_transform(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """END_OF_SOURCE flush with single mode routes result through downstream transforms.

        Verifies that the single aggregated output token from single mode
        correctly continues through remaining transforms in the pipeline.
        """

        class FastSource(_TestSourceBase):
            """Source that completes immediately."""

            name = "fast_source_single_ds"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})

            def close(self) -> None:
                pass

        class SingleAgg(BaseTransform):
            """First transform: single mode aggregation."""

            name = "single_agg_ds"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class DownstreamTransform(BaseTransform):
            """Second transform: adds 'processed' field."""

            name = "downstream_single"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = False

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(
                    {**row, "processed": True, "doubled_total": row.get("total", 0) * 2}, success_reason={"action": "test"}
                )

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_single_ds"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        agg_transform = as_transform(SingleAgg())
        downstream = as_transform(DownstreamTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[agg_transform, downstream],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        agg_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="test_single_ds",
            plugin="single_agg_ds",
            trigger=TriggerConfig(count=100),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[agg_transform, downstream],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={agg_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_single_ds", options={}),
            sinks={"output": SinkSettings(plugin="collector_single_ds", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Single mode with downstream should produce 1 row
        assert len(sink.rows) == 1, f"Expected 1 row, got {len(sink.rows)}"

        output = sink.rows[0]
        # From aggregation:
        assert output["total"] == 300  # 100+200
        assert output["count"] == 2
        # From downstream transform:
        assert output["processed"] is True
        assert output["doubled_total"] == 600  # 300*2


class TestTimeoutFlushErrorHandling:
    """Tests for timeout/end-of-source flush error handling bugs.

    These tests verify correctness of error handling in timeout flush paths:
    - Bug 1: Duplicate terminal outcomes when flush fails in single/transform modes
    - Bug 2: Routed/quarantined counts not tracked for timeout flushes
    """

    def test_timeout_flush_error_single_mode_no_duplicate_outcomes(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Timeout flush errors in single mode must not record duplicate terminal outcomes.

        Bug: When handle_timeout_flush is called and the flush fails:
        - In single/transform modes, buffered tokens already have CONSUMED_IN_BATCH recorded
        - The error path tries to record FAILED for all buffered tokens
        - This violates the unique terminal outcome constraint → IntegrityError

        Expected behavior: The batch failure is recorded in the batch table.
        The tokens' CONSUMED_IN_BATCH outcome remains valid (they were consumed
        into a batch that failed). The pipeline should complete without crashing.

        Test setup:
        - Aggregation with short timeout (0.1s), high count (never triggers by count)
        - Single output mode (tokens get CONSUMED_IN_BATCH when buffered)
        - Transform that FAILS when batch is flushed
        - Source emits row, waits for timeout, emits another row to trigger timeout check
        """
        flush_calls: list[str] = []

        class FailingBatchTransform(BaseTransform):
            """Batch transform that fails on flush."""

            name = "failing_batch"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Batch flush - FAIL
                    flush_calls.append("flush_failed")
                    return TransformResult.error({"reason": "deliberate_failure"})
                # Single row passthrough (shouldn't happen in batch mode)
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        # Create mock clock for deterministic timeout testing
        clock = MockClock(start=0.0)

        def advance_after_first_row(row_idx: int) -> None:
            """Advance clock after row 1 to trigger timeout before row 2."""
            if row_idx == 0:
                clock.advance(0.15)  # Past 0.1s timeout

        # CallbackSource with clock advancement
        timeout_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Gets buffered, marked CONSUMED_IN_BATCH
                {"id": 2, "value": 200},  # Triggers timeout check → flush fails
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_first_row,
            source_name="timeout_trigger_source",
        )

        class CollectorSink(_TestSinkBase):
            """Sink that collects results."""

            name = "collector_timeout_err"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://test",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        source = as_source(timeout_source)
        transform = as_transform(FailingBatchTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="failing_agg",
            plugin="failing_batch",
            trigger=TriggerConfig(
                timeout_seconds=0.1,  # Short timeout
                count=100,  # High count - won't trigger by count
            ),
            output_mode="transform",  # Tokens get CONSUMED_IN_BATCH when buffered
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={transform_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="timeout_trigger_source", options={}),
            sinks={"output": SinkSettings(plugin="collector_timeout_err", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        # Inject MockClock for deterministic timeout testing
        orchestrator = Orchestrator(db=landscape_db, clock=clock)

        # BUG: This should NOT crash with IntegrityError
        # The fix should handle the fact that tokens in single mode already
        # have CONSUMED_IN_BATCH recorded when the flush fails.
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Verify the flush was attempted and failed
        assert len(flush_calls) >= 1, "Batch flush should have been called"

        # Pipeline should complete (possibly with failures) but NOT crash
        assert result.status in [RunStatus.COMPLETED, RunStatus.FAILED], (
            f"Pipeline should complete gracefully even with flush failures, got {result.status}"
        )

        # Verify failure was recorded appropriately
        # Note: Since the batch failed, we expect rows_failed to include the failed batch
        # The exact count depends on implementation details, but there should be failures
        assert result.rows_failed >= 1, f"Failed flush should result in at least 1 failed row, got {result.rows_failed}"

    def test_timeout_flush_downstream_routed_counts_tracked(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Downstream ROUTED outcomes from timeout flush must be tracked in stats.

        Bug: In _check_aggregation_timeouts and _flush_remaining_aggregation_buffers,
        when downstream processing results in ROUTED outcomes:
        - The code adds to pending_tokens correctly
        - But increments rows_succeeded instead of rows_routed
        - And doesn't update routed_destinations dict

        Expected: rows_routed should be incremented and routed_destinations
        should track where rows were routed.

        Test setup:
        - Aggregation with short timeout, single mode
        - Downstream gate that routes rows to named sink
        - Verify RunResult.rows_routed reflects timeout-flushed routed rows
        """
        from elspeth.core.config import GateSettings

        class SimpleAggTransform(BaseTransform):
            """Simple aggregation transform."""

            name = "simple_agg_for_routing"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Batch mode - combine rows
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        # Create mock clock for deterministic timeout testing
        clock = MockClock(start=0.0)

        def advance_after_first_row(row_idx: int) -> None:
            """Advance clock after row 1 to trigger timeout before row 2."""
            if row_idx == 0:
                clock.advance(0.15)  # Past 0.1s timeout

        # CallbackSource with clock advancement
        timeout_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Gets buffered
                {"id": 2, "value": 200},  # Triggers timeout flush → aggregated result goes to gate → ROUTED
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_first_row,
            source_name="timeout_source_routing",
        )

        class RoutedSink(_TestSinkBase):
            """Sink for routed rows."""

            name = "routed_sink"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://routed",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        class DefaultSink(_TestSinkBase):
            """Default sink."""

            name = "default_sink_routing"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://default",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        source = as_source(timeout_source)
        agg_transform = as_transform(SimpleAggTransform())
        routed_sink = as_sink(RoutedSink())
        default_sink = as_sink(DefaultSink())

        # Use GateSettings with expression - all rows route to "routed_sink"
        gate_settings = GateSettings(
            name="route_all",
            condition="True",  # Always route
            routes={"true": "routed_sink", "false": "output"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[agg_transform],
            sinks={"output": default_sink, "routed_sink": routed_sink},
            gates=[gate_settings],
            aggregation_settings={},  # Will be set after graph build
            coalesce_settings={},
        )

        from tests.engine.orchestrator_test_helpers import build_production_graph

        graph = build_production_graph(config)

        # Get transform node_id from graph
        transform_id_map = graph.get_transform_id_map()
        agg_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="agg_for_routing",
            plugin="simple_agg_for_routing",
            trigger=TriggerConfig(
                timeout_seconds=0.1,
                count=100,  # Won't trigger by count
            ),
            output_mode="transform",
        )

        # Rebuild config with aggregation settings
        config = PipelineConfig(
            source=source,
            transforms=[agg_transform],
            sinks={"output": default_sink, "routed_sink": routed_sink},
            gates=[gate_settings],
            aggregation_settings={agg_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="timeout_source_routing", options={}),
            sinks={
                "output": SinkSettings(plugin="default_sink_routing", options={}),
                "routed_sink": SinkSettings(plugin="routed_sink", options={}),
            },
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        # Inject MockClock for deterministic timeout testing
        orchestrator = Orchestrator(db=landscape_db, clock=clock)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED, f"Run failed: {result}"

        # BUG: rows_routed should be > 0 for rows that went through the gate
        # The current code incorrectly counts them as rows_succeeded
        assert result.rows_routed >= 1, (
            f"Timeout-flushed rows going through gate should increment rows_routed. "
            f"Got rows_routed={result.rows_routed}, rows_succeeded={result.rows_succeeded}. "
            f"Bug: _check_aggregation_timeouts counts ROUTED as succeeded."
        )

        # Verify routed_destinations is populated
        assert len(result.routed_destinations) > 0, (
            "routed_destinations should track where rows were routed. "
            "Got empty dict. Bug: _check_aggregation_timeouts doesn't update routed_destinations."
        )

    def test_passthrough_flush_failure_marks_all_buffered_tokens_failed(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Passthrough flush failure must mark ALL buffered tokens as FAILED.

        Bug: P1-2026-01-21-aggregation-passthrough-failure-buffered

        In passthrough mode, tokens get BUFFERED outcome (non-terminal) when buffered.
        When flush fails, only the triggering token gets FAILED outcome.
        The previously buffered tokens remain BUFFERED forever - violating the
        invariant that every token must reach a terminal state.

        Expected behavior:
        - ALL buffered tokens (including triggering token) get FAILED outcome
        - No tokens should remain with BUFFERED outcome after flush failure
        - rows_failed count should match total buffer size

        Test setup:
        - 3 rows emitted, count trigger = 3 (flush on 3rd row)
        - Passthrough mode (tokens get BUFFERED when buffered)
        - Transform returns error on flush
        """

        from elspeth.contracts.enums import RowOutcome

        flush_calls: list[str] = []

        class FailingPassthroughTransform(BaseTransform):
            """Batch transform that fails on flush in passthrough mode."""

            name = "failing_passthrough"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Batch flush - FAIL with error result
                    flush_calls.append("flush_failed")
                    return TransformResult.error({"reason": "deliberate_failure", "error": "deliberate_passthrough_failure"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CountTriggerSource(_TestSourceBase):
            """Source that emits exactly 3 rows to trigger count-based flush."""

            name = "count_trigger_source"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                # Row 1: Gets buffered, marked BUFFERED
                yield SourceRow.valid({"id": 1, "value": 100})
                # Row 2: Gets buffered, marked BUFFERED
                yield SourceRow.valid({"id": 2, "value": 200})
                # Row 3: Triggers count flush (count=3) → flush fails
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class CollectorSink(_TestSinkBase):
            """Sink that collects results."""

            name = "collector_passthrough_err"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://test",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        source = as_source(CountTriggerSource())
        transform = as_transform(FailingPassthroughTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="failing_passthrough_agg",
            plugin="failing_passthrough",
            trigger=TriggerConfig(
                count=3,  # Flush after 3 rows
            ),
            output_mode="passthrough",  # Tokens get BUFFERED when buffered
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={transform_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="count_trigger_source", options={}),
            sinks={"output": SinkSettings(plugin="collector_passthrough_err", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Verify flush was called and failed
        assert len(flush_calls) == 1, f"Expected 1 flush call, got {len(flush_calls)}"
        assert flush_calls[0] == "flush_failed"

        # BUG ASSERTION: All 3 tokens should be marked FAILED
        # Currently only the triggering token (row 3) gets FAILED
        assert result.rows_failed == 3, (
            f"All 3 buffered tokens should be marked FAILED when passthrough flush fails. "
            f"Got rows_failed={result.rows_failed}. "
            f"Bug: Only triggering token marked FAILED, buffered tokens left in BUFFERED state."
        )

        # Verify all tokens have a terminal outcome
        # Note: BUFFERED records remain in DB as audit trail, but tokens should ALSO
        # have a terminal outcome (FAILED in this case). Check for tokens that have
        # BUFFERED but no terminal outcome.
        with landscape_db.connection() as conn:
            from sqlalchemy import func, select

            from elspeth.core.landscape.schema import token_outcomes_table

            # Count FAILED outcomes (should be 3 - one per buffered token)
            failed_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(token_outcomes_table.c.outcome == RowOutcome.FAILED.value)
                .where(token_outcomes_table.c.run_id == result.run_id)
            ).scalar()

            assert failed_count == 3, (
                f"All 3 buffered tokens should have FAILED outcome recorded. "
                f"Found {failed_count} FAILED outcomes. "
                f"Bug: Not all buffered tokens received terminal outcome."
            )

    def test_passthrough_end_of_source_flush_failure_marks_all_failed(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """END_OF_SOURCE passthrough flush failure must mark ALL buffered tokens FAILED.

        Bug: P1-2026-01-21-aggregation-passthrough-failure-buffered

        This tests the most dangerous scenario: source completes with buffered rows,
        triggering END_OF_SOURCE flush. If flush fails, tokens would be stuck
        forever (no more rows will arrive to trigger recovery).

        Expected behavior:
        - ALL buffered tokens get FAILED outcome
        - Pipeline completes (doesn't hang)
        - No tokens left in BUFFERED state
        """
        from elspeth.contracts.enums import RowOutcome

        flush_calls: list[str] = []

        class FailingPassthroughTransform(BaseTransform):
            """Batch transform that fails on END_OF_SOURCE flush."""

            name = "failing_passthrough_eos"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    flush_calls.append("flush_failed")
                    return TransformResult.error({"reason": "deliberate_failure", "error": "eos_failure"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class ShortSource(_TestSourceBase):
            """Source that emits 2 rows (won't trigger count=10)."""

            name = "short_source"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                # Source completes → END_OF_SOURCE flush

            def close(self) -> None:
                pass

        class CollectorSink(_TestSinkBase):
            """Sink for collecting output."""

            name = "collector_eos"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://test",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        source = as_source(ShortSource())
        transform = as_transform(FailingPassthroughTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="failing_passthrough_eos_agg",
            plugin="failing_passthrough_eos",
            trigger=TriggerConfig(
                count=10,  # High count - won't trigger by count
            ),
            output_mode="passthrough",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={transform_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="short_source", options={}),
            sinks={"output": SinkSettings(plugin="collector_eos", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Verify END_OF_SOURCE flush was triggered and failed
        assert len(flush_calls) == 1, f"Expected 1 END_OF_SOURCE flush, got {len(flush_calls)}"

        # BUG ASSERTION: Both tokens should be marked FAILED
        assert result.rows_failed == 2, (
            f"Both buffered tokens should be marked FAILED on END_OF_SOURCE flush failure. "
            f"Got rows_failed={result.rows_failed}. "
            f"Bug: Tokens stuck in BUFFERED state forever."
        )

        # Verify all tokens have a terminal FAILED outcome
        # Note: BUFFERED records remain in DB as audit trail, but tokens should ALSO
        # have a terminal outcome (FAILED in this case).
        with landscape_db.connection() as conn:
            from sqlalchemy import func, select

            from elspeth.core.landscape.schema import token_outcomes_table

            # Count FAILED outcomes for this run (should be 2 - one per buffered token)
            failed_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(token_outcomes_table.c.outcome == RowOutcome.FAILED.value)
                .where(token_outcomes_table.c.run_id == result.run_id)
            ).scalar()

            assert failed_count == 2, f"Both buffered tokens should have FAILED outcome recorded. Found {failed_count} FAILED outcomes."

    def test_single_mode_count_flush_failure_triggering_token_has_outcome(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Count-triggered flush failure in single mode must record CONSUMED_IN_BATCH for triggering token.

        Bug: P2-2026-01-28-aggregation-triggering-token-no-outcome

        In single/transform modes, when a count-triggered flush FAILS:
        - Previously buffered tokens have CONSUMED_IN_BATCH (recorded when buffered via non-flushing path)
        - The triggering token (the one that caused count threshold to be reached) has NO outcome

        This happens because:
        1. Triggering token is buffered (line 598)
        2. should_flush() returns True (count threshold reached)
        3. Non-flushing path (lines 893-899 which records CONSUMED_IN_BATCH) is SKIPPED
        4. execute_flush() fails
        5. Failure path (lines 652-667) doesn't record any outcome for triggering token

        The triggering token ends up with no terminal outcome, violating the invariant
        that every token must reach exactly one terminal state.

        Expected behavior:
        - ALL tokens (including triggering token) have CONSUMED_IN_BATCH outcome
        - Batch failure is recorded in batches table (semantic: "consumed into failed batch")

        Test setup:
        - 3 rows emitted, count trigger = 3 (flush on 3rd row)
        - Single mode (tokens get CONSUMED_IN_BATCH when buffered)
        - Transform returns error on flush
        """
        from elspeth.contracts.enums import RowOutcome

        flush_calls: list[str] = []

        class FailingSingleTransform(BaseTransform):
            """Batch transform that fails on flush in single mode."""

            name = "failing_single"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Batch flush - FAIL with error result
                    flush_calls.append("flush_failed")
                    return TransformResult.error({"reason": "deliberate_failure", "error": "deliberate_single_mode_failure"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CountTriggerSource(_TestSourceBase):
            """Source that emits exactly 3 rows to trigger count-based flush."""

            name = "count_trigger_source_single"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                # Row 1: Gets buffered, marked CONSUMED_IN_BATCH (non-flushing path)
                yield SourceRow.valid({"id": 1, "value": 100})
                # Row 2: Gets buffered, marked CONSUMED_IN_BATCH (non-flushing path)
                yield SourceRow.valid({"id": 2, "value": 200})
                # Row 3: Triggers count flush (count=3) → flush fails
                # BUG: This token has NO outcome recorded!
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class CollectorSink(_TestSinkBase):
            """Sink that collects results."""

            name = "collector_single_err"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://test",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        source = as_source(CountTriggerSource())
        transform = as_transform(FailingSingleTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="failing_single_agg",
            plugin="failing_single",
            trigger=TriggerConfig(
                count=3,  # Flush after 3 rows
            ),
            output_mode="transform",  # Tokens get CONSUMED_IN_BATCH when buffered
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={transform_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="count_trigger_source_single", options={}),
            sinks={"output": SinkSettings(plugin="collector_single_err", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Verify flush was called and failed
        assert len(flush_calls) == 1, f"Expected 1 flush call, got {len(flush_calls)}"
        assert flush_calls[0] == "flush_failed"

        # BUG ASSERTION: All 3 tokens should have CONSUMED_IN_BATCH outcome
        # Currently only rows 1 and 2 have it (from non-flushing path when buffered)
        # Row 3 (triggering token) has NO outcome!
        with landscape_db.connection() as conn:
            from sqlalchemy import func, select

            from elspeth.core.landscape.schema import token_outcomes_table

            # Count CONSUMED_IN_BATCH outcomes (should be 3 - one per token)
            consumed_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(token_outcomes_table.c.outcome == RowOutcome.CONSUMED_IN_BATCH.value)
                .where(token_outcomes_table.c.run_id == result.run_id)
            ).scalar()

            assert consumed_count == 3, (
                f"All 3 tokens should have CONSUMED_IN_BATCH outcome recorded. "
                f"Found {consumed_count} CONSUMED_IN_BATCH outcomes. "
                f"Bug: Triggering token (row 3) has no terminal outcome - "
                f"it went straight from buffer to failed flush, skipping outcome recording."
            )

    def test_transform_mode_count_flush_failure_triggering_token_has_outcome(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Count-triggered flush failure in transform mode must record CONSUMED_IN_BATCH for triggering token.

        Same bug as single mode, but tests transform mode specifically since it has
        slightly different semantics (triggering token is replaced by expanded tokens on success).

        Expected behavior:
        - ALL tokens have CONSUMED_IN_BATCH outcome
        - Transform mode success path records CONSUMED_IN_BATCH at line 816-821
        - Transform mode failure path should also record CONSUMED_IN_BATCH for triggering token
        """
        from elspeth.contracts.enums import RowOutcome

        flush_calls: list[str] = []

        class FailingTransformModeTransform(BaseTransform):
            """Batch transform that fails on flush in transform mode."""

            name = "failing_transform_mode"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Batch flush - FAIL
                    flush_calls.append("flush_failed")
                    return TransformResult.error({"reason": "deliberate_failure", "error": "deliberate_transform_mode_failure"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CountTriggerSource(_TestSourceBase):
            """Source that emits exactly 3 rows to trigger count-based flush."""

            name = "count_trigger_source_transform"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class CollectorSink(_TestSinkBase):
            """Sink that collects results."""

            name = "collector_transform_err"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(
                    path="memory://test",
                    size_bytes=0,
                    content_hash="test",
                )

            def close(self) -> None:
                pass

        source = as_source(CountTriggerSource())
        transform = as_transform(FailingTransformModeTransform())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="failing_transform_mode_agg",
            plugin="failing_transform_mode",
            trigger=TriggerConfig(
                count=3,
            ),
            output_mode="transform",  # Transform mode
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={transform_node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="count_trigger_source_transform", options={}),
            sinks={"output": SinkSettings(plugin="collector_transform_err", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Verify flush was called and failed
        assert len(flush_calls) == 1, f"Expected 1 flush call, got {len(flush_calls)}"

        # All 3 tokens should have CONSUMED_IN_BATCH outcome
        with landscape_db.connection() as conn:
            from sqlalchemy import func, select

            from elspeth.core.landscape.schema import token_outcomes_table

            consumed_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(token_outcomes_table.c.outcome == RowOutcome.CONSUMED_IN_BATCH.value)
                .where(token_outcomes_table.c.run_id == result.run_id)
            ).scalar()

            assert consumed_count == 3, (
                f"All 3 tokens should have CONSUMED_IN_BATCH outcome. "
                f"Found {consumed_count}. "
                f"Bug: Triggering token has no terminal outcome in transform mode."
            )


class TestTimeoutFlushStepIndexing:
    """Test that timeout/end-of-source flushes record correct 1-indexed step.

    Bug: Brief 1 - Timeout/End-of-Source Flush Step Indexing (P2)

    Root cause: _find_aggregation_transform returns 0-indexed position which
    flows directly to audit recording, but normal transform execution uses
    1-indexed steps (step = start_step + step_offset + 1).

    Impact: Audit trails for timeout/end-of-source batches have step_index=0
    for the first aggregation, while normal processing records step_index=1.
    """

    def test_timeout_flush_records_1_indexed_step(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Timeout flush at first transform should record step_index=1, not 0.

        The aggregation is at position 0 in transforms list (first transform).
        Normal processing would record step_index=1 (1-indexed).
        Timeout flush should record the same step_index=1.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table

        # Create mock clock for deterministic timeout testing
        clock = MockClock(start=0.0)

        def advance_after_first_row(row_idx: int) -> None:
            """Advance clock after row 1 to trigger timeout before row 2."""
            if row_idx == 0:
                clock.advance(0.8)  # Past 0.5s timeout

        # CallbackSource with clock advancement
        timeout_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Emit first row, it will be buffered
                {"id": 2, "value": 200},  # Emit second row - timeout check happens before this is processed
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_first_row,
            source_name="slow_source_step_test",
        )

        class BatchAgg(BaseTransform):
            """Simple aggregation that returns batch sum."""

            name = "batch_agg_step_test"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_step_test"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(timeout_source)
        transform = as_transform(BatchAgg())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],  # First (and only) transform at position 0
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        agg_settings = AggregationSettings(
            name="test_step",
            plugin="batch_agg_step_test",
            trigger=TriggerConfig(count=100, timeout_seconds=0.5),  # Timeout trigger (0.5s with CI margin)
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="slow_source_step_test", options={}),
            sinks={"output": SinkSettings(plugin="collector_step_test", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        # Inject MockClock for deterministic timeout testing
        orchestrator = Orchestrator(db=landscape_db, clock=clock)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED, f"Run failed: {result}"

        # Query node_states to find the timeout-triggered flush's step_index
        with landscape_db.connection() as conn:
            # Get all node_states for this transform node
            states = conn.execute(
                select(node_states_table.c.step_index, node_states_table.c.node_id)
                .where(node_states_table.c.run_id == result.run_id)
                .where(node_states_table.c.node_id == node_id)
            ).fetchall()

        # There should be at least one state (the timeout flush)
        assert len(states) >= 1, f"Expected at least 1 node_state, got {len(states)}"

        # All step_index values should be 1 (1-indexed for first transform)
        for step_index, _state_node_id in states:
            assert step_index == 1, (
                f"Timeout flush step_index should be 1 (1-indexed), got {step_index}. "
                f"Bug: _find_aggregation_transform returns 0-indexed position which "
                f"flows directly to audit recording without +1 conversion."
            )

    def test_end_of_source_flush_records_1_indexed_step(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """End-of-source flush at first transform should record step_index=1, not 0.

        Same as timeout test but using END_OF_SOURCE trigger (no timeout, just
        source completion with buffered rows).
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table

        class FastSource(_TestSourceBase):
            """Source that completes quickly with buffered rows."""

            name = "fast_source_step_test"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                # Emit rows that will be buffered (count trigger won't fire)
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                # Source completes - end-of-source flush triggers

            def close(self) -> None:
                pass

        class BatchAgg(BaseTransform):
            """Simple aggregation that returns batch sum."""

            name = "batch_agg_eos_step_test"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class CollectorSink(_TestSinkBase):
            """Sink that collects output."""

            name = "collector_eos_step_test"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(BatchAgg())
        sink = as_sink(CollectorSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],  # First (and only) transform at position 0
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        agg_settings = AggregationSettings(
            name="test_eos_step",
            plugin="batch_agg_eos_step_test",
            trigger=TriggerConfig(count=100),  # High count, no timeout - forces END_OF_SOURCE
            output_mode="transform",
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_step_test", options={}),
            sinks={"output": SinkSettings(plugin="collector_eos_step_test", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED, f"Run failed: {result}"

        # Query node_states to find the END_OF_SOURCE flush's step_index
        with landscape_db.connection() as conn:
            states = conn.execute(
                select(node_states_table.c.step_index, node_states_table.c.node_id)
                .where(node_states_table.c.run_id == result.run_id)
                .where(node_states_table.c.node_id == node_id)
            ).fetchall()

        assert len(states) >= 1, f"Expected at least 1 node_state, got {len(states)}"

        for step_index, _state_node_id in states:
            assert step_index == 1, (
                f"END_OF_SOURCE flush step_index should be 1 (1-indexed), got {step_index}. "
                f"Bug: _find_aggregation_transform returns 0-indexed position which "
                f"flows directly to audit recording without +1 conversion."
            )


class TestExpectedOutputCountEnforcement:
    """Test runtime enforcement of expected_output_count for aggregations.

    When expected_output_count is configured, the processor must validate
    that the aggregation produces exactly that many output rows. This is
    a plugin contract violation (hard error) if mismatched.

    Covers:
    - _process_batch_aggregation_node() path (count trigger flush)
    - handle_timeout_flush() path (timeout/end-of-source flush)
    """

    def test_expected_output_count_matches_passes(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Aggregation with matching expected_output_count completes successfully."""

        class FastSource(_TestSourceBase):
            """Source that emits rows for aggregation."""

            name = "fast_source_count_match"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class SingleRowAgg(BaseTransform):
            """Aggregation that produces exactly 1 output row."""

            name = "single_row_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total, "count": len(row)}, success_reason={"action": "test"})
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class SimpleSink(_TestSinkBase):
            """Simple sink."""

            name = "simple_sink_count_match"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(SingleRowAgg())
        sink = as_sink(SimpleSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        # expected_output_count=1 matches what SingleRowAgg produces
        agg_settings = AggregationSettings(
            name="test_count_match",
            plugin="single_row_agg",
            trigger=TriggerConfig(count=3),  # Flush after 3 rows
            output_mode="transform",
            expected_output_count=1,  # Expect exactly 1 output row
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_count_match", options={}),
            sinks={"output": SinkSettings(plugin="simple_sink_count_match", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Should complete successfully since output count matches
        assert result.status == RunStatus.COMPLETED, f"Run should complete when output count matches: {result}"
        assert len(sink.rows) == 1, f"Should have exactly 1 output row, got {len(sink.rows)}"

    def test_expected_output_count_mismatch_raises_runtime_error(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Aggregation with mismatched expected_output_count raises RuntimeError.

        This tests the _process_batch_aggregation_node() path where a count
        trigger causes the flush.
        """

        class FastSource(_TestSourceBase):
            """Source that emits rows for aggregation."""

            name = "fast_source_count_mismatch"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class MultiRowAgg(BaseTransform):
            """Aggregation that produces multiple output rows (violates expected_output_count=1)."""

            name = "multi_row_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Return 2 rows instead of 1 - this violates expected_output_count=1
                    return TransformResult.success_multi(
                        [{"part": 1, "total": sum(r.get("value", 0) for r in row)}, {"part": 2, "count": len(row)}],
                        success_reason={"action": "test"},
                    )
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class SimpleSink(_TestSinkBase):
            """Simple sink."""

            name = "simple_sink_count_mismatch"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(MultiRowAgg())
        sink = as_sink(SimpleSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        # expected_output_count=1 but MultiRowAgg produces 2 rows
        agg_settings = AggregationSettings(
            name="test_count_mismatch",
            plugin="multi_row_agg",
            trigger=TriggerConfig(count=3),  # Flush after 3 rows
            output_mode="transform",
            expected_output_count=1,  # Expect 1, but plugin produces 2
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_count_mismatch", options={}),
            sinks={"output": SinkSettings(plugin="simple_sink_count_mismatch", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)

        # Should raise RuntimeError due to output count mismatch
        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        error_msg = str(exc_info.value)
        assert "test_count_mismatch" in error_msg, f"Error should mention aggregation name: {error_msg}"
        assert "2" in error_msg, f"Error should mention actual count (2): {error_msg}"
        assert "expected_output_count=1" in error_msg, f"Error should mention expected count: {error_msg}"
        assert "plugin contract violation" in error_msg.lower(), f"Error should mention contract violation: {error_msg}"

    def test_expected_output_count_timeout_flush_path(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Test expected_output_count enforcement in handle_timeout_flush path.

        Uses MockClock to trigger timeout-based flush, which exercises
        handle_timeout_flush() rather than _process_batch_aggregation_node().
        """
        clock = MockClock(start=0.0)

        def advance_after_first_row(row_idx: int) -> None:
            if row_idx == 0:
                clock.advance(0.5)  # Advance past timeout

        callback_source = CallbackSource(
            rows=[
                {"id": 1, "value": 100},  # Buffered, then timeout triggers
                {"id": 2, "value": 200},  # This row triggers timeout check
            ],
            output_schema=_TestSchema,
            after_yield_callback=advance_after_first_row,
            source_name="timeout_count_source",
        )

        class MultiRowAgg(BaseTransform):
            """Aggregation that produces 2 output rows (violates expected_output_count=1)."""

            name = "timeout_multi_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Return 2 rows - violates expected_output_count=1
                    return TransformResult.success_multi(
                        [{"part": 1}, {"part": 2}],
                        success_reason={"action": "test"},
                    )
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class SimpleSink(_TestSinkBase):
            """Simple sink."""

            name = "timeout_sink"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(callback_source)
        transform = as_transform(MultiRowAgg())
        sink = as_sink(SimpleSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        # Timeout triggers flush, which will produce 2 rows vs expected 1
        agg_settings = AggregationSettings(
            name="timeout_test",
            plugin="timeout_multi_agg",
            trigger=TriggerConfig(timeout_seconds=0.1, count=100),  # Timeout will fire
            output_mode="transform",
            expected_output_count=1,  # Plugin produces 2, should fail
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="timeout_count_source", options={}),
            sinks={"output": SinkSettings(plugin="timeout_sink", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db, clock=clock)

        # Should raise RuntimeError due to output count mismatch on timeout flush
        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        error_msg = str(exc_info.value)
        assert "timeout_test" in error_msg, f"Error should mention aggregation name: {error_msg}"
        assert "expected_output_count=1" in error_msg, f"Error should mention expected count: {error_msg}"

    def test_expected_output_count_none_skips_validation(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """When expected_output_count is None, no validation is performed.

        This is the default behavior - aggregations can produce any number
        of output rows.
        """

        class FastSource(_TestSourceBase):
            """Source that emits rows."""

            name = "fast_source_no_count"
            output_schema = _TestSchema

            def __init__(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"id": 1, "value": 100})
                yield SourceRow.valid({"id": 2, "value": 200})
                yield SourceRow.valid({"id": 3, "value": 300})

            def close(self) -> None:
                pass

        class VariableOutputAgg(BaseTransform):
            """Aggregation that produces variable number of rows."""

            name = "variable_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    # Return N rows where N = len(input) - could be any number
                    return TransformResult.success_multi(
                        [{"idx": i, "value": r.get("value", 0)} for i, r in enumerate(row)],
                        success_reason={"action": "test"},
                    )
                return TransformResult.success(dict(row), success_reason={"action": "test"})

        class SimpleSink(_TestSinkBase):
            """Simple sink."""

            name = "simple_sink_no_count"

            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                for row in rows:
                    self.rows.append(row)
                return ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test")

            def close(self) -> None:
                pass

        source = as_source(FastSource())
        transform = as_transform(VariableOutputAgg())
        sink = as_sink(SimpleSink())

        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
            coalesce_settings=None,
        )

        node_id = graph.get_transform_id_map()[0]

        # No expected_output_count - any output count is acceptable
        agg_settings = AggregationSettings(
            name="test_no_count",
            plugin="variable_agg",
            trigger=TriggerConfig(count=3),
            output_mode="transform",
            # expected_output_count not set (None)
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=[],
            aggregation_settings={node_id: agg_settings},
            coalesce_settings={},
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="fast_source_no_count", options={}),
            sinks={"output": SinkSettings(plugin="simple_sink_no_count", options={})},
            default_sink="output",
            transforms=[],
            gates=[],
            aggregation={},
        )

        orchestrator = Orchestrator(db=landscape_db)
        result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Should complete successfully without validation
        assert result.status == RunStatus.COMPLETED, f"Run should complete when expected_output_count is None: {result}"
        # Variable output agg produces 3 rows (one per input)
        assert len(sink.rows) == 3, f"Should have 3 output rows, got {len(sink.rows)}"
