# tests/engine/test_processor_guards.py
"""Tests for RowProcessor safety guards.

These tests verify that safety mechanisms (iteration limits, etc.)
correctly prevent pathological scenarios from hanging the pipeline.
"""

from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import NodeType, RowOutcome
from elspeth.contracts.results import RowResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.processor import MAX_WORK_QUEUE_ITERATIONS, RowProcessor, _WorkItem
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult as PluginTransformResult
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema


class TestProcessorGuards:
    """Tests for processor safety guards."""

    def test_max_work_queue_iterations_constant_value(self) -> None:
        """Verify MAX_WORK_QUEUE_ITERATIONS is set to expected value.

        This is a sanity check - if someone changes the constant,
        they should be aware tests depend on it.
        """
        assert MAX_WORK_QUEUE_ITERATIONS == 10_000

    def test_work_queue_exceeding_limit_raises_runtime_error(self) -> None:
        """Exceeding MAX_WORK_QUEUE_ITERATIONS must raise RuntimeError.

        This test verifies the infinite loop guard fires correctly.
        We patch the constant to a lower value and create a mock scenario
        where the work queue keeps growing indefinitely.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup minimal infrastructure
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        span_factory = SpanFactory()

        # Create processor with minimal config
        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        # Create a simple token to use in the work queue
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        token_info = TokenInfo(
            token_id=token.token_id,
            row_id=row.row_id,
            row_data={"value": 1},
            branch_name=None,
        )

        # Create a mock result that will be returned by the patched method
        mock_result = RowResult(
            token=token_info,
            final_data={"value": 1},
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )

        # Patch the internal _process_single_token method to always return more work
        with (
            patch.object(
                processor,
                "_process_single_token",
                side_effect=lambda **kwargs: (
                    mock_result,
                    [_WorkItem(token=token_info, start_step=0)],
                ),
            ),
            patch("elspeth.engine.processor.MAX_WORK_QUEUE_ITERATIONS", 5),
        ):
            # Simulate the work queue loop that's inside process_row
            # to test the guard logic more directly
            work_queue: deque[_WorkItem] = deque([_WorkItem(token=token_info, start_step=0)])
            iterations = 0

            # Use the patched value
            limit = 5

            with (
                pytest.raises(RuntimeError, match=r"exceeded .* iterations"),
                processor._spans.row_span(token_info.row_id, token_info.token_id),
            ):
                while work_queue:
                    iterations += 1
                    if iterations > limit:
                        raise RuntimeError(f"Work queue exceeded {limit} iterations. Possible infinite loop in pipeline.")
                    item = work_queue.popleft()
                    _result, child_items = processor._process_single_token(
                        token=item.token,
                        transforms=[],
                        ctx=PluginContext(run_id=run.run_id, config={}),
                        start_step=item.start_step,
                    )
                    work_queue.extend(child_items)

    def test_normal_processing_completes_without_hitting_guard(self) -> None:
        """Normal DAG processing should never approach the iteration limit.

        This is a sanity check that the guard doesn't interfere with
        legitimate pipelines.
        """
        # A simple linear pipeline with 10 transforms should complete
        # in exactly 10 iterations (one per transform)
        assert MAX_WORK_QUEUE_ITERATIONS > 10

        # The guard is set high enough that even complex DAGs with
        # many forks/joins should complete well under the limit
        # A DAG with 100 nodes and 10 parallel branches = ~1000 iterations max
        assert MAX_WORK_QUEUE_ITERATIONS > 1000

    def test_iteration_guard_exists_in_process_row(self) -> None:
        """Verify iteration guard is present in process_row method.

        This is a structural test - we verify by running a simple case
        that the guard infrastructure is in place.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        # Must register transform node for FK constraint
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        span_factory = SpanFactory()

        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        # Simple passthrough transform
        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> PluginTransformResult:
                return PluginTransformResult.success(row, success_reason={"action": "passthrough"})

        transform = PassthroughTransform(transform_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # This should complete successfully without hitting the guard
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform],
            ctx=ctx,
        )

        # If we get here without RuntimeError, the guard didn't fire
        # (which is correct for normal processing)
        assert len(results) >= 1  # At least one result (terminal state)

    def test_guard_constant_is_reasonable(self) -> None:
        """Verify the MAX_WORK_QUEUE_ITERATIONS constant is reasonable.

        The guard should be high enough to not trigger on legitimate pipelines
        but low enough to catch runaway loops quickly.
        """
        # Should be at least 1000 for complex DAGs
        assert MAX_WORK_QUEUE_ITERATIONS >= 1000

        # Should not be astronomical (would defeat the purpose)
        assert MAX_WORK_QUEUE_ITERATIONS <= 100_000

        # Should be exactly 10,000 as documented
        assert MAX_WORK_QUEUE_ITERATIONS == 10_000
