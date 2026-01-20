# tests/engine/test_processor_outcomes.py
"""Integration tests for processor outcome recording (AUD-001).

These tests verify that the processor records token outcomes at determination
points, creating entries in the token_outcomes table for audit trail completeness.
"""

from typing import Any, ClassVar

import pytest

from elspeth.contracts import NodeType, PluginSchema, RowOutcome
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class _TestSchema(PluginSchema):
    """Dynamic schema for test plugins."""

    model_config: ClassVar[dict[str, Any]] = {"extra": "allow"}


class TestProcessorRecordsOutcomes:
    """Test that processor records outcomes at determination points."""

    @pytest.fixture
    def setup_pipeline(self):
        """Set up minimal pipeline for testing outcome recording."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        return db, recorder

    def test_completed_outcome_recorded_at_pipeline_end(self, setup_pipeline) -> None:
        """Default COMPLETED outcome is recorded when row reaches end."""
        _db, recorder = setup_pipeline
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        # Create run
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register minimal nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="src",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create a simple passthrough transform
        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create processor
        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # Process a row
        results = processor.process_row(
            row_index=0,
            row_data={"x": 1},
            transforms=[PassthroughTransform(transform_node.node_id)],
            ctx=ctx,
        )

        # Should get COMPLETED result
        assert len(results) == 1
        result = results[0]
        assert result.outcome == RowOutcome.COMPLETED

        # Verify outcome was recorded in audit trail
        outcome = recorder.get_token_outcome(result.token.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.COMPLETED
        assert outcome.is_terminal is True

    def test_completed_outcome_without_transforms(self, setup_pipeline) -> None:
        """COMPLETED outcome recorded even when no transforms in pipeline."""
        _db, recorder = setup_pipeline
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="src",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # Process with no transforms - goes straight to COMPLETED
        results = processor.process_row(
            row_index=0,
            row_data={"x": 42},
            transforms=[],
            ctx=ctx,
        )

        assert len(results) == 1
        result = results[0]
        assert result.outcome == RowOutcome.COMPLETED

        # Verify outcome was recorded
        outcome = recorder.get_token_outcome(result.token.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.COMPLETED
        assert outcome.is_terminal is True

    def test_outcome_api_works_directly(self, setup_pipeline) -> None:
        """Verify the record_token_outcome API works as expected."""
        _db, recorder = setup_pipeline

        # Create run
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register minimal nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="src",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create row and token
        row = recorder.create_row(run.run_id, source.node_id, 0, {"x": 1})
        token = recorder.create_token(row.row_id)

        # Record COMPLETED outcome directly
        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="sink")

        # Verify outcome recorded
        outcome = recorder.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.COMPLETED
        assert outcome.sink_name == "sink"
        assert outcome.is_terminal is True
