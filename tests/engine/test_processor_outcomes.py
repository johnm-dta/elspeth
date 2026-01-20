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


class TestAllOutcomeTypesRecorded:
    """Verify all 9 outcome types are properly recorded in the audit trail.

    This is the comprehensive integration test for AUD-001, proving that:
    1. All 9 outcome types can be recorded with appropriate context
    2. The recorded outcomes can be retrieved via get_token_outcome()
    3. The is_terminal flag is correctly set based on outcome type
    """

    @pytest.mark.parametrize(
        "outcome,kwargs",
        [
            (RowOutcome.COMPLETED, {"sink_name": "test_sink"}),
            (RowOutcome.ROUTED, {"sink_name": "error_sink"}),
            (RowOutcome.FORKED, {"fork_group_id": "fork_123"}),
            (RowOutcome.FAILED, {"error_hash": "abc123"}),
            (RowOutcome.QUARANTINED, {"error_hash": "def456"}),
            (RowOutcome.COALESCED, {"join_group_id": "join_123"}),
            (RowOutcome.EXPANDED, {"expand_group_id": "expand_123"}),
        ],
    )
    def test_outcome_type_can_be_recorded(self, outcome: RowOutcome, kwargs: dict[str, str]) -> None:
        """Each outcome type (non-batch) should be recordable with appropriate context."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)

        outcome_id = recorder.record_token_outcome(run.run_id, token.token_id, outcome, **kwargs)

        assert outcome_id is not None
        recorded = recorder.get_token_outcome(token.token_id)
        assert recorded is not None
        assert recorded.outcome == outcome
        assert recorded.is_terminal == outcome.is_terminal

    @pytest.mark.parametrize(
        "outcome",
        [
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.BUFFERED,
        ],
    )
    def test_batch_outcome_type_can_be_recorded(self, outcome: RowOutcome) -> None:
        """Batch-related outcomes require a real batch (FK constraint)."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        # Need an aggregation node for the batch
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)

        # Create a real batch for FK constraint
        batch = recorder.create_batch(run.run_id, "agg")

        outcome_id = recorder.record_token_outcome(run.run_id, token.token_id, outcome, batch_id=batch.batch_id)

        assert outcome_id is not None
        recorded = recorder.get_token_outcome(token.token_id)
        assert recorded is not None
        assert recorded.outcome == outcome
        assert recorded.batch_id == batch.batch_id
        assert recorded.is_terminal == outcome.is_terminal


class TestTerminalUniquenessConstraint:
    """Test the partial unique index enforces exactly one terminal outcome per token."""

    def test_only_one_terminal_outcome_per_token(self) -> None:
        """Partial unique index enforces exactly one terminal outcome.

        The token_outcomes table has a partial unique index on token_id
        where is_terminal=1. This means:
        - Multiple BUFFERED outcomes are allowed (non-terminal)
        - Only one terminal outcome (COMPLETED, ROUTED, etc.) is allowed
        - Second terminal outcome raises IntegrityError
        """
        from sqlalchemy.exc import IntegrityError

        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)

        # First terminal outcome succeeds
        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out")

        # Second terminal outcome fails with IntegrityError
        with pytest.raises(IntegrityError):
            recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.ROUTED, sink_name="err")

    def test_multiple_buffered_outcomes_allowed(self) -> None:
        """Multiple BUFFERED (non-terminal) outcomes are allowed.

        BUFFERED is non-terminal, so the partial unique index doesn't apply.
        A token can be buffered multiple times before reaching its terminal state.
        """
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        # Need an aggregation node for batches
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)

        # Create real batches for FK constraint
        batch1 = recorder.create_batch(run.run_id, "agg")
        batch2 = recorder.create_batch(run.run_id, "agg")

        # Multiple BUFFERED outcomes should succeed
        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.BUFFERED, batch_id=batch1.batch_id)
        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.BUFFERED, batch_id=batch2.batch_id)

        # Then terminal outcome should also succeed
        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out")

        # Verify terminal outcome is returned (preferred over non-terminal)
        recorded = recorder.get_token_outcome(token.token_id)
        assert recorded is not None
        assert recorded.outcome == RowOutcome.COMPLETED
        assert recorded.is_terminal is True


class TestExplainShowsRecordedOutcome:
    """Test end-to-end flow: record outcome -> explain() includes it."""

    def test_explain_shows_recorded_outcome(self) -> None:
        """Full flow: record outcome -> explain() includes it."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"data": "test"})
        token = recorder.create_token(row.row_id)

        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.ROUTED, sink_name="errors")

        lineage = explain(recorder, run.run_id, token_id=token.token_id)

        assert lineage is not None
        assert lineage.outcome is not None
        assert lineage.outcome.outcome == RowOutcome.ROUTED
        assert lineage.outcome.sink_name == "errors"
        assert lineage.outcome.is_terminal is True

    def test_explain_shows_outcome_context_fields(self) -> None:
        """Explain returns all outcome-specific context fields."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"data": "test"})
        token = recorder.create_token(row.row_id)

        # Record FAILED outcome with error_hash
        recorder.record_token_outcome(
            run.run_id,
            token.token_id,
            RowOutcome.FAILED,
            error_hash="err_abc123",
        )

        lineage = explain(recorder, run.run_id, token_id=token.token_id)

        assert lineage is not None
        assert lineage.outcome is not None
        assert lineage.outcome.outcome == RowOutcome.FAILED
        assert lineage.outcome.error_hash == "err_abc123"
        assert lineage.outcome.is_terminal is True

    def test_explain_returns_none_outcome_when_not_recorded(self) -> None:
        """Explain returns None for outcome when none recorded."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"data": "test"})
        token = recorder.create_token(row.row_id)
        # No outcome recorded

        lineage = explain(recorder, run.run_id, token_id=token.token_id)

        assert lineage is not None
        assert lineage.outcome is None
