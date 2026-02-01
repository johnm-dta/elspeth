# tests/engine/test_processor_outcomes.py
"""Integration tests for processor outcome recording (AUD-001).

These tests verify that the processor records token outcomes at determination
points, creating entries in the token_outcomes table for audit trail completeness.
"""

from typing import Any

import pytest

from elspeth.contracts import NodeType, RoutingMode, RowOutcome, RunStatus
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import GateName, NodeID

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestProcessorRecordsOutcomes:
    """Test that processor records outcomes at determination points."""

    @pytest.fixture
    def setup_pipeline(self, landscape_db):
        """Set up minimal pipeline for testing outcome recording."""
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        return landscape_db, recorder

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
    def test_outcome_type_can_be_recorded(self, landscape_db, outcome: RowOutcome, kwargs: dict[str, Any]) -> None:
        """Each outcome type (non-batch) should be recordable with appropriate context."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

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

        # === P1: Assert outcome-specific context fields ===
        if outcome == RowOutcome.COMPLETED:
            assert recorded.sink_name == kwargs["sink_name"], "COMPLETED should have sink_name"
        elif outcome == RowOutcome.ROUTED:
            assert recorded.sink_name == kwargs["sink_name"], "ROUTED should have sink_name"
        elif outcome == RowOutcome.FORKED:
            assert recorded.fork_group_id == kwargs["fork_group_id"], "FORKED should have fork_group_id"
        elif outcome == RowOutcome.COALESCED:
            assert recorded.join_group_id == kwargs["join_group_id"], "COALESCED should have join_group_id"
        elif outcome == RowOutcome.FAILED:
            assert recorded.error_hash == kwargs["error_hash"], "FAILED should have error_hash"
        elif outcome == RowOutcome.QUARANTINED:
            assert recorded.error_hash == kwargs["error_hash"], "QUARANTINED should have error_hash"
        elif outcome == RowOutcome.EXPANDED:
            assert recorded.expand_group_id == kwargs["expand_group_id"], "EXPANDED should have expand_group_id"

    @pytest.mark.parametrize(
        "outcome",
        [
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.BUFFERED,
        ],
    )
    def test_batch_outcome_type_can_be_recorded(self, landscape_db, outcome: RowOutcome) -> None:
        """Batch-related outcomes require a real batch (FK constraint)."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

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

    def test_only_one_terminal_outcome_per_token(self, landscape_db) -> None:
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
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

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

    def test_multiple_buffered_outcomes_allowed(self, landscape_db) -> None:
        """Multiple BUFFERED (non-terminal) outcomes are allowed.

        BUFFERED is non-terminal, so the partial unique index doesn't apply.
        A token can be buffered multiple times before reaching its terminal state.
        """
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

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

    def test_explain_shows_recorded_outcome(self, landscape_db) -> None:
        """Full flow: record outcome -> explain() includes it."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        recorder = LandscapeRecorder(landscape_db)

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

    def test_explain_shows_outcome_context_fields(self, landscape_db) -> None:
        """Explain returns all outcome-specific context fields."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        recorder = LandscapeRecorder(landscape_db)

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

    def test_explain_returns_none_outcome_when_not_recorded(self, landscape_db) -> None:
        """Explain returns None for outcome when none recorded."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        recorder = LandscapeRecorder(landscape_db)

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


class TestEngineIntegrationOutcomes:
    """Engine-level tests that run processor and verify outcomes in audit trail.

    Unlike the direct LandscapeRecorder tests above, these tests exercise the full
    RowProcessor path to verify outcomes are recorded correctly end-to-end.
    """

    def test_processor_records_completed_outcome_with_context(self, landscape_db) -> None:
        """RowProcessor should record COMPLETED outcome with correct context."""
        from typing import Any, ClassVar

        from pydantic import ConfigDict

        from elspeth.contracts import PluginSchema, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class _TestSchema(PluginSchema):
            model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        class EnricherTransform(BaseTransform):
            name = "enricher"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "enriched": True}, success_reason={"action": "enrich"})

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[EnricherTransform(transform.node_id)],
            ctx=ctx,
        )

        # Verify result
        assert len(results) == 1
        result = results[0]
        assert result.outcome == RowOutcome.COMPLETED

        # Note: COMPLETED token_outcomes are recorded by orchestrator at sink level,
        # not by the processor. The processor records node_states for transforms.
        # We verify the node_states were recorded correctly with hashes.
        states = recorder.get_node_states_for_token(result.token.token_id)
        assert len(states) == 1, "Should have node_state for the transform"
        state = states[0]
        assert state.status == RunStatus.COMPLETED, "Transform should complete successfully"
        assert state.input_hash is not None, "Input hash should be recorded"
        assert hasattr(state, "output_hash") and state.output_hash is not None, "Output hash should be recorded"

    def test_processor_records_quarantined_outcome_with_error_hash(self) -> None:
        """RowProcessor should record QUARANTINED outcome with error_hash."""
        from typing import Any, ClassVar

        from pydantic import ConfigDict

        from elspeth.contracts import NodeType, PluginSchema, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class _TestSchema(PluginSchema):
            model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="validator",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        class ValidatingTransform(BaseTransform):
            name = "validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Quarantine on error

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                if row["value"] < 0:
                    return TransformResult.error({"reason": "validation_failed", "error": "negative_value"})
                return TransformResult.success(row, success_reason={"action": "test"})

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        results = processor.process_row(
            row_index=0,
            row_data={"value": -5},
            transforms=[ValidatingTransform(transform.node_id)],
            ctx=ctx,
        )

        # Verify result
        assert len(results) == 1
        result = results[0]
        assert result.outcome == RowOutcome.QUARANTINED

        # Query the audit trail directly and verify outcome with error_hash
        outcome = recorder.get_token_outcome(result.token.token_id)
        assert outcome is not None, "Token outcome should be recorded by processor"
        assert outcome.outcome == RowOutcome.QUARANTINED, "Outcome should be QUARANTINED"
        assert outcome.error_hash is not None, "Error hash should be recorded for quarantine"
        assert outcome.is_terminal is True, "QUARANTINED is terminal"

    def test_processor_records_forked_outcome_with_fork_group_id(self) -> None:
        """RowProcessor should record FORKED outcome with fork_group_id and parent lineage."""
        from elspeth.contracts import NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        path_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        path_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        # Register edges for fork
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=path_a.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=path_b.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Config-driven fork gate
        splitter_gate = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            edge_map={
                (NodeID(gate.node_id), "path_a"): edge_a.edge_id,
                (NodeID(gate.node_id), "path_b"): edge_b.edge_id,
            },
            config_gates=[splitter_gate],
            config_gate_id_map={GateName("splitter"): NodeID(gate.node_id)},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[],
            ctx=ctx,
        )

        # Should have parent (FORKED) + 2 children (COMPLETED)
        assert len(results) == 3

        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        completed_results = [r for r in results if r.outcome == RowOutcome.COMPLETED]

        assert len(forked_results) == 1
        assert len(completed_results) == 2

        parent = forked_results[0]

        # Verify FORKED outcome has fork_group_id
        parent_outcome = recorder.get_token_outcome(parent.token.token_id)
        assert parent_outcome is not None, "Parent token outcome should be recorded"
        assert parent_outcome.outcome == RowOutcome.FORKED, "Should be FORKED"
        assert parent_outcome.fork_group_id is not None, "Fork group ID should be set"

        # Verify children have parent lineage
        for child in completed_results:
            parents = recorder.get_token_parents(child.token.token_id)
            assert len(parents) == 1, "Each child should have exactly 1 parent"
            assert parents[0].parent_token_id == parent.token.token_id, "Parent should be forked token"
