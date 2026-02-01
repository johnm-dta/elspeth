# tests/engine/test_processor_coalesce.py
"""Tests for RowProcessor coalesce functionality.

This module contains tests for coalesce-related functionality extracted from
test_processor.py, including:
- TestRowProcessorCoalesce: Tests for fork->coalesce flows, policies, and audit trails
- TestCoalesceLinkage: Tests for branch-to-coalesce mapping in RowProcessor

Test plugins inherit from base classes (BaseTransform)
because the processor uses isinstance() for type-safe plugin detection.
Gates are config-driven using GateSettings.
"""

from typing import Any

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    RowOutcome,
    TransformResult,
)
from tests.conftest import as_transform
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema


class TestRowProcessorCoalesce:
    """Test RowProcessor integration with CoalesceExecutor."""

    def test_processor_accepts_coalesce_executor(self, landscape_db: "LandscapeDB") -> None:
        """RowProcessor should accept coalesce_executor parameter."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        # Should not raise
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            coalesce_executor=coalesce_executor,
        )
        assert processor._coalesce_executor is coalesce_executor

    def test_fork_then_coalesce_require_all(self, landscape_db: "LandscapeDB") -> None:
        """Fork children should coalesce when all branches arrive.

        Pipeline: source -> enrich_a -> enrich_b -> fork_gate -> coalesce -> completed

        This test verifies the full fork->coalesce flow using config gates:
        1. Transforms enrich data (sentiment, entities)
        2. Gate forks to two paths (path_a, path_b) - children inherit enriched data
        3. Coalesce merges both paths with require_all policy
        4. Parent token becomes FORKED, children become COALESCED
        5. Merged token has fields from both transforms
        """
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes (transforms before gate since config gates run after transforms)
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        fork_gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Setup coalesce executor
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Transforms enrich data before the fork
        class EnrichA(BaseTransform):
            name = "enrich_a"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "sentiment": "positive"}, success_reason={"action": "enrich"})

        class EnrichB(BaseTransform):
            name = "enrich_b"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "entities": ["ACME"]}, success_reason={"action": "enrich"})

        # Config-driven fork gate
        fork_gate_config = GateSettings(
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
                (NodeID(fork_gate.node_id), "path_a"): edge_a.edge_id,
                (NodeID(fork_gate.node_id), "path_b"): edge_b.edge_id,
            },
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merger"): NodeID(coalesce_node.node_id)},
            config_gates=[fork_gate_config],
            config_gate_id_map={GateName("splitter"): NodeID(fork_gate.node_id)},
            branch_to_coalesce={
                BranchName("path_a"): CoalesceName("merger"),
                BranchName("path_b"): CoalesceName("merger"),
            },
            coalesce_step_map={CoalesceName("merger"): 3},  # transforms(2) + gate(1)
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process should:
        # 1. EnrichA adds sentiment
        # 2. EnrichB adds entities
        # 3. Fork at config gate (parent FORKED with both fields)
        # 4. Coalesce both paths (merged token COALESCED)
        results = processor.process_row(
            row_index=0,
            row_data={"text": "ACME earnings"},
            transforms=[
                EnrichA(transform_a.node_id),
                EnrichB(transform_b.node_id),
            ],
            ctx=ctx,
        )

        # Verify outcomes
        outcomes = {r.outcome for r in results}
        assert RowOutcome.FORKED in outcomes
        assert RowOutcome.COALESCED in outcomes

        # Find the coalesced result
        coalesced = [r for r in results if r.outcome == RowOutcome.COALESCED]
        assert len(coalesced) == 1

        # Verify merged data (both fields present from transforms before fork)
        merged_data = coalesced[0].final_data
        assert merged_data["sentiment"] == "positive"
        assert merged_data["entities"] == ["ACME"]

    def test_coalesced_token_audit_trail_complete(self, landscape_db: "LandscapeDB") -> None:
        """Coalesced tokens should have complete audit trail for explain().

        After enrich -> fork -> coalesce, querying explain() on the merged
        token should show:
        - Original source row
        - Transform processing steps
        - Fork point (parent token for forked children)
        - Both branch paths
        - Coalesce point with parent relationships

        This test verifies the audit infrastructure captures the complete
        lineage for a coalesced token, enabling explain() queries.
        """
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes (transforms before gate since config gates run after transforms)
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        fork_gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Setup coalesce executor
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Transforms enrich data before the fork
        class EnrichA(BaseTransform):
            name = "enrich_a"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "sentiment": "positive"}, success_reason={"action": "enrich"})

        class EnrichB(BaseTransform):
            name = "enrich_b"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "entities": ["ACME"]}, success_reason={"action": "enrich"})

        # Config-driven fork gate
        fork_gate_config = GateSettings(
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
                (NodeID(fork_gate.node_id), "path_a"): edge_a.edge_id,
                (NodeID(fork_gate.node_id), "path_b"): edge_b.edge_id,
            },
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merger"): NodeID(coalesce_node.node_id)},
            config_gates=[fork_gate_config],
            config_gate_id_map={GateName("splitter"): NodeID(fork_gate.node_id)},
            branch_to_coalesce={
                BranchName("path_a"): CoalesceName("merger"),
                BranchName("path_b"): CoalesceName("merger"),
            },
            coalesce_step_map={CoalesceName("merger"): 3},  # transforms(2) + gate(1)
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process the row through enrich -> fork -> coalesce
        results = processor.process_row(
            row_index=0,
            row_data={"text": "ACME earnings"},
            transforms=[
                EnrichA(transform_a.node_id),
                EnrichB(transform_b.node_id),
            ],
            ctx=ctx,
        )

        # === Verify outcomes ===
        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        coalesced_results = [r for r in results if r.outcome == RowOutcome.COALESCED]

        assert len(forked_results) == 1, "Should have exactly 1 FORKED result"
        assert len(coalesced_results) == 1, "Should have exactly 1 COALESCED result"

        forked = forked_results[0]
        coalesced = coalesced_results[0]

        # === Audit Trail: Verify source row exists ===
        row = recorder.get_row(forked.token.row_id)
        assert row is not None, "Source row should be recorded"
        assert row.row_index == 0
        assert row.source_node_id == source.node_id

        # === Audit Trail: Verify merged token has parent relationships ===
        # The merged token's parents are the consumed child tokens (with branch names)
        merged_token = coalesced.token
        merged_parents = recorder.get_token_parents(merged_token.token_id)
        assert len(merged_parents) == 2, "Merged token should have 2 parents (the consumed children)"

        # Get child token IDs from the merged token's parents
        child_token_ids = {p.parent_token_id for p in merged_parents}

        # Verify child tokens have branch names
        for child_token_id in child_token_ids:
            child_token = recorder.get_token(child_token_id)
            assert child_token is not None, "Child token should exist"
            assert child_token.branch_name in (
                "path_a",
                "path_b",
            ), f"Child token should have branch name, got {child_token.branch_name}"

        # Verify child tokens have parent relationships pointing to forked token
        for child_token_id in child_token_ids:
            parents = recorder.get_token_parents(child_token_id)
            assert len(parents) == 1, "Child token should have 1 parent"
            assert parents[0].parent_token_id == forked.token.token_id, "Parent should be the forked token"

        # === Audit Trail: Verify consumed tokens have node_states at coalesce ===
        # The CoalesceExecutor records node_states for consumed tokens
        for child_token_id in child_token_ids:
            states = recorder.get_node_states_for_token(child_token_id)
            # Should have states: gate evaluation + transform processing + coalesce
            assert len(states) >= 1, f"Child token {child_token_id} should have node states"

            # Check that at least one state is at the coalesce node
            coalesce_states = [s for s in states if s.node_id == coalesce_node.node_id]
            assert len(coalesce_states) == 1, "Child token should have exactly one coalesce node_state"

            coalesce_state = coalesce_states[0]
            assert coalesce_state.status == RunStatus.COMPLETED

        # === Audit Trail: Verify merged token has join_group_id ===
        merged_token_record = recorder.get_token(merged_token.token_id)
        assert merged_token_record is not None
        assert merged_token_record.join_group_id is not None, "Merged token should have join_group_id"

        # === Audit Trail: Verify complete lineage back to source ===
        # Follow the chain: merged_token -> children -> forked parent -> source row
        assert merged_token.row_id == row.row_id, "Merged token traces back to source row"

    def test_coalesce_best_effort_with_quarantined_child(self, landscape_db: "LandscapeDB") -> None:
        """best_effort policy should merge available children even if one quarantines.

        Scenario:
        - Fork to 3 paths: sentiment, entities, summary
        - summary path quarantines (transform returns TransformResult.error())
        - best_effort timeout triggers, merges sentiment + entities
        - Result should include FORKED, QUARANTINED, and COALESCED outcomes

        This test verifies the end-to-end flow using CoalesceExecutor directly
        to simulate the scenario where one branch is quarantined and never
        reaches the coalesce point.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.clock import MockClock
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        # Deterministic clock for timeout testing
        clock = MockClock(start=100.0)

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register minimal nodes needed for coalesce testing
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Setup coalesce with best_effort policy and short timeout
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["sentiment", "entities", "summary"],
            policy="best_effort",
            timeout_seconds=0.1,  # Short timeout for testing
            merge="union",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
            clock=clock,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Create tokens to simulate fork scenario
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            row_index=0,
            row_data={"text": "ACME earnings report"},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["sentiment", "entities", "summary"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate processing: sentiment and entities complete, summary is quarantined
        # sentiment child completes with enriched data
        sentiment_token = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"text": "ACME earnings report", "sentiment": "positive"},
            branch_name="sentiment",
        )
        outcome1 = coalesce_executor.accept(sentiment_token, "merger", step_in_pipeline=3)
        assert outcome1.held is True

        # entities child completes with enriched data
        entities_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"text": "ACME earnings report", "entities": ["ACME"]},
            branch_name="entities",
        )
        outcome2 = coalesce_executor.accept(entities_token, "merger", step_in_pipeline=3)
        assert outcome2.held is True  # Still waiting (need all 3 or timeout)

        # summary child is QUARANTINED - it never arrives at coalesce
        # (simulated by simply not calling accept for it)

        # Advance clock past timeout
        clock.advance(0.15)

        # Check timeouts - should trigger best_effort merge
        timed_out = coalesce_executor.check_timeouts("merger", step_in_pipeline=3)

        # Should have merged sentiment + entities (without summary)
        assert len(timed_out) == 1
        outcome = timed_out[0]
        assert outcome.held is False
        assert outcome.merged_token is not None
        assert outcome.failure_reason is None  # Not a failure, just partial merge

        # Verify merged data contains sentiment and entities but not summary
        merged_data = outcome.merged_token.row_data
        assert "sentiment" in merged_data
        assert merged_data["sentiment"] == "positive"
        assert "entities" in merged_data
        assert merged_data["entities"] == ["ACME"]
        # summary never arrived, so its data is NOT in merged result
        # (The original text field should be there from union merge)
        assert "text" in merged_data

        # Verify coalesce metadata shows partial merge
        assert outcome.coalesce_metadata is not None
        assert outcome.coalesce_metadata["policy"] == "best_effort"
        assert set(outcome.coalesce_metadata["branches_arrived"]) == {
            "sentiment",
            "entities",
        }
        assert "summary" not in outcome.coalesce_metadata["branches_arrived"]

    def test_coalesce_quorum_merges_at_threshold(self, landscape_db: "LandscapeDB") -> None:
        """Quorum policy should merge when quorum_count branches arrive.

        Setup: Fork to 3 paths (fast, medium, slow), quorum=2
        - When 2 of 3 arrive, merge immediately
        - 3rd branch result is discarded (arrives after merge)

        This test uses CoalesceExecutor directly to verify:
        1. First branch (fast) is held
        2. Second branch (medium) triggers merge at quorum=2
        3. Merged data contains only fast and medium
        4. Late arrival (slow) starts a new pending entry (doesn't crash)
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register minimal nodes needed for coalesce testing
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Setup coalesce with quorum policy (2 of 3)
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["fast", "medium", "slow"],
            policy="quorum",
            quorum_count=2,
            merge="nested",  # Use nested to see which branches contributed
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Create tokens to simulate fork scenario
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            row_index=0,
            row_data={"text": "test input"},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["fast", "medium", "slow"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate: fast arrives first with enriched data
        fast_token = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"text": "test input", "fast_result": "fast done"},
            branch_name="fast",
        )
        outcome1 = coalesce_executor.accept(fast_token, "merger", step_in_pipeline=3)

        # First arrival: should be held (quorum not met yet)
        assert outcome1.held is True
        assert outcome1.merged_token is None

        # Simulate: medium arrives second with enriched data
        medium_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"text": "test input", "medium_result": "medium done"},
            branch_name="medium",
        )
        outcome2 = coalesce_executor.accept(medium_token, "merger", step_in_pipeline=3)

        # Second arrival: quorum met (2 of 3), merge triggers immediately
        assert outcome2.held is False
        assert outcome2.merged_token is not None
        assert outcome2.failure_reason is None  # Not a failure

        # Verify merged data using nested strategy
        merged_data = outcome2.merged_token.row_data
        assert "fast" in merged_data, "Merged data should have 'fast' branch"
        assert "medium" in merged_data, "Merged data should have 'medium' branch"
        assert "slow" not in merged_data, "Merged data should NOT have 'slow' branch"

        # Check nested structure contains expected data
        assert merged_data["fast"]["fast_result"] == "fast done"
        assert merged_data["medium"]["medium_result"] == "medium done"

        # Verify coalesce metadata shows quorum merge
        assert outcome2.coalesce_metadata is not None
        assert outcome2.coalesce_metadata["policy"] == "quorum"
        assert set(outcome2.coalesce_metadata["branches_arrived"]) == {"fast", "medium"}
        assert outcome2.coalesce_metadata["expected_branches"] == [
            "fast",
            "medium",
            "slow",
        ]

        # Verify consumed tokens
        assert len(outcome2.consumed_tokens) == 2
        consumed_ids = {t.token_id for t in outcome2.consumed_tokens}
        assert fast_token.token_id in consumed_ids
        assert medium_token.token_id in consumed_ids

        # Verify arrival order is recorded (fast came before medium)
        arrival_order = outcome2.coalesce_metadata["arrival_order"]
        assert len(arrival_order) == 2
        assert arrival_order[0]["branch"] == "fast"  # First arrival
        assert arrival_order[1]["branch"] == "medium"  # Second arrival

        # === Late arrival behavior ===
        # The slow branch arrives after merge is complete.
        # With Gap #2 fix, late arrivals are now rejected with failure_reason.
        # This prevents orphan pending entries and confusing duplicate audit records.
        slow_token = TokenInfo(
            row_id=children[2].row_id,
            token_id=children[2].token_id,
            row_data={"text": "test input", "slow_result": "slow done"},
            branch_name="slow",
        )
        outcome3 = coalesce_executor.accept(slow_token, "merger", step_in_pipeline=3)

        # Late arrival gets rejected with proper failure outcome
        assert outcome3.held is False
        assert outcome3.merged_token is None
        assert outcome3.failure_reason == "late_arrival_after_merge"
        assert len(outcome3.consumed_tokens) == 1
        assert outcome3.consumed_tokens[0].token_id == slow_token.token_id

    def test_nested_fork_coalesce(self, landscape_db: "LandscapeDB") -> None:
        """Test fork within fork, with coalesce at each level.

        DAG structure:
        source -> gate1 (fork A,B) -> [
            path_a -> gate2 (fork A1,A2) -> [A1, A2] -> coalesce_inner -> ...
            path_b -> transform_b
        ] -> coalesce_outer

        Should produce:
        - 1 parent FORKED (gate1)
        - 2 level-1 children (path_a FORKED, path_b continues)
        - 2 level-2 children from path_a (A1, A2)
        - 1 inner COALESCED (A1+A2)
        - 1 outer COALESCED (inner+path_b)

        This test uses CoalesceExecutor directly to simulate the nested DAG flow,
        providing clear control over the token hierarchy at each level.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes for the nested DAG
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        inner_coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="inner_merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        outer_coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="outer_merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # === Setup two coalesce points: inner (A1+A2) and outer (inner+path_b) ===
        inner_coalesce_settings = CoalesceSettings(
            name="inner_merger",
            branches=["path_a1", "path_a2"],
            policy="require_all",
            merge="nested",  # Use nested to see branch structure
        )
        outer_coalesce_settings = CoalesceSettings(
            name="outer_merger",
            branches=["path_a_merged", "path_b"],  # inner result + path_b
            policy="require_all",
            merge="nested",
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(inner_coalesce_settings, inner_coalesce_node.node_id)
        coalesce_executor.register_coalesce(outer_coalesce_settings, outer_coalesce_node.node_id)

        # === Level 0: Create initial token (source row) ===
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            row_index=0,
            row_data={"text": "Document for nested processing"},
        )

        # === Level 1: Fork to path_a and path_b (gate1) ===
        level1_children, _fork_group_id1 = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )
        assert len(level1_children) == 2
        path_a_token = level1_children[0]  # branch_name="path_a"
        path_b_token = level1_children[1]  # branch_name="path_b"

        # Verify initial token is now the FORKED parent
        initial_token_record = recorder.get_token(initial_token.token_id)
        assert initial_token_record is not None

        # Verify children have correct branch names
        assert path_a_token.branch_name == "path_a"
        assert path_b_token.branch_name == "path_b"

        # === Level 2: path_a forks again to A1 and A2 (gate2) ===
        level2_children, _fork_group_id2 = token_manager.fork_token(
            parent_token=path_a_token,
            branches=["path_a1", "path_a2"],
            step_in_pipeline=2,
            run_id=run.run_id,
        )
        assert len(level2_children) == 2
        path_a1_token = level2_children[0]  # branch_name="path_a1"
        path_a2_token = level2_children[1]  # branch_name="path_a2"

        # path_a token is now FORKED (has children)
        path_a_record = recorder.get_token(path_a_token.token_id)
        assert path_a_record is not None

        # Verify level 2 branch names
        assert path_a1_token.branch_name == "path_a1"
        assert path_a2_token.branch_name == "path_a2"

        # === Process level 2 children (simulate transform enrichment) ===
        # A1 adds sentiment analysis
        enriched_a1 = TokenInfo(
            row_id=path_a1_token.row_id,
            token_id=path_a1_token.token_id,
            row_data={
                "text": "Document for nested processing",
                "sentiment": "positive",
            },
            branch_name="path_a1",
        )
        # A2 adds entity extraction
        enriched_a2 = TokenInfo(
            row_id=path_a2_token.row_id,
            token_id=path_a2_token.token_id,
            row_data={
                "text": "Document for nested processing",
                "entities": ["ACME", "2024"],
            },
            branch_name="path_a2",
        )

        # === Inner coalesce: merge A1 + A2 ===
        inner_outcome1 = coalesce_executor.accept(enriched_a1, "inner_merger", step_in_pipeline=3)
        assert inner_outcome1.held is True  # First arrival, waiting for A2

        inner_outcome2 = coalesce_executor.accept(enriched_a2, "inner_merger", step_in_pipeline=3)
        assert inner_outcome2.held is False  # Both arrived, merge triggered
        assert inner_outcome2.merged_token is not None
        assert inner_outcome2.failure_reason is None

        inner_merged_token = inner_outcome2.merged_token

        # Verify inner merge consumed both A1 and A2
        assert len(inner_outcome2.consumed_tokens) == 2
        consumed_inner_ids = {t.token_id for t in inner_outcome2.consumed_tokens}
        assert enriched_a1.token_id in consumed_inner_ids
        assert enriched_a2.token_id in consumed_inner_ids

        # Verify inner merged data has nested structure
        inner_merged_data = inner_merged_token.row_data
        assert "path_a1" in inner_merged_data
        assert "path_a2" in inner_merged_data
        assert inner_merged_data["path_a1"]["sentiment"] == "positive"
        assert inner_merged_data["path_a2"]["entities"] == ["ACME", "2024"]

        # === Process path_b (simulate transform enrichment) ===
        enriched_b = TokenInfo(
            row_id=path_b_token.row_id,
            token_id=path_b_token.token_id,
            row_data={
                "text": "Document for nested processing",
                "category": "financial",
            },
            branch_name="path_b",
        )

        # === Outer coalesce: merge inner_merged + path_b ===
        # First, prepare inner merged token for outer coalesce
        # It needs branch_name="path_a_merged" to match outer coalesce config
        inner_for_outer = TokenInfo(
            row_id=inner_merged_token.row_id,
            token_id=inner_merged_token.token_id,
            row_data=inner_merged_token.row_data,
            branch_name="path_a_merged",  # Assign branch for outer coalesce
        )

        outer_outcome1 = coalesce_executor.accept(inner_for_outer, "outer_merger", step_in_pipeline=4)
        assert outer_outcome1.held is True  # Waiting for path_b

        outer_outcome2 = coalesce_executor.accept(enriched_b, "outer_merger", step_in_pipeline=4)
        assert outer_outcome2.held is False  # Both arrived, final merge triggered
        assert outer_outcome2.merged_token is not None
        assert outer_outcome2.failure_reason is None

        outer_merged_token = outer_outcome2.merged_token

        # Verify outer merge consumed both inner_merged and path_b
        assert len(outer_outcome2.consumed_tokens) == 2
        consumed_outer_ids = {t.token_id for t in outer_outcome2.consumed_tokens}
        assert inner_for_outer.token_id in consumed_outer_ids
        assert enriched_b.token_id in consumed_outer_ids

        # === Verify final merged data has complete nested hierarchy ===
        final_data = outer_merged_token.row_data
        assert "path_a_merged" in final_data
        assert "path_b" in final_data

        # path_b branch has category
        assert final_data["path_b"]["category"] == "financial"

        # path_a_merged branch has the inner merge results (nested A1+A2)
        inner_result = final_data["path_a_merged"]
        assert "path_a1" in inner_result
        assert "path_a2" in inner_result
        assert inner_result["path_a1"]["sentiment"] == "positive"
        assert inner_result["path_a2"]["entities"] == ["ACME", "2024"]

        # === Verify token hierarchy through audit trail ===
        # All tokens should trace back to the same row_id
        assert initial_token.row_id == path_a_token.row_id
        assert initial_token.row_id == path_b_token.row_id
        assert initial_token.row_id == path_a1_token.row_id
        assert initial_token.row_id == path_a2_token.row_id
        assert initial_token.row_id == inner_merged_token.row_id
        assert initial_token.row_id == outer_merged_token.row_id

        # Verify parent-child relationships at each level
        # Level 1 children (path_a, path_b) should have initial_token as parent
        path_a_parents = recorder.get_token_parents(path_a_token.token_id)
        assert len(path_a_parents) == 1
        assert path_a_parents[0].parent_token_id == initial_token.token_id

        path_b_parents = recorder.get_token_parents(path_b_token.token_id)
        assert len(path_b_parents) == 1
        assert path_b_parents[0].parent_token_id == initial_token.token_id

        # Level 2 children (A1, A2) should have path_a as parent
        a1_parents = recorder.get_token_parents(path_a1_token.token_id)
        assert len(a1_parents) == 1
        assert a1_parents[0].parent_token_id == path_a_token.token_id

        a2_parents = recorder.get_token_parents(path_a2_token.token_id)
        assert len(a2_parents) == 1
        assert a2_parents[0].parent_token_id == path_a_token.token_id

        # Inner merged token should have A1 and A2 as parents
        inner_merged_parents = recorder.get_token_parents(inner_merged_token.token_id)
        assert len(inner_merged_parents) == 2
        inner_parent_ids = {p.parent_token_id for p in inner_merged_parents}
        assert path_a1_token.token_id in inner_parent_ids
        assert path_a2_token.token_id in inner_parent_ids

        # Outer merged token should have inner_merged and path_b as parents
        outer_merged_parents = recorder.get_token_parents(outer_merged_token.token_id)
        assert len(outer_merged_parents) == 2
        outer_parent_ids = {p.parent_token_id for p in outer_merged_parents}
        assert inner_merged_token.token_id in outer_parent_ids
        assert path_b_token.token_id in outer_parent_ids

        # === Verify coalesce metadata captures the hierarchy ===
        assert inner_outcome2.coalesce_metadata is not None
        assert inner_outcome2.coalesce_metadata["policy"] == "require_all"
        assert set(inner_outcome2.coalesce_metadata["branches_arrived"]) == {
            "path_a1",
            "path_a2",
        }

        assert outer_outcome2.coalesce_metadata is not None
        assert outer_outcome2.coalesce_metadata["policy"] == "require_all"
        assert set(outer_outcome2.coalesce_metadata["branches_arrived"]) == {
            "path_a_merged",
            "path_b",
        }

        # === Verify merged tokens have join_group_id ===
        inner_merged_record = recorder.get_token(inner_merged_token.token_id)
        assert inner_merged_record is not None
        assert inner_merged_record.join_group_id is not None

        outer_merged_record = recorder.get_token(outer_merged_token.token_id)
        assert outer_merged_record is not None
        assert outer_merged_record.join_group_id is not None

    def test_late_arrival_coalesce_returns_failed_outcome(self, landscape_db: "LandscapeDB") -> None:
        """Late arrivals at coalesce points return FAILED outcome from processor.

        This is an INTEGRATION CONTRACT TEST verifying the processor correctly
        handles CoalesceOutcome.failure_reason field.

        Scenario:
        - Fork creates 2 children (fast, slow branches)
        - Coalesce policy is FIRST (merges immediately on first arrival)
        - Fast branch arrives -> triggers merge -> returns COALESCED
        - Slow branch arrives AFTER merge complete -> returns FAILED

        Verifies:
        1. processor.process_row(slow_token) returns RowResult with outcome=FAILED
        2. RowResult.error field contains structured FailureInfo
        3. RowResult.final_data is slow token's data (unchanged)
        4. FAILED outcome recorded in token_outcomes table
        5. NO COMPLETED outcome recorded for late arrival

        This test prevents regression of the bug where late arrivals fell through
        to COMPLETED path because processor didn't check failure_reason field.
        """
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        # Setup: Create recorder and span factory
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        span_factory = SpanFactory()

        # Register nodes (source and coalesce)
        from elspeth.contracts import NodeType

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Setup: Create processor with coalesce executor
        token_manager = TokenManager(recorder)
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )

        # Register coalesce with FIRST policy (merges on first arrival)
        coalesce_settings = CoalesceSettings(
            name="test_coalesce",
            branches=["fast", "slow"],
            policy="first",
            strategy="overwrite",
            primary_branch="fast",
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            coalesce_executor=coalesce_executor,
        )

        # Create initial token and fork it
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            row_index=0,
            row_data={"text": "test"},
        )

        # Fork into fast and slow branches
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["fast", "slow"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate processing: create enriched tokens for each branch
        fast_token = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"text": "test", "fast_result": "done"},
            branch_name="fast",
        )
        slow_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"text": "test", "slow_result": "done"},
            branch_name="slow",
        )

        # Need to create context for processing
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id=run.run_id, config={})

        # Coalesce triggers when completed_step >= coalesce_at_step
        # With no transforms, completed_step = 0, so set coalesce_at_step=0
        # This means coalesce happens immediately (step 0 complete -> step 1 is coalesce)

        # Process fast token -> triggers merge (FIRST policy)
        result1, _ = processor._process_single_token(
            token=fast_token,
            transforms=[],  # No transforms needed for this test
            ctx=ctx,
            start_step=0,
            coalesce_name=CoalesceName("test_coalesce"),
            coalesce_at_step=0,  # Changed from 1 to 0 (coalesce immediately)
        )

        # Verify fast token merged successfully
        assert result1 is not None
        assert not isinstance(result1, list), "Expected single RowResult, not list"
        assert result1.outcome == RowOutcome.COALESCED

        # Process slow token (late arrival - merge already happened)
        result2, _ = processor._process_single_token(
            token=slow_token,
            transforms=[],
            ctx=ctx,
            start_step=0,
            coalesce_name=CoalesceName("test_coalesce"),
            coalesce_at_step=0,  # Changed from 1 to 0 (same as fast token)
        )

        # === CRITICAL ASSERTIONS ===

        # 1. Result is not None (token not held)
        assert result2 is not None, "Late arrival should return result, not be held"
        assert not isinstance(result2, list), "Expected single RowResult, not list"

        # 2. Outcome is FAILED, not COMPLETED
        assert result2.outcome == RowOutcome.FAILED, f"Late arrival should return FAILED, got {result2.outcome}"

        # 3. Token identity preserved
        assert result2.token.token_id == slow_token.token_id
        assert result2.token.branch_name == "slow"

        # 4. Row data unchanged (not merged)
        assert result2.final_data == slow_token.row_data
        assert result2.final_data["slow_result"] == "done"

        # 5. Structured error details present
        assert result2.error is not None, "FAILED outcome must include FailureInfo"
        assert result2.error.exception_type == "CoalesceFailure"
        assert "late_arrival" in result2.error.message.lower()

        # 6. Verify audit trail: FAILED outcome recorded
        # Query token_outcomes table to verify FAILED was recorded
        from sqlalchemy import text

        with landscape_db.engine.connect() as conn:
            outcomes = conn.execute(
                text("""
                    SELECT outcome, error_hash
                    FROM token_outcomes
                    WHERE run_id = :run_id
                    AND token_id = :token_id
                """),
                {"run_id": run.run_id, "token_id": slow_token.token_id},
            ).fetchall()

        assert len(outcomes) > 0, "Should have at least one outcome recorded"

        # Find FAILED outcome (note: stored as lowercase in database)
        failed_outcomes = [o for o in outcomes if o[0].upper() == "FAILED"]  # o[0] is outcome
        assert len(failed_outcomes) == 1, f"Should have exactly one FAILED outcome. Got outcomes: {[(o[0], o[1]) for o in outcomes]}"
        assert failed_outcomes[0][1] is not None, "FAILED outcome must have error_hash"  # o[1] is error_hash

        # 7. Verify NO COMPLETED outcome recorded
        completed_outcomes = [o for o in outcomes if o[0].upper() == "COMPLETED"]
        assert len(completed_outcomes) == 0, (
            f"Late arrival should NOT have COMPLETED outcome (bug symptom). Got: {[(o[0], o[1]) for o in outcomes]}"
        )


class TestCoalesceLinkage:
    """Test fork -> coalesce linkage."""

    def test_processor_accepts_coalesce_mapping_params(self, landscape_db: "LandscapeDB") -> None:
        """RowProcessor should accept branch_to_coalesce and coalesce_step_map."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Should not raise - params are accepted
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge_point")},
            coalesce_step_map={CoalesceName("merge_point"): 3},
        )

        assert processor._branch_to_coalesce == {BranchName("path_a"): CoalesceName("merge_point")}
        assert processor._coalesce_step_map == {CoalesceName("merge_point"): 3}


class TestAggregationCoalesceMetadataPropagation:
    """UNIT TEST: Aggregation continuation paths propagate coalesce metadata.

    Scope: Tests RowProcessor._process_batch_aggregation_node and handle_timeout_flush
    internal behavior. Does NOT test full orchestrator/graph integration.

    Bug: Brief 2 - Coalesce Metadata Dropped on Aggregation Continuation (P2)

    Root cause: _WorkItem carries coalesce_at_step/coalesce_name, but when
    _process_batch_aggregation_node queues work items for continuation, it
    creates them without these fields. Same for handle_timeout_flush.

    Impact: A forked branch that aggregates and then continues in single mode
    will skip the coalesce point, leaving the other branch to hang.

    Note: This is a unit test of RowProcessor internals. For production-path
    integration tests of fork+aggregation+coalesce, see the orchestrator tests
    once full fork+coalesce fixtures are available.
    """

    def test_aggregation_single_mode_preserves_coalesce_metadata(
        self,
        landscape_db: LandscapeDB,
    ) -> None:
        """Aggregation continuation should preserve coalesce metadata.

        Scenario:
        - Token has branch_name "path_a" (from a prior fork)
        - Token processes through aggregation (single mode)
        - Aggregation flushes and creates continuation _WorkItem
        - The continuation should have coalesce_at_step and coalesce_name

        Without the fix, the continuation _WorkItem lacks coalesce metadata,
        causing the token to skip the coalesce point.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.types import BranchName, CoalesceName, NodeID
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Aggregation transform
        class BatchAggForCoalesce(BaseTransform):
            """Aggregation that sums values."""

            name = "batch_agg_coalesce"
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any] | list[dict[str, Any]], ctx: Any) -> TransformResult:
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"total": total}, success_reason={"action": "aggregate"})
                return TransformResult.success(dict(row), success_reason={"action": "passthrough"})

        agg_transform = as_transform(BatchAggForCoalesce())

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg_coalesce",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="coalesce_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Set node_id on transform
        agg_transform.node_id = agg_node.node_id

        # Coalesce settings: merge path_a and path_b
        coalesce_settings = CoalesceSettings(
            name="merge",
            branches=["path_a", "path_b"],
            policy="require_all",
        )

        # Aggregation settings - triggers after 2 rows
        agg_settings = AggregationSettings(
            name="batch_agg_coalesce",
            plugin="batch_agg_coalesce",
            trigger=TriggerConfig(count=2),
            output_mode="transform",
        )

        # Create coalesce executor
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        coalesce_executor.register_coalesce(
            settings=coalesce_settings,
            node_id=coalesce_node.node_id,
        )

        # Branch to coalesce mapping
        branch_to_coalesce = {
            BranchName("path_a"): CoalesceName("merge"),
            BranchName("path_b"): CoalesceName("merge"),
        }

        # Coalesce step is at step 1 (after aggregation at step 0)
        coalesce_step_map = {CoalesceName("merge"): 1}

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merge"): NodeID(coalesce_node.node_id)},
            branch_to_coalesce=branch_to_coalesce,
            coalesce_step_map=coalesce_step_map,
            aggregation_settings={NodeID(agg_node.node_id): agg_settings},
        )

        # Create source token
        source_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            row_index=0,
            row_data={"id": 1, "value": 100},
        )

        # Fork to create a token with branch_name
        forked_tokens, _fork_group_id = token_manager.fork_token(
            parent_token=source_token,
            branches=["path_a"],
            step_in_pipeline=0,
            run_id=run.run_id,
        )
        forked_token = forked_tokens[0]

        assert forked_token.branch_name == "path_a"

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process first token - gets buffered
        results = processor.process_token(
            token=forked_token,
            transforms=[agg_transform],
            ctx=ctx,
            start_step=0,
            coalesce_at_step=1,
            coalesce_name=CoalesceName("merge"),
        )

        # Create second forked token
        source_token2 = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            row_index=1,
            row_data={"id": 2, "value": 200},
        )
        forked_tokens2, _fork_group_id2 = token_manager.fork_token(
            parent_token=source_token2,
            branches=["path_a"],
            step_in_pipeline=0,
            run_id=run.run_id,
        )
        forked_token2 = forked_tokens2[0]

        # Process second token - triggers flush
        results2 = processor.process_token(
            token=forked_token2,
            transforms=[agg_transform],
            ctx=ctx,
            start_step=0,
            coalesce_at_step=1,
            coalesce_name=CoalesceName("merge"),
        )

        all_results = results + results2

        # Check that no token has COMPLETED outcome (which would mean it
        # bypassed coalesce). All tokens should either be:
        # - CONSUMED_IN_BATCH (buffered for aggregation)
        # - None (held for coalesce, waiting for path_b)
        #
        # If the bug exists, we'd see COMPLETED because the continuation
        # _WorkItem lacks coalesce metadata, so it skips coalesce check.

        for result in all_results:
            if result.outcome == RowOutcome.COMPLETED:
                raise AssertionError(
                    f"Aggregation output has COMPLETED outcome, which means it "
                    f"bypassed the coalesce point. Bug: aggregation continuation "
                    f"_WorkItem is missing coalesce_at_step and coalesce_name. "
                    f"Result: {result}"
                )


class TestCoalesceSelectBranchFailure:
    """Tests for bug 9z8: Double terminal outcome recording on select-merge failure.

    When coalesce uses merge="select" and the selected branch hasn't arrived,
    CoalesceExecutor records FAILED outcomes for arrived tokens, then
    RowProcessor._maybe_coalesce_token also tries to record FAILED for the
    current token, causing a unique constraint violation.
    """

    def test_select_merge_failure_records_single_outcome(self, landscape_db: "LandscapeDB") -> None:
        """Select merge failure should record exactly one terminal outcome per token.

        Bug 9z8: When select_branch not arrived with first policy:
        1. CoalesceExecutor records FAILED for all arrived tokens (including current)
        2. RowProcessor also records FAILED for current token
        3. CRASH: Unique constraint violation on token_outcomes

        Expected: Only ONE FAILED outcome recorded for current token.
        """
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        fork_gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="branch_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="branch_b",
            mode=RoutingMode.COPY,
        )

        # Setup coalesce with:
        # - policy="first" (merge on first arrival)
        # - merge="select" with select_branch="branch_a"
        # When branch_b arrives first, it should fail (select_branch not arrived)
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["branch_a", "branch_b"],
            policy="first",
            merge="select",
            select_branch="branch_a",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Config-driven fork gate that forks to both paths
        # CRITICAL: fork_to order determines work queue order
        # We want branch_b to arrive FIRST so select_branch="branch_a" is NOT arrived
        fork_gate_config = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["branch_b", "branch_a"],  # branch_b arrives first!
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            edge_map={
                (NodeID(fork_gate.node_id), "branch_a"): edge_a.edge_id,
                (NodeID(fork_gate.node_id), "branch_b"): edge_b.edge_id,
            },
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merger"): NodeID(coalesce_node.node_id)},
            config_gates=[fork_gate_config],
            config_gate_id_map={GateName("splitter"): NodeID(fork_gate.node_id)},
            branch_to_coalesce={
                BranchName("branch_a"): CoalesceName("merger"),
                BranchName("branch_b"): CoalesceName("merger"),
            },
            coalesce_step_map={CoalesceName("merger"): 1},  # After gate
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process a row - this will:
        # 1. Fork at gate to branch_a and branch_b
        # 2. Process branch_b first (implementation dependent, but both will try to coalesce)
        # 3. First arrival triggers merge (policy=first)
        # 4. But select_branch="branch_a" is not arrived
        # 5. Failure should record FAILED for arrived token
        # 6. BUG: RowProcessor ALSO records FAILED → unique constraint violation

        # This should NOT raise an IntegrityError
        results = processor.process_row(
            row_index=0,
            row_data={"text": "test"},
            transforms=[],
            ctx=ctx,
        )

        # Verify we got results (not a crash)
        assert len(results) > 0

        # Verify at least one FAILED outcome for the coalesce failure
        failed_results = [r for r in results if r.outcome == RowOutcome.FAILED]
        assert len(failed_results) >= 1, "Should have at least one FAILED result from coalesce failure"

        # Verify the failure reasons are correct:
        # - branch_b (first to arrive): fails with "select_branch_not_arrived"
        # - branch_a (late arrival): fails with "late_arrival_after_merge"
        failure_messages = {r.error.message for r in failed_results if r.error and r.error.exception_type == "CoalesceFailure"}
        assert "select_branch_not_arrived" in failure_messages or "late_arrival_after_merge" in failure_messages, (
            f"Expected coalesce failures, got: {failure_messages}"
        )

        # Query the database to verify no duplicate outcomes
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import token_outcomes_table

        with landscape_db.engine.connect() as conn:
            # Count outcomes per token_id - should never be > 1
            stmt = (
                select(
                    token_outcomes_table.c.token_id,
                    func.count().label("count"),
                )
                .group_by(token_outcomes_table.c.token_id)
                .having(func.count() > 1)
            )
            duplicates = conn.execute(stmt).fetchall()
            assert len(duplicates) == 0, (
                f"Bug 9z8: Found duplicate terminal outcomes for tokens: {duplicates}. "
                f"CoalesceExecutor and RowProcessor both recorded outcomes for the same token."
            )
