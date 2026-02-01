"""Tests for coalesce audit gaps identified in deep dive analysis.

These tests verify that fork/coalesce operations maintain complete audit trail:
- Gap 1a: Successful merge - consumed tokens must have token_outcome records
- Gap 1b: Failed coalesce - consumed tokens must have token_outcome records
- Gap 2: Fork child fails before coalesce - siblings should not be stranded

References:
- Deep dive analysis by Opus agent (acb14c0)
- Integration seam analysis Issue #1 (partially fixed)
"""

import pytest

from elspeth.contracts import NodeType, Run, TokenInfo
from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.coalesce_executor import CoalesceExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


@pytest.fixture
def db() -> LandscapeDB:
    return LandscapeDB.in_memory()


@pytest.fixture
def recorder(db: LandscapeDB) -> LandscapeRecorder:
    return LandscapeRecorder(db)


@pytest.fixture
def run(recorder: LandscapeRecorder) -> Run:
    return recorder.begin_run(config={}, canonical_version="v1")


class TestCoalesceAuditGap1a:
    """Gap 1a: Successful merge must record token_outcome for consumed tokens.

    Current behavior: Consumed tokens get node_state records but NO token_outcome.
    Expected: Each consumed token should have RowOutcome.COALESCED recorded.

    This is inconsistent with the rest of ELSPETH where terminal states are
    recorded in BOTH node_state (for lineage) AND token_outcome (for O(1) lookup).
    """

    def test_successful_merge_records_token_outcomes_for_consumed_tokens(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When coalesce succeeds, each consumed token MUST have token_outcome record.

        Audit contract violation: get_token_outcome() currently returns None for
        consumed tokens, forcing queries to infer terminal state from node_states.

        Expected: Each consumed token should have:
        - token_outcome with RowOutcome.COALESCED
        - join_group_id pointing to merged token
        """
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="coalesce_1",
            plugin_name="merge_all",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Configure coalesce
        settings = CoalesceSettings(
            name="merge_all",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create and fork initial token
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 100},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Accept both tokens (triggers merge)
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"value": 100, "a_result": 1},
            branch_name="path_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"value": 100, "b_result": 2},
            branch_name="path_b",
        )

        outcome_a = executor.accept(token_a, "merge_all", step_in_pipeline=2)
        assert outcome_a.held is True

        outcome_b = executor.accept(token_b, "merge_all", step_in_pipeline=2)
        assert outcome_b.held is False
        assert outcome_b.merged_token is not None

        merged_token = outcome_b.merged_token
        consumed_tokens = outcome_b.consumed_tokens

        # CRITICAL: Each consumed token MUST have token_outcome record
        for consumed_token in consumed_tokens:
            token_outcome = recorder.get_token_outcome(consumed_token.token_id)

            assert token_outcome is not None, (
                f"Gap 1a: Consumed token {consumed_token.token_id} has NO token_outcome! "
                f"get_token_outcome() returns None, violating audit contract. "
                f"Expected: RowOutcome.COALESCED with join_group_id={merged_token.join_group_id}"
            )

            # Verify outcome details
            assert token_outcome.outcome == RowOutcome.COALESCED, (
                f"Consumed token {consumed_token.token_id} should have outcome=COALESCED, got {token_outcome.outcome}"
            )

            # Verify consumed token has the canonical join_group_id (same as merged token)
            assert token_outcome.join_group_id == merged_token.join_group_id, (
                f"Consumed token {consumed_token.token_id} should have same join_group_id as merged token "
                f"(canonical ID={merged_token.join_group_id}), got {token_outcome.join_group_id}"
            )


class TestCoalesceAuditGap1b:
    """Gap 1b: Failed coalesce must record token_outcome for consumed tokens.

    Current behavior: Consumed tokens get node_state(failed) + error_json but NO token_outcome.
    Expected: Each consumed token should have RowOutcome.FAILED recorded.

    This was part of Issue #1 fix but we only added node_state records, not token_outcome.
    """

    def test_failed_coalesce_records_token_outcomes_for_consumed_tokens(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When coalesce fails, each consumed token MUST have token_outcome record.

        Scenario: Quorum not met (need 3 branches, only 2 arrive)

        Expected: Each consumed token should have:
        - token_outcome with RowOutcome.FAILED
        - error_hash explaining failure reason
        """
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="coalesce_1",
            plugin_name="quorum_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Configure quorum coalesce
        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["path_a", "path_b", "path_c"],
            policy="quorum",
            quorum_count=3,  # Need all 3
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create and fork initial token
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 100},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b", "path_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Accept only 2 of 3 (quorum not met)
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"value": 100, "a_result": 1},
            branch_name="path_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"value": 100, "b_result": 2},
            branch_name="path_b",
        )

        executor.accept(token_a, "quorum_merge", step_in_pipeline=2)
        executor.accept(token_b, "quorum_merge", step_in_pipeline=2)

        # Flush at end-of-source (triggers failure)
        flushed = executor.flush_pending(step_map={"quorum_merge": 2})
        assert len(flushed) == 1

        failure_outcome = flushed[0]
        assert failure_outcome.failure_reason == "quorum_not_met"
        consumed_tokens = failure_outcome.consumed_tokens

        # CRITICAL: Each consumed token MUST have token_outcome record
        for consumed_token in consumed_tokens:
            token_outcome = recorder.get_token_outcome(consumed_token.token_id)

            assert token_outcome is not None, (
                f"Gap 1b: Consumed token {consumed_token.token_id} from FAILED coalesce has NO token_outcome! "
                f"get_token_outcome() returns None, violating audit contract. "
                f"Expected: RowOutcome.FAILED with error explaining quorum_not_met"
            )

            # Verify outcome details
            assert token_outcome.outcome == RowOutcome.FAILED, (
                f"Consumed token {consumed_token.token_id} from failed coalesce should have outcome=FAILED, got {token_outcome.outcome}"
            )

            assert token_outcome.error_hash is not None, (
                f"Consumed token {consumed_token.token_id} failed outcome should have error_hash "
                f"explaining why coalesce failed (quorum_not_met)"
            )


class TestCoalesceAuditGap2:
    """Gap 2: Fork child fails before coalesce - siblings should not be stranded.

    Current behavior: If one fork child fails, siblings are held until flush_pending.
    Expected: This is handled correctly, but test verifies audit trail completeness.

    Scenario:
    1. Fork creates children A, B, C
    2. Child A fails at transform (before reaching coalesce)
    3. Children B, C arrive at coalesce (require_all policy)
    4. Coalesce waits for A (which will never arrive)
    5. flush_pending() should record failure for B and C

    This test verifies that the audit trail correctly shows:
    - Child A: FAILED outcome (at transform)
    - Children B, C: FAILED outcome (at coalesce, with incomplete_branches reason)
    """

    def test_fork_child_fails_before_coalesce_siblings_recorded_at_flush(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When fork child fails before coalesce, siblings should have complete audit trail.

        This tests the "sibling stranding" scenario identified in the deep dive.

        Expected audit trail:
        - Parent: FORKED outcome
        - Failed child A: FAILED outcome (at transform where it failed)
        - Held children B, C: FAILED outcomes (at coalesce, incomplete_branches)
        """
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        # Note: transform_node registered but not used in this test
        # (simulates child A failing at transform before reaching coalesce)
        recorder.register_node(
            run_id=run.run_id,
            node_id="transform_1",
            plugin_name="filter_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="coalesce_1",
            plugin_name="merge_all",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Configure coalesce
        settings = CoalesceSettings(
            name="merge_all",
            branches=["path_a", "path_b", "path_c"],
            policy="require_all",
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create and fork initial token
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 100},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b", "path_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate child A failing at transform (before coalesce)
        child_a = children[0]
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=child_a.token_id,
            outcome=RowOutcome.FAILED,
            error_hash="simulated_transform_failure",
        )

        # Children B and C arrive at coalesce (waiting for A)
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"value": 100, "b_result": 2},
            branch_name="path_b",
        )
        token_c = TokenInfo(
            row_id=children[2].row_id,
            token_id=children[2].token_id,
            row_data={"value": 100, "c_result": 3},
            branch_name="path_c",
        )

        executor.accept(token_b, "merge_all", step_in_pipeline=2)
        executor.accept(token_c, "merge_all", step_in_pipeline=2)

        # End-of-source: flush pending (A will never arrive)
        flushed = executor.flush_pending(step_map={"merge_all": 2})
        assert len(flushed) == 1

        failure_outcome = flushed[0]
        assert failure_outcome.failure_reason == "incomplete_branches"

        # AUDIT TRAIL VERIFICATION
        # Child A should have FAILED outcome (from transform)
        outcome_a = recorder.get_token_outcome(child_a.token_id)
        assert outcome_a is not None
        assert outcome_a.outcome == RowOutcome.FAILED

        # Children B and C should have FAILED outcomes (from coalesce flush)
        for consumed_token in failure_outcome.consumed_tokens:
            token_outcome = recorder.get_token_outcome(consumed_token.token_id)

            assert token_outcome is not None, (
                f"Gap 2: Stranded sibling {consumed_token.token_id} has NO token_outcome! "
                f"When sibling A failed before coalesce, B and C were held until flush. "
                f"Expected: RowOutcome.FAILED recorded at flush_pending"
            )

            assert token_outcome.outcome == RowOutcome.FAILED, (
                f"Stranded sibling {consumed_token.token_id} should have outcome=FAILED, got {token_outcome.outcome}"
            )

        # Verify all three children reached terminal state
        all_outcomes = recorder.get_token_outcomes_for_row(run.run_id, initial_token.row_id)
        child_outcomes = [o for o in all_outcomes if o.token_id in [c.token_id for c in children]]

        assert len(child_outcomes) == 3, (
            f"Expected 3 child tokens to have outcomes, got {len(child_outcomes)}. "
            f"Missing outcomes means tokens disappeared from audit trail."
        )

        # All should be FAILED (A from transform, B+C from coalesce)
        assert all(o.outcome == RowOutcome.FAILED for o in child_outcomes), (
            f"All fork children should eventually reach FAILED outcome. Got outcomes: {[o.outcome for o in child_outcomes]}"
        )


class TestCoalesceTimeoutAuditGap:
    """Test that timeout-triggered merges have complete audit trail.

    DESIGN CLARIFICATION: Token outcomes work as follows:
    - Consumed tokens (branch tokens absorbed by merge): COALESCED (terminal)
    - Merged token: Gets COMPLETED when reaching sink (not COALESCED)

    The merged token does NOT get COALESCED because:
    1. Each token can only have ONE terminal outcome (unique constraint)
    2. In nested coalesces, a merged token becomes a consumed token in outer merge
    3. Recording COALESCED for merged token would violate unique constraint

    The audit trail is complete because:
    - Consumed tokens show COALESCED with join_group_id
    - Merged token will show COMPLETED when it reaches the sink
    - The link between consumed and merged is through join_group_id
    """

    def test_timeout_consumed_tokens_have_coalesced_outcome(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Timeout-triggered merges must record COALESCED for consumed tokens.

        This verifies that when a timeout fires and triggers a best_effort merge,
        all consumed branch tokens (the ones that arrived before timeout) have
        COALESCED outcomes recorded with proper join_group_id.
        """
        from elspeth.engine.clock import MockClock

        # Deterministic clock for timeout testing
        clock = MockClock(start=100.0)

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="coalesce_1",
            plugin_name="timeout_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Configure best_effort coalesce with timeout
        settings = CoalesceSettings(
            name="timeout_merge",
            branches=["path_a", "path_b"],
            policy="best_effort",  # Merges whatever arrived on timeout
            merge="union",
            timeout_seconds=0.01,  # Very short for testing
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
            clock=clock,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create and fork initial token
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 100},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Only path_a arrives (path_b is "slow")
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"value": 100, "a_result": 1},
            branch_name="path_a",
        )

        outcome_a = executor.accept(token_a, "timeout_merge", step_in_pipeline=2)
        assert outcome_a.held is True  # Waiting for path_b

        # Advance clock past timeout
        clock.advance(0.02)

        # Check timeouts - should trigger merge with what arrived
        timed_out = executor.check_timeouts(
            coalesce_name="timeout_merge",
            step_in_pipeline=2,
        )

        # Verify timeout triggered
        assert len(timed_out) == 1
        timeout_outcome = timed_out[0]
        assert timeout_outcome.merged_token is not None
        merged_token = timeout_outcome.merged_token

        # Verify consumed tokens in the outcome
        assert len(timeout_outcome.consumed_tokens) == 1
        assert timeout_outcome.consumed_tokens[0].token_id == token_a.token_id

        # CRITICAL: The consumed token (path_a) must have COALESCED outcome
        # This is recorded by _execute_merge() for each consumed token
        consumed_outcome = recorder.get_token_outcome(token_a.token_id)
        assert consumed_outcome is not None, (
            f"Consumed token {token_a.token_id} has NO token_outcome! _execute_merge should record COALESCED for all consumed tokens."
        )
        assert consumed_outcome.outcome == RowOutcome.COALESCED, (
            f"Consumed token should have outcome=COALESCED, got {consumed_outcome.outcome}"
        )
        assert consumed_outcome.join_group_id is not None, "Consumed token COALESCED outcome must have join_group_id"

        # Verify join_group_id links consumed to merged
        assert consumed_outcome.join_group_id == merged_token.join_group_id, (
            f"Consumed token's join_group_id ({consumed_outcome.join_group_id}) "
            f"should match merged token's join_group_id ({merged_token.join_group_id})"
        )

        # Merged token has NO outcome yet - it will get COMPLETED when reaching sink
        # This is correct behavior: merged token's outcome is recorded by orchestrator
        # when it reaches the sink, not here in check_timeouts
