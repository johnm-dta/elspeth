"""Tests for CoalesceExecutor."""

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import Run
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


@pytest.fixture
def db() -> LandscapeDB:
    return LandscapeDB.in_memory()


@pytest.fixture
def recorder(db: LandscapeDB) -> LandscapeRecorder:
    """Shared recorder for all tests."""
    return LandscapeRecorder(db)


@pytest.fixture
def run(recorder: LandscapeRecorder) -> Run:
    """Create a run for testing."""
    return recorder.begin_run(config={}, canonical_version="v1")


@pytest.fixture
def executor_setup(recorder: LandscapeRecorder, run: Run) -> tuple[LandscapeRecorder, SpanFactory, "TokenManager", str]:
    """Common setup for executor tests - reduces boilerplate.

    Returns:
        Tuple of (recorder, span_factory, token_manager, run_id)
    """
    from elspeth.engine.tokens import TokenManager

    span_factory = SpanFactory()
    token_manager = TokenManager(recorder)
    return recorder, span_factory, token_manager, run.run_id


class TestCoalesceExecutorInit:
    """Test CoalesceExecutor initialization."""

    def test_executor_initializes(self, executor_setup: tuple[LandscapeRecorder, SpanFactory, Any, str]) -> None:
        """Executor should initialize with recorder and span factory."""
        from elspeth.engine.coalesce_executor import CoalesceExecutor

        recorder, span_factory, token_manager, run_id = executor_setup

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run_id,
        )

        assert executor is not None


class TestCoalesceExecutorRequireAll:
    """Test require_all policy."""

    def test_accept_holds_first_token(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """First token should be held, waiting for others."""
        from elspeth.contracts import NodeType
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register source and coalesce nodes
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
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="merge_results",
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

        # Create a token from path_a
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 42},
        )
        # Fork creates children with branch names
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )
        token_a = children[0]  # path_a

        # Accept first token
        outcome = executor.accept(
            token=token_a,
            coalesce_name="merge_results",
            step_in_pipeline=2,
        )

        # Should be held
        assert outcome.held is True
        assert outcome.merged_token is None
        assert outcome.consumed_tokens == []

    def test_accept_merges_when_all_arrive(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When all branches arrive, should merge and return merged token."""
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

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
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="merge_results",
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

        # Create tokens from both paths with different data
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"original": True},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate different processing on each branch
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"sentiment": "positive"},
            branch_name="path_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"entities": ["ACME"]},
            branch_name="path_b",
        )

        # Accept first token - should hold
        outcome1 = executor.accept(token_a, "merge_results", step_in_pipeline=2)
        assert outcome1.held is True

        # Accept second token - should merge
        outcome2 = executor.accept(token_b, "merge_results", step_in_pipeline=2)
        assert outcome2.held is False
        assert outcome2.merged_token is not None
        assert outcome2.merged_token.row_data == {
            "sentiment": "positive",
            "entities": ["ACME"],
        }
        assert len(outcome2.consumed_tokens) == 2


class TestCoalesceExecutorFirst:
    """Test FIRST policy."""

    def test_first_merges_immediately(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """FIRST policy should merge as soon as one token arrives."""
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="first_wins",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="first_wins",
            branches=["fast", "slow"],
            policy="first",
            merge="union",  # Union merge takes first arrival's data (policy=first means one token)
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"original": True},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["fast", "slow"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate slow arriving first (fast is delayed)
        token_slow = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"result": "from_slow"},
            branch_name="slow",
        )

        # Accept slow token - should merge immediately with FIRST policy
        outcome = executor.accept(token_slow, "first_wins", step_in_pipeline=2)

        assert outcome.held is False
        assert outcome.merged_token is not None
        assert outcome.merged_token.row_data == {"result": "from_slow"}

    def test_late_arrival_after_first_merge_handled_gracefully(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Late arrivals after FIRST policy merge should be handled gracefully.

        This test verifies the fix for Gap #2 from fork/coalesce deep dive.

        Scenario:
        - Fork creates 2 child tokens (branches fast, slow)
        - fast branch arrives first at coalesce (FIRST policy)
        - Merge happens immediately (first branch triggers merge)
        - slow branch arrives AFTER merge already completed

        Expected behavior for late arrival:
        - Late arrival should NOT create orphan pending entry
        - Late arrival should get failure outcome with reason "late_arrival_after_merge"
        - Late arrival should have audit trail showing it was rejected as late

        Gap #2 problem: Previously late arrivals created NEW pending entries that:
        1. Would never merge (siblings already processed)
        2. Would fail at flush_pending() with incomplete_branches
        3. Created confusing duplicate audit entries for the same fork operation
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="first_wins",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="first_wins",
            branches=["fast", "slow"],
            policy="first",
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create parent token and fork it
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 100},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["fast", "slow"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Fast token arrives first - should merge immediately
        token_fast = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"value": 100, "fast_result": 1},
            branch_name="fast",
        )
        outcome_fast = executor.accept(token_fast, "first_wins", step_in_pipeline=2)
        assert outcome_fast.held is False
        assert outcome_fast.merged_token is not None

        # Slow token arrives AFTER merge completed - late arrival!
        token_slow = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"value": 100, "slow_result": 2},
            branch_name="slow",
        )
        outcome_slow = executor.accept(token_slow, "first_wins", step_in_pipeline=2)

        # Late arrival should get rejected with failure outcome
        assert outcome_slow.held is False, "Late arrival should not be held"
        assert outcome_slow.merged_token is None, "Late arrival should not create merged token"
        assert outcome_slow.failure_reason == "late_arrival_after_merge", (
            f"Late arrival should have failure_reason='late_arrival_after_merge', got '{outcome_slow.failure_reason}'"
        )

        # Late arrival should have consumed_tokens populated
        assert outcome_slow.consumed_tokens is not None
        assert len(outcome_slow.consumed_tokens) == 1
        assert outcome_slow.consumed_tokens[0].token_id == token_slow.token_id


class TestCoalesceExecutorQuorum:
    """Test QUORUM policy."""

    def test_quorum_merges_at_threshold(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """QUORUM should merge when quorum_count branches arrive."""
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["model_a", "model_b", "model_c"],
            policy="quorum",
            quorum_count=2,  # Merge when 2 of 3 arrive
            merge="nested",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"score": 0.9},
            branch_name="model_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"score": 0.85},
            branch_name="model_b",
        )

        # Accept first - should hold (1 < 2)
        outcome1 = executor.accept(token_a, "quorum_merge", step_in_pipeline=2)
        assert outcome1.held is True

        # Accept second - should merge (2 >= 2)
        outcome2 = executor.accept(token_b, "quorum_merge", step_in_pipeline=2)
        assert outcome2.held is False
        assert outcome2.merged_token is not None
        # Nested merge strategy
        assert outcome2.merged_token.row_data == {
            "model_a": {"score": 0.9},
            "model_b": {"score": 0.85},
        }

    def test_quorum_records_failure_on_timeout_if_quorum_not_met(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """QUORUM should record failure on timeout if quorum not met.

        Note: This test was updated from asserting len(timed_out) == 0 (buggy behavior)
        to asserting len(timed_out) == 1 with failure outcome (correct behavior).
        Bug 6tb fix ensures stranded tokens are properly recorded as failed.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.clock import MockClock
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        # Deterministic clock for timeout testing
        clock = MockClock(start=100.0)

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["model_a", "model_b", "model_c"],
            policy="quorum",
            quorum_count=2,  # Need 2 of 3
            merge="nested",
            timeout_seconds=0.1,
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
            clock=clock,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Accept only ONE token (quorum needs 2)
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"score": 0.9},
            branch_name="model_a",
        )

        outcome = executor.accept(token_a, "quorum_merge", step_in_pipeline=2)
        assert outcome.held is True

        # Advance clock past timeout (0.1s + 0.05s margin)
        clock.advance(0.15)  # Now at 100.15

        # check_timeouts should record failure and return outcome
        # (Bug 6tb fix: no longer returns empty list)
        timed_out = executor.check_timeouts("quorum_merge", step_in_pipeline=2)
        assert len(timed_out) == 1
        assert timed_out[0].failure_reason == "quorum_not_met_at_timeout"

    def test_flush_pending_quorum_failure_records_audit_trail(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When coalesce fails (quorum not met), consumed tokens MUST have audit records.

        This test verifies the fix for Issue #1 from integration seam analysis.

        Audit gap scenario:
        - Fork creates 3 child tokens (branches A, B, C)
        - Only branches A and B arrive at coalesce (need all 3 for quorum)
        - flush_pending() called at end-of-source
        - Result: quorum_not_met failure

        Expected audit trail:
        - consumed_tokens list populated in CoalesceOutcome
        - Each consumed token has node_state recorded (status="failed")

        Without this, tokens "disappear" from audit trail - violating auditability contract.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["path_a", "path_b", "path_c"],
            policy="quorum",
            quorum_count=3,  # Need all 3 branches
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create parent token and fork it
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

        # Accept only 2 of 3 tokens (quorum not met)
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

        outcome_a = executor.accept(token_a, "quorum_merge", step_in_pipeline=2)
        assert outcome_a.held is True  # Waiting for siblings

        outcome_b = executor.accept(token_b, "quorum_merge", step_in_pipeline=2)
        assert outcome_b.held is True  # Still waiting (need 3)

        # End-of-source: flush pending coalesces
        flushed = executor.flush_pending(step_map={"quorum_merge": 2})

        # Verify failure outcome returned
        assert len(flushed) == 1
        failure_outcome = flushed[0]
        assert failure_outcome.held is False
        assert failure_outcome.merged_token is None
        assert failure_outcome.failure_reason == "quorum_not_met"

        # CRITICAL: consumed_tokens MUST be populated
        assert failure_outcome.consumed_tokens is not None, (
            "CoalesceOutcome.consumed_tokens must be populated on failure (currently returns empty list, losing tokens)"
        )
        assert len(failure_outcome.consumed_tokens) == 2, (
            f"Expected 2 consumed tokens (A and B), got {len(failure_outcome.consumed_tokens)}"
        )

        # Verify the tokens are the ones we submitted
        consumed_ids = {t.token_id for t in failure_outcome.consumed_tokens}
        assert token_a.token_id in consumed_ids
        assert token_b.token_id in consumed_ids

        # AUDIT TRAIL VERIFICATION:
        # Each consumed token MUST have node state recorded
        for token in failure_outcome.consumed_tokens:
            node_states = recorder.get_node_states_for_token(token.token_id)

            # Find the coalesce node state
            coalesce_states = [ns for ns in node_states if ns.node_id == coalesce_node.node_id]

            assert len(coalesce_states) > 0, (
                f"Token {token.token_id} has NO node state for coalesce_1 - audit gap! Cannot trace what happened to this token."
            )

            coalesce_state = coalesce_states[0]

            # Verify node state indicates failure
            from elspeth.contracts import NodeStateStatus as _NodeStateStatus

            assert coalesce_state.status == _NodeStateStatus.FAILED, (
                f"Token {token.token_id} node state should be 'failed', got '{coalesce_state.status}'"
            )

            # Verify error_json explains the failure (type narrow to NodeStateFailed)
            from elspeth.contracts import NodeStateFailed

            assert isinstance(coalesce_state, NodeStateFailed), "Failed state should be NodeStateFailed"
            assert coalesce_state.error_json is not None, f"Token {token.token_id} failed node state must have error_json populated"

            import json

            error_data = json.loads(coalesce_state.error_json)
            assert "failure_reason" in error_data, f"Token {token.token_id} node state missing failure explanation in error_json"
            assert error_data["failure_reason"] == "quorum_not_met"

    def test_held_tokens_have_audit_trail(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Held tokens (waiting for siblings) MUST have audit trail showing open state.

        This test verifies the fix for Gap #1 from fork/coalesce deep dive.

        Scenario:
        - Fork creates 3 child tokens (branches A, B, C)
        - Branch A arrives at coalesce (need all 3 for require_all)
        - Branch A is held waiting for B and C
        - Branch B arrives (still need C)
        - Branch B is held waiting for C

        Expected audit trail while held:
        - Each held token has node_state with status="open" (in-progress)
        - Node state shows input_data (what token brought to coalesce)
        - Node state is NOT completed yet (no completed_at, no duration)
        - When merge/fail happens, these open states are completed

        Gap #1 problem: Previously held tokens had NO audit record until merge/fail.
        If pipeline crashed or was queried mid-run, held tokens were invisible.
        This violated ELSPETH's core contract: "I don't know what happened is never acceptable"
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="require_all_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="require_all_merge",
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

        # Create parent token and fork it
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

        # Accept first token - should be held
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"value": 100, "a_result": 1},
            branch_name="path_a",
        )
        outcome_a = executor.accept(token_a, "require_all_merge", step_in_pipeline=2)
        assert outcome_a.held is True

        # CRITICAL: Token A should now have audit trail showing it's held
        node_states_a = recorder.get_node_states_for_token(token_a.token_id)
        coalesce_states_a = [ns for ns in node_states_a if ns.node_id == coalesce_node.node_id]

        assert len(coalesce_states_a) > 0, (
            f"Held token {token_a.token_id} has NO node state for coalesce - "
            f"audit gap! Cannot trace that this token is waiting for siblings."
        )

        held_state_a = coalesce_states_a[0]
        # Held tokens have status="open" (in-progress, not completed yet)
        from elspeth.contracts import NodeStateStatus

        assert held_state_a.status == NodeStateStatus.OPEN, (
            f"Held token should have status='open' (in-progress), got '{held_state_a.status}'"
        )

        # Accept second token - should also be held (2 < 3)
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"value": 100, "b_result": 2},
            branch_name="path_b",
        )
        outcome_b = executor.accept(token_b, "require_all_merge", step_in_pipeline=2)
        assert outcome_b.held is True

        # Token B should also have audit trail
        node_states_b = recorder.get_node_states_for_token(token_b.token_id)
        coalesce_states_b = [ns for ns in node_states_b if ns.node_id == coalesce_node.node_id]

        assert len(coalesce_states_b) > 0, f"Held token {token_b.token_id} has NO node state - audit gap!"

        held_state_b = coalesce_states_b[0]
        assert held_state_b.status == NodeStateStatus.OPEN, (
            f"Held token should have status='open' (in-progress), got '{held_state_b.status}'"
        )


class TestCoalesceExecutorBestEffort:
    """Test BEST_EFFORT policy with timeout."""

    def test_best_effort_merges_on_timeout(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """BEST_EFFORT should merge whatever arrived when timeout expires."""
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.clock import MockClock
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        # Deterministic clock for timeout testing
        clock = MockClock(start=100.0)

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="best_effort_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="best_effort_merge",
            branches=["path_a", "path_b", "path_c"],
            policy="best_effort",
            timeout_seconds=0.1,  # Short timeout for testing
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
            clock=clock,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b", "path_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"a_result": 1},
            branch_name="path_a",
        )

        # Accept one token
        outcome1 = executor.accept(token_a, "best_effort_merge", step_in_pipeline=2)
        assert outcome1.held is True

        # Advance clock past timeout (0.1s + 0.05s margin)
        clock.advance(0.15)  # Now at 100.15

        # Check timeout and force merge
        timed_out = executor.check_timeouts("best_effort_merge", step_in_pipeline=2)

        # Should have one merged result
        assert len(timed_out) == 1
        assert timed_out[0].merged_token is not None
        assert timed_out[0].merged_token.row_data == {"a_result": 1}

    def test_check_timeouts_unregistered_raises(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """check_timeouts should raise ValueError for unregistered coalesce."""
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )

        with pytest.raises(ValueError, match="not registered"):
            executor.check_timeouts("nonexistent", step_in_pipeline=2)


class TestCoalesceAuditMetadata:
    """Test coalesce audit metadata recording."""

    def test_coalesce_records_audit_metadata(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Coalesce should record audit metadata with policy, strategy, and timing."""
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

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
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="merge_results",
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

        # Create tokens from both paths
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"original": True},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate different processing on each branch
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"sentiment": "positive"},
            branch_name="path_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"entities": ["ACME"]},
            branch_name="path_b",
        )

        # Accept first token - should hold
        outcome1 = executor.accept(token_a, "merge_results", step_in_pipeline=2)
        assert outcome1.held is True

        # Accept second token - should merge
        outcome2 = executor.accept(token_b, "merge_results", step_in_pipeline=2)
        assert outcome2.held is False
        assert outcome2.merged_token is not None

        # Verify coalesce_metadata is populated
        assert outcome2.coalesce_metadata is not None
        metadata = outcome2.coalesce_metadata

        # Check required fields
        assert metadata["policy"] == "require_all"
        assert metadata["merge_strategy"] == "union"
        assert metadata["expected_branches"] == ["path_a", "path_b"]
        assert set(metadata["branches_arrived"]) == {"path_a", "path_b"}
        assert metadata["wait_duration_ms"] >= 0

        # Check arrival_order structure
        assert "arrival_order" in metadata
        assert len(metadata["arrival_order"]) == 2
        for entry in metadata["arrival_order"]:
            assert "branch" in entry
            assert "arrival_offset_ms" in entry
            assert entry["arrival_offset_ms"] >= 0

        # P1: Verify audit trail - node_states for consumed tokens
        from elspeth.contracts.enums import NodeStateStatus, RowOutcome
        from elspeth.core.canonical import stable_hash

        for token in outcome2.consumed_tokens:
            node_states = recorder.get_node_states_for_token(token.token_id)

            # Find the coalesce node state
            coalesce_states = [ns for ns in node_states if ns.node_id == coalesce_node.node_id]
            assert len(coalesce_states) == 1, f"Token {token.token_id} should have exactly 1 node state for coalesce node"

            state = coalesce_states[0]
            # Verify status is COMPLETED for successful merge
            assert state.status == NodeStateStatus.COMPLETED, (
                f"Token {token.token_id} coalesce node state should be COMPLETED, got {state.status}"
            )
            # Verify input_hash is present (audit trail integrity)
            assert state.input_hash is not None, f"Token {token.token_id} node state must have input_hash for audit trail"
            # Verify input_hash matches the token's row_data
            expected_input_hash = stable_hash(token.row_data)
            assert state.input_hash == expected_input_hash, (
                f"Token {token.token_id} input_hash mismatch: expected {expected_input_hash}, got {state.input_hash}"
            )
            # Verify output_hash is present for completed state
            assert state.output_hash is not None, f"Token {token.token_id} completed node state must have output_hash"

            # Verify token outcome is COALESCED
            token_outcome = recorder.get_token_outcome(token.token_id)
            assert token_outcome is not None, f"Token {token.token_id} must have outcome recorded"
            assert token_outcome.outcome == RowOutcome.COALESCED, (
                f"Token {token.token_id} outcome should be COALESCED, got {token_outcome.outcome}"
            )

        # P1: Verify token_parents lineage for merged token
        merged_token_id = outcome2.merged_token.token_id
        parents = recorder.get_token_parents(merged_token_id)
        assert len(parents) == 2, f"Merged token should have 2 parents, got {len(parents)}"

        # Verify ordinals are 0 and 1 (ordered correctly)
        ordinals = sorted([p.ordinal for p in parents])
        assert ordinals == [0, 1], f"Parent ordinals should be [0, 1], got {ordinals}"

        # Verify parent token_ids match consumed tokens
        parent_ids = {p.parent_token_id for p in parents}
        consumed_ids = {t.token_id for t in outcome2.consumed_tokens}
        assert parent_ids == consumed_ids, f"Merged token parents {parent_ids} should match consumed tokens {consumed_ids}"


class TestCoalesceIntegration:
    """Integration tests for full fork -> process -> coalesce flow."""

    def test_fork_process_coalesce_full_flow(self, recorder: LandscapeRecorder, run: Run) -> None:
        """Full flow: fork -> different transforms -> coalesce."""
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register all nodes (use existing pattern with plugin_version, config, schema_config)
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="merge",
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="merge_results",
            branches=["sentiment", "entities"],
            policy="require_all",
            merge="nested",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Simulate source row
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"text": "ACME Corp reported positive earnings"},
        )

        # Simulate fork
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["sentiment", "entities"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Simulate different processing on each branch
        sentiment_token = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"sentiment": "positive", "confidence": 0.92},
            branch_name="sentiment",
        )
        entities_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"entities": [{"name": "ACME Corp", "type": "ORG"}]},
            branch_name="entities",
        )

        # Coalesce
        outcome1 = executor.accept(sentiment_token, "merge_results", step_in_pipeline=3)
        assert outcome1.held is True

        outcome2 = executor.accept(entities_token, "merge_results", step_in_pipeline=3)
        assert outcome2.held is False
        assert outcome2.merged_token is not None

        # Verify merged data has nested structure
        merged = outcome2.merged_token.row_data
        assert merged == {
            "sentiment": {"sentiment": "positive", "confidence": 0.92},
            "entities": {"entities": [{"name": "ACME Corp", "type": "ORG"}]},
        }

        # Verify consumed tokens
        assert len(outcome2.consumed_tokens) == 2
        consumed_branches = {t.branch_name for t in outcome2.consumed_tokens}
        assert consumed_branches == {"sentiment", "entities"}


class TestFlushPending:
    """Test flush_pending() for graceful shutdown."""

    @pytest.mark.parametrize(
        (
            "policy",
            "quorum_count",
            "branches",
            "tokens_to_accept",
            "expected_flushed_count",
            "expected_should_merge",
            "expected_failure_reason",
        ),
        [
            pytest.param(
                "best_effort",
                None,
                ["path_a", "path_b", "path_c"],
                [
                    {"row_data": {"a_result": 1}, "branch_name": "path_a"},
                    {"row_data": {"b_result": 2}, "branch_name": "path_b"},
                ],
                1,
                True,
                None,
                id="best_effort-incomplete-merges",
            ),
            pytest.param(
                "quorum",
                3,
                ["model_a", "model_b", "model_c", "model_d"],
                [
                    {"row_data": {"score": 0.9}, "branch_name": "model_a"},
                ],
                1,
                False,
                "quorum_not_met",
                id="quorum-not-met-fails",
            ),
            pytest.param(
                "require_all",
                None,
                ["path_a", "path_b"],
                [
                    {"row_data": {"a_result": 1}, "branch_name": "path_a"},
                ],
                1,
                False,
                "incomplete_branches",
                id="require_all-incomplete-fails",
            ),
            pytest.param(
                "quorum",
                2,
                ["model_a", "model_b", "model_c", "model_d"],
                [
                    {"row_data": {"score": 0.9}, "branch_name": "model_a"},
                    {"row_data": {"score": 0.85}, "branch_name": "model_b"},
                ],
                0,
                None,  # Not applicable - quorum met on accept, nothing flushed
                None,
                id="quorum-met-on-accept-nothing-to-flush",
            ),
        ],
    )
    def test_flush_pending_policy_behavior(
        self,
        recorder: LandscapeRecorder,
        run: Run,
        policy: str,
        quorum_count: int | None,
        branches: list[str],
        tokens_to_accept: list[dict[str, Any]],
        expected_flushed_count: int,
        expected_should_merge: bool | None,
        expected_failure_reason: str | None,
    ) -> None:
        """flush_pending behavior varies by policy and arrival count.

        Covers:
        - best_effort: always merges whatever arrived
        - quorum (not met): fails with quorum_not_met
        - require_all (incomplete): fails with incomplete_branches
        - quorum (met on accept): nothing pending to flush
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        coalesce_name = f"{policy}_merge"
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
            plugin_name=coalesce_name,
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Build settings based on policy
        settings_kwargs: dict[str, Any] = {
            "name": coalesce_name,
            "branches": branches,
            "policy": policy,
            "merge": "union",
        }
        if policy == "best_effort":
            settings_kwargs["timeout_seconds"] = 60.0
        if quorum_count is not None:
            settings_kwargs["quorum_count"] = quorum_count

        settings = CoalesceSettings(**settings_kwargs)

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=branches,
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Map branch names to forked children
        branch_to_child = {branches[i]: children[i] for i in range(len(branches))}

        # Accept tokens as specified
        last_outcome = None
        for token_spec in tokens_to_accept:
            branch = token_spec["branch_name"]
            child = branch_to_child[branch]
            token = TokenInfo(
                row_id=child.row_id,
                token_id=child.token_id,
                row_data=token_spec["row_data"],
                branch_name=branch,
            )
            last_outcome = executor.accept(token, coalesce_name, step_in_pipeline=2)

        # Special case: quorum met on accept (nothing to flush)
        if expected_flushed_count == 0:
            # Verify quorum merged on final accept
            assert last_outcome is not None
            assert last_outcome.held is False
            assert last_outcome.merged_token is not None

            flushed = executor.flush_pending(step_map={coalesce_name: 2})
            assert len(flushed) == 0
            return

        # General case: call flush_pending and verify results
        flushed = executor.flush_pending(step_map={coalesce_name: 2})
        assert len(flushed) == expected_flushed_count

        result = flushed[0]
        assert result.held is False

        if expected_should_merge:
            assert result.merged_token is not None
            # Verify merged data contains all accepted tokens' data
            for token_spec in tokens_to_accept:
                for key, value in token_spec["row_data"].items():
                    assert result.merged_token.row_data.get(key) == value
        else:
            assert result.merged_token is None
            assert result.failure_reason == expected_failure_reason

        # Additional check for require_all: verify metadata
        if policy == "require_all":
            assert result.coalesce_metadata is not None
            assert result.coalesce_metadata["policy"] == "require_all"

    def test_flush_pending_empty_when_no_pending(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """flush_pending should return empty list when nothing is pending."""
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )

        # No coalesces registered, nothing pending
        flushed = executor.flush_pending(step_map={})
        assert len(flushed) == 0


class TestDuplicateBranchDetection:
    """Tests for bug x5a: Duplicate branch arrivals should be detected and rejected.

    Current behavior (BUG): Silent overwrite of first token
    Expected behavior: Raise ValueError to catch upstream bugs immediately

    Per ELSPETH's "Plugin Ownership" principle: bugs in our code should crash,
    not be silently hidden. Duplicate arrivals indicate a bug in fork, retry,
    or checkpoint/resume logic.
    """

    def test_duplicate_branch_arrival_raises_error(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Accepting same branch twice for same row_id must raise ValueError.

        This detects bugs in:
        - Fork creating duplicate branch names
        - Retry logic re-sending already-coalesced tokens
        - Checkpoint/resume replaying tokens incorrectly

        The silent overwrite in current code violates audit integrity:
        first token is lost without any record.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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

        # Create tokens for testing
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

        # First arrival for path_a - should be held
        token_a_first = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"value": 100, "result": "first"},
            branch_name="path_a",
        )
        outcome = executor.accept(token_a_first, "merge_all", step_in_pipeline=2)
        assert outcome.held is True

        # Second arrival for path_a (DUPLICATE) - should raise ValueError
        # Using SAME token to simulate retry/checkpoint bug that replays token
        token_a_duplicate = TokenInfo(
            row_id=children[0].row_id,  # Same row_id
            token_id=children[0].token_id,  # Same token - replayed
            row_data={"value": 100, "result": "duplicate_data"},
            branch_name="path_a",  # Same branch - THIS IS THE BUG
        )

        # BUG x5a: Currently this silently overwrites first token's data
        # Expected: Should raise ValueError to catch upstream bugs
        with pytest.raises(ValueError, match=r"Duplicate arrival.*path_a"):
            executor.accept(token_a_duplicate, "merge_all", step_in_pipeline=2)


class TestTimeoutFailureRecording:
    """Tests for bug 6tb: Timeout-triggered failures must be recorded in audit trail.

    Current behavior (BUG): check_timeouts() returns empty list when quorum not met,
    leaving tokens stranded in _pending with no failure recorded.

    Expected behavior: Return CoalesceOutcome with failure_reason, record FAILED
    outcomes for held tokens, clean up pending entry.
    """

    def test_check_timeouts_records_failure_when_quorum_not_met(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When timeout fires and quorum not met, must record FAILED outcomes.

        Bug 6tb: Currently check_timeouts() silently ignores quorum-not-met,
        leaving tokens stranded until flush_pending() at end-of-source.
        For streaming sources that never end, tokens are stranded indefinitely.

        Expected: Record failure just like flush_pending() does.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.enums import RowOutcome
        from elspeth.engine.clock import MockClock
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        clock = MockClock(start=100.0)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["model_a", "model_b", "model_c"],
            policy="quorum",
            quorum_count=2,  # Need 2 of 3
            merge="nested",
            timeout_seconds=0.1,
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
            clock=clock,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Accept only ONE token (quorum needs 2)
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"score": 0.9},
            branch_name="model_a",
        )
        outcome = executor.accept(token_a, "quorum_merge", step_in_pipeline=2)
        assert outcome.held is True

        # Advance clock past timeout
        clock.advance(0.15)

        # BUG 6tb: Currently returns empty list, leaving token stranded
        # Expected: Return failure outcome with proper audit trail
        timed_out = executor.check_timeouts("quorum_merge", step_in_pipeline=2)

        # Should return ONE failure outcome
        assert len(timed_out) == 1, (
            f"Bug 6tb: check_timeouts() returned {len(timed_out)} outcomes, expected 1 failure outcome. "
            f"When timeout fires and quorum not met, must record failure."
        )

        failure_outcome = timed_out[0]
        assert failure_outcome.failure_reason == "quorum_not_met_at_timeout", (
            f"Expected failure_reason='quorum_not_met_at_timeout', got {failure_outcome.failure_reason}"
        )

        # Verify consumed token was returned
        assert len(failure_outcome.consumed_tokens) == 1
        assert failure_outcome.consumed_tokens[0].token_id == token_a.token_id

        # Verify FAILED outcome recorded in audit trail
        token_outcome = recorder.get_token_outcome(token_a.token_id)
        assert token_outcome is not None, (
            "Bug 6tb: Token has no outcome recorded. check_timeouts() should record FAILED outcome like flush_pending() does."
        )
        assert token_outcome.outcome == RowOutcome.FAILED

        # Verify pending entry was cleaned up
        key = ("quorum_merge", token_a.row_id)
        assert key not in executor._pending, (
            "Bug 6tb: Pending entry not cleaned up after timeout failure. Token would be stranded indefinitely."
        )

    def test_check_timeouts_records_failure_for_require_all(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """When timeout fires for require_all policy, must record FAILED outcomes.

        Bug P1-2026-01-30: check_timeouts() has no handling for require_all policy,
        so timeout_seconds is silently ignored and pending coalesces persist
        indefinitely until flush_pending() at end-of-source.

        For streaming sources that never end, tokens are stranded indefinitely.

        Expected: Record failure with "incomplete_branches" like flush_pending() does.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.enums import RowOutcome
        from elspeth.engine.clock import MockClock
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        clock = MockClock(start=100.0)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="require_all_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="require_all_merge",
            branches=["model_a", "model_b", "model_c"],
            policy="require_all",  # Requires ALL branches
            merge="nested",
            timeout_seconds=0.1,
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
            clock=clock,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children, _fork_group_id = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Accept only TWO tokens (require_all needs all 3)
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"score": 0.9},
            branch_name="model_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"score": 0.8},
            branch_name="model_b",
        )

        outcome_a = executor.accept(token_a, "require_all_merge", step_in_pipeline=2)
        assert outcome_a.held is True

        outcome_b = executor.accept(token_b, "require_all_merge", step_in_pipeline=2)
        assert outcome_b.held is True

        # Advance clock past timeout
        clock.advance(0.15)

        # check_timeouts should record failure and return outcome
        timed_out = executor.check_timeouts("require_all_merge", step_in_pipeline=2)

        # Should return ONE failure outcome (for the pending coalesce)
        assert len(timed_out) == 1, (
            f"Bug P1-2026-01-30: check_timeouts() returned {len(timed_out)} outcomes, expected 1. "
            f"require_all policy ignores timeout_seconds - tokens stranded indefinitely."
        )

        failure_outcome = timed_out[0]
        assert failure_outcome.failure_reason == "incomplete_branches", (
            f"Expected failure_reason='incomplete_branches', got {failure_outcome.failure_reason}"
        )

        # Verify both consumed tokens were returned
        assert len(failure_outcome.consumed_tokens) == 2
        consumed_ids = {t.token_id for t in failure_outcome.consumed_tokens}
        assert token_a.token_id in consumed_ids
        assert token_b.token_id in consumed_ids

        # Verify FAILED outcomes recorded in audit trail
        for token in [token_a, token_b]:
            token_outcome = recorder.get_token_outcome(token.token_id)
            assert token_outcome is not None, (
                f"Token {token.token_id} has no outcome recorded. check_timeouts() should record FAILED outcome like flush_pending()."
            )
            assert token_outcome.outcome == RowOutcome.FAILED

        # Verify pending entry was cleaned up
        key = ("require_all_merge", token_a.row_id)
        assert key not in executor._pending, "Pending entry not cleaned up after timeout failure."

        # Verify coalesce_metadata includes expected info
        assert failure_outcome.coalesce_metadata is not None
        assert failure_outcome.coalesce_metadata["policy"] == "require_all"
        assert set(failure_outcome.coalesce_metadata["branches_arrived"]) == {"model_a", "model_b"}
        assert set(failure_outcome.coalesce_metadata["expected_branches"]) == {"model_a", "model_b", "model_c"}


class TestSelectBranchValidation:
    """Tests for bug 2ho: Select merge must fail when select_branch not arrived.

    Current behavior (BUG): Silent fallback to first arrived branch
    Expected behavior: Return failure outcome when select_branch missing

    The silent fallback violates audit integrity: metadata says "select: slow_model"
    but data came from "fast_model".
    """

    def test_select_merge_fails_when_select_branch_not_arrived(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Select merge must fail if select_branch hasn't arrived.

        Scenario:
        - branches: [slow_model, fast_model]
        - policy: first (merge on first arrival)
        - merge: select, select_branch: slow_model

        Bug 2ho: If fast_model arrives first, triggers merge, returns fast_model
        data silently (fallback on line 396).

        Expected: Return failure since slow_model (the selected branch) not present.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="select_merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Configure: "first" policy with "select" merge - problematic combo
        settings = CoalesceSettings(
            name="select_merge",
            branches=["slow_model", "fast_model"],
            policy="first",  # Merge on first arrival
            merge="select",
            select_branch="slow_model",  # We want slow_model's output
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"query": "test"},
        )
        children, _ = token_manager.fork_token(
            parent_token=initial_token,
            branches=["slow_model", "fast_model"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # fast_model arrives first - triggers merge due to "first" policy
        # But select_branch is "slow_model" which hasn't arrived!
        fast_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"query": "test", "result": "fast_result"},
            branch_name="fast_model",
        )

        outcome = executor.accept(fast_token, "select_merge", step_in_pipeline=2)

        # BUG 2ho: Currently outcome.held=False, outcome.merged_token contains
        # fast_model data (silent fallback), violating the select contract.
        #
        # Expected: failure_reason because select_branch not in arrived
        assert outcome.held is False, "First policy should not hold"
        assert outcome.failure_reason == "select_branch_not_arrived", (
            f"Bug 2ho: Expected failure_reason='select_branch_not_arrived', "
            f"but got {outcome.failure_reason}. "
            f"Merged token contains wrong data: {outcome.merged_token.row_data if outcome.merged_token else 'None'}"
        )
        assert outcome.merged_token is None, "Bug 2ho: Should NOT return merged_token when select_branch missing"


class TestCoalesceMetadataRecording:
    """Tests for bug l4h: Coalesce metadata must be persisted in audit trail.

    Current behavior (BUG): coalesce_metadata is computed in _execute_merge()
    and returned in CoalesceOutcome, but never recorded to the audit database.

    Expected behavior: coalesce_metadata should be included in node_state
    output_data for consumed tokens, enabling complete lineage queries.
    """

    def test_successful_merge_records_coalesce_metadata_in_node_state(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """Successful merge must persist coalesce_metadata in audit trail.

        Bug l4h: Currently output_data only contains {"merged_into": token_id}.
        The rich metadata (policy, branches, timing) is computed but discarded.

        Expected: output_data should include coalesce_context with:
        - policy
        - merge_strategy
        - branches_arrived
        - arrival_order
        - wait_duration_ms
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.enums import NodeStateStatus
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

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
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = CoalesceSettings(
            name="merge_results",
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

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 100},
        )
        children, _ = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
            run_id=run.run_id,
        )

        # Accept both tokens to trigger merge
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

        executor.accept(token_a, "merge_results", step_in_pipeline=2)
        outcome = executor.accept(token_b, "merge_results", step_in_pipeline=2)

        assert outcome.merged_token is not None

        # Query node state for consumed token (path_a)
        node_states = recorder.get_node_states_for_token(token_a.token_id)
        coalesce_state = next(s for s in node_states if s.node_id == coalesce_node.node_id)

        assert coalesce_state.status == NodeStateStatus.COMPLETED

        # BUG l4h: context_after_json should contain coalesce_context with merge metadata
        # Currently it's None because _execute_merge() doesn't pass context_after
        import json

        context_after_json = coalesce_state.context_after_json
        assert context_after_json is not None, (
            "Bug l4h: context_after_json is None. coalesce_metadata was computed but never persisted via context_after."
        )

        context_after = json.loads(context_after_json)
        assert "coalesce_context" in context_after, (
            f"Bug l4h: coalesce_context missing from context_after. "
            f"Got: {context_after}. "
            f"coalesce_metadata was computed but never persisted."
        )

        context = context_after["coalesce_context"]
        assert context["policy"] == "require_all", f"Expected policy='require_all', got {context.get('policy')}"
        assert context["merge_strategy"] == "union", f"Expected merge_strategy='union', got {context.get('merge_strategy')}"
        assert set(context["branches_arrived"]) == {"path_a", "path_b"}, (
            f"Expected branches_arrived=['path_a', 'path_b'], got {context.get('branches_arrived')}"
        )
