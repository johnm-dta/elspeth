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
def executor_setup(
    recorder: LandscapeRecorder, run: Run
) -> tuple[LandscapeRecorder, SpanFactory, "TokenManager", str]:
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

    def test_executor_initializes(
        self, executor_setup: tuple[LandscapeRecorder, SpanFactory, Any, str]
    ) -> None:
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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
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
            merge="select",
            select_branch="fast",  # Prefer fast, but take whatever arrives first
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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["fast", "slow"],
            step_in_pipeline=1,
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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
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

    def test_quorum_does_not_merge_on_timeout_if_quorum_not_met(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """QUORUM should NOT merge on timeout if quorum not met."""
        import time

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
            quorum_count=2,  # Need 2 of 3
            merge="nested",
            timeout_seconds=0.1,
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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
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

        # Wait for timeout
        time.sleep(0.15)

        # check_timeouts should return empty list (quorum not met)
        timed_out = executor.check_timeouts("quorum_merge", step_in_pipeline=2)
        assert len(timed_out) == 0


class TestCoalesceExecutorBestEffort:
    """Test BEST_EFFORT policy with timeout."""

    def test_best_effort_merges_on_timeout(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """BEST_EFFORT should merge whatever arrived when timeout expires."""
        import time

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
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={},
        )
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b", "path_c"],
            step_in_pipeline=1,
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

        # Wait for timeout
        time.sleep(0.15)

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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
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


class TestCoalesceIntegration:
    """Integration tests for full fork -> process -> coalesce flow."""

    def test_fork_process_coalesce_full_flow(
        self, recorder: LandscapeRecorder, run: Run
    ) -> None:
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
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["sentiment", "entities"],
            step_in_pipeline=1,
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
