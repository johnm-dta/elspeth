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
