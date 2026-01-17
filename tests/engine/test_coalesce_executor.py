"""Tests for CoalesceExecutor."""

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import Run
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager


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
