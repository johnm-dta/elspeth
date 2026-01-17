"""Tests for CoalesceExecutor."""

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager


@pytest.fixture
def db() -> LandscapeDB:
    return LandscapeDB.in_memory()


@pytest.fixture
def run(db: LandscapeDB) -> Any:
    recorder = LandscapeRecorder(db)
    return recorder.begin_run(config={}, canonical_version="v1")


@pytest.fixture
def executor_setup(
    db: LandscapeDB, run: Any
) -> tuple[LandscapeRecorder, SpanFactory, "TokenManager", str]:
    """Common setup for executor tests - reduces boilerplate.

    Returns:
        Tuple of (recorder, span_factory, token_manager, run_id)
    """
    from elspeth.engine.tokens import TokenManager

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()
    token_manager = TokenManager(recorder)
    return recorder, span_factory, token_manager, run.run_id


class TestCoalesceExecutorInit:
    """Test CoalesceExecutor initialization."""

    def test_executor_initializes(self, db: LandscapeDB, run: Any) -> None:
        """Executor should initialize with recorder and span factory."""
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )

        assert executor is not None
