"""Integration tests for guided-mode step commit handlers.

These tests use real CompositionState and the real tools.py mutation
helpers; only the catalog is constructed via the public test seam
(create_catalog_service()).
"""

from __future__ import annotations

from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SourceResolved,
)
from elspeth.web.composer.guided.steps import StepHandlerResult, handle_step_1_source
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.dependencies import create_catalog_service


def _empty_state() -> CompositionState:
    """Construct an empty CompositionState per errata C3 (6-arg required ctor)."""
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


class TestStep1Handler:
    def test_commits_source_to_state_on_success(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog = create_catalog_service()

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "data.csv", "schema": {"mode": "observed"}},
                observed_columns=("a", "b"),
                sample_rows=({"a": "1", "b": "2"},),
            ),
            catalog=catalog,
        )

        assert isinstance(result, StepHandlerResult)
        assert result.tool_result.success is True
        assert result.state.source is not None
        assert result.state.source.plugin == "csv"
        assert result.session.step_1_result is not None
        assert result.session.step_1_result.plugin == "csv"
        # Session step pointer is NOT advanced here — the dispatcher does that.
        assert result.session.step == session.step

    def test_returns_failure_unchanged_session_when_plugin_unknown(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog = create_catalog_service()

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="DEFINITELY_NOT_A_REAL_PLUGIN_xyzzy",
                options={},
                observed_columns=(),
                sample_rows=(),
            ),
            catalog=catalog,
        )

        assert result.tool_result.success is False
        assert result.state is state  # unchanged on failure
        assert result.session.step_1_result is None
