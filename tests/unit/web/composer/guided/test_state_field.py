"""Tests for CompositionState.guided_session field."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.state import CompositionState, PipelineMetadata


def _empty_state(**overrides: object) -> CompositionState:
    """Return a minimal valid CompositionState, with optional field overrides."""
    base: dict[str, object] = {
        "source": None,
        "nodes": (),
        "edges": (),
        "outputs": (),
        "metadata": PipelineMetadata(),
        "version": 1,
    }
    base.update(overrides)
    return CompositionState(**base)  # type: ignore[arg-type]


class TestGuidedSessionField:
    def test_default_is_none(self) -> None:
        state = _empty_state()
        assert state.guided_session is None

    def test_can_attach_initial_session(self) -> None:
        sess = GuidedSession.initial()
        state = _empty_state(guided_session=sess)
        assert state.guided_session is sess

    def test_session_history_remains_immutable(self) -> None:
        sess = GuidedSession.initial()
        state = _empty_state(guided_session=sess)
        with pytest.raises(AttributeError):
            state.guided_session = None  # type: ignore[misc]
