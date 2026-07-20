"""_initial_composition_state_with_guided_session threads the WorkflowProfile."""

from __future__ import annotations

from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
from elspeth.web.sessions.routes._helpers import (
    _initial_composition_state_with_guided_session,
)


def test_default_is_empty_profile() -> None:
    state = _initial_composition_state_with_guided_session()
    assert state.guided_session is not None
    assert state.guided_session.profile == EMPTY_PROFILE


def test_threads_tutorial_profile() -> None:
    state = _initial_composition_state_with_guided_session(profile=TUTORIAL_PROFILE)
    assert state.guided_session is not None
    assert state.guided_session.profile == TUTORIAL_PROFILE
