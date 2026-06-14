"""Verify default-guided invariant: fresh sessions get GuidedSession.initial()
attached on first lazy-create in any endpoint.

Per errata C7 / spec §5.2 — new sessions default to guided mode. The route
layer's lazy-create branch is the only place fresh CompositionState is
built (SessionServiceImpl.create_session does not eagerly persist state);
the _initial_composition_state_with_guided_session helper applied
uniformly enforces the invariant.

HTTP-level test (driving lazy-create via send_message or recompose) is
deferred to Task 3.5, which mounts the /guided/respond endpoint and will
have a natural trigger for state lazy-create. The unit test of the helper
is sufficient for Task 3.4's acceptance criterion.
"""

from __future__ import annotations

from elspeth.web.composer.guided.protocol import GuidedStep


def test_helper_attaches_initial_guided_session() -> None:
    """The factory helper returns CompositionState with GuidedSession.initial()."""
    from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session

    state = _initial_composition_state_with_guided_session()

    assert state.sources.get("source") is None
    assert state.nodes == ()
    assert state.edges == ()
    assert state.outputs == ()
    assert state.version == 1
    assert state.guided_session is not None
    assert state.guided_session.step is GuidedStep.STEP_1_SOURCE
    assert state.guided_session.terminal is None
    assert state.guided_session.history == ()


def test_helper_is_deterministic() -> None:
    """Two calls produce equal (but distinct) GuidedSession instances."""
    from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session

    a = _initial_composition_state_with_guided_session()
    b = _initial_composition_state_with_guided_session()

    # Equal: same shape, same step
    assert a.guided_session == b.guided_session
    # Distinct CompositionState instances (no shared state)
    assert a is not b
