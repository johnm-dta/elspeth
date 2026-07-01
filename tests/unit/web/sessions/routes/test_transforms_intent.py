"""Unit test: transforms-intent extraction from guided chat history.

In the staged orchestrator each phase is driven by its own ``/guided/chat`` send
(source -> sink -> transforms). A reject / advisor / repair re-solve of the
transform chain must rebuild from the TRANSFORMS-phase request, NOT the
source-phase opening turn. ``_transforms_intent_from_chat_history`` is the reader
that supplies it: the LAST USER turn's content (``None`` when no user turn exists
yet). Reject/advisor/repair gestures arrive on ``/guided/respond`` control
signals, not chat, so they add no later user chat turn — the transforms send
stays the last one.
"""

from __future__ import annotations

from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, GuidedStep
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.sessions.routes._helpers import _transforms_intent_from_chat_history


def _session(*turns: ChatTurn) -> GuidedSession:
    return GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(),
        step_1_result=None,
        step_2_result=None,
        step_3_proposal=None,
        chat_history=tuple(turns),
    )


_TS = "2026-06-27T00:00:00Z"


def test_returns_last_user_turn_content() -> None:
    # Staged flow: a Source-phase send, then a Transforms-phase send. A transform
    # re-solve must rebuild from the TRANSFORMS send (the last user turn), never
    # the source opening — feeding the source intent re-solves blind to the
    # transform the operator actually asked for.
    session = _session(
        ChatTurn(role=ChatRole.USER, content="Create a source of url rows", seq=0, step=GuidedStep.STEP_1_SOURCE, ts_iso=_TS),
        ChatTurn(role=ChatRole.ASSISTANT, content="I set up a source", seq=1, step=GuidedStep.STEP_1_SOURCE, ts_iso=_TS),
        ChatTurn(role=ChatRole.USER, content="Fetch each page and summarise it", seq=2, step=GuidedStep.STEP_3_TRANSFORMS, ts_iso=_TS),
        ChatTurn(role=ChatRole.ASSISTANT, content="Proposed a chain", seq=3, step=GuidedStep.STEP_3_TRANSFORMS, ts_iso=_TS),
    )
    assert _transforms_intent_from_chat_history(session) == "Fetch each page and summarise it"


def test_returns_none_when_no_user_turn() -> None:
    assert _transforms_intent_from_chat_history(_session()) is None
    assistant_only = _session(
        ChatTurn(role=ChatRole.ASSISTANT, content="hello", seq=0, step=GuidedStep.STEP_1_SOURCE, ts_iso=_TS),
    )
    assert _transforms_intent_from_chat_history(assistant_only) is None
