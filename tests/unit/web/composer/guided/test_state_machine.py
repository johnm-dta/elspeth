"""Tests for guided-mode state machine dataclasses."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
)


class TestTerminalState:
    def test_completed_kind_has_no_reason(self) -> None:
        ts = TerminalState(kind=TerminalKind.COMPLETED, reason=None)
        assert ts.kind == TerminalKind.COMPLETED
        assert ts.reason is None

    def test_exited_to_freeform_requires_reason(self) -> None:
        ts = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_CHOSE_FREEFORM,
        )
        assert ts.kind == TerminalKind.EXITED_TO_FREEFORM
        assert ts.reason == TerminalReason.USER_CHOSE_FREEFORM

    def test_terminal_state_is_frozen(self) -> None:
        ts = TerminalState(kind=TerminalKind.COMPLETED, reason=None)
        with pytest.raises(AttributeError):
            ts.kind = TerminalKind.EXITED_TO_FREEFORM  # type: ignore[misc]


class TestTurnRecord:
    def test_turn_record_carries_emitted_and_response(self) -> None:
        from elspeth.web.composer.guided.protocol import Turn, TurnResponse

        turn: Turn = {
            "type": TurnType.SINGLE_SELECT,
            "step_index": 0,
            "payload": {
                "question": "Choose a plugin",
                "options": [{"id": "csv", "label": "CSV", "hint": None}],
                "allow_custom": False,
            },
        }
        response: TurnResponse = {
            "chosen": ["csv"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        rec = TurnRecord(turn=turn, response=response)
        assert rec.turn == turn
        assert rec.response == response

    def test_turn_record_frozen(self) -> None:
        from elspeth.web.composer.guided.protocol import Turn, TurnResponse

        turn: Turn = {
            "type": TurnType.SINGLE_SELECT,
            "step_index": 0,
            "payload": {
                "question": "Choose a plugin",
                "options": [{"id": "csv", "label": "CSV", "hint": None}],
                "allow_custom": False,
            },
        }
        response: TurnResponse = {
            "chosen": ["csv"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        rec = TurnRecord(turn=turn, response=response)
        with pytest.raises(AttributeError):
            rec.turn = turn  # type: ignore[misc]


class TestGuidedSession:
    def test_initial_session_at_step_1(self) -> None:
        sess = GuidedSession.initial()
        assert sess.step == GuidedStep.STEP_1_SOURCE
        assert sess.history == ()
        assert sess.step_1_result is None
        assert sess.step_2_result is None
        assert sess.step_3_proposal is None
        assert sess.terminal is None

    def test_session_history_is_immutable_tuple(self) -> None:
        sess = GuidedSession.initial()
        assert isinstance(sess.history, tuple)
        with pytest.raises(AttributeError):
            sess.history = ()  # type: ignore[misc]

    def test_session_with_terminal_set(self) -> None:
        term = TerminalState(kind=TerminalKind.COMPLETED, reason=None)
        sess = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=term,
        )
        assert sess.terminal is term
        assert sess.terminal.kind == TerminalKind.COMPLETED
