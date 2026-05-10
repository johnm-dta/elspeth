# tests/unit/web/composer/guided/test_state_machine.py
"""Tests for GuidedSession, TerminalState, TurnRecord — state machine data."""

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
        t = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline:\n")
        assert t.kind is TerminalKind.COMPLETED
        assert t.reason is None

    def test_exited_to_freeform_requires_reason(self) -> None:
        t = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        assert t.reason is TerminalReason.USER_PRESSED_EXIT

    def test_terminal_state_is_frozen(self) -> None:
        t = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml=None)
        with pytest.raises(AttributeError):
            t.kind = TerminalKind.EXITED_TO_FREEFORM  # type: ignore[misc]


class TestTurnRecord:
    def test_turn_record_carries_emitted_and_response(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc123",
            response_hash="def456",
            emitter="server",
        )
        assert rec.emitter == "server"

    def test_turn_record_frozen(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc",
            response_hash=None,
            emitter="server",
        )
        with pytest.raises(AttributeError):
            rec.emitter = "llm"  # type: ignore[misc]


class TestGuidedSession:
    def test_initial_session_at_step_1(self) -> None:
        s = GuidedSession.initial()
        assert s.step is GuidedStep.STEP_1_SOURCE
        assert s.terminal is None
        assert s.history == ()

    def test_session_history_is_immutable_tuple(self) -> None:
        s = GuidedSession.initial()
        with pytest.raises(AttributeError):
            s.history.append(None)  # type: ignore[attr-defined]

    def test_session_with_terminal_set(self) -> None:
        s = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            terminal=TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="x:\n"),
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
        )
        assert s.terminal is not None
        assert s.terminal.kind is TerminalKind.COMPLETED
