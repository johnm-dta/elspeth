# tests/unit/web/composer/guided/test_state_machine.py
"""Tests for GuidedSession, TerminalState, TurnRecord — state machine data."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.protocol import GuidedStep, TurnResponse, TurnType
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
    step_advance,
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


# ---------------------------------------------------------------------------
# Helper: build a minimal TurnResponse
# ---------------------------------------------------------------------------


def _make_response(
    *,
    control_signal: str | None = None,
    edited_values: dict[str, object] | None = None,
    chosen: list[str] | None = None,
) -> TurnResponse:
    return TurnResponse(
        chosen=chosen,
        edited_values=edited_values,
        custom_inputs=None,
        accepted_step_index=None,
        edit_step_index=None,
        control_signal=control_signal,
    )


# ---------------------------------------------------------------------------
# Task 2.1 tests: step_advance dispatcher + Step 1 → Step 2 branch
# ---------------------------------------------------------------------------


class TestStepAdvance:
    def test_initial_session_advances_after_source_confirmed(self) -> None:
        """INSPECT_AND_CONFIRM with valid edited_values advances to STEP_2_SINK."""
        session = GuidedSession.initial()
        response = _make_response(
            edited_values={
                "plugin": "csv",
                "options": {"path": "/data/input.csv"},
                "observed_columns": ["id", "name", "score"],
                "sample_rows": [{"id": 1, "name": "Alice", "score": 99}],
            },
        )
        new_sess, next_turn, terminal, directives = step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)
        assert new_sess.step is GuidedStep.STEP_2_SINK
        assert new_sess.step_1_result is not None
        assert new_sess.step_1_result.plugin == "csv"
        assert terminal is None
        assert next_turn is None
        assert any(d.tool_name == "guided_step_advanced" for d in directives)

    def test_inspect_and_confirm_without_edited_values_raises(self) -> None:
        """edited_values=None on an INSPECT_AND_CONFIRM response must raise ValueError."""
        session = GuidedSession.initial()
        response = _make_response(edited_values=None)
        with pytest.raises(ValueError, match="inspect_and_confirm"):
            step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)

    def test_exit_to_freeform_terminates_with_user_pressed_exit(self) -> None:
        """control_signal='exit_to_freeform' produces USER_PRESSED_EXIT terminal."""
        session = GuidedSession.initial()
        response = _make_response(control_signal="exit_to_freeform")
        _new_sess, next_turn, terminal, directives = step_advance(session, response, current_turn_type=TurnType.SINGLE_SELECT)
        assert terminal is not None
        assert terminal.kind is TerminalKind.EXITED_TO_FREEFORM
        assert terminal.reason is TerminalReason.USER_PRESSED_EXIT
        assert terminal.pipeline_yaml is None
        assert next_turn is None
        assert any(d.tool_name == "guided_dropped_to_freeform" for d in directives)
        # The directive must record the drop_reason value so Phase 3 can reconstruct it
        drop_directive = next(d for d in directives if d.tool_name == "guided_dropped_to_freeform")
        assert drop_directive.arguments["drop_reason"] == "user_pressed_exit"

    def test_step_1_non_inspect_turn_does_not_advance(self) -> None:
        """A SINGLE_SELECT response in Step 1 is an intra-step turn — no advance."""
        session = GuidedSession.initial()
        response = _make_response(chosen=["csv"])
        new_sess, next_turn, terminal, directives = step_advance(session, response, current_turn_type=TurnType.SINGLE_SELECT)
        assert new_sess is session  # same object — no state change
        assert next_turn is None
        assert terminal is None
        assert directives == []

    # ---------------------------------------------------------------------------
    # Task 2.2 tests: Step 2 → Step 2.5 branch
    # ---------------------------------------------------------------------------

    def test_step_2_advances_after_required_fields_declared(self) -> None:
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a", "b"),
                sample_rows=({"a": "1", "b": "2"},),
            ),
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": {
                "outputs": [
                    {
                        "plugin": "json",
                        "options": {"path": "out.jsonl"},
                        "required_fields": ["a"],
                        "schema_mode": "fixed",
                    },
                ],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }

        new_sess, _next, terminal, _events = step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)

        assert new_sess.step is GuidedStep.STEP_2_5_RECIPE_MATCH
        assert new_sess.step_2_result is not None
        assert len(new_sess.step_2_result.outputs) == 1
        assert terminal is None

    def test_step_2_let_source_decide_sets_observed_mode(self) -> None:
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a", "b"),
                sample_rows=({},),
            ),
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": {
                "outputs": [
                    {
                        "plugin": "json",
                        "options": {"path": "out.jsonl"},
                        "required_fields": [],
                        "schema_mode": "observed",
                    },
                ],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, _next, _terminal, _events = step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)
        assert new_sess.step_2_result is not None
        assert new_sess.step_2_result.outputs[0].schema_mode == "observed"

    def test_step_2_without_edited_values_raises(self) -> None:
        """edited_values=None on a MULTI_SELECT_WITH_CUSTOM response must raise ValueError."""
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a",),
                sample_rows=({},),
            ),
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
        response = _make_response(edited_values=None)
        with pytest.raises(ValueError, match="multi_select_with_custom"):
            step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)

    # ---------------------------------------------------------------------------
    # Task 2.4 tests: Step 2.5 → Step 3 (or passthrough on recipe accept)
    # ---------------------------------------------------------------------------

    def test_step_2_5_recipe_accepted_session_unchanged(self) -> None:
        """chosen=["accept"] at STEP_2_5: session stays at STEP_2_5, no directive.

        The endpoint handler (Errata C2) reads response["chosen"] == ["accept"]
        and invokes _execute_apply_pipeline_recipe to commit the recipe and
        produce the COMPLETED terminal. step_advance is pure and does not run
        apply_recipe; it leaves the session unchanged for the handler to act on.
        """
        sess = GuidedSession(
            step=GuidedStep.STEP_2_5_RECIPE_MATCH,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={"blob_id": "blob-1"},
                observed_columns=("a",),
                sample_rows=({},),
            ),
            step_2_result=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl"},
                        required_fields=("category",),
                        schema_mode="fixed",
                    ),
                )
            ),
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": ["accept"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, next_turn, terminal, directives = step_advance(
            sess,
            response,
            current_turn_type=TurnType.RECIPE_OFFER,
        )
        # Session stays at STEP_2_5 — endpoint handler advances to COMPLETED
        assert new_sess.step is GuidedStep.STEP_2_5_RECIPE_MATCH
        assert new_sess is sess  # pure: no state change
        assert next_turn is None
        assert terminal is None
        assert directives == []

    def test_step_2_5_build_manually_advances_to_step_3(self) -> None:
        """chosen=["build_manually"] at STEP_2_5: advance to STEP_3_TRANSFORMS."""
        sess = GuidedSession(
            step=GuidedStep.STEP_2_5_RECIPE_MATCH,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a",),
                sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": ["build_manually"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, next_turn, terminal, directives = step_advance(
            sess,
            response,
            current_turn_type=TurnType.RECIPE_OFFER,
        )
        assert new_sess.step is GuidedStep.STEP_3_TRANSFORMS
        assert next_turn is None
        assert terminal is None
        assert len(directives) == 1
        assert directives[0].tool_name == "guided_step_advanced"
        assert directives[0].arguments["prev_step"] == GuidedStep.STEP_2_5_RECIPE_MATCH.value
        assert directives[0].arguments["next_step"] == GuidedStep.STEP_3_TRANSFORMS.value
        assert directives[0].arguments["reason"] == "user_advanced"

    def test_step_2_5_unexpected_chosen_raises(self) -> None:
        """Any chosen value other than 'accept' or 'build_manually' is a protocol violation."""
        sess = GuidedSession(
            step=GuidedStep.STEP_2_5_RECIPE_MATCH,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a",),
                sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=None,
            terminal=None,
        )
        response = _make_response(chosen=["nonsense"])
        with pytest.raises(ValueError, match="unexpected chosen for recipe_offer"):
            step_advance(sess, response, current_turn_type=TurnType.RECIPE_OFFER)
