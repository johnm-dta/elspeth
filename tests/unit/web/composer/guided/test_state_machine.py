# tests/unit/web/composer/guided/test_state_machine.py
"""Tests for GuidedSession, TerminalState, TurnRecord — state machine data."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, TurnResponse, TurnType
from elspeth.web.composer.guided.state_machine import (
    GUIDED_SESSION_SCHEMA_VERSION,
    GuidedSession,
    SinkIntent,
    SourceIntent,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
    mark_protocol_violation,
    mark_solver_exhausted,
    step_advance,
)
from elspeth.web.composer.source_inspection import facts_from_dict

_SOURCE_ID = "11111111-1111-4111-8111-111111111111"
_OUTPUT_ID = "22222222-2222-4222-8222-222222222222"


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
        t = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline: {}\n")
        with pytest.raises(AttributeError):
            t.kind = TerminalKind.EXITED_TO_FREEFORM  # type: ignore[misc]


class TestTurnRecord:
    def test_turn_record_carries_emitted_and_response(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="a" * 64,
            response_hash="b" * 64,
            summary="Selected source: csv",
            emitter="server",
        )
        assert rec.emitter == "server"
        assert rec.summary == "Selected source: csv"

    def test_turn_record_frozen(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="a" * 64,
            response_hash=None,
            emitter="server",
        )
        with pytest.raises(AttributeError):
            rec.emitter = "llm"  # type: ignore[misc]

    def test_turn_record_summary_roundtrip(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="a" * 64,
            response_hash="b" * 64,
            summary="Selected source: csv",
            emitter="server",
        )
        assert TurnRecord.from_dict(rec.to_dict()) == rec


class TestChatTurn:
    def test_valid_chat_turn_is_frozen_dataclass(self) -> None:
        turn = ChatTurn(
            role=ChatRole.USER,
            content="What does this step need?",
            seq=0,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:00+00:00",
        )

        assert turn.role is ChatRole.USER
        with pytest.raises(AttributeError):
            turn.content = "mutated"  # type: ignore[misc]

    def test_rejects_wire_role_from_audit_initiator_namespace(self) -> None:
        with pytest.raises(TypeError, match="role"):
            ChatTurn(
                role="step_entry_opener",  # type: ignore[arg-type]
                content="What does this step need?",
                seq=0,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
            )

    def test_rejects_negative_seq(self) -> None:
        with pytest.raises(ValueError, match="seq"):
            ChatTurn(
                role=ChatRole.USER,
                content="What does this step need?",
                seq=-1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
            )

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValueError, match="content"):
            ChatTurn(
                role=ChatRole.USER,
                content="",
                seq=0,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
            )

    def test_rejects_wrong_step_type(self) -> None:
        with pytest.raises(TypeError, match="step"):
            ChatTurn(
                role=ChatRole.USER,
                content="What does this step need?",
                seq=0,
                step="step_1_source",  # type: ignore[arg-type]
                ts_iso="2026-05-13T12:00:00+00:00",
            )

    def test_assistant_message_kind_is_required(self) -> None:
        with pytest.raises(ValueError, match="assistant_message_kind is required"):
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content="Here is some advice.",
                seq=1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
            )

    def test_accepts_real_assistant_reply_kind(self) -> None:
        turn = ChatTurn(
            role=ChatRole.ASSISTANT,
            content="Here is some advice.",
            seq=1,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:00+00:00",
            assistant_message_kind="assistant",
        )
        assert turn.assistant_message_kind == "assistant"
        assert turn.synthetic_failure_reason is None

    def test_accepts_synthetic_failure_kind_with_reason(self) -> None:
        turn = ChatTurn(
            role=ChatRole.ASSISTANT,
            content="That reply didn't pass a quality check.",
            seq=1,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:00+00:00",
            assistant_message_kind="synthetic_failure",
            synthetic_failure_reason="quality_guard",
        )
        assert turn.assistant_message_kind == "synthetic_failure"
        assert turn.synthetic_failure_reason == "quality_guard"

    def test_accepts_not_applied_synthetic_failure_reason(self) -> None:
        turn = ChatTurn(
            role=ChatRole.ASSISTANT,
            content="I did not apply that generated source.",
            seq=1,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:00+00:00",
            assistant_message_kind="synthetic_failure",
            synthetic_failure_reason="not_applied",
        )
        assert turn.synthetic_failure_reason == "not_applied"

    def test_rejects_synthetic_failure_without_reason(self) -> None:
        with pytest.raises(ValueError, match="synthetic_failure_reason is required"):
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content="I could not apply that response.",
                seq=1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
                assistant_message_kind="synthetic_failure",
            )

    def test_rejects_unknown_assistant_message_kind(self) -> None:
        with pytest.raises(ValueError, match="assistant_message_kind"):
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content="x",
                seq=1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
                assistant_message_kind="bogus",  # type: ignore[arg-type]
            )

    def test_rejects_unknown_synthetic_failure_reason(self) -> None:
        with pytest.raises(ValueError, match="synthetic_failure_reason"):
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content="x",
                seq=1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
                assistant_message_kind="synthetic_failure",
                synthetic_failure_reason="bogus",  # type: ignore[arg-type]
            )

    def test_rejects_reason_without_synthetic_failure_kind(self) -> None:
        with pytest.raises(ValueError, match="synthetic_failure_reason is set"):
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content="x",
                seq=1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
                assistant_message_kind="assistant",
                synthetic_failure_reason="quality_guard",  # type: ignore[arg-type]
            )

    def test_rejects_kind_on_user_turn(self) -> None:
        with pytest.raises(ValueError, match="not applicable to a USER turn"):
            ChatTurn(
                role=ChatRole.USER,
                content="x",
                seq=0,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-05-13T12:00:00+00:00",
                assistant_message_kind="assistant",
            )


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
        )
        assert s.terminal is not None
        assert s.terminal.kind is TerminalKind.COMPLETED

    def test_initial_session_has_empty_plural_state(self) -> None:
        sess = GuidedSession.initial()
        restored = GuidedSession.from_dict(sess.to_dict())

        assert restored == sess
        assert restored.source_order == ()
        assert restored.reviewed_sources == {}
        assert restored.pending_source_intents == {}
        assert restored.output_order == ()
        assert restored.reviewed_outputs == {}
        assert restored.pending_output_intents == {}
        assert restored.active_proposal is None
        assert restored.active_edit_target is None

    def test_source_inspection_facts_from_dict_is_tier1_strict(self) -> None:
        d = {
            "source_kind": "csv",
            "redacted_identity": {"filename": "input.csv"},
            "byte_range_inspected": [0, 128],
            "sample_row_count": 10,
            "observed_headers": ["name", "age"],
            "inferred_types": {"name": "str", "age": "int"},
            "url_candidates": [],
            "warnings": [],
        }
        restored = facts_from_dict(d)
        assert restored.observed_headers == ("name", "age")

    def test_guided_session_schema_version_is_8(self) -> None:
        assert GUIDED_SESSION_SCHEMA_VERSION == 8

    def test_guided_session_to_dict_includes_schema_version(self) -> None:
        sess = GuidedSession.initial()
        assert sess.to_dict()["schema_version"] == 8

    def test_guided_session_requires_schema_version(self) -> None:
        current = GuidedSession.initial().to_dict()
        del current["schema_version"]

        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(current)

    def test_guided_session_rejects_old_schema_version(self) -> None:
        old = GuidedSession.initial().to_dict()
        old["schema_version"] = 4

        with pytest.raises(InvariantError, match="unsupported schema_version 4"):
            GuidedSession.from_dict(old)

    def test_guided_session_rejects_prior_v6_record_carrying_entry_seed(self) -> None:
        # Regression for the v6->v7 migration hazard: a session persisted by the
        # pre-entry_seed-removal build carries schema_version=6 and a now-dropped
        # ``entry_seed`` key in its nested profile sub-dict. The schema_version
        # gate must reject it cleanly *before* the profile parse, so the operator
        # sees the actionable "unsupported schema_version" signal rather than the
        # deep, misleading "malformed profile" error the orphaned key would raise
        # in WorkflowProfile.from_dict's closed-key-set check.
        old = GuidedSession.initial().to_dict()
        old["schema_version"] = 6
        old["profile"] = {**old["profile"], "entry_seed": "legacy framing prompt"}

        with pytest.raises(InvariantError, match="unsupported schema_version 6"):
            GuidedSession.from_dict(old)

    def test_guided_session_current_history_requires_summary(self) -> None:
        current = GuidedSession.initial().to_dict()
        current["history"] = [
            {
                "step": GuidedStep.STEP_1_SOURCE.value,
                "turn_type": TurnType.SINGLE_SELECT.value,
                "payload_hash": "abc",
                "response_hash": "def",
                "emitter": "server",
            }
        ]

        with pytest.raises(InvariantError, match=r"history\[0\].*summary.*exact dict"):
            GuidedSession.from_dict(current)

    def test_guided_session_schema8_requires_source_order(self) -> None:
        current = GuidedSession.initial().to_dict()
        del current["source_order"]

        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(current)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("schema_version", "8", "schema_version"),
            ("schema_version", 8.0, "schema_version"),
            ("schema_version", True, "schema_version"),
            ("transition_consumed", "false", "transition_consumed"),
            ("transition_consumed", 1, "transition_consumed"),
            ("chat_history", (), "chat_history"),
            ("chat_turn_seq", "0", "chat_turn_seq"),
            ("chat_turn_seq", True, "chat_turn_seq"),
            ("chat_turn_seq", -1, "chat_turn_seq"),
        ],
    )
    def test_guided_session_from_dict_rejects_coerced_scalar_fields(
        self,
        field: str,
        bad_value: object,
        match: str,
    ) -> None:
        d = GuidedSession.initial().to_dict()
        d[field] = bad_value

        with pytest.raises(InvariantError, match=match):
            GuidedSession.from_dict(d)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("role", 1, "GuidedSession\\.from_dict"),
            ("content", 42, "GuidedSession\\.from_dict"),
            ("seq", "0", "GuidedSession\\.from_dict"),
            ("seq", True, "GuidedSession\\.from_dict"),
            ("seq", -1, "GuidedSession\\.from_dict"),
            ("step", 1, "GuidedSession\\.from_dict"),
            ("ts_iso", None, "GuidedSession\\.from_dict"),
        ],
    )
    def test_guided_session_from_dict_rejects_malformed_chat_history_entry(
        self,
        field: str,
        bad_value: object,
        match: str,
    ) -> None:
        d = GuidedSession.initial().to_dict()
        chat_entry = {
            "role": ChatRole.USER.value,
            "content": "Need help with this step.",
            "seq": 0,
            "step": GuidedStep.STEP_1_SOURCE.value,
            "ts_iso": "2026-05-13T12:00:00+00:00",
            "assistant_message_kind": None,
            "synthetic_failure_reason": None,
        }
        chat_entry[field] = bad_value
        d["chat_history"] = [chat_entry]

        with pytest.raises(InvariantError, match=match):
            GuidedSession.from_dict(d)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("assistant_message_kind", 1),
            ("synthetic_failure_reason", 1),
        ],
    )
    def test_guided_session_from_dict_rejects_malformed_chat_history_kind_type(
        self,
        field: str,
        bad_value: object,
    ) -> None:
        """assistant_message_kind/synthetic_failure_reason: str or None, never another type."""
        d = GuidedSession.initial().to_dict()
        chat_entry = {
            "role": ChatRole.ASSISTANT.value,
            "content": "Here is some advice.",
            "seq": 0,
            "step": GuidedStep.STEP_1_SOURCE.value,
            "ts_iso": "2026-05-13T12:00:00+00:00",
            "assistant_message_kind": None,
            "synthetic_failure_reason": None,
        }
        chat_entry[field] = bad_value
        d["chat_history"] = [chat_entry]

        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(d)

    def test_guided_session_from_dict_rejects_unknown_chat_history_kind_value(self) -> None:
        """A closed-set violation on the persisted value still crashes loudly
        (ChatTurn.__post_init__'s ValueError, wrapped as InvariantError by the
        same broad except every other malformed chat_history case hits)."""
        d = GuidedSession.initial().to_dict()
        chat_entry = {
            "role": ChatRole.ASSISTANT.value,
            "content": "Here is some advice.",
            "seq": 0,
            "step": GuidedStep.STEP_1_SOURCE.value,
            "ts_iso": "2026-05-13T12:00:00+00:00",
            "assistant_message_kind": "bogus",
            "synthetic_failure_reason": None,
        }
        d["chat_history"] = [chat_entry]

        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(d)

    def test_guided_session_roundtrip_chat_history_with_synthetic_failure_kind(self) -> None:
        """A persisted synthetic-failure turn survives to_dict/from_dict intact.

        fp-review C-2 persisted-history closure: the discriminator must not be
        lost across the persist/restore cycle (the whole point of adding it).
        """
        user_turn = ChatTurn(
            role=ChatRole.USER,
            content="what should I do?",
            seq=0,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:00+00:00",
        )
        assistant_turn = ChatTurn(
            role=ChatRole.ASSISTANT,
            content="That reply didn't pass a quality check, so it wasn't shown.",
            seq=1,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:01+00:00",
            assistant_message_kind="synthetic_failure",
            synthetic_failure_reason="quality_guard",
        )
        sess = dataclasses.replace(
            GuidedSession.initial(),
            chat_history=(user_turn, assistant_turn),
            chat_turn_seq=2,
        )
        d = sess.to_dict()
        assert d["chat_history"][0]["assistant_message_kind"] is None
        assert d["chat_history"][1]["assistant_message_kind"] == "synthetic_failure"
        assert d["chat_history"][1]["synthetic_failure_reason"] == "quality_guard"

        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.chat_history[0].assistant_message_kind is None
        assert restored.chat_history[1].assistant_message_kind == "synthetic_failure"
        assert restored.chat_history[1].synthetic_failure_reason == "quality_guard"

    def test_guided_session_from_dict_rejects_chat_entry_without_kind_fields(self) -> None:
        """Missing schema-8 discriminator fields fail the integrity check."""
        d = GuidedSession.initial().to_dict()
        incomplete_entry = {
            "role": ChatRole.ASSISTANT.value,
            "content": "an incomplete reply",
            "seq": 0,
            "step": GuidedStep.STEP_1_SOURCE.value,
            "ts_iso": "2026-05-13T12:00:00+00:00",
            # No "assistant_message_kind" / "synthetic_failure_reason" keys.
        }
        d["chat_history"] = [incomplete_entry]

        with pytest.raises(InvariantError, match="missing keys"):
            GuidedSession.from_dict(d)


class TestGuidedSessionProfileFields:
    def test_initial_defaults_to_empty_profile(self) -> None:
        sess = GuidedSession.initial()
        assert sess.profile == EMPTY_PROFILE
        assert sess.advisor_checkpoint_passes_used == 0
        assert sess.advisor_signoff_escape_offered is False

    def test_initial_accepts_profile_argument(self) -> None:
        sess = GuidedSession.initial(profile=TUTORIAL_PROFILE)
        assert sess.profile == TUTORIAL_PROFILE
        assert sess.advisor_checkpoint_passes_used == 0

    def test_to_dict_emits_profile_and_pass_counter(self) -> None:
        sess = GuidedSession.initial(profile=TUTORIAL_PROFILE)
        d = sess.to_dict()
        assert d["profile"] == TUTORIAL_PROFILE.to_dict()
        assert d["advisor_checkpoint_passes_used"] == 0
        assert d["advisor_signoff_escape_offered"] is False

    def test_roundtrip_with_tutorial_profile(self) -> None:
        sess = dataclasses.replace(
            GuidedSession.initial(profile=TUTORIAL_PROFILE),
            advisor_checkpoint_passes_used=2,
            advisor_signoff_escape_offered=True,
        )
        restored = GuidedSession.from_dict(sess.to_dict())
        assert restored == sess
        assert restored.profile == TUTORIAL_PROFILE
        assert restored.advisor_checkpoint_passes_used == 2
        assert restored.advisor_signoff_escape_offered is True

    def test_roundtrip_with_empty_profile(self) -> None:
        sess = GuidedSession.initial()
        assert GuidedSession.from_dict(sess.to_dict()) == sess

    def test_from_dict_rejects_missing_profile_key(self) -> None:
        d = GuidedSession.initial().to_dict()
        del d["profile"]
        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_missing_pass_counter_key(self) -> None:
        d = GuidedSession.initial().to_dict()
        del d["advisor_checkpoint_passes_used"]
        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_missing_escape_flag_key(self) -> None:
        d = GuidedSession.initial().to_dict()
        del d["advisor_signoff_escape_offered"]
        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_malformed_profile(self) -> None:
        d = GuidedSession.initial().to_dict()
        d["profile"] = {"coaching": True}
        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_string_pass_counter(self) -> None:
        d = GuidedSession.initial().to_dict()
        d["advisor_checkpoint_passes_used"] = "1"
        with pytest.raises(InvariantError, match=r"advisor_checkpoint_passes_used"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_bool_as_int_pass_counter(self) -> None:
        d = GuidedSession.initial().to_dict()
        d["advisor_checkpoint_passes_used"] = True
        with pytest.raises(InvariantError, match=r"advisor_checkpoint_passes_used"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_negative_pass_counter(self) -> None:
        d = GuidedSession.initial().to_dict()
        d["advisor_checkpoint_passes_used"] = -1
        with pytest.raises(InvariantError, match=r"advisor_checkpoint_passes_used"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_string_escape_flag(self) -> None:
        d = GuidedSession.initial().to_dict()
        d["advisor_signoff_escape_offered"] = "false"
        with pytest.raises(InvariantError, match=r"advisor_signoff_escape_offered"):
            GuidedSession.from_dict(d)

    def test_from_dict_rejects_number_escape_flag(self) -> None:
        d = GuidedSession.initial().to_dict()
        d["advisor_signoff_escape_offered"] = 1
        with pytest.raises(InvariantError, match=r"advisor_signoff_escape_offered"):
            GuidedSession.from_dict(d)


# ---------------------------------------------------------------------------
# Helper: build a minimal TurnResponse
# ---------------------------------------------------------------------------


def _make_response(
    *,
    control_signal: ControlSignal | None = None,
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


def _wire_response(control: ControlSignal | None = None) -> TurnResponse:
    return TurnResponse(
        chosen=None,
        edited_values=None,
        custom_inputs=None,
        accepted_step_index=None,
        edit_step_index=None,
        control_signal=control,
    )


# ---------------------------------------------------------------------------
# Task 2.1 tests: step_advance dispatcher + Step 1 → Step 2 branch
# ---------------------------------------------------------------------------


class TestStepAdvance:
    def test_inspect_and_confirm_is_intra_step_no_advance(self) -> None:
        """Step 1 remains unchanged until the schema-8 transition is settled.

        ``step_advance`` is pure for this source-stage response. The route
        projects the response with the source transition authority and commits
        the resulting guided state through the atomic operation settlement.
        """
        intent = SourceIntent(
            name="primary",
            phase="plugin_options",
            plugin="csv",
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        )
        from dataclasses import replace

        session = replace(
            GuidedSession.initial(),
            source_order=(_SOURCE_ID,),
            pending_source_intents={_SOURCE_ID: intent},
        )
        response = _make_response(
            edited_values={
                "columns": ["id", "name", "score"],
            },
        )
        new_sess, next_turn, terminal, directives = step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)
        assert new_sess is session  # pure: no state change
        assert new_sess.step is GuidedStep.STEP_1_SOURCE
        assert new_sess.pending_source_intents[_SOURCE_ID] is intent
        assert terminal is None
        assert next_turn is None
        assert directives == []

    def test_exit_to_freeform_terminates_with_user_pressed_exit(self) -> None:
        """control_signal='exit_to_freeform' produces USER_PRESSED_EXIT terminal."""
        session = GuidedSession.initial()
        response = _make_response(control_signal=ControlSignal.EXIT_TO_FREEFORM)
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

    def test_step_2_multi_select_is_intra_step_no_advance(self) -> None:
        """Step 2 remains unchanged until the schema-8 transition is settled.

        ``step_advance`` is pure for this output-stage response. The route
        projects the response with the sink transition authority and commits
        the resulting guided state through the atomic operation settlement.
        """
        intent = SinkIntent(name="main", phase="field_review", plugin="json", options={"path": "out.jsonl"})
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            output_order=(_OUTPUT_ID,),
            pending_output_intents={_OUTPUT_ID: intent},
        )
        response: TurnResponse = {
            "chosen": ["a"],
            "edited_values": None,
            "custom_inputs": [],
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }

        new_sess, next_turn, terminal, directives = step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)

        assert new_sess is sess  # pure: no state change
        assert new_sess.step is GuidedStep.STEP_2_SINK
        assert new_sess.pending_output_intents[_OUTPUT_ID] is intent
        assert next_turn is None
        assert terminal is None
        assert directives == []

    def test_step_2_single_select_and_schema_form_are_also_intra_step(self) -> None:
        """SINGLE_SELECT and SCHEMA_FORM at STEP_2_SINK are likewise pure self-loops."""
        sess = GuidedSession(step=GuidedStep.STEP_2_SINK)
        for turn_type in (TurnType.SINGLE_SELECT, TurnType.SCHEMA_FORM):
            response = _make_response(chosen=["json"])
            new_sess, next_turn, terminal, directives = step_advance(sess, response, current_turn_type=turn_type)
            assert new_sess is sess
            assert next_turn is None
            assert terminal is None
            assert directives == []

    # ---------------------------------------------------------------------------
    # Task 2.5 tests: Step 3 accept/edit/reject + terminal-failure paths
    # ---------------------------------------------------------------------------

    def test_step_3_legacy_turn_is_still_a_pure_self_loop_during_cutover(self) -> None:
        """Task 3 replaces the protocol enum; Task 1 keeps the pure dispatcher safe meanwhile."""
        sess = GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS)
        response = _make_response()
        new_sess, next_turn, terminal, directives = step_advance(
            sess,
            response,
            current_turn_type=TurnType.PROPOSE_CHAIN,
        )
        assert new_sess is sess  # pure: no state change
        assert next_turn is None
        assert terminal is None
        assert directives == []

    def test_step_3_unexpected_turn_type_raises(self) -> None:
        """Any turn type outside the Step 3 closed set is an InvariantError.

        Step 3 only ever emits PROPOSE_CHAIN, SINGLE_SELECT, and SCHEMA_FORM
        turns from the server. A different turn type in the history record means
        the emitter stamped an invalid type — that is a server-side bug
        (InvariantError), not a client protocol violation (ValueError).
        """
        sess = GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS)
        response = _make_response()
        with pytest.raises(InvariantError, match="_advance_step_3"):
            step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)

    def test_step_3_schema_form_is_intra_step_for_edit_flow(self) -> None:
        sess = GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS)
        response = _make_response(edited_values={"plugin": "rename", "options": {}})

        new_sess, next_turn, terminal, directives = step_advance(
            sess,
            response,
            current_turn_type=TurnType.SCHEMA_FORM,
        )

        assert new_sess is sess
        assert next_turn is None
        assert terminal is None
        assert directives == []

    def test_step_advance_unhandled_step_raises_invariant_error_not_assertion(self) -> None:
        """Dispatcher fall-through is gated by InvariantError, not AssertionError.

        Regression test for PR-review finding B2: AssertionError is stripped by
        ``python -O`` and would silently fall through under optimized
        deployment.  ``InvariantError`` subclasses ``Exception`` directly and
        survives ``-O`` — see ``elspeth.web.composer.guided.errors``.

        The closed enum ``GuidedStep`` makes this branch unreachable from valid
        state, so we inject a sentinel via ``object.__setattr__`` on the frozen
        dataclass to drive the dispatcher into the fall-through.
        """
        sess = GuidedSession.initial()
        # Inject a step value the dispatcher cannot match.  ``object.__setattr__``
        # bypasses ``frozen=True``; that's intentional here — the production code
        # never mutates step in place, only via dataclasses.replace.
        object.__setattr__(sess, "step", "not-a-real-step")
        response = _make_response()
        with pytest.raises(InvariantError, match="unhandled step"):
            step_advance(sess, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)
        # And explicitly: it is NOT an AssertionError subclass.  This pins the
        # python -O contract: AssertionError raises are elided; InvariantError
        # raises are not.
        try:
            object.__setattr__(sess, "step", "also-not-real")
            step_advance(sess, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)
        except InvariantError as exc:
            assert not isinstance(exc, AssertionError), (
                "InvariantError must NOT inherit from AssertionError; "
                "python -O strips AssertionError raises and would silently "
                "skip the dispatcher gate."
            )


class TestAdvanceStep4Wire:
    def test_confirm_wiring_is_a_self_loop(self) -> None:
        session = dataclasses.replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
        new_session, turn, terminal, directives = step_advance(
            session,
            _wire_response(),
            current_turn_type=TurnType.CONFIRM_WIRING,
        )
        assert new_session.step is GuidedStep.STEP_4_WIRE
        assert terminal is None
        assert turn is None
        assert directives == []

    def test_wire_stage_rejects_illegal_turn_type(self) -> None:
        session = dataclasses.replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
        with pytest.raises(InvariantError, match="STEP_4_WIRE"):
            step_advance(session, _wire_response(), current_turn_type=TurnType.PROPOSE_CHAIN)

    def test_wire_stage_exit_to_freeform_still_terminates(self) -> None:
        session = dataclasses.replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
        _new, _turn, terminal, _directives = step_advance(
            session,
            _wire_response(control=ControlSignal.EXIT_TO_FREEFORM),
            current_turn_type=TurnType.CONFIRM_WIRING,
        )
        assert terminal is not None


# ---------------------------------------------------------------------------
# Task 2.5 tests: standalone terminal-failure helpers (spec §5.4)
# ---------------------------------------------------------------------------


class TestTerminalHelpers:
    def test_mark_solver_exhausted_sets_terminal_and_emits_directive(self) -> None:
        sess = GuidedSession.initial()
        new_sess, terminal, directives = mark_solver_exhausted(
            sess,
            validation_result={"errors": ["..."]},
        )
        assert new_sess.terminal is terminal
        assert terminal.kind is TerminalKind.EXITED_TO_FREEFORM
        assert terminal.reason is TerminalReason.SOLVER_EXHAUSTED
        assert terminal.pipeline_yaml is None
        # freeze_fields deep-freezes the arguments Mapping: list → tuple.
        # The stored validation_result is MappingProxyType({'errors': ('...',)}).
        assert any(
            d.tool_name == "guided_dropped_to_freeform"
            and d.arguments["drop_reason"] == "solver_exhausted"
            and d.arguments["validation_result"] == {"errors": ("...",)}
            for d in directives
        )

    def test_mark_solver_exhausted_with_none_validation(self) -> None:
        sess = GuidedSession.initial()
        _, _, directives = mark_solver_exhausted(sess, validation_result=None)
        assert directives[0].arguments["validation_result"] is None

    def test_mark_protocol_violation_sets_terminal_and_emits_directive(self) -> None:
        sess = GuidedSession.initial()
        _new_sess, terminal, directives = mark_protocol_violation(sess)
        assert terminal.kind is TerminalKind.EXITED_TO_FREEFORM
        assert terminal.reason is TerminalReason.PROTOCOL_VIOLATION
        assert any(d.tool_name == "guided_dropped_to_freeform" and d.arguments["drop_reason"] == "protocol_violation" for d in directives)


class TestStateMachineInvariants:
    """Hypothesis property tests for invariants that must hold across all step states."""

    @given(st.sampled_from(list(GuidedStep)))
    def test_exit_to_freeform_always_terminates(self, starting_step: GuidedStep) -> None:
        """control_signal='exit_to_freeform' terminates from ANY step, regardless of
        intra-step state. The terminal kind is EXITED_TO_FREEFORM and the reason is
        USER_PRESSED_EXIT — spec §5.3."""
        sess = GuidedSession(step=starting_step)
        response: TurnResponse = {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": ControlSignal.EXIT_TO_FREEFORM,
        }
        new_sess, _next, terminal, _directives = step_advance(
            sess,
            response,
            current_turn_type=TurnType.SINGLE_SELECT,
        )
        assert terminal is not None
        assert new_sess.terminal is not None
        assert new_sess.terminal.kind is TerminalKind.EXITED_TO_FREEFORM
        assert new_sess.terminal.reason is TerminalReason.USER_PRESSED_EXIT
