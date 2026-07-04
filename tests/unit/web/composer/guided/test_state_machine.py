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
    ChainProposal,
    GuidedSession,
    SinkIntent,
    SinkResolved,
    SourceIntent,
    SourceResolved,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
    mark_protocol_violation,
    mark_solver_exhausted,
    step_advance,
)
from elspeth.web.composer.source_inspection import SourceInspectionFacts, facts_from_dict


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
            summary="Selected source: csv",
            emitter="server",
        )
        assert rec.emitter == "server"
        assert rec.summary == "Selected source: csv"

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

    def test_turn_record_summary_roundtrip(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc",
            response_hash="def",
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

    def test_assistant_message_kind_and_reason_default_to_none(self) -> None:
        """Legacy-compatible construction: omitting the new fields is valid."""
        turn = ChatTurn(
            role=ChatRole.ASSISTANT,
            content="Here is some advice.",
            seq=1,
            step=GuidedStep.STEP_1_SOURCE,
            ts_iso="2026-05-13T12:00:00+00:00",
        )
        assert turn.assistant_message_kind is None
        assert turn.synthetic_failure_reason is None

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
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
        )
        assert s.terminal is not None
        assert s.terminal.kind is TerminalKind.COMPLETED

    def test_guided_session_roundtrip_with_sink_intent(self) -> None:
        """GuidedSession with step_2_sink_intent survives to_dict/from_dict round-trip.

        Exercises the Tier-1 serialisation boundary: the session is persisted to
        composer_meta["guided_session"] between turns, so the new staging field must
        survive the round-trip without loss or corruption.
        """
        intent = SinkIntent(
            plugin="json",
            options={"path": "/data/out.jsonl", "schema": {"mode": "observed"}},
        )
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={"path": "/data/in.csv"},
                observed_columns=("col_a", "col_b"),
                sample_rows=({"col_a": "x", "col_b": "y"},),
            ),
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
            step_2_sink_intent=intent,
        )
        d = sess.to_dict()
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_2_sink_intent is not None
        assert restored.step_2_sink_intent.plugin == "json"
        assert restored.step_2_sink_intent.options["path"] == "/data/out.jsonl"

    def test_guided_session_roundtrip_with_sink_intent_none(self) -> None:
        """GuidedSession with step_2_sink_intent=None round-trips cleanly.

        Ensures the None case is serialised as None and reconstructed as None
        (not absent or missing), which would crash from_dict's strict key read.
        """
        sess = GuidedSession.initial()
        d = sess.to_dict()
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_2_sink_intent is None

    def test_guided_session_roundtrip_with_source_intent(self) -> None:
        """GuidedSession with step_1_source_intent survives to_dict/from_dict round-trip.

        Exercises the Tier-1 serialisation boundary: the session is persisted to
        composer_meta["guided_session"] between turns, so the new staging field must
        survive the round-trip without loss or corruption.
        """
        intent = SourceIntent(
            plugin="csv",
            options={"path": "/data/in.csv", "schema": {"mode": "observed"}},
            observed_columns=("col_a", "col_b"),
            sample_rows=({"col_a": "x", "col_b": "y"},),
        )
        from dataclasses import replace

        sess = replace(GuidedSession.initial(), step_1_source_intent=intent)
        d = sess.to_dict()
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_1_source_intent is not None
        assert restored.step_1_source_intent.plugin == "csv"
        assert restored.step_1_source_intent.observed_columns == ("col_a", "col_b")
        assert restored.step_1_source_intent.sample_rows == ({"col_a": "x", "col_b": "y"},)

    def test_guided_session_roundtrip_with_source_intent_none(self) -> None:
        """GuidedSession with step_1_source_intent=None round-trips cleanly.

        Ensures the None case is serialised as None and reconstructed as None
        (not absent or missing), which would crash from_dict's strict key read.
        """
        sess = GuidedSession.initial()
        d = sess.to_dict()
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_1_source_intent is None

    def test_guided_session_roundtrip_with_step_2_chosen_plugin(self) -> None:
        """GuidedSession with step_2_chosen_plugin survives to_dict/from_dict round-trip.

        Exercises the Tier-1 serialisation boundary for the new staging field.
        The field is set in the SINGLE_SELECT→SCHEMA_FORM window at STEP_2_SINK;
        it must survive the persist/restore cycle so GET /guided can rebuild
        the correct SCHEMA_FORM on refresh.
        """
        from dataclasses import replace

        sess = replace(GuidedSession.initial(), step_2_chosen_plugin="json")
        d = sess.to_dict()
        assert d["step_2_chosen_plugin"] == "json"  # serialised
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_2_chosen_plugin == "json"

    def test_guided_session_roundtrip_with_step_2_chosen_plugin_none(self) -> None:
        """GuidedSession with step_2_chosen_plugin=None round-trips cleanly.

        Ensures the None case is serialised as None (not absent), which would
        crash from_dict's strict key read.
        """
        sess = GuidedSession.initial()
        d = sess.to_dict()
        assert d["step_2_chosen_plugin"] is None  # serialised as null
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_2_chosen_plugin is None

    def test_guided_session_roundtrip_with_step_1_chosen_plugin(self) -> None:
        from dataclasses import replace

        sess = replace(GuidedSession.initial(), step_1_chosen_plugin="csv")
        d = sess.to_dict()
        assert d["step_1_chosen_plugin"] == "csv"
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_1_chosen_plugin == "csv"

    def test_guided_session_roundtrip_with_step_1_chosen_plugin_none(self) -> None:
        sess = GuidedSession.initial()
        d = sess.to_dict()
        assert d["step_1_chosen_plugin"] is None
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_1_chosen_plugin is None

    def test_guided_session_rejects_mutable_step_1_chosen_plugin(self) -> None:
        with pytest.raises(TypeError, match="step_1_chosen_plugin must be str or None"):
            dataclasses.replace(GuidedSession.initial(), step_1_chosen_plugin=[])

    def test_guided_session_round_trips_inspection_facts(self) -> None:
        facts = SourceInspectionFacts(
            source_kind="csv",
            redacted_identity={"filename": "input.csv"},
            byte_range_inspected=(0, 128),
            observed_headers=("name", "age"),
            inferred_types={"name": "str", "age": "int"},
            url_candidates=(),
            sample_row_count=10,
            warnings=(),
        )
        sess = dataclasses.replace(GuidedSession.initial(), step_1_inspection_facts=facts)
        d = sess.to_dict()

        restored = GuidedSession.from_dict(d)

        assert restored.step_1_inspection_facts == facts

    def test_guided_session_inspection_facts_default_none(self) -> None:
        sess = GuidedSession.initial()
        assert sess.step_1_inspection_facts is None

    def test_guided_session_rejects_mutable_inspection_facts(self) -> None:
        with pytest.raises(TypeError, match="step_1_inspection_facts must be SourceInspectionFacts or None"):
            dataclasses.replace(GuidedSession.initial(), step_1_inspection_facts={})

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

    def test_guided_session_schema_version_is_7(self) -> None:
        assert GUIDED_SESSION_SCHEMA_VERSION == 7

    def test_guided_session_to_dict_includes_schema_version(self) -> None:
        sess = GuidedSession.initial()
        assert sess.to_dict()["schema_version"] == 7

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

        with pytest.raises(InvariantError, match=r"TurnRecord\.from_dict"):
            GuidedSession.from_dict(current)

    def test_guided_session_v3_requires_step_3_edit_index(self) -> None:
        current = GuidedSession.initial().to_dict()
        del current["step_3_edit_index"]

        with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
            GuidedSession.from_dict(current)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("schema_version", "7", "schema_version"),
            ("schema_version", 7.0, "schema_version"),
            ("schema_version", True, "schema_version"),
            ("step_1_chosen_plugin", 123, "step_1_chosen_plugin"),
            ("step_2_chosen_plugin", ["json"], "step_2_chosen_plugin"),
            ("transition_consumed", "false", "transition_consumed"),
            ("transition_consumed", 1, "transition_consumed"),
            ("step_3_edit_index", "0", "step_3_edit_index"),
            ("step_3_edit_index", True, "step_3_edit_index"),
            ("step_3_edit_index", -1, "step_3_edit_index"),
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
            field: bad_value,
        }
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

    def test_guided_session_from_dict_chat_history_legacy_entry_without_kind_fields(self) -> None:
        """A turn persisted BEFORE this field existed has no key at all.

        Requirement: absence stays absence on the wire — None, never a
        fabricated default — and from_dict must not crash on the missing keys
        (unlike role/content/seq/step/ts_iso, these two are genuinely optional).
        """
        d = GuidedSession.initial().to_dict()
        legacy_entry = {
            "role": ChatRole.ASSISTANT.value,
            "content": "a pre-migration reply",
            "seq": 0,
            "step": GuidedStep.STEP_1_SOURCE.value,
            "ts_iso": "2026-05-13T12:00:00+00:00",
            # No "assistant_message_kind" / "synthetic_failure_reason" keys.
        }
        d["chat_history"] = [legacy_entry]

        restored = GuidedSession.from_dict(d)
        assert len(restored.chat_history) == 1
        assert restored.chat_history[0].assistant_message_kind is None
        assert restored.chat_history[0].synthetic_failure_reason is None


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
        """_advance_step_1 is a pure self-loop for INSPECT_AND_CONFIRM — it never
        advances step or sets step_1_result.

        elspeth-948eb9c0b8 C-3(b) mirror fix (same shape as the Step 2 fix):
        resolving step_1_source_intent + edited_values["columns"] into a
        SourceResolved, and the handle_step_1_source commit, both moved to the
        dispatcher (_dispatch_guided_respond's STEP_1_SOURCE intra-step
        INSPECT_AND_CONFIRM branch in sessions/routes/_helpers.py) so that
        guided.step / step_1_result are only ever set once the commit is known
        to have succeeded — mirroring _advance_step_2/_advance_step_3/_advance_step_4.
        Coverage for the resolve/validate/commit behaviour itself lives in
        tests/integration/web/composer/guided/test_respond.py
        (TestStep1InspectAndConfirmAccept).
        """
        intent = SourceIntent(
            plugin="csv",
            options={"path": "/data/input.csv"},
            observed_columns=("id", "name", "score"),
            sample_rows=({"id": 1, "name": "Alice", "score": 99},),
        )
        from dataclasses import replace

        session = replace(GuidedSession.initial(), step_1_source_intent=intent)
        response = _make_response(
            edited_values={
                "columns": ["id", "name", "score"],
            },
        )
        new_sess, next_turn, terminal, directives = step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)
        assert new_sess is session  # pure: no state change
        assert new_sess.step is GuidedStep.STEP_1_SOURCE
        assert new_sess.step_1_result is None
        assert new_sess.step_1_source_intent is intent  # untouched — not consumed here
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
        """_advance_step_2 is a pure self-loop for MULTI_SELECT_WITH_CUSTOM — it
        never advances step or sets step_2_result.

        elspeth-948eb9c0b8 C-3(b): resolving chosen + custom_inputs +
        step_2_sink_intent into a SinkOutputResolved, the fail-closed
        passthrough validation, and the handle_step_2_sink commit all moved to
        the dispatcher (_dispatch_guided_respond's STEP_2_SINK intra-step
        MULTI_SELECT_WITH_CUSTOM branch in sessions/routes/_helpers.py) so
        that guided.step / step_2_result are only ever set once the commit is
        known to have succeeded — mirroring _advance_step_3 / _advance_step_4.
        Coverage for the resolve/validate/commit behaviour itself lives in
        tests/integration/web/composer/guided/test_respond.py.
        """
        intent = SinkIntent(plugin="json", options={"path": "out.jsonl"})
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
            step_2_sink_intent=intent,
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
        assert new_sess.step_2_result is None
        assert new_sess.step_2_sink_intent is intent  # untouched — not consumed here
        assert next_turn is None
        assert terminal is None
        assert directives == []

    def test_step_2_single_select_and_schema_form_are_also_intra_step(self) -> None:
        """SINGLE_SELECT and SCHEMA_FORM at STEP_2_SINK are likewise pure self-loops."""
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

    def test_step_3_accept_chain_marks_session_with_proposal(self) -> None:
        """PROPOSE_CHAIN at STEP_3: step_advance is a no-op; handler interprets.

        The session must pass through unchanged (step_advance is pure; the
        endpoint handler runs preview_pipeline and commits via tools.py).
        The step_3_proposal must remain on the session if it was already set.
        """
        proposal = ChainProposal(
            steps=({"plugin": "rename", "options": {}, "rationale": "normalise names"},),
            why="The source columns need renaming before sink.",
        )
        sess = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a",),
                sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=proposal,
            terminal=None,
        )
        response = _make_response()
        new_sess, next_turn, terminal, directives = step_advance(
            sess,
            response,
            current_turn_type=TurnType.PROPOSE_CHAIN,
        )
        assert new_sess is sess  # pure: no state change
        assert new_sess.step_3_proposal is proposal
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
        sess = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
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
        response = _make_response()
        with pytest.raises(InvariantError, match="_advance_step_3"):
            step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)

    def test_step_3_schema_form_is_intra_step_for_edit_flow(self) -> None:
        proposal = ChainProposal(
            steps=({"plugin": "rename", "options": {}, "rationale": "normalise names"},),
            why="The source columns need renaming before sink.",
        )
        sess = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a",),
                sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=proposal,
            terminal=None,
            step_3_edit_index=0,
        )
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
        sess = GuidedSession(
            step=starting_step,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
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
