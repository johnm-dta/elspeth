# tests/unit/web/composer/guided/test_state_machine.py
"""Tests for GuidedSession, TerminalState, TurnRecord — state machine data."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, TurnResponse, TurnType
from elspeth.web.composer.guided.state_machine import (
    GUIDED_SESSION_SCHEMA_VERSION,
    ChainProposal,
    GuidedSession,
    SinkIntent,
    SinkOutputResolved,
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

    def test_guided_session_roundtrip_with_recipe_offer(self) -> None:
        """GuidedSession with step_2_5_recipe_offer survives to_dict/from_dict round-trip.

        Exercises the Tier-1 serialisation boundary: the staged offer is persisted
        to composer_meta["guided_session"] when the recipe_offer turn is emitted and
        must survive the round-trip so the POST /respond accept branch can verify
        the recipe_name.
        """
        from dataclasses import replace

        from elspeth.web.composer.guided.recipe_match import RecipeMatch
        from elspeth.web.composer.recipes import SlotSpec

        offer = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={"source_blob_id": "blob-abc", "output_path": "out.jsonl", "label_field": "category"},
            unsatisfied_slots={
                "classifier_template": SlotSpec(slot_type="str", description="Jinja2 template", required=True),
                "model": SlotSpec(slot_type="str", description="LLM model name", required=True),
                "api_key_secret": SlotSpec(slot_type="str", description="Secret name", required=True),
            },
        )
        sess = replace(GuidedSession.initial(), step_2_5_recipe_offer=offer)
        d = sess.to_dict()
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_2_5_recipe_offer is not None
        assert restored.step_2_5_recipe_offer.recipe_name == "classify-rows-llm-jsonl"
        assert restored.step_2_5_recipe_offer.slots["source_blob_id"] == "blob-abc"
        assert "classifier_template" in restored.step_2_5_recipe_offer.unsatisfied_slots

    def test_guided_session_roundtrip_with_recipe_offer_none(self) -> None:
        """GuidedSession with step_2_5_recipe_offer=None round-trips cleanly.

        Ensures the None case is serialised as None (not absent), which would
        crash from_dict's strict key read.
        """
        sess = GuidedSession.initial()
        d = sess.to_dict()
        restored = GuidedSession.from_dict(d)
        assert restored == sess
        assert restored.step_2_5_recipe_offer is None

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

    def test_guided_session_schema_version_bumped_for_inspection_facts(self) -> None:
        assert GUIDED_SESSION_SCHEMA_VERSION == 6

    def test_guided_session_to_dict_includes_schema_version(self) -> None:
        sess = GuidedSession.initial()
        assert sess.to_dict()["schema_version"] == 6

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


# ---------------------------------------------------------------------------
# Task 2.1 tests: step_advance dispatcher + Step 1 → Step 2 branch
# ---------------------------------------------------------------------------


class TestStepAdvance:
    def test_initial_session_advances_after_source_confirmed(self) -> None:
        """INSPECT_AND_CONFIRM with step_1_source_intent set advances to STEP_2_SINK.

        The new wire contract is narrow: edited_values = {"columns": list}.
        Plugin, options, and sample_rows are recovered from step_1_source_intent.
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
        assert new_sess.step is GuidedStep.STEP_2_SINK
        assert new_sess.step_1_result is not None
        assert new_sess.step_1_result.plugin == "csv"
        assert new_sess.step_1_result.observed_columns == ("id", "name", "score")
        assert new_sess.step_1_source_intent is None  # consumed; cleared atomically
        assert terminal is None
        assert next_turn is None
        assert any(d.tool_name == "guided_step_advanced" for d in directives)

    def test_advance_step_1_raises_when_intent_missing(self) -> None:
        """INSPECT_AND_CONFIRM without step_1_source_intent raises InvariantError.

        The intent must be set by the emit site before the turn is sent.
        Arriving at INSPECT_AND_CONFIRM with intent=None is a state-machine
        invariant violation (server bug, not client fault).
        """
        session = GuidedSession.initial()
        # Confirm intent is None (initial state — no emit site set it).
        assert session.step_1_source_intent is None
        response = _make_response(edited_values={"columns": ["id", "name"]})
        with pytest.raises(InvariantError, match="step_1_source_intent is None"):
            step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)

    def test_advance_step_1_columns_are_coerced_to_str(self) -> None:
        """Numeric column values in edited_values["columns"] are coerced to str.

        Tier-3 coercion at the HTTP boundary: ``str(c)`` applied to each element.
        """
        intent = SourceIntent(
            plugin="csv",
            options={"path": "/data/input.csv"},
            observed_columns=("col_a",),
            sample_rows=(),
        )
        from dataclasses import replace

        session = replace(GuidedSession.initial(), step_1_source_intent=intent)
        response = _make_response(edited_values={"columns": [42, "name", True]})
        new_sess, _, _, _ = step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)
        assert new_sess.step_1_result is not None
        assert new_sess.step_1_result.observed_columns == ("42", "name", "True")

    def test_inspect_and_confirm_without_edited_values_raises(self) -> None:
        """edited_values=None on an INSPECT_AND_CONFIRM response must raise ValueError.

        step_1_source_intent must be set so the intent-guard passes; the
        edited_values-None guard is the second check.
        """
        intent = SourceIntent(
            plugin="csv",
            options={"path": "/data/input.csv"},
            observed_columns=("id",),
            sample_rows=(),
        )
        from dataclasses import replace

        session = replace(GuidedSession.initial(), step_1_source_intent=intent)
        response = _make_response(edited_values=None)
        with pytest.raises(ValueError, match="inspect_and_confirm"):
            step_advance(session, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM)

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

    def test_step_2_advances_after_required_fields_declared(self) -> None:
        """_advance_step_2 reads chosen + custom_inputs from response and plugin + options
        from GuidedSession.step_2_sink_intent to construct SinkOutputResolved."""
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

        new_sess, _next, terminal, _events = step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)

        assert new_sess.step is GuidedStep.STEP_2_5_RECIPE_MATCH
        assert new_sess.step_2_result is not None
        assert len(new_sess.step_2_result.outputs) == 1
        output = new_sess.step_2_result.outputs[0]
        assert output.plugin == "json"
        assert output.required_fields == ("a",)
        assert output.schema_mode == "observed"
        # step_2_sink_intent must be cleared after consumption
        assert new_sess.step_2_sink_intent is None
        assert terminal is None

    def test_step_2_let_source_decide_sets_observed_mode(self) -> None:
        """schema_mode is always 'observed' — the backend sets it, not the frontend."""
        intent = SinkIntent(plugin="json", options={"path": "out.jsonl"})
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
            step_2_sink_intent=intent,
        )
        response: TurnResponse = {
            "chosen": [],
            "edited_values": None,
            "custom_inputs": [],
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, _next, _terminal, _events = step_advance(sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM)
        assert new_sess.step_2_result is not None
        assert new_sess.step_2_result.outputs[0].schema_mode == "observed"

    def test_step_2_without_sink_intent_raises(self) -> None:
        """MULTI_SELECT_WITH_CUSTOM with step_2_sink_intent=None is a state-machine
        invariant violation — raises InvariantError immediately (server bug)."""
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
            step_2_sink_intent=None,  # missing — SCHEMA_FORM dispatcher didn't run
        )
        response: TurnResponse = {
            "chosen": ["a"],
            "edited_values": None,
            "custom_inputs": [],
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        with pytest.raises(InvariantError, match="step_2_sink_intent is None"):
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
