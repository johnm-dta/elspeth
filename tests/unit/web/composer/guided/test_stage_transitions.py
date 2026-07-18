"""Pure schema-8 RESPOND stage transitions.

The route owns I/O, policy/catalog lookup, audit emission, and CAS persistence.
These tests pin the smaller pure layer that moves one persisted guided intent
through its legal phases without constructing executable topology.
"""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID

import pytest

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.stage_transitions import (
    GUIDED_TURN_TOKEN_SCHEMA,
    AnsweredTurn,
    FieldSelectionResponse,
    InspectionResponse,
    PluginSelectionResponse,
    SchemaFormAuthority,
    SchemaFormResponse,
    SinkFieldReviewPreparation,
    SinkPluginSelectionPreparation,
    SinkSchemaFormPreparation,
    SourceInspectionPreparation,
    SourceSchemaFormPreparation,
    guided_turn_token,
    transition_sink_field_review,
    transition_sink_plugin_selection,
    transition_sink_schema_form,
    transition_source_inspection_review,
    transition_source_plugin_selection,
    transition_source_schema_form,
)
from elspeth.web.composer.guided.state_machine import GuidedSession, SinkIntent, SourceIntent, TurnRecord
from elspeth.web.composer.source_inspection import SourceInspectionFacts, facts_to_dict

SOURCE_A = "11111111-1111-4111-8111-111111111111"
SOURCE_B = "22222222-2222-4222-8222-222222222222"
OUTPUT_A = "33333333-3333-4333-8333-333333333333"
OUTPUT_B = "44444444-4444-4444-8444-444444444444"
PAYLOAD_A = "a" * 64
PAYLOAD_B = "b" * 64


def _inspection(*, headers: tuple[str, ...] = ("id", "name")) -> SourceInspectionFacts:
    return SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={
            "filename": "input.csv",
            "mime_type": "text/csv",
            "blob_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        },
        byte_range_inspected=(0, 32),
        sample_row_count=2,
        observed_headers=headers,
        inferred_types=dict.fromkeys(headers, "str"),
        url_candidates=(),
        warnings=("sample warning",),
    )


def _source(name: str, columns: tuple[str, ...]) -> SourceResolved:
    return SourceResolved(
        name=name,
        plugin="csv",
        options={"path": f"/data/{name}.csv"},
        observed_columns=columns,
        sample_rows=(),
        on_validation_failure="discard",
    )


def _output(name: str) -> SinkOutputResolved:
    return SinkOutputResolved(
        name=name,
        plugin="json",
        options={"path": f"/data/{name}.jsonl"},
        required_fields=("id",),
        schema_mode="observed",
        on_write_failure="discard",
    )


def _with_unanswered_turn(
    session: GuidedSession,
    turn_type: TurnType,
    *,
    payload_hash: str = PAYLOAD_A,
) -> tuple[GuidedSession, AnsweredTurn]:
    record = TurnRecord(
        step=session.step,
        turn_type=turn_type,
        payload_hash=payload_hash,
        response_hash=None,
        emitter="server",
    )
    session = replace(session, history=(*session.history, record))
    history_index = len(session.history) - 1
    return session, AnsweredTurn(history_index=history_index)


def _source_options_session(*, facts: SourceInspectionFacts | None) -> tuple[GuidedSession, AnsweredTurn]:
    session = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_A,),
        pending_source_intents={
            SOURCE_A: SourceIntent(
                name="source",
                phase="plugin_options",
                plugin="csv",
                options=None,
                inspection_facts=facts,
                observed_columns=(),
                sample_rows=(),
            )
        },
    )
    return _with_unanswered_turn(session, TurnType.SCHEMA_FORM)


def _source_review_session() -> tuple[GuidedSession, AnsweredTurn]:
    facts = _inspection()
    session = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_A,),
        pending_source_intents={
            SOURCE_A: SourceIntent(
                name="source",
                phase="inspection_review",
                plugin="csv",
                options={
                    "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "on_validation_failure": "discard",
                },
                inspection_facts=facts,
                observed_columns=facts.observed_headers or (),
                sample_rows=(),
            )
        },
    )
    return _with_unanswered_turn(session, TurnType.INSPECT_AND_CONFIRM)


def _sink_options_session() -> tuple[GuidedSession, AnsweredTurn]:
    session = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        source_order=(SOURCE_A, SOURCE_B),
        reviewed_sources={
            SOURCE_A: _source("source", ("id", "name")),
            SOURCE_B: _source("source_2", ("name", "email")),
        },
        output_order=(OUTPUT_A,),
        pending_output_intents={OUTPUT_A: SinkIntent(name="output", phase="plugin_options", plugin="json", options=None)},
    )
    return _with_unanswered_turn(session, TurnType.SCHEMA_FORM)


def _sink_review_session() -> tuple[GuidedSession, AnsweredTurn]:
    session = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        source_order=(SOURCE_A, SOURCE_B),
        reviewed_sources={
            SOURCE_A: _source("source", ("id", "name")),
            SOURCE_B: _source("source_2", ("name", "email")),
        },
        output_order=(OUTPUT_A,),
        pending_output_intents={
            OUTPUT_A: SinkIntent(
                name="output",
                phase="field_review",
                plugin="json",
                options={"path": "/data/out.jsonl", "on_write_failure": "discard"},
            )
        },
    )
    return _with_unanswered_turn(session, TurnType.MULTI_SELECT_WITH_CUSTOM)


SOURCE_KNOBS = {
    "fields": [
        {"name": "mode", "kind": "enum", "enum": ["csv", "json"], "required": True, "nullable": False},
        {"name": "path", "kind": "text", "required": False, "nullable": False},
        {"name": "blob_id", "kind": "blob-ref", "required": False, "nullable": True},
        {
            "name": "delimiter",
            "kind": "text",
            "required": False,
            "nullable": False,
            "visible_when": {"field": "mode", "equals": "csv"},
        },
    ]
}
SINK_KNOBS = {
    "fields": [
        {"name": "path", "kind": "text", "required": False, "nullable": False},
        {"name": "indent", "kind": "number-int", "required": False, "nullable": False},
        {"name": "schema", "kind": "json-object", "required": False, "nullable": False},
    ]
}


def test_source_intent_plugin_options_can_retain_exact_inspection_facts() -> None:
    facts = _inspection()

    intent = SourceIntent(
        name="source",
        phase="plugin_options",
        plugin="csv",
        options=None,
        inspection_facts=facts,
        observed_columns=(),
        sample_rows=(),
    )

    restored = SourceIntent.from_dict(intent.to_dict())
    assert restored.to_dict() == intent.to_dict()
    assert restored.inspection_facts is not facts
    with pytest.raises(ValueError, match="plugin_options"):
        replace(intent, options={"path": "/client"})
    with pytest.raises(ValueError, match="plugin_options"):
        replace(intent, observed_columns=("client",))


def test_source_selection_allocates_stable_id_once_and_retains_server_facts() -> None:
    session, turn = _with_unanswered_turn(GuidedSession.initial(), TurnType.SINGLE_SELECT)
    facts = _inspection()

    result = transition_source_plugin_selection(
        session,
        turn=turn,
        response=PluginSelectionResponse(chosen=("csv",)),
        permitted_plugins=("csv", "json"),
        inspection_facts=facts,
        new_stable_id=UUID(SOURCE_A),
    )

    assert session.source_order == ()
    assert result.session.source_order == (SOURCE_A,)
    intent = result.session.pending_source_intents[SOURCE_A]
    assert (intent.name, intent.phase, intent.plugin) == ("source", "plugin_options", "csv")
    assert intent.inspection_facts is not facts
    assert facts_to_dict(intent.inspection_facts) == facts_to_dict(facts)  # type: ignore[arg-type]
    assert result.next_turn == SourceSchemaFormPreparation(stable_id=SOURCE_A, plugin="csv", inspection_facts=facts)


def test_source_selection_reuses_target_and_preserves_multi_source_order() -> None:
    pending = SourceIntent(
        name="source_2",
        phase="plugin_selection",
        plugin=None,
        options=None,
        inspection_facts=None,
        observed_columns=(),
        sample_rows=(),
    )
    session = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_A, SOURCE_B),
        reviewed_sources={SOURCE_A: _source("source", ("id",))},
        pending_source_intents={SOURCE_B: pending},
    )
    session, turn = _with_unanswered_turn(session, TurnType.SINGLE_SELECT)

    result = transition_source_plugin_selection(
        session,
        turn=turn,
        response=PluginSelectionResponse(chosen=("json",)),
        permitted_plugins=("csv", "json"),
        inspection_facts=None,
        target_id=SOURCE_B,
    )

    assert result.session.source_order == (SOURCE_A, SOURCE_B)
    assert result.session.pending_source_intents[SOURCE_B].name == "source_2"
    assert result.session.reviewed_sources[SOURCE_A] == session.reviewed_sources[SOURCE_A]


@pytest.mark.parametrize("chosen", [(), ("csv", "json"), ("blocked",)])
def test_source_selection_rejects_non_exact_or_unpermitted_choice(chosen: tuple[str, ...]) -> None:
    session, turn = _with_unanswered_turn(GuidedSession.initial(), TurnType.SINGLE_SELECT)
    with pytest.raises(ValueError):
        transition_source_plugin_selection(
            session,
            turn=turn,
            response=PluginSelectionResponse(chosen=chosen),
            permitted_plugins=("csv", "json"),
            inspection_facts=None,
            new_stable_id=UUID(SOURCE_A),
        )


def test_source_schema_form_uses_held_plugin_and_custody_options_then_enters_inspection() -> None:
    facts = _inspection()
    session, turn = _source_options_session(facts=facts)
    authority = SchemaFormAuthority(
        knobs=SOURCE_KNOBS,
        model_validated_options={
            "mode": "csv",
            "delimiter": ",",
            "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        },
        server_options={"path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
    )

    result = transition_source_schema_form(
        session,
        target_id=SOURCE_A,
        turn=turn,
        response=SchemaFormResponse(
            plugin="csv",
            options={"mode": "csv", "delimiter": ",", "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        ),
        authority=authority,
    )

    assert session.pending_source_intents[SOURCE_A].phase == "plugin_options"
    intent = result.session.pending_source_intents[SOURCE_A]
    assert intent.phase == "inspection_review"
    assert intent.plugin == "csv"
    assert dict(intent.options or {}) == {
        "mode": "csv",
        "delimiter": ",",
        "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "on_validation_failure": "discard",
    }
    assert intent.observed_columns == ("id", "name")
    assert intent.sample_rows == ()
    assert facts_to_dict(intent.inspection_facts) == facts_to_dict(facts)  # type: ignore[arg-type]
    assert result.next_turn == SourceInspectionPreparation(stable_id=SOURCE_A, inspection_facts=facts)


def test_inspection_facts_survive_restart_between_selection_form_and_review() -> None:
    facts = _inspection()
    session, turn = _source_options_session(facts=facts)
    restored = GuidedSession.from_dict(session.to_dict())

    result = transition_source_schema_form(
        restored,
        target_id=SOURCE_A,
        turn=turn,
        response=SchemaFormResponse(
            plugin="csv",
            options={"mode": "csv", "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        ),
        authority=SchemaFormAuthority(
            knobs=SOURCE_KNOBS,
            model_validated_options={"mode": "csv", "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
            server_options={"path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        ),
    )
    restored_review = GuidedSession.from_dict(result.session.to_dict())

    review_intent = restored_review.pending_source_intents[SOURCE_A]
    assert review_intent.phase == "inspection_review"
    assert facts_to_dict(review_intent.inspection_facts) == facts_to_dict(facts)  # type: ignore[arg-type]
    assert review_intent.observed_columns == facts.observed_headers


def test_source_schema_form_without_inspection_reviews_same_id_and_advances() -> None:
    session, turn = _source_options_session(facts=None)

    result = transition_source_schema_form(
        session,
        target_id=SOURCE_A,
        turn=turn,
        response=SchemaFormResponse(plugin="csv", options={"mode": "csv", "path": "/data/input.csv"}),
        authority=SchemaFormAuthority(
            knobs=SOURCE_KNOBS,
            model_validated_options={"mode": "csv", "path": "/data/input.csv"},
        ),
    )

    assert result.session.step is GuidedStep.STEP_2_SINK
    assert result.session.source_order == (SOURCE_A,)
    assert not result.session.pending_source_intents
    source = result.session.reviewed_sources[SOURCE_A]
    assert source.name == "source"
    assert source.plugin == "csv"
    assert source.observed_columns == ()
    assert result.next_turn == SinkPluginSelectionPreparation()


@pytest.mark.parametrize(
    ("response", "authority"),
    [
        (
            SchemaFormResponse(plugin="json", options={"mode": "csv"}),
            SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        ),
        (
            SchemaFormResponse(plugin="csv", options={"secret": "x"}),
            SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        ),
        (
            SchemaFormResponse(plugin="csv", options={"mode": "json", "delimiter": ","}),
            SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        ),
        (
            SchemaFormResponse(plugin="csv", options={"blob_id": "not-a-uuid"}),
            SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        ),
        (
            SchemaFormResponse(plugin="csv", options={"path": "bad\x00path"}),
            SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        ),
        (
            SchemaFormResponse(plugin="csv", options={"path": "/client/changed"}),
            SchemaFormAuthority(
                knobs=SOURCE_KNOBS,
                model_validated_options={"path": "/server/held"},
                server_options={"path": "/server/held"},
            ),
        ),
    ],
)
def test_source_schema_form_rejects_plugin_hidden_blob_path_and_custody_tampering(
    response: SchemaFormResponse,
    authority: SchemaFormAuthority,
) -> None:
    session, turn = _source_options_session(facts=None)
    with pytest.raises(ValueError):
        transition_source_schema_form(session, target_id=SOURCE_A, turn=turn, response=response, authority=authority)
    assert session.pending_source_intents[SOURCE_A].phase == "plugin_options"


def test_inspection_review_uses_only_edited_columns_and_held_intent() -> None:
    session, turn = _source_review_session()

    result = transition_source_inspection_review(
        session,
        target_id=SOURCE_A,
        turn=turn,
        response=InspectionResponse(columns=("record_id", "display_name")),
    )

    assert result.session.step is GuidedStep.STEP_2_SINK
    assert result.session.source_order == (SOURCE_A,)
    assert not result.session.pending_source_intents
    source = result.session.reviewed_sources[SOURCE_A]
    assert source.plugin == "csv"
    assert dict(source.options)["path"].startswith("blob:")
    assert source.observed_columns == ("record_id", "display_name")
    assert source.sample_rows == ()
    assert result.next_turn == SinkPluginSelectionPreparation()


@pytest.mark.parametrize("columns", [(), ("id", "id"), ("",), (1,)])
def test_inspection_review_rejects_empty_duplicate_or_malformed_columns(columns: tuple[object, ...]) -> None:
    session, turn = _source_review_session()
    with pytest.raises((TypeError, ValueError)):
        transition_source_inspection_review(
            session,
            target_id=SOURCE_A,
            turn=turn,
            response=InspectionResponse(columns=columns),  # type: ignore[arg-type]
        )


def test_source_selection_rejects_malformed_server_inspection_facts() -> None:
    malformed = SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={},
        byte_range_inspected=(0, 1),
        sample_row_count=1,
        observed_headers=("id",),
        inferred_types={"other": "str"},
        url_candidates=(),
        warnings=(),
    )
    session, turn = _with_unanswered_turn(GuidedSession.initial(), TurnType.SINGLE_SELECT)
    with pytest.raises(InvariantError):
        transition_source_plugin_selection(
            session,
            turn=turn,
            response=PluginSelectionResponse(chosen=("csv",)),
            permitted_plugins=("csv",),
            inspection_facts=malformed,
            new_stable_id=UUID(SOURCE_A),
        )


def test_source_selection_rejects_inspection_facts_for_a_different_plugin_kind() -> None:
    session, turn = _with_unanswered_turn(GuidedSession.initial(), TurnType.SINGLE_SELECT)
    with pytest.raises(ValueError, match="inspection facts"):
        transition_source_plugin_selection(
            session,
            turn=turn,
            response=PluginSelectionResponse(chosen=("json",)),
            permitted_plugins=("csv", "json"),
            inspection_facts=_inspection(),
            new_stable_id=UUID(SOURCE_A),
        )


def test_source_schema_form_rejects_blob_custody_that_differs_from_inspection() -> None:
    session, turn = _source_options_session(facts=_inspection())
    with pytest.raises(ValueError, match="custody"):
        transition_source_schema_form(
            session,
            target_id=SOURCE_A,
            turn=turn,
            response=SchemaFormResponse(
                plugin="csv",
                options={"mode": "csv", "path": "blob:bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"},
            ),
            authority=SchemaFormAuthority(
                knobs=SOURCE_KNOBS,
                model_validated_options={
                    "mode": "csv",
                    "path": "blob:bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                },
                server_options={"path": "blob:bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"},
            ),
        )


def test_source_schema_form_rejects_inspection_without_blob_custody() -> None:
    session, turn = _source_options_session(facts=_inspection())
    with pytest.raises(ValueError, match="custody"):
        transition_source_schema_form(
            session,
            target_id=SOURCE_A,
            turn=turn,
            response=SchemaFormResponse(plugin="csv", options={"mode": "csv", "path": "/data/uninspected.csv"}),
            authority=SchemaFormAuthority(
                knobs=SOURCE_KNOBS,
                model_validated_options={"mode": "csv", "path": "/data/uninspected.csv"},
            ),
        )


def test_sink_selection_allocates_stable_id_and_reuses_pending_target() -> None:
    base = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        source_order=(SOURCE_A,),
        reviewed_sources={SOURCE_A: _source("source", ("id",))},
    )
    session, turn = _with_unanswered_turn(base, TurnType.SINGLE_SELECT)
    result = transition_sink_plugin_selection(
        session,
        turn=turn,
        response=PluginSelectionResponse(chosen=("json",)),
        permitted_plugins=("json",),
        new_stable_id=UUID(OUTPUT_A),
    )
    assert result.session.output_order == (OUTPUT_A,)
    assert result.session.pending_output_intents[OUTPUT_A].name == "output"
    assert result.next_turn == SinkSchemaFormPreparation(stable_id=OUTPUT_A, plugin="json")

    pending = SinkIntent(name="output_2", phase="plugin_selection", plugin=None, options=None)
    resumed = replace(
        result.session,
        history=(),
        output_order=(OUTPUT_A, OUTPUT_B),
        reviewed_outputs={OUTPUT_A: _output("output")},
        pending_output_intents={OUTPUT_B: pending},
    )
    resumed, resumed_turn = _with_unanswered_turn(resumed, TurnType.SINGLE_SELECT)
    resumed_result = transition_sink_plugin_selection(
        resumed,
        target_id=OUTPUT_B,
        turn=resumed_turn,
        response=PluginSelectionResponse(chosen=("json",)),
        permitted_plugins=("json",),
    )
    assert resumed_result.session.output_order == (OUTPUT_A, OUTPUT_B)
    assert resumed_result.session.reviewed_outputs[OUTPUT_A].name == "output"


def test_sink_selection_rejects_any_unreviewed_ordered_source() -> None:
    pending_source = SourceIntent(
        name="source_2",
        phase="plugin_selection",
        plugin=None,
        options=None,
        inspection_facts=None,
        observed_columns=(),
        sample_rows=(),
    )
    session = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        source_order=(SOURCE_A, SOURCE_B),
        reviewed_sources={SOURCE_A: _source("source", ("id",))},
        pending_source_intents={SOURCE_B: pending_source},
    )
    session, turn = _with_unanswered_turn(session, TurnType.SINGLE_SELECT)
    with pytest.raises(InvariantError, match="reviewed"):
        transition_sink_plugin_selection(
            session,
            turn=turn,
            response=PluginSelectionResponse(chosen=("json",)),
            permitted_plugins=("json",),
            new_stable_id=UUID(OUTPUT_A),
        )


def test_sink_schema_form_holds_options_and_emits_stable_deduplicated_candidates() -> None:
    session, turn = _sink_options_session()

    result = transition_sink_schema_form(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=SchemaFormResponse(plugin="json", options={"path": "/data/out.jsonl", "indent": 2}),
        authority=SchemaFormAuthority(
            knobs=SINK_KNOBS,
            model_validated_options={"path": "/data/out.jsonl", "indent": 2},
        ),
    )

    assert session.pending_output_intents[OUTPUT_A].phase == "plugin_options"
    intent = result.session.pending_output_intents[OUTPUT_A]
    assert intent.phase == "field_review"
    assert intent.plugin == "json"
    assert result.session.output_order == (OUTPUT_A,)
    assert result.next_turn == SinkFieldReviewPreparation(stable_id=OUTPUT_A, candidate_fields=("id", "name", "email"))


def test_schema_forms_reject_wrong_knob_types() -> None:
    sink_session, sink_turn = _sink_options_session()
    with pytest.raises(ValueError, match="number-int"):
        transition_sink_schema_form(
            sink_session,
            target_id=OUTPUT_A,
            turn=sink_turn,
            response=SchemaFormResponse(plugin="json", options={"indent": "not-an-int"}),
            authority=SchemaFormAuthority(knobs=SINK_KNOBS, model_validated_options={}),
        )


def test_source_structural_policy_survives_inspection_and_is_removed_from_plugin_options() -> None:
    source_session, source_turn = _source_options_session(facts=_inspection())
    policy_knobs = {
        "fields": [
            *SOURCE_KNOBS["fields"],
            {
                "name": "on_validation_failure",
                "kind": "enum",
                "enum": ["discard", "quarantine"],
                "required": False,
                "nullable": False,
            },
        ]
    }
    staged = transition_source_schema_form(
        source_session,
        target_id=SOURCE_A,
        turn=source_turn,
        response=SchemaFormResponse(
            plugin="csv",
            options={
                "mode": "csv",
                "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "on_validation_failure": "quarantine",
            },
        ),
        authority=SchemaFormAuthority(
            knobs=policy_knobs,
            model_validated_options={
                "mode": "csv",
                "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "on_validation_failure": "quarantine",
            },
            server_options={"path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        ),
    ).session
    assert staged.pending_source_intents[SOURCE_A].options is not None
    assert staged.pending_source_intents[SOURCE_A].options["on_validation_failure"] == "quarantine"
    staged = replace(staged, history=())
    staged, inspection_turn = _with_unanswered_turn(staged, TurnType.INSPECT_AND_CONFIRM)

    resolved = transition_source_inspection_review(
        staged,
        target_id=SOURCE_A,
        turn=inspection_turn,
        response=InspectionResponse(columns=("id", "name")),
    ).session.reviewed_sources[SOURCE_A]

    assert resolved.on_validation_failure == "quarantine"
    assert "on_validation_failure" not in resolved.options


def test_sink_structural_policy_survives_field_review_and_is_removed_from_plugin_options() -> None:
    session, turn = _sink_options_session()
    policy_knobs = {
        "fields": [
            *SINK_KNOBS["fields"],
            {
                "name": "on_write_failure",
                "kind": "enum",
                "enum": ["discard", "quarantine"],
                "required": False,
                "nullable": False,
            },
        ]
    }
    staged = transition_sink_schema_form(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=SchemaFormResponse(
            plugin="json",
            options={"path": "/data/out.jsonl", "on_write_failure": "quarantine"},
        ),
        authority=SchemaFormAuthority(
            knobs=policy_knobs,
            model_validated_options={"path": "/data/out.jsonl", "on_write_failure": "quarantine"},
        ),
    ).session
    staged = replace(staged, history=())
    staged, field_turn = _with_unanswered_turn(staged, TurnType.MULTI_SELECT_WITH_CUSTOM)

    resolved = transition_sink_field_review(
        staged,
        target_id=OUTPUT_A,
        turn=field_turn,
        response=FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=None),
    ).session.reviewed_outputs[OUTPUT_A]

    assert resolved.on_write_failure == "quarantine"
    assert "on_write_failure" not in resolved.options


def test_sink_schema_mode_is_preserved_from_validated_options() -> None:
    session, turn = _sink_options_session()
    staged = transition_sink_schema_form(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=SchemaFormResponse(
            plugin="json",
            options={"path": "/data/out.jsonl", "schema": {"mode": "flexible"}},
        ),
        authority=SchemaFormAuthority(
            knobs=SINK_KNOBS,
            model_validated_options={"path": "/data/out.jsonl", "schema": {"mode": "flexible"}},
        ),
    ).session
    staged = replace(staged, history=())
    staged, field_turn = _with_unanswered_turn(staged, TurnType.MULTI_SELECT_WITH_CUSTOM)

    resolved = transition_sink_field_review(
        staged,
        target_id=OUTPUT_A,
        turn=field_turn,
        response=FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=None),
    )

    assert resolved.session.reviewed_outputs[OUTPUT_A].schema_mode == "flexible"


def test_sink_field_review_resolves_same_id_and_advances_without_topology() -> None:
    session, turn = _sink_review_session()

    result = transition_sink_field_review(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=FieldSelectionResponse(chosen=("id", "email"), custom_inputs=("derived",), control_signal=None),
    )

    assert session.pending_output_intents[OUTPUT_A].phase == "field_review"
    assert result.session.step is GuidedStep.STEP_3_TRANSFORMS
    assert result.session.output_order == (OUTPUT_A,)
    assert not result.session.pending_output_intents
    output = result.session.reviewed_outputs[OUTPUT_A]
    assert output.required_fields == ("id", "email", "derived")
    assert output.schema_mode == "observed"
    assert result.next_turn is None
    assert set(result.__dataclass_fields__) == {"session", "next_turn"}
    assert not hasattr(result, "topology")


def test_sink_field_review_explicit_passthrough_is_the_only_empty_selection() -> None:
    session, turn = _sink_review_session()
    result = transition_sink_field_review(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=FieldSelectionResponse(chosen=(), custom_inputs=(), control_signal=ControlSignal.PASSTHROUGH),
    )
    assert result.session.reviewed_outputs[OUTPUT_A].required_fields == ()


@pytest.mark.parametrize(
    "response",
    [
        FieldSelectionResponse(chosen=(), custom_inputs=(), control_signal=None),
        FieldSelectionResponse(chosen=("missing",), custom_inputs=(), control_signal=None),
        FieldSelectionResponse(chosen=("id", "id"), custom_inputs=(), control_signal=None),
        FieldSelectionResponse(chosen=("id",), custom_inputs=("id",), control_signal=None),
        FieldSelectionResponse(chosen=(), custom_inputs=("name",), control_signal=None),
        FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=ControlSignal.PASSTHROUGH),
        FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=ControlSignal.BACK),
    ],
)
def test_sink_field_review_rejects_empty_unknown_duplicate_overlap_and_signal_conflicts(
    response: FieldSelectionResponse,
) -> None:
    session, turn = _sink_review_session()
    with pytest.raises(ValueError):
        transition_sink_field_review(session, target_id=OUTPUT_A, turn=turn, response=response)


def test_sink_field_review_preserves_two_output_order_and_unrelated_identity() -> None:
    pending = SinkIntent(
        name="output_2",
        phase="field_review",
        plugin="json",
        options={"path": "/data/second.jsonl", "on_write_failure": "discard"},
    )
    session = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        source_order=(SOURCE_A,),
        reviewed_sources={SOURCE_A: _source("source", ("id",))},
        output_order=(OUTPUT_A, OUTPUT_B),
        reviewed_outputs={OUTPUT_A: _output("output")},
        pending_output_intents={OUTPUT_B: pending},
    )
    session, turn = _with_unanswered_turn(session, TurnType.MULTI_SELECT_WITH_CUSTOM)

    result = transition_sink_field_review(
        session,
        target_id=OUTPUT_B,
        turn=turn,
        response=FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=None),
    )

    assert result.session.output_order == (OUTPUT_A, OUTPUT_B)
    assert result.session.reviewed_outputs[OUTPUT_A] is session.reviewed_outputs[OUTPUT_A]
    assert result.session.reviewed_outputs[OUTPUT_B].name == "output_2"


def test_turn_token_is_restart_stable_and_aba_sensitive() -> None:
    kwargs = {
        "schema": GUIDED_TURN_TOKEN_SCHEMA,
        "step": GuidedStep.STEP_1_SOURCE,
        "turn_type": TurnType.SCHEMA_FORM,
        "payload_hash": PAYLOAD_A,
    }
    first = guided_turn_token(history_index=4, **kwargs)
    restored = guided_turn_token(history_index=4, **kwargs)
    later_same_payload = guided_turn_token(history_index=6, **kwargs)

    assert first == restored
    assert first != later_same_payload
    assert first != guided_turn_token(history_index=4, **{**kwargs, "payload_hash": PAYLOAD_B})


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "guided.turn-token.v0"},
        {"history_index": True},
        {"history_index": -1},
        {"step": "step_1_source"},
        {"turn_type": "schema_form"},
        {"payload_hash": "A" * 64},
        {"payload_hash": "short"},
    ],
)
def test_turn_token_rejects_noncanonical_inputs(overrides: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "schema": GUIDED_TURN_TOKEN_SCHEMA,
        "history_index": 0,
        "step": GuidedStep.STEP_1_SOURCE,
        "turn_type": TurnType.SCHEMA_FORM,
        "payload_hash": PAYLOAD_A,
    }
    kwargs.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        guided_turn_token(**kwargs)  # type: ignore[arg-type]


def test_transition_derives_current_turn_and_rejects_stale_index_wrong_type_and_component() -> None:
    session, turn = _source_options_session(facts=None)
    stale = AnsweredTurn(history_index=turn.history_index + 1)
    wrong_record = replace(session.history[-1], turn_type=TurnType.SINGLE_SELECT)
    wrong_type = replace(session, history=(*session.history[:-1], wrong_record))
    response = SchemaFormResponse(plugin="csv", options={"mode": "csv"})
    authority = SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={})

    with pytest.raises(ValueError, match="stale"):
        transition_source_schema_form(session, target_id=SOURCE_A, turn=stale, response=response, authority=authority)
    with pytest.raises(ValueError, match="turn type"):
        transition_source_schema_form(wrong_type, target_id=SOURCE_A, turn=turn, response=response, authority=authority)
    with pytest.raises(ValueError, match="target"):
        transition_source_schema_form(session, target_id=SOURCE_B, turn=turn, response=response, authority=authority)


def test_transition_rejects_cross_stage_and_already_answered_turn() -> None:
    session, turn = _source_options_session(facts=None)
    cross_stage = replace(session, step=GuidedStep.STEP_2_SINK)
    with pytest.raises(InvariantError, match="step"):
        transition_source_schema_form(
            cross_stage,
            target_id=SOURCE_A,
            turn=turn,
            response=SchemaFormResponse(plugin="csv", options={"mode": "csv"}),
            authority=SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        )

    answered_record = replace(session.history[-1], response_hash=PAYLOAD_B)
    answered = replace(session, history=(*session.history[:-1], answered_record))
    with pytest.raises(ValueError, match="answered"):
        transition_source_schema_form(
            answered,
            target_id=SOURCE_A,
            turn=turn,
            response=SchemaFormResponse(plugin="csv", options={"mode": "csv"}),
            authority=SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        )


def test_transition_rejects_ambiguous_sibling_target_for_one_turn_occurrence() -> None:
    first = SourceIntent(
        name="source",
        phase="plugin_options",
        plugin="csv",
        options=None,
        inspection_facts=None,
        observed_columns=(),
        sample_rows=(),
    )
    second = replace(first, name="source_2")
    session = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_A, SOURCE_B),
        pending_source_intents={SOURCE_A: first, SOURCE_B: second},
    )
    session, turn = _with_unanswered_turn(session, TurnType.SCHEMA_FORM)

    with pytest.raises(ValueError, match="ambiguous"):
        transition_source_schema_form(
            session,
            target_id=SOURCE_A,
            turn=turn,
            response=SchemaFormResponse(plugin="csv", options={"mode": "csv"}),
            authority=SchemaFormAuthority(knobs=SOURCE_KNOBS, model_validated_options={}),
        )
