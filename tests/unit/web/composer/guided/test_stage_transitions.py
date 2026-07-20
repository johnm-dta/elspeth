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
    AnsweredTurn,
    FieldSelectionResponse,
    InspectionResponse,
    PluginSelectionResponse,
    SchemaFormAuthority,
    SchemaFormResponse,
    add_component_intent,
    begin_component_edit,
    finish_component_review,
    remove_reviewed_component,
    reorder_reviewed_components,
    transition_sink_field_review,
    transition_sink_plugin_selection,
    transition_sink_schema_form,
    transition_source_inspection_review,
    transition_source_plugin_selection,
    transition_source_schema_form,
)
from elspeth.web.composer.guided.state_machine import ComponentTarget, GuidedSession, SinkIntent, SourceIntent, TurnRecord
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


def _review_session(*, step: GuidedStep, two_components: bool = True) -> GuidedSession:
    sources = {
        SOURCE_A: _source("source", ("id", "name")),
        **({SOURCE_B: _source("source_2", ("email",))} if two_components else {}),
    }
    source_order = (SOURCE_A, SOURCE_B) if two_components else (SOURCE_A,)
    if step is GuidedStep.STEP_1_SOURCE:
        return GuidedSession(step=step, source_order=source_order, reviewed_sources=sources)
    outputs = {
        OUTPUT_A: _output("output"),
        **({OUTPUT_B: _output("output_2")} if two_components else {}),
    }
    output_order = (OUTPUT_A, OUTPUT_B) if two_components else (OUTPUT_A,)
    return GuidedSession(
        step=step,
        source_order=source_order,
        reviewed_sources=sources,
        output_order=output_order,
        reviewed_outputs=outputs,
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
    assert result.source_order == (SOURCE_A,)
    intent = result.pending_source_intents[SOURCE_A]
    assert (intent.name, intent.phase, intent.plugin) == ("source", "plugin_options", "csv")
    assert intent.inspection_facts is not facts
    assert facts_to_dict(intent.inspection_facts) == facts_to_dict(facts)  # type: ignore[arg-type]


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

    assert result.source_order == (SOURCE_A, SOURCE_B)
    assert result.pending_source_intents[SOURCE_B].name == "source_2"
    assert result.reviewed_sources[SOURCE_A] == session.reviewed_sources[SOURCE_A]


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
    intent = result.pending_source_intents[SOURCE_A]
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
    restored_review = GuidedSession.from_dict(result.to_dict())

    review_intent = restored_review.pending_source_intents[SOURCE_A]
    assert review_intent.phase == "inspection_review"
    assert facts_to_dict(review_intent.inspection_facts) == facts_to_dict(facts)  # type: ignore[arg-type]
    assert review_intent.observed_columns == facts.observed_headers


def test_source_schema_form_without_inspection_reviews_same_id_and_stays_in_source_stage() -> None:
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

    assert result.step is GuidedStep.STEP_1_SOURCE
    assert result.source_order == (SOURCE_A,)
    assert not result.pending_source_intents
    assert not result.pending_output_intents
    source = result.reviewed_sources[SOURCE_A]
    assert source.name == "source"
    assert source.plugin == "csv"
    assert source.observed_columns == ()


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

    assert result.step is GuidedStep.STEP_1_SOURCE
    assert result.source_order == (SOURCE_A,)
    assert not result.pending_source_intents
    assert not result.pending_output_intents
    source = result.reviewed_sources[SOURCE_A]
    assert source.plugin == "csv"
    assert dict(source.options)["path"].startswith("blob:")
    assert source.observed_columns == ("record_id", "display_name")
    assert source.sample_rows == ()


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
    assert result.output_order == (OUTPUT_A,)
    assert result.pending_output_intents[OUTPUT_A] == SinkIntent(
        name="output",
        phase="plugin_options",
        plugin="json",
        options=None,
    )

    pending = SinkIntent(name="output_2", phase="plugin_selection", plugin=None, options=None)
    resumed = replace(
        result,
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
    assert resumed_result.output_order == (OUTPUT_A, OUTPUT_B)
    assert resumed_result.reviewed_outputs[OUTPUT_A].name == "output"


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
    intent = result.pending_output_intents[OUTPUT_A]
    assert intent.phase == "field_review"
    assert intent.plugin == "json"
    assert result.output_order == (OUTPUT_A,)
    assert tuple(
        dict.fromkeys(column for stable_id in result.source_order for column in result.reviewed_sources[stable_id].observed_columns)
    ) == ("id", "name", "email")


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
    )
    staged_options = staged.pending_source_intents[SOURCE_A].options
    assert staged_options is not None
    assert staged_options["on_validation_failure"] == "quarantine"
    staged = replace(staged, history=())
    staged, inspection_turn = _with_unanswered_turn(staged, TurnType.INSPECT_AND_CONFIRM)

    resolved = transition_source_inspection_review(
        staged,
        target_id=SOURCE_A,
        turn=inspection_turn,
        response=InspectionResponse(columns=("id", "name")),
    ).reviewed_sources[SOURCE_A]

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
    )
    staged = replace(staged, history=())
    staged, field_turn = _with_unanswered_turn(staged, TurnType.MULTI_SELECT_WITH_CUSTOM)

    resolved = transition_sink_field_review(
        staged,
        target_id=OUTPUT_A,
        turn=field_turn,
        response=FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=None),
    ).reviewed_outputs[OUTPUT_A]

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
    )
    staged = replace(staged, history=())
    staged, field_turn = _with_unanswered_turn(staged, TurnType.MULTI_SELECT_WITH_CUSTOM)

    resolved = transition_sink_field_review(
        staged,
        target_id=OUTPUT_A,
        turn=field_turn,
        response=FieldSelectionResponse(chosen=("id",), custom_inputs=(), control_signal=None),
    )

    assert resolved.reviewed_outputs[OUTPUT_A].schema_mode == "flexible"


def test_sink_field_review_resolves_same_id_and_stays_in_output_stage() -> None:
    session, turn = _sink_review_session()

    result = transition_sink_field_review(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=FieldSelectionResponse(chosen=("id", "email"), custom_inputs=("derived",), control_signal=None),
    )

    assert session.pending_output_intents[OUTPUT_A].phase == "field_review"
    assert result.step is GuidedStep.STEP_2_SINK
    assert result.output_order == (OUTPUT_A,)
    assert not result.pending_output_intents
    output = result.reviewed_outputs[OUTPUT_A]
    assert output.required_fields == ("id", "email", "derived")
    assert output.schema_mode == "observed"
    assert not hasattr(result, "topology")


def test_sink_field_review_explicit_passthrough_is_the_only_empty_selection() -> None:
    session, turn = _sink_review_session()
    result = transition_sink_field_review(
        session,
        target_id=OUTPUT_A,
        turn=turn,
        response=FieldSelectionResponse(chosen=(), custom_inputs=(), control_signal=ControlSignal.PASSTHROUGH),
    )
    assert result.reviewed_outputs[OUTPUT_A].required_fields == ()


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

    assert result.output_order == (OUTPUT_A, OUTPUT_B)
    assert result.reviewed_outputs[OUTPUT_A] is session.reviewed_outputs[OUTPUT_A]
    assert result.reviewed_outputs[OUTPUT_B].name == "output_2"


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


@pytest.mark.parametrize(
    ("step", "component_kind", "new_id", "expected_order", "expected_name"),
    [
        (
            GuidedStep.STEP_1_SOURCE,
            "source",
            UUID("55555555-5555-4555-8555-555555555555"),
            (SOURCE_A, SOURCE_B, "55555555-5555-4555-8555-555555555555"),
            "source_3",
        ),
        (
            GuidedStep.STEP_2_SINK,
            "output",
            UUID("66666666-6666-4666-8666-666666666666"),
            (OUTPUT_A, OUTPUT_B, "66666666-6666-4666-8666-666666666666"),
            "output_3",
        ),
    ],
)
def test_add_component_intent_reuses_preallocated_id_and_appends_order(
    step: GuidedStep,
    component_kind: str,
    new_id: UUID,
    expected_order: tuple[str, ...],
    expected_name: str,
) -> None:
    session = _review_session(step=step)

    result = add_component_intent(session, component_kind, new_id)

    if component_kind == "source":
        assert result.source_order == expected_order
        intent = result.pending_source_intents[str(new_id)]
        assert result.reviewed_sources[SOURCE_A] is session.reviewed_sources[SOURCE_A]
    else:
        assert result.output_order == expected_order
        intent = result.pending_output_intents[str(new_id)]
        assert result.reviewed_sources[SOURCE_A] is session.reviewed_sources[SOURCE_A]
    assert intent.name == expected_name
    assert intent.phase == "plugin_selection"
    assert session.to_dict() == _review_session(step=step).to_dict()


def test_add_component_intent_rejects_cross_stage_duplicate_and_non_uuid() -> None:
    source_session = _review_session(step=GuidedStep.STEP_1_SOURCE)

    with pytest.raises(ValueError, match="step"):
        add_component_intent(source_session, "output", UUID(OUTPUT_A))
    with pytest.raises(InvariantError, match="already"):
        add_component_intent(source_session, "source", UUID(SOURCE_A))
    with pytest.raises(InvariantError, match="UUID"):
        add_component_intent(source_session, "source", SOURCE_B)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("step", "target"),
    [
        (GuidedStep.STEP_1_SOURCE, ComponentTarget(kind="source", stable_id=SOURCE_B)),
        (GuidedStep.STEP_2_SINK, ComponentTarget(kind="output", stable_id=OUTPUT_B)),
    ],
)
def test_begin_component_edit_preserves_reviewed_record_as_prefill_authority(
    step: GuidedStep,
    target: ComponentTarget,
) -> None:
    session = _review_session(step=step)
    original_dict = session.to_dict()

    result = begin_component_edit(session, target)

    if target.kind == "source":
        reviewed = result.reviewed_sources[target.stable_id]
        assert reviewed is session.reviewed_sources[target.stable_id]
        assert reviewed.name == "source_2"
        assert reviewed.plugin == "csv"
        assert dict(reviewed.options) == {"path": "/data/source_2.csv"}
        assert reviewed.on_validation_failure == "discard"
        assert not result.pending_source_intents
        assert result.reviewed_sources[SOURCE_A] is session.reviewed_sources[SOURCE_A]
    else:
        reviewed = result.reviewed_outputs[target.stable_id]
        assert reviewed is session.reviewed_outputs[target.stable_id]
        assert reviewed.name == "output_2"
        assert reviewed.plugin == "json"
        assert dict(reviewed.options) == {"path": "/data/output_2.jsonl"}
        assert reviewed.on_write_failure == "discard"
        assert not result.pending_output_intents
        assert result.reviewed_outputs[OUTPUT_A] is session.reviewed_outputs[OUTPUT_A]
    assert result.active_edit_target == target
    assert session.to_dict() == original_dict


def test_source_edit_replaces_same_reviewed_id_and_clears_target_after_direct_review() -> None:
    session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    target = ComponentTarget(kind="source", stable_id=SOURCE_A)
    editing = begin_component_edit(session, target)
    editing, turn = _with_unanswered_turn(editing, TurnType.SCHEMA_FORM)
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

    result = transition_source_schema_form(
        editing,
        target_id=SOURCE_A,
        turn=turn,
        response=SchemaFormResponse(
            plugin="csv",
            options={"mode": "csv", "path": "/data/revised.csv", "on_validation_failure": "quarantine"},
        ),
        authority=SchemaFormAuthority(
            knobs=policy_knobs,
            model_validated_options={
                "mode": "csv",
                "path": "/data/revised.csv",
                "on_validation_failure": "quarantine",
            },
        ),
    )

    assert result.step is GuidedStep.STEP_1_SOURCE
    assert result.source_order == editing.source_order
    assert result.reviewed_sources[SOURCE_B] is editing.reviewed_sources[SOURCE_B]
    revised = result.reviewed_sources[SOURCE_A]
    assert revised.name == "source"
    assert dict(revised.options) == {"mode": "csv", "path": "/data/revised.csv"}
    assert revised.on_validation_failure == "quarantine"
    assert result.active_edit_target is None
    assert not result.pending_source_intents


def test_inspection_backed_source_edit_survives_restart_until_final_review() -> None:
    session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    target = ComponentTarget(kind="source", stable_id=SOURCE_A)
    editing = begin_component_edit(session, target)
    editing, turn = _with_unanswered_turn(editing, TurnType.SCHEMA_FORM)
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
        editing,
        target_id=SOURCE_A,
        turn=turn,
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
        edit_inspection_facts=_inspection(),
    )

    assert staged.active_edit_target == target
    assert staged.reviewed_sources[SOURCE_A] is editing.reviewed_sources[SOURCE_A]
    assert staged.pending_source_intents[SOURCE_A].phase == "inspection_review"
    restored = GuidedSession.from_dict(staged.to_dict())
    restored = replace(restored, history=())
    restored, inspection_turn = _with_unanswered_turn(restored, TurnType.INSPECT_AND_CONFIRM)
    result = transition_source_inspection_review(
        restored,
        target_id=SOURCE_A,
        turn=inspection_turn,
        response=InspectionResponse(columns=("record_id", "display_name")),
    )

    assert result.step is GuidedStep.STEP_1_SOURCE
    assert result.source_order == session.source_order
    assert result.reviewed_sources[SOURCE_B] == session.reviewed_sources[SOURCE_B]
    revised = result.reviewed_sources[SOURCE_A]
    assert revised.name == "source"
    assert revised.observed_columns == ("record_id", "display_name")
    assert revised.on_validation_failure == "quarantine"
    assert result.active_edit_target is None
    assert not result.pending_source_intents


def test_output_edit_preserves_identity_policy_and_target_until_field_review() -> None:
    session = _review_session(step=GuidedStep.STEP_2_SINK)
    reviewed_outputs = dict(session.reviewed_outputs)
    reviewed_outputs[OUTPUT_A] = replace(reviewed_outputs[OUTPUT_A], on_write_failure="failures")
    session = replace(session, reviewed_outputs=reviewed_outputs)
    target = ComponentTarget(kind="output", stable_id=OUTPUT_A)
    editing = begin_component_edit(session, target)
    editing, turn = _with_unanswered_turn(editing, TurnType.SCHEMA_FORM)
    staged = transition_sink_schema_form(
        editing,
        target_id=OUTPUT_A,
        turn=turn,
        response=SchemaFormResponse(
            plugin="json",
            options={"path": "/data/revised.jsonl"},
        ),
        authority=SchemaFormAuthority(
            knobs=SINK_KNOBS,
            model_validated_options={"path": "/data/revised.jsonl"},
        ),
    )

    assert staged.active_edit_target == target
    assert staged.reviewed_outputs[OUTPUT_A] is editing.reviewed_outputs[OUTPUT_A]
    assert staged.pending_output_intents[OUTPUT_A].phase == "field_review"
    restored = GuidedSession.from_dict(staged.to_dict())
    restored = replace(restored, history=())
    restored, field_turn = _with_unanswered_turn(restored, TurnType.MULTI_SELECT_WITH_CUSTOM)
    result = transition_sink_field_review(
        restored,
        target_id=OUTPUT_A,
        turn=field_turn,
        response=FieldSelectionResponse(chosen=("id", "name"), custom_inputs=(), control_signal=None),
    )

    assert result.step is GuidedStep.STEP_2_SINK
    assert result.output_order == session.output_order
    assert result.reviewed_outputs[OUTPUT_B] == session.reviewed_outputs[OUTPUT_B]
    revised = result.reviewed_outputs[OUTPUT_A]
    assert revised.name == "output"
    assert dict(revised.options) == {"path": "/data/revised.jsonl"}
    assert revised.required_fields == ("id", "name")
    assert revised.on_write_failure == "failures"
    assert result.active_edit_target is None
    assert not result.pending_output_intents


def test_output_edit_rejects_hidden_structural_policy_override() -> None:
    session = _review_session(step=GuidedStep.STEP_2_SINK)
    reviewed_outputs = dict(session.reviewed_outputs)
    reviewed_outputs[OUTPUT_A] = replace(reviewed_outputs[OUTPUT_A], on_write_failure="failures")
    session = replace(session, reviewed_outputs=reviewed_outputs)
    editing = begin_component_edit(session, ComponentTarget(kind="output", stable_id=OUTPUT_A))
    editing, turn = _with_unanswered_turn(editing, TurnType.SCHEMA_FORM)

    with pytest.raises(ValueError, match="server-held structural policy"):
        transition_sink_schema_form(
            editing,
            target_id=OUTPUT_A,
            turn=turn,
            response=SchemaFormResponse(
                plugin="json",
                options={"path": "/data/revised.jsonl", "on_write_failure": "discard"},
            ),
            authority=SchemaFormAuthority(
                knobs=SINK_KNOBS,
                model_validated_options={"path": "/data/revised.jsonl", "on_write_failure": "discard"},
            ),
        )


def test_begin_component_edit_rejects_node_edge_cross_kind_missing_and_pending() -> None:
    session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    missing = ComponentTarget(kind="source", stable_id="77777777-7777-4777-8777-777777777777")

    for target in (
        ComponentTarget(kind="node", stable_id=SOURCE_A),
        ComponentTarget(kind="edge", stable_id=SOURCE_A),
        ComponentTarget(kind="output", stable_id=OUTPUT_A),
        missing,
    ):
        with pytest.raises(ValueError):
            begin_component_edit(session, target)

    pending = add_component_intent(session, "source", UUID("55555555-5555-4555-8555-555555555555"))
    with pytest.raises(InvariantError, match="pending"):
        begin_component_edit(pending, ComponentTarget(kind="source", stable_id=SOURCE_A))


@pytest.mark.parametrize(
    ("step", "target", "remaining_order"),
    [
        (GuidedStep.STEP_1_SOURCE, ComponentTarget(kind="source", stable_id=SOURCE_A), (SOURCE_B,)),
        (GuidedStep.STEP_2_SINK, ComponentTarget(kind="output", stable_id=OUTPUT_A), (OUTPUT_B,)),
    ],
)
def test_remove_reviewed_component_changes_mapping_and_order_atomically(
    step: GuidedStep,
    target: ComponentTarget,
    remaining_order: tuple[str, ...],
) -> None:
    session = _review_session(step=step)
    original_dict = session.to_dict()

    result = remove_reviewed_component(session, target)

    if target.kind == "source":
        assert result.source_order == remaining_order
        assert tuple(result.reviewed_sources) == remaining_order
        assert result.reviewed_sources[SOURCE_B] is session.reviewed_sources[SOURCE_B]
    else:
        assert result.output_order == remaining_order
        assert tuple(result.reviewed_outputs) == remaining_order
        assert result.reviewed_outputs[OUTPUT_B] is session.reviewed_outputs[OUTPUT_B]
    assert session.to_dict() == original_dict


@pytest.mark.parametrize(
    ("step", "target"),
    [
        (GuidedStep.STEP_1_SOURCE, ComponentTarget(kind="source", stable_id=SOURCE_A)),
        (GuidedStep.STEP_2_SINK, ComponentTarget(kind="output", stable_id=OUTPUT_A)),
    ],
)
def test_remove_reviewed_component_rejects_last_required_item(step: GuidedStep, target: ComponentTarget) -> None:
    session = _review_session(step=step, two_components=False)

    with pytest.raises(ValueError, match="last"):
        remove_reviewed_component(session, target)

    assert target.stable_id in (session.reviewed_sources if target.kind == "source" else session.reviewed_outputs)


def test_remove_reviewed_component_rejects_cross_kind_and_unresolved_target() -> None:
    session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    with pytest.raises(ValueError, match="step"):
        remove_reviewed_component(session, ComponentTarget(kind="output", stable_id=OUTPUT_A))
    with pytest.raises(ValueError, match="reviewed"):
        remove_reviewed_component(
            session,
            ComponentTarget(kind="source", stable_id="77777777-7777-4777-8777-777777777777"),
        )


@pytest.mark.parametrize(
    ("step", "component_kind", "stable_ids", "expected"),
    [
        (GuidedStep.STEP_1_SOURCE, "source", (UUID(SOURCE_B), UUID(SOURCE_A)), (SOURCE_B, SOURCE_A)),
        (GuidedStep.STEP_2_SINK, "output", (UUID(OUTPUT_B), UUID(OUTPUT_A)), (OUTPUT_B, OUTPUT_A)),
    ],
)
def test_reorder_reviewed_components_requires_stable_id_exact_permutation(
    step: GuidedStep,
    component_kind: str,
    stable_ids: tuple[UUID, ...],
    expected: tuple[str, ...],
) -> None:
    session = _review_session(step=step)

    result = reorder_reviewed_components(session, component_kind, stable_ids)

    assert (result.source_order if component_kind == "source" else result.output_order) == expected
    assert result.reviewed_sources[SOURCE_A] is session.reviewed_sources[SOURCE_A]
    if step is GuidedStep.STEP_2_SINK:
        assert result.reviewed_outputs[OUTPUT_A] is session.reviewed_outputs[OUTPUT_A]


@pytest.mark.parametrize(
    "stable_ids",
    [
        (UUID(SOURCE_A),),
        (UUID(SOURCE_A), UUID(SOURCE_A)),
        (UUID(SOURCE_A), UUID("77777777-7777-4777-8777-777777777777")),
    ],
    ids=["missing", "duplicate", "foreign"],
)
def test_reorder_reviewed_components_rejects_non_permutations(stable_ids: tuple[UUID, ...]) -> None:
    session = _review_session(step=GuidedStep.STEP_1_SOURCE)

    with pytest.raises(ValueError, match="permutation"):
        reorder_reviewed_components(session, "source", stable_ids)


def test_reorder_reviewed_components_rejects_cross_kind_and_pending_collection() -> None:
    session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    with pytest.raises(ValueError, match="step"):
        reorder_reviewed_components(session, "output", (UUID(OUTPUT_A), UUID(OUTPUT_B)))
    pending = add_component_intent(session, "source", UUID("55555555-5555-4555-8555-555555555555"))
    with pytest.raises(InvariantError, match="pending"):
        reorder_reviewed_components(pending, "source", (UUID(SOURCE_B), UUID(SOURCE_A)))


def test_finish_component_review_advances_only_valid_source_then_output_collections() -> None:
    source_session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    output_session = _review_session(step=GuidedStep.STEP_2_SINK)

    source_result = finish_component_review(source_session, "source")
    output_result = finish_component_review(output_session, "output")

    assert source_result.step is GuidedStep.STEP_2_SINK
    assert output_result.step is GuidedStep.STEP_3_TRANSFORMS
    assert source_result.reviewed_sources[SOURCE_A] is source_session.reviewed_sources[SOURCE_A]
    assert output_result.reviewed_outputs[OUTPUT_A] is output_session.reviewed_outputs[OUTPUT_A]


def test_finish_component_review_rejects_cross_kind_empty_and_pending_collections() -> None:
    source_session = _review_session(step=GuidedStep.STEP_1_SOURCE)
    with pytest.raises(ValueError, match="step"):
        finish_component_review(source_session, "output")
    with pytest.raises(InvariantError, match="at least one"):
        finish_component_review(GuidedSession.initial(), "source")
    pending = add_component_intent(source_session, "source", UUID("55555555-5555-4555-8555-555555555555"))
    with pytest.raises(InvariantError, match="pending"):
        finish_component_review(pending, "source")
