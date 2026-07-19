"""Schema-8 guided checkpoint contract tests.

These tests intentionally exercise the persisted JSON boundary rather than
only constructing dataclasses.  Guided composer metadata is audit-tier state:
unknown keys, coercion, stale aliases, and altered integrity fields must fail
closed when a process restores the checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any
from uuid import UUID

import pytest

from elspeth.web.composer.guided import stage_subjects, state_machine
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, GuidedStep, TurnType
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.stage_subjects import (
    ComponentCountConstraint,
    EdgeRouteConstraint,
    FailureRouteConstraint,
    OptionValueConstraint,
    PluginSubject,
    StableSubject,
    SubjectPresenceConstraint,
)
from elspeth.web.composer.guided.state_machine import (
    GUIDED_SESSION_SCHEMA_VERSION,
    ComponentTarget,
    DeferredStageIntent,
    GuidedProposalRef,
    GuidedSession,
    SinkIntent,
    SourceIntent,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
    guided_reviewed_anchor_hash,
)
from elspeth.web.composer.pipeline_proposal import AbsentBase, PresentBase
from elspeth.web.composer.source_inspection import SourceInspectionFacts

SOURCE_A = "11111111-1111-4111-8111-111111111111"
SOURCE_B = "22222222-2222-4222-8222-222222222222"
OUTPUT_A = "33333333-3333-4333-8333-333333333333"
OUTPUT_B = "44444444-4444-4444-8444-444444444444"
NODE_A = "55555555-5555-4555-8555-555555555555"
NODE_B = "66666666-6666-4666-8666-666666666666"
INTENT_A = "77777777-7777-4777-8777-777777777777"
INTENT_B = "88888888-8888-4888-8888-888888888888"
MESSAGE_A = "99999999-9999-4999-8999-999999999999"
PLUGIN_SUBJECT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
PROPOSAL_A = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
PROPOSAL_B = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
STATE_A = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _inspection() -> SourceInspectionFacts:
    return SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={"filename": "input.csv", "mime_type": "text/csv"},
        byte_range_inspected=(0, 32),
        sample_row_count=2,
        observed_headers=("id", "name"),
        inferred_types={"id": "int", "name": "str"},
        url_candidates=(),
        warnings=(),
    )


def _source(name: str, path: str) -> SourceResolved:
    return SourceResolved(
        name=name,
        plugin="csv",
        options={"path": path, "dialect": {"delimiter": ","}},
        observed_columns=("id", "name"),
        sample_rows=({"id": 1, "name": "Ada"},),
        on_validation_failure="discard",
    )


def _output(name: str, path: str) -> SinkOutputResolved:
    return SinkOutputResolved(
        name=name,
        plugin="json",
        options={"path": path, "format": {"indent": 2}},
        required_fields=("id", "name"),
        schema_mode="fixed",
        on_write_failure="discard",
    )


def _source_intent(*, name: str = "incoming") -> SourceIntent:
    return SourceIntent(
        name=name,
        phase="inspection_review",
        plugin="csv",
        options={"path": "/data/incoming.csv"},
        inspection_facts=_inspection(),
        observed_columns=("id", "name"),
        sample_rows=({"id": 1, "name": "Ada"},),
    )


def _sink_intent(*, name: str = "archive") -> SinkIntent:
    return SinkIntent(
        name=name,
        phase="field_review",
        plugin="json",
        options={"path": "/data/archive.jsonl"},
    )


def _deferred(*, intent_id: str = INTENT_A) -> DeferredStageIntent:
    stable = StableSubject(kind="stable", component_kind="source", stable_id=SOURCE_A)
    plugin = PluginSubject(
        kind="plugin",
        subject_id=PLUGIN_SUBJECT,
        plugin_kind="transform",
        plugin_name="rename",
    )
    return DeferredStageIntent.create(
        intent_id=intent_id,
        receiving_stage="source",
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="rename",
        redacted_summary="Rename name to display_name before output.",
        originating_message_id=MESSAGE_A,
        message_content_hash=HASH_A,
        constraints=(
            SubjectPresenceConstraint(kind="subject_presence", subject=plugin, present=True),
            OptionValueConstraint(
                kind="option_value",
                subject=plugin,
                option_path=("mapping", "name"),
                operator="equals",
                value="display_name",
            ),
            ComponentCountConstraint(
                kind="component_count",
                component_kind="node",
                plugin_kind="transform",
                plugin_name="rename",
                operator="at_least",
                count=1,
            ),
            EdgeRouteConstraint(
                kind="edge_route",
                from_subject=stable,
                edge_type="on_success",
                to_subject=plugin,
                present=True,
            ),
            FailureRouteConstraint(
                kind="failure_route",
                subject=stable,
                failure_kind="source_validation",
                operator="equals",
                target="discard",
            ),
        ),
    )


def _reviewed_facts(
    *,
    source_order: tuple[str, ...],
    reviewed_sources: dict[str, SourceResolved],
    output_order: tuple[str, ...],
    reviewed_outputs: dict[str, SinkOutputResolved],
) -> dict[str, Any]:
    return {
        "source_order": list(source_order),
        "reviewed_sources": {stable_id: reviewed_sources[stable_id].to_dict() for stable_id in source_order},
        "output_order": list(output_order),
        "reviewed_outputs": {stable_id: reviewed_outputs[stable_id].to_dict() for stable_id in output_order},
    }


def _proposal(
    *,
    source_order: tuple[str, ...],
    reviewed_sources: dict[str, SourceResolved],
    output_order: tuple[str, ...],
    reviewed_outputs: dict[str, SinkOutputResolved],
    covered: tuple[str, ...] = (INTENT_A,),
) -> GuidedProposalRef:
    anchor = guided_reviewed_anchor_hash(
        source_order=source_order,
        reviewed_sources=reviewed_sources,
        output_order=output_order,
        reviewed_outputs=reviewed_outputs,
    )
    return GuidedProposalRef(
        proposal_id=PROPOSAL_A,
        draft_hash=HASH_B,
        base=PresentBase(state_id=STATE_A, composition_content_hash=HASH_C),
        reviewed_anchor_hash=anchor,
        covered_deferred_intent_ids=covered,
        creation_event_schema="pipeline_proposal_created.v1",
    )


def _full_session() -> GuidedSession:
    sources = {
        SOURCE_A: _source("primary", "/data/primary.csv"),
        SOURCE_B: _source("secondary", "/data/secondary.csv"),
    }
    outputs = {
        OUTPUT_A: _output("archive", "/data/archive.jsonl"),
        OUTPUT_B: _output("errors", "/data/errors.jsonl"),
    }
    source_order = (SOURCE_B, SOURCE_A)
    output_order = (OUTPUT_B, OUTPUT_A)
    deferred = _deferred()
    return GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.PROPOSE_PIPELINE,
                payload_hash=HASH_A,
                response_hash=None,
                emitter="llm",
            ),
        ),
        source_order=source_order,
        reviewed_sources=sources,
        pending_source_intents={},
        output_order=output_order,
        reviewed_outputs=outputs,
        pending_output_intents={},
        deferred_intents=(deferred,),
        active_proposal=_proposal(
            source_order=source_order,
            reviewed_sources=sources,
            output_order=output_order,
            reviewed_outputs=outputs,
        ),
        active_edit_target=ComponentTarget(kind="node", stable_id=NODE_A),
        root_intent_message_id=MESSAGE_A,
    )


def test_schema8_round_trip_retains_plural_order_and_stable_ids_after_restart() -> None:
    session = _full_session()

    encoded = session.to_dict()
    restored = GuidedSession.from_dict(encoded)

    assert GUIDED_SESSION_SCHEMA_VERSION == 9
    assert restored == session
    assert restored.source_order == (SOURCE_B, SOURCE_A)
    assert restored.output_order == (OUTPUT_B, OUTPUT_A)
    assert tuple(restored.reviewed_sources) == (SOURCE_A, SOURCE_B)
    assert restored.reviewed_sources[SOURCE_A].name == "primary"
    assert restored.reviewed_outputs[OUTPUT_B].name == "errors"
    assert restored.active_edit_target == ComponentTarget(kind="node", stable_id=NODE_A)
    assert set(encoded) == {
        "schema_version",
        "step",
        "history",
        "profile",
        "advisor_checkpoint_passes_used",
        "advisor_signoff_escape_offered",
        "terminal",
        "transition_consumed",
        "chat_history",
        "chat_turn_seq",
        "source_order",
        "reviewed_sources",
        "pending_source_intents",
        "output_order",
        "reviewed_outputs",
        "pending_output_intents",
        "deferred_intents",
        "active_proposal",
        "active_edit_target",
        "root_intent_message_id",
    }


@pytest.mark.parametrize("step", list(GuidedStep))
def test_every_guided_stage_round_trips(step: GuidedStep) -> None:
    session = replace(GuidedSession.initial(), step=step)
    assert GuidedSession.from_dict(session.to_dict()) == session


def test_pending_source_and_output_phases_round_trip() -> None:
    pending_sources = {
        SOURCE_A: SourceIntent(
            name="primary",
            phase="plugin_selection",
            plugin=None,
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        ),
        SOURCE_B: SourceIntent(
            name="secondary",
            phase="plugin_options",
            plugin="csv",
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        ),
    }
    pending_outputs = {
        OUTPUT_A: SinkIntent(name="archive", phase="plugin_selection", plugin=None, options=None),
        OUTPUT_B: SinkIntent(name="errors", phase="plugin_options", plugin="json", options=None),
    }
    session = replace(
        GuidedSession.initial(),
        source_order=(SOURCE_B, SOURCE_A),
        pending_source_intents=pending_sources,
        output_order=(OUTPUT_A, OUTPUT_B),
        pending_output_intents=pending_outputs,
    )

    restored = GuidedSession.from_dict(session.to_dict())

    assert restored == session
    assert restored.pending_source_intents[SOURCE_A].phase == "plugin_selection"
    assert restored.pending_output_intents[OUTPUT_B].plugin == "json"


def test_proposal_ref_round_trips_absent_base_and_paired_supersession() -> None:
    ref = GuidedProposalRef(
        proposal_id=PROPOSAL_A,
        draft_hash=HASH_A,
        base=AbsentBase(),
        reviewed_anchor_hash=HASH_B,
        covered_deferred_intent_ids=(),
        creation_event_schema="pipeline_proposal_created.v1",
        supersedes_proposal_id=PROPOSAL_B,
        supersedes_draft_hash=HASH_C,
    )
    assert GuidedProposalRef.from_dict(ref.to_dict()) == ref


def test_constraint_union_round_trips_with_subject_target() -> None:
    plugin = PluginSubject(
        kind="plugin",
        subject_id=PLUGIN_SUBJECT,
        plugin_kind="sink",
        plugin_name="json",
    )
    deferred = DeferredStageIntent.create(
        intent_id=INTENT_B,
        receiving_stage="topology",
        target_stage="wire_review",
        catalog_kind="sink",
        catalog_name="json",
        redacted_summary="Send output write failures to the error sink.",
        originating_message_id=MESSAGE_A,
        message_content_hash=HASH_A,
        constraints=(
            FailureRouteConstraint(
                kind="failure_route",
                subject=plugin,
                failure_kind="output_write",
                operator="not_equals",
                target=StableSubject(kind="stable", component_kind="output", stable_id=OUTPUT_B),
            ),
        ),
    )
    assert DeferredStageIntent.from_dict(deferred.to_dict()) == deferred


@pytest.mark.parametrize("bad_version", [7, 8, "9", 9.0, True])
def test_wrong_or_coerced_schema_version_fails_closed(bad_version: object) -> None:
    encoded = GuidedSession.initial().to_dict()
    encoded["schema_version"] = bad_version
    with pytest.raises(InvariantError, match="schema_version"):
        GuidedSession.from_dict(encoded)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "step_1_result",
        "step_2_result",
        "step_3_proposal",
        "step_3_edit_index",
        "step_1_source_intent",
        "step_2_sink_intent",
        "step_1_inspection_facts",
        "step_1_chosen_plugin",
        "step_2_chosen_plugin",
    ],
)
def test_v7_aliases_are_not_accepted(legacy_key: str) -> None:
    encoded = GuidedSession.initial().to_dict()
    encoded[legacy_key] = None
    with pytest.raises(InvariantError, match="unexpected keys"):
        GuidedSession.from_dict(encoded)


def test_all_persisted_records_reject_missing_and_extra_keys() -> None:
    encoded = _full_session().to_dict()
    del encoded["source_order"]
    with pytest.raises(InvariantError, match="missing keys"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["reviewed_sources"][SOURCE_A]["extra"] = True
    with pytest.raises(InvariantError, match="unexpected keys"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["active_proposal"]["extra"] = True
    with pytest.raises(InvariantError, match="unexpected keys"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["deferred_intents"][0]["constraints"][0]["extra"] = True
    with pytest.raises(InvariantError, match="unexpected keys"):
        GuidedSession.from_dict(encoded)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("step",), "unknown"),
        (("active_edit_target", "kind"), "transform"),
        (("deferred_intents", 0, "target_stage"), "deployment"),
        (("deferred_intents", 0, "catalog_kind"), "database"),
        (("deferred_intents", 0, "constraints", 0, "kind"), "unknown"),
    ],
)
def test_closed_enums_fail_closed(path: tuple[str | int, ...], value: object) -> None:
    encoded: Any = _full_session().to_dict()
    cursor = encoded
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(InvariantError):
        GuidedSession.from_dict(encoded)


@pytest.mark.parametrize("bad_id", ["not-a-uuid", str(PROPOSAL_A).upper(), "{bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb}"])
def test_stable_ids_are_canonical_lowercase_uuid_text(bad_id: str) -> None:
    encoded = GuidedSession.initial().to_dict()
    encoded["source_order"] = [bad_id]
    encoded["pending_source_intents"] = {bad_id: _source_intent().to_dict()}
    with pytest.raises(InvariantError, match="canonical lowercase UUID"):
        GuidedSession.from_dict(encoded)


def test_plural_orders_are_exact_duplicate_free_permutations_and_names_are_unique() -> None:
    encoded = _full_session().to_dict()
    encoded["source_order"] = [SOURCE_A, SOURCE_A]
    with pytest.raises(InvariantError, match="source_order"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["output_order"] = [OUTPUT_A]
    with pytest.raises(InvariantError, match="output_order"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["reviewed_sources"][SOURCE_B]["name"] = "primary"
    with pytest.raises(InvariantError, match="source names"):
        GuidedSession.from_dict(encoded)

    encoded = GuidedSession.initial().to_dict()
    encoded["source_order"] = [SOURCE_A]
    encoded["output_order"] = [SOURCE_A]
    encoded["pending_source_intents"] = {SOURCE_A: _source_intent().to_dict()}
    encoded["pending_output_intents"] = {SOURCE_A: _sink_intent().to_dict()}
    with pytest.raises(InvariantError, match="globally unique"):
        GuidedSession.from_dict(encoded)


def test_reviewed_and_pending_keysets_must_be_disjoint() -> None:
    encoded = _full_session().to_dict()
    encoded["pending_source_intents"] = {SOURCE_A: _source_intent(name="pending").to_dict()}
    with pytest.raises(InvariantError, match="disjoint"):
        GuidedSession.from_dict(encoded)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["reviewed_sources"][SOURCE_A]["options"]["dialect"].__setitem__("delimiter", "|"),
        lambda d: d["reviewed_outputs"][OUTPUT_A]["options"]["format"].__setitem__("indent", 4),
    ],
)
def test_active_proposal_recomputes_reviewed_anchor(mutate: Any) -> None:
    encoded = _full_session().to_dict()
    mutate(encoded)
    with pytest.raises(InvariantError, match="reviewed_anchor_hash"):
        GuidedSession.from_dict(encoded)


def test_active_proposal_requires_no_pending_intents_and_valid_covered_subsequence() -> None:
    encoded = _full_session().to_dict()
    encoded["pending_output_intents"] = {"eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee": _sink_intent(name="late").to_dict()}
    encoded["output_order"].append("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
    with pytest.raises(InvariantError, match="active_proposal"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["active_proposal"]["covered_deferred_intent_ids"] = [INTENT_B]
    with pytest.raises(InvariantError, match="covered_deferred_intent_ids"):
        GuidedSession.from_dict(encoded)

    encoded = _full_session().to_dict()
    encoded["deferred_intents"][0]["constraints"] = []
    with pytest.raises(InvariantError, match="empty constraints"):
        GuidedSession.from_dict(encoded)


def test_active_proposal_requires_one_trailing_unanswered_proposal_or_wire_turn() -> None:
    valid_step_3 = _full_session()
    assert GuidedSession.from_dict(valid_step_3.to_dict()) == valid_step_3

    valid_step_4 = replace(
        valid_step_3,
        step=GuidedStep.STEP_4_WIRE,
        history=(
            replace(valid_step_3.history[-1], response_hash=HASH_B),
            TurnRecord(
                step=GuidedStep.STEP_4_WIRE,
                turn_type=TurnType.CONFIRM_WIRING,
                payload_hash=HASH_C,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    assert GuidedSession.from_dict(valid_step_4.to_dict()) == valid_step_4

    invalid_records = []
    missing = valid_step_3.to_dict()
    missing["history"] = []
    invalid_records.append(missing)
    answered = valid_step_3.to_dict()
    answered["history"][-1]["response_hash"] = HASH_B
    invalid_records.append(answered)
    cross_stage = valid_step_3.to_dict()
    cross_stage["step"] = GuidedStep.STEP_4_WIRE.value
    invalid_records.append(cross_stage)
    ambiguous = valid_step_4.to_dict()
    ambiguous["history"].append(dict(ambiguous["history"][-1]))
    invalid_records.append(ambiguous)
    for invalid in invalid_records:
        with pytest.raises(InvariantError, match="active_proposal"):
            GuidedSession.from_dict(invalid)


def test_active_edit_target_must_resolve_or_have_active_proposal() -> None:
    encoded = GuidedSession.initial().to_dict()
    encoded["active_edit_target"] = {"kind": "source", "stable_id": SOURCE_A}
    with pytest.raises(InvariantError, match="active_edit_target"):
        GuidedSession.from_dict(encoded)

    encoded = GuidedSession.initial().to_dict()
    encoded["active_edit_target"] = {"kind": "edge", "stable_id": NODE_A}
    with pytest.raises(InvariantError, match="active_proposal"):
        GuidedSession.from_dict(encoded)


def test_terminal_state_clears_proposal_and_edit_target() -> None:
    with pytest.raises((InvariantError, ValueError), match="terminal"):
        replace(
            _full_session(),
            terminal=TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline: {}\n"),
        )


@pytest.mark.parametrize(
    "terminal",
    [
        TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline: {}\n"),
        TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        ),
    ],
)
def test_completed_and_exited_states_round_trip(terminal: TerminalState) -> None:
    session = replace(GuidedSession.initial(), terminal=terminal)
    assert GuidedSession.from_dict(session.to_dict()) == session


def test_terminal_state_cross_field_rules_are_enforced() -> None:
    with pytest.raises((InvariantError, ValueError)):
        TerminalState(kind=TerminalKind.COMPLETED, reason=TerminalReason.USER_PRESSED_EXIT, pipeline_yaml="pipeline: {}\n")
    with pytest.raises((InvariantError, ValueError)):
        TerminalState(kind=TerminalKind.EXITED_TO_FREEFORM, reason=None, pipeline_yaml=None)
    with pytest.raises((InvariantError, ValueError)):
        TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml="pipeline: {}\n",
        )


def test_source_intent_phase_presence_rules_are_strict() -> None:
    with pytest.raises((InvariantError, ValueError), match="plugin_selection"):
        SourceIntent(
            name="bad",
            phase="plugin_selection",
            plugin="csv",
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        )
    with pytest.raises((InvariantError, ValueError), match="inspection_review"):
        SourceIntent(
            name="bad",
            phase="inspection_review",
            plugin="csv",
            options={},
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        )


def test_sink_intent_phase_presence_rules_are_strict() -> None:
    with pytest.raises((InvariantError, ValueError), match="plugin_selection"):
        SinkIntent(name="bad", phase="plugin_selection", plugin="json", options=None)
    with pytest.raises((InvariantError, ValueError), match="field_review"):
        SinkIntent(name="bad", phase="field_review", plugin="json", options=None)


def test_deferred_stage_is_forward_only_and_catalog_fields_are_paired() -> None:
    with pytest.raises((InvariantError, ValueError), match="forward"):
        replace(_deferred(), receiving_stage="topology", target_stage="output")
    with pytest.raises((InvariantError, ValueError), match="catalog"):
        replace(_deferred(), catalog_name=None)


def test_deferred_summary_hash_and_message_hash_are_verified() -> None:
    encoded = _deferred().to_dict()
    encoded["redacted_summary"] = "Altered summary"
    with pytest.raises(InvariantError, match="summary_hash"):
        DeferredStageIntent.from_dict(encoded)

    encoded = _deferred().to_dict()
    encoded["message_content_hash"] = "not-a-hash"
    with pytest.raises(InvariantError, match="message_content_hash"):
        DeferredStageIntent.from_dict(encoded)


def test_option_path_is_bounded_and_option_values_are_strict_json_scalars() -> None:
    subject = StableSubject(kind="stable", component_kind="node", stable_id=NODE_A)
    with pytest.raises((InvariantError, ValueError), match="option_path"):
        OptionValueConstraint(
            kind="option_value",
            subject=subject,
            option_path=(),
            operator="equals",
            value=1,
        )
    with pytest.raises((InvariantError, ValueError), match="option_path"):
        OptionValueConstraint(
            kind="option_value",
            subject=subject,
            option_path=tuple(f"part{index}" for index in range(17)),
            operator="equals",
            value=1,
        )
    with pytest.raises((InvariantError, ValueError), match="JSON scalar"):
        OptionValueConstraint(
            kind="option_value",
            subject=subject,
            option_path=("threshold",),
            operator="equals",
            value={"nested": "not scalar"},
        )
    with pytest.raises((InvariantError, ValueError), match="JSON"):
        OptionValueConstraint(
            kind="option_value",
            subject=subject,
            option_path=("threshold",),
            operator="equals",
            value=math.nan,
        )


def test_component_count_pairing_and_non_negative_exact_int() -> None:
    with pytest.raises((InvariantError, ValueError), match="paired"):
        ComponentCountConstraint(
            kind="component_count",
            component_kind="node",
            plugin_kind="transform",
            plugin_name=None,
            operator="equals",
            count=1,
        )
    with pytest.raises((InvariantError, TypeError, ValueError), match="count"):
        ComponentCountConstraint(
            kind="component_count",
            component_kind="node",
            plugin_kind=None,
            plugin_name=None,
            operator="equals",
            count=True,
        )


def test_proposal_ref_requires_hashes_uuid_types_and_paired_supersession() -> None:
    with pytest.raises((InvariantError, TypeError, ValueError), match="proposal_id"):
        replace(
            _proposal(source_order=(), reviewed_sources={}, output_order=(), reviewed_outputs={}, covered=()), proposal_id=str(PROPOSAL_A)
        )
    with pytest.raises((InvariantError, ValueError), match="supersedes"):
        replace(
            _proposal(source_order=(), reviewed_sources={}, output_order=(), reviewed_outputs={}, covered=()),
            supersedes_proposal_id=PROPOSAL_B,
            supersedes_draft_hash=None,
        )


def test_schema8_state_is_recursively_immutable_and_detached() -> None:
    mutable_source_options = {"path": "/data/in.csv", "nested": {"delimiter": ","}}
    mutable_sample = {"id": 1}
    source = SourceResolved(
        name="primary",
        plugin="csv",
        options=mutable_source_options,
        observed_columns=("id",),
        sample_rows=(mutable_sample,),
        on_validation_failure="discard",
    )
    mutable_sources = {SOURCE_A: source}
    session = replace(
        GuidedSession.initial(),
        source_order=(SOURCE_A,),
        reviewed_sources=mutable_sources,
    )

    mutable_source_options["nested"]["delimiter"] = "|"
    mutable_sample["id"] = 2
    mutable_sources.clear()

    assert session.reviewed_sources[SOURCE_A].options["nested"]["delimiter"] == ","
    assert session.reviewed_sources[SOURCE_A].sample_rows[0]["id"] == 1
    with pytest.raises(TypeError):
        session.reviewed_sources[SOURCE_A].options["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        session.reviewed_sources[SOURCE_A].options["nested"]["delimiter"] = "|"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        session.active_proposal = None  # type: ignore[misc]


def test_turn_record_requires_closed_exact_audit_fields() -> None:
    class _StrSubclass(str):
        pass

    record = TurnRecord(
        step=GuidedStep.STEP_1_SOURCE,
        turn_type=TurnType.SINGLE_SELECT,
        payload_hash=HASH_A,
        response_hash=None,
        emitter="server",
        summary=None,
    )
    assert TurnRecord.from_dict(record.to_dict()) == record

    for change, match in (
        ({"payload_hash": "short"}, "payload_hash"),
        ({"response_hash": "short"}, "response_hash"),
        ({"emitter": "worker"}, "emitter"),
        ({"emitter": _StrSubclass("server")}, "emitter"),
        ({"summary": 7}, "summary"),
    ):
        with pytest.raises((InvariantError, TypeError, ValueError), match=match):
            replace(record, **change)


@pytest.mark.parametrize(
    "bad_value",
    [b"bytes", bytearray(b"bytes"), {"set"}, ("tuple",), {1: "non-string-key"}, math.inf, -math.inf, math.nan],
)
def test_persisted_options_reject_non_strict_json_values(bad_value: object) -> None:
    with pytest.raises((InvariantError, TypeError, ValueError), match=r"JSON|key|number"):
        SourceResolved(
            name="primary",
            plugin="csv",
            options={"bad": bad_value},
            observed_columns=(),
            sample_rows=(),
            on_validation_failure="discard",
        )


def test_persisted_json_is_depth_item_and_string_bounded() -> None:
    too_deep: dict[str, object] = {}
    cursor = too_deep
    for _ in range(65):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child

    for options, match in (
        (too_deep, "depth"),
        ({"items": list(range(10_001))}, "item"),
        ({"text": "x" * 65_537}, "string"),
    ):
        with pytest.raises((InvariantError, TypeError, ValueError), match=match):
            SinkIntent(name="archive", phase="field_review", plugin="json", options=options)


def test_persisted_json_has_shared_aggregate_text_and_utf8_budgets() -> None:
    legal_piece = "x" * 65_000
    with pytest.raises(InvariantError, match=r"aggregate.*character"):
        SourceResolved(
            name="primary",
            plugin="csv",
            options={f"option_{index}": legal_piece for index in range(8)},
            observed_columns=(),
            sample_rows=tuple({f"sample_{index}": legal_piece} for index in range(9)),
            on_validation_failure="discard",
        )

    emoji_piece = "😀" * 65_000
    with pytest.raises(InvariantError, match="UTF-8"):
        SinkIntent(
            name="archive",
            phase="field_review",
            plugin="json",
            options={f"text_{index}": emoji_piece for index in range(5)},
        )

    with pytest.raises(InvariantError, match=r"aggregate.*character"):
        SourceResolved(
            name="primary",
            plugin="csv",
            options={},
            observed_columns=tuple(f"column-{index}-{'x' * 64_980}" for index in range(17)),
            sample_rows=(),
            on_validation_failure="discard",
        )

    with pytest.raises(InvariantError, match="UTF-8"):
        SinkOutputResolved(
            name="archive",
            plugin="json",
            options={},
            required_fields=tuple("😀" * 60_000 for _ in range(5)),
            schema_mode="observed",
            on_write_failure="discard",
        )


def test_pending_intent_json_and_inspection_facts_are_detached_snapshots() -> None:
    nested = {"items": [{"value": 1}]}
    headers = ["id"]
    urls = ["https://example.invalid/"]
    warnings = ["sample warning"]
    facts = SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={"filename": "input.csv"},
        byte_range_inspected=(0, 10),
        sample_row_count=1,
        observed_headers=headers,  # type: ignore[arg-type]
        inferred_types={"id": "int"},
        url_candidates=urls,  # type: ignore[arg-type]
        warnings=warnings,  # type: ignore[arg-type]
    )
    intent = SourceIntent(
        name="incoming",
        phase="inspection_review",
        plugin="csv",
        options=nested,
        inspection_facts=facts,
        observed_columns=headers,
        sample_rows=({"id": 1},),
    )
    encoded_before = intent.to_dict()

    nested["items"][0]["value"] = 2
    headers.append("mutated")
    urls.append("https://mutated.invalid/")
    warnings.append("mutated")

    assert intent.to_dict() == encoded_before
    assert intent.inspection_facts is not facts
    assert intent.inspection_facts is not None
    assert intent.inspection_facts.observed_headers == ("id",)
    assert intent.inspection_facts.url_candidates == ("https://example.invalid/",)
    assert intent.inspection_facts.warnings == ("sample warning",)


def test_reviewed_anchor_rejects_missing_duplicate_and_unreviewed_order_entries() -> None:
    source = _source("primary", "/data/primary.csv")
    with pytest.raises(InvariantError, match="source_order"):
        guided_reviewed_anchor_hash(
            source_order=(SOURCE_A, SOURCE_A),
            reviewed_sources={SOURCE_A: source},
            output_order=(),
            reviewed_outputs={},
        )
    with pytest.raises(InvariantError, match="source_order"):
        guided_reviewed_anchor_hash(
            source_order=(SOURCE_A,),
            reviewed_sources={},
            output_order=(),
            reviewed_outputs={},
        )
    with pytest.raises(InvariantError, match="reviewed_sources"):
        guided_reviewed_anchor_hash(
            source_order=(),
            reviewed_sources={SOURCE_A: source},
            output_order=(),
            reviewed_outputs={},
        )


def test_source_and_output_edit_targets_resolve_only_reviewed_components() -> None:
    pending_source_session = replace(
        GuidedSession.initial(),
        source_order=(SOURCE_A,),
        pending_source_intents={SOURCE_A: _source_intent()},
    )
    with pytest.raises(InvariantError, match="active_edit_target"):
        replace(pending_source_session, active_edit_target=ComponentTarget(kind="source", stable_id=SOURCE_A))

    pending_output_session = replace(
        GuidedSession.initial(),
        output_order=(OUTPUT_A,),
        pending_output_intents={OUTPUT_A: _sink_intent()},
    )
    with pytest.raises(InvariantError, match="active_edit_target"):
        replace(pending_output_session, active_edit_target=ComponentTarget(kind="output", stable_id=OUTPUT_A))


@pytest.mark.parametrize(
    ("active_kind", "pending_kind", "pending_id", "pending_phase"),
    [
        ("source", "source", SOURCE_B, "inspection_review"),
        ("source", "output", OUTPUT_B, "field_review"),
        ("source", "source", SOURCE_A, "plugin_options"),
        ("output", "output", OUTPUT_B, "field_review"),
        ("output", "source", SOURCE_B, "inspection_review"),
        ("output", "output", OUTPUT_A, "plugin_options"),
    ],
)
def test_active_component_edit_rejects_unrelated_or_wrong_phase_pending_intent(
    active_kind: str,
    pending_kind: str,
    pending_id: str,
    pending_phase: str,
) -> None:
    source = _source("primary", "/data/primary.csv")
    output = _output("archive", "/data/archive.jsonl")
    source_order = [SOURCE_A]
    output_order = [OUTPUT_A]
    pending_sources: dict[str, SourceIntent] = {}
    pending_outputs: dict[str, SinkIntent] = {}
    if pending_kind == "source":
        if pending_id not in source_order:
            source_order.append(pending_id)
        pending_sources[pending_id] = SourceIntent(
            name="primary" if pending_id == SOURCE_A else "secondary",
            phase=pending_phase,  # type: ignore[arg-type]
            plugin="csv",
            options=None if pending_phase == "plugin_options" else {"path": "/data/pending.csv"},
            inspection_facts=None if pending_phase == "plugin_options" else _inspection(),
            observed_columns=(),
            sample_rows=(),
        )
    else:
        if pending_id not in output_order:
            output_order.append(pending_id)
        pending_outputs[pending_id] = SinkIntent(
            name="archive" if pending_id == OUTPUT_A else "errors",
            phase=pending_phase,  # type: ignore[arg-type]
            plugin="json",
            options=None if pending_phase == "plugin_options" else {"path": "/data/pending.jsonl"},
        )

    with pytest.raises(InvariantError, match=r"active .* edit"):
        GuidedSession(
            step=GuidedStep.STEP_1_SOURCE if active_kind == "source" else GuidedStep.STEP_2_SINK,
            source_order=tuple(source_order),
            reviewed_sources={SOURCE_A: source},
            pending_source_intents=pending_sources,
            output_order=tuple(output_order),
            reviewed_outputs={OUTPUT_A: output},
            pending_output_intents=pending_outputs,
            active_edit_target=ComponentTarget(kind=active_kind, stable_id=SOURCE_A if active_kind == "source" else OUTPUT_A),
        )


def test_legal_active_component_edit_overlaps_roundtrip_exactly() -> None:
    source = _source("primary", "/data/primary.csv")
    source_edit = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_A,),
        reviewed_sources={SOURCE_A: source},
        pending_source_intents={
            SOURCE_A: SourceIntent(
                name=source.name,
                phase="inspection_review",
                plugin=source.plugin,
                options={"path": "/data/revised.csv", "on_validation_failure": "discard"},
                inspection_facts=_inspection(),
                observed_columns=("id", "name"),
                sample_rows=(),
            )
        },
        active_edit_target=ComponentTarget(kind="source", stable_id=SOURCE_A),
    )
    output = _output("archive", "/data/archive.jsonl")
    output_edit = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        output_order=(OUTPUT_A,),
        reviewed_outputs={OUTPUT_A: output},
        pending_output_intents={
            OUTPUT_A: SinkIntent(
                name=output.name,
                phase="field_review",
                plugin=output.plugin,
                options={"path": "/data/revised.jsonl", "on_write_failure": "discard"},
            )
        },
        active_edit_target=ComponentTarget(kind="output", stable_id=OUTPUT_A),
    )

    for legal_overlap in (source_edit, output_edit):
        restored = GuidedSession.from_dict(legal_overlap.to_dict())
        assert restored == legal_overlap
        assert restored.to_dict() == legal_overlap.to_dict()


def test_guided_component_deferred_constraint_and_prose_collections_are_bounded() -> None:
    with pytest.raises(InvariantError, match="source components"):
        replace(GuidedSession.initial(), source_order=tuple(object() for _ in range(257)))  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match="output components"):
        replace(GuidedSession.initial(), output_order=tuple(object() for _ in range(257)))  # type: ignore[arg-type]

    source_ids = tuple(str(UUID(int=index + 1)) for index in range(257))
    pending_sources = {
        stable_id: SourceIntent(
            name=f"source-{index}",
            phase="plugin_selection",
            plugin=None,
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        )
        for index, stable_id in enumerate(source_ids)
    }
    with pytest.raises(InvariantError, match="source components"):
        replace(GuidedSession.initial(), source_order=source_ids, pending_source_intents=pending_sources)

    output_ids = tuple(str(UUID(int=index + 1_000)) for index in range(257))
    pending_outputs = {
        stable_id: SinkIntent(name=f"output-{index}", phase="plugin_selection", plugin=None, options=None)
        for index, stable_id in enumerate(output_ids)
    }
    with pytest.raises(InvariantError, match="output components"):
        replace(GuidedSession.initial(), output_order=output_ids, pending_output_intents=pending_outputs)

    with pytest.raises(InvariantError, match="deferred_intents"):
        replace(GuidedSession.initial(), deferred_intents=(_deferred(),) * 257)

    with pytest.raises(InvariantError, match="constraints"):
        replace(
            _deferred(),
            constraints=tuple(
                SubjectPresenceConstraint(
                    kind="subject_presence",
                    subject=StableSubject(kind="stable", component_kind="source", stable_id=SOURCE_A),
                    present=True,
                )
                for _ in range(65)
            ),
        )

    with pytest.raises(InvariantError, match="redacted_summary"):
        DeferredStageIntent.create(
            intent_id=INTENT_A,
            receiving_stage="source",
            target_stage="output",
            catalog_kind=None,
            catalog_name=None,
            redacted_summary="x" * 4097,
            originating_message_id=MESSAGE_A,
            message_content_hash=HASH_A,
            constraints=(),
        )


def test_guided_history_and_chat_counts_and_aggregate_prose_are_bounded() -> None:
    history_record = TurnRecord(
        step=GuidedStep.STEP_1_SOURCE,
        turn_type=TurnType.SINGLE_SELECT,
        payload_hash=HASH_A,
        response_hash=None,
        emitter="server",
        summary="x" * 4_096,
    )
    with pytest.raises(InvariantError, match=r"history.*record"):
        replace(GuidedSession.initial(), history=(history_record,) * 4_097)
    with pytest.raises(InvariantError, match=r"history.*aggregate"):
        replace(GuidedSession.initial(), history=(history_record,) * 257)

    chat_turn = ChatTurn(
        role=ChatRole.USER,
        content="x",
        seq=0,
        step=GuidedStep.STEP_1_SOURCE,
        ts_iso="2026-07-18T00:00:00Z",
    )
    with pytest.raises(InvariantError, match=r"chat_history.*turn"):
        replace(GuidedSession.initial(), chat_history=(chat_turn,) * 4_097, chat_turn_seq=4_097)

    chat_history = tuple(replace(chat_turn, seq=index, content="x" * 65_536) for index in range(65))
    with pytest.raises(InvariantError, match=r"chat_history.*aggregate"):
        replace(GuidedSession.initial(), chat_history=chat_history, chat_turn_seq=65)


def test_from_dict_rejects_oversized_collections_before_child_decoders(monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_decode(*_args: object, **_kwargs: object) -> object:
        pytest.fail("child decoder ran before collection bound")

    monkeypatch.setattr(TurnRecord, "from_dict", classmethod(_must_not_decode))
    encoded = GuidedSession.initial().to_dict()
    encoded["history"] = [{}] * 4_097
    with pytest.raises(InvariantError, match=r"history.*record"):
        GuidedSession.from_dict(encoded)

    monkeypatch.setattr(state_machine, "_chat_turn_from_guided_dict", _must_not_decode)
    encoded = GuidedSession.initial().to_dict()
    encoded["chat_history"] = [{}] * 4_097
    with pytest.raises(InvariantError, match=r"chat_history.*turn"):
        GuidedSession.from_dict(encoded)

    encoded = GuidedSession.initial().to_dict()
    encoded["source_order"] = [None] * 257
    with pytest.raises(InvariantError, match="source components"):
        GuidedSession.from_dict(encoded)

    encoded = GuidedSession.initial().to_dict()
    encoded["output_order"] = [None] * 257
    with pytest.raises(InvariantError, match="output components"):
        GuidedSession.from_dict(encoded)

    for field_name, decoder_owner in (
        ("reviewed_sources", SourceResolved),
        ("pending_source_intents", SourceIntent),
        ("reviewed_outputs", SinkOutputResolved),
        ("pending_output_intents", SinkIntent),
    ):
        monkeypatch.setattr(decoder_owner, "from_dict", classmethod(_must_not_decode))
        encoded = GuidedSession.initial().to_dict()
        encoded[field_name] = {str(UUID(int=index + 1)): {} for index in range(257)}
        with pytest.raises(InvariantError, match="components"):
            GuidedSession.from_dict(encoded)

    monkeypatch.setattr(DeferredStageIntent, "from_dict", classmethod(_must_not_decode))
    encoded = GuidedSession.initial().to_dict()
    encoded["deferred_intents"] = [{}] * 257
    with pytest.raises(InvariantError, match="deferred_intents"):
        GuidedSession.from_dict(encoded)


def test_from_dict_rejects_aggregate_prose_before_child_decoders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TurnRecord, "from_dict", classmethod(lambda *_args: pytest.fail("history child decoder ran")))
    encoded = GuidedSession.initial().to_dict()
    history_record = TurnRecord(
        step=GuidedStep.STEP_1_SOURCE,
        turn_type=TurnType.SINGLE_SELECT,
        payload_hash=HASH_A,
        response_hash=None,
        emitter="server",
        summary="x" * 4_096,
    ).to_dict()
    encoded["history"] = [history_record] * 257
    with pytest.raises(InvariantError, match=r"history.*aggregate"):
        GuidedSession.from_dict(encoded)

    monkeypatch.setattr(
        state_machine,
        "_chat_turn_from_guided_dict",
        lambda *_args: pytest.fail("chat child decoder ran"),
    )
    encoded = GuidedSession.initial().to_dict()
    encoded["chat_history"] = [
        {
            "role": "user",
            "content": "x" * 65_536,
            "seq": index,
            "step": "step_1_source",
            "ts_iso": "2026-07-18T00:00:00Z",
            "assistant_message_kind": None,
            "synthetic_failure_reason": None,
        }
        for index in range(65)
    ]
    encoded["chat_turn_seq"] = 65
    with pytest.raises(InvariantError, match=r"chat_history.*aggregate"):
        GuidedSession.from_dict(encoded)


def test_from_dict_rejects_total_deferred_constraints_before_child_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        DeferredStageIntent,
        "from_dict",
        classmethod(lambda *_args: pytest.fail("deferred child decoder ran")),
    )
    encoded = GuidedSession.initial().to_dict()
    deferred = _deferred().to_dict()
    constraint = deferred["constraints"][0]
    encoded["deferred_intents"] = [
        {**deferred, "intent_id": str(UUID(int=index + 1)), "constraints": [constraint] * 64} for index in range(65)
    ]

    with pytest.raises(InvariantError, match="deferred constraints"):
        GuidedSession.from_dict(encoded)


def test_deferred_from_dict_rejects_constraint_count_before_child_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_subjects, "constraint_from_dict", lambda _value: pytest.fail("constraint decoder ran"))
    encoded = _deferred().to_dict()
    encoded["constraints"] = [{}] * 65

    with pytest.raises(InvariantError, match="constraints"):
        DeferredStageIntent.from_dict(encoded)


def test_reviewed_components_have_no_name_or_failure_compatibility_defaults() -> None:
    with pytest.raises(TypeError, match=r"name.*on_validation_failure|on_validation_failure.*name"):
        SourceResolved(  # type: ignore[call-arg]
            plugin="csv",
            options={},
            observed_columns=(),
            sample_rows=(),
        )
    with pytest.raises(TypeError, match=r"name.*on_write_failure|on_write_failure.*name"):
        SinkOutputResolved(  # type: ignore[call-arg]
            plugin="json",
            options={},
            required_fields=(),
            schema_mode="observed",
        )


def test_constraint_subjects_are_semantically_compatible_with_constraint_kind() -> None:
    stable_source = StableSubject(kind="stable", component_kind="source", stable_id=SOURCE_A)
    stable_node = StableSubject(kind="stable", component_kind="node", stable_id=NODE_A)
    stable_edge = StableSubject(kind="stable", component_kind="edge", stable_id=NODE_B)
    stable_output = StableSubject(kind="stable", component_kind="output", stable_id=OUTPUT_A)

    with pytest.raises(InvariantError, match="plugin_kind"):
        ComponentCountConstraint(
            kind="component_count",
            component_kind="node",
            plugin_kind="source",
            plugin_name="csv",
            operator="equals",
            count=1,
        )
    with pytest.raises(InvariantError, match="edge"):
        OptionValueConstraint(kind="option_value", subject=stable_edge, option_path=("x",), operator="equals", value=1)
    with pytest.raises(InvariantError, match="from_subject"):
        EdgeRouteConstraint(kind="edge_route", from_subject=stable_output, edge_type="on_success", to_subject=stable_node, present=True)
    with pytest.raises(InvariantError, match="to_subject"):
        EdgeRouteConstraint(kind="edge_route", from_subject=stable_node, edge_type="on_success", to_subject=stable_source, present=True)
    with pytest.raises(InvariantError, match="failure_kind"):
        FailureRouteConstraint(
            kind="failure_route",
            subject=stable_node,
            failure_kind="source_validation",
            operator="equals",
            target="discard",
        )
    with pytest.raises(InvariantError, match="target"):
        FailureRouteConstraint(
            kind="failure_route",
            subject=stable_source,
            failure_kind="source_validation",
            operator="equals",
            target=stable_node,
        )


def test_chat_history_sequence_is_unique_increasing_and_not_ahead_of_counter() -> None:
    first = ChatTurn(
        role=ChatRole.USER,
        content="first",
        seq=1,
        step=GuidedStep.STEP_1_SOURCE,
        ts_iso="2026-07-18T00:00:00Z",
    )
    duplicate = replace(
        first,
        role=ChatRole.ASSISTANT,
        content="duplicate",
        assistant_message_kind="assistant",
    )
    earlier = replace(
        first,
        role=ChatRole.ASSISTANT,
        content="earlier",
        seq=0,
        assistant_message_kind="assistant",
    )

    with pytest.raises(InvariantError, match="strictly increasing"):
        replace(GuidedSession.initial(), chat_history=(first, duplicate), chat_turn_seq=1)
    with pytest.raises(InvariantError, match="strictly increasing"):
        replace(GuidedSession.initial(), chat_history=(first, earlier), chat_turn_seq=1)
    with pytest.raises(InvariantError, match="chat_turn_seq"):
        replace(GuidedSession.initial(), chat_history=(first,), chat_turn_seq=0)
    with pytest.raises(InvariantError, match="chat_turn_seq"):
        replace(GuidedSession.initial(), chat_history=(first,), chat_turn_seq=1)
    with pytest.raises(InvariantError, match="chat_turn_seq"):
        replace(GuidedSession.initial(), chat_history=(first,), chat_turn_seq=3)
    with pytest.raises(InvariantError, match="chat_turn_seq"):
        replace(GuidedSession.initial(), chat_turn_seq=1)


def test_proposal_ref_rejects_self_supersession() -> None:
    proposal = _proposal(source_order=(), reviewed_sources={}, output_order=(), reviewed_outputs={}, covered=())
    with pytest.raises(InvariantError, match="self"):
        replace(
            proposal,
            supersedes_proposal_id=proposal.proposal_id,
            supersedes_draft_hash=HASH_C,
        )
