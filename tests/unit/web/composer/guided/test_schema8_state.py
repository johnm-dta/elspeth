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

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.state_machine import (
    GUIDED_SESSION_SCHEMA_VERSION,
    ComponentCountConstraint,
    ComponentTarget,
    DeferredStageIntent,
    EdgeRouteConstraint,
    FailureRouteConstraint,
    GuidedProposalRef,
    GuidedSession,
    OptionValueConstraint,
    PluginSubject,
    SinkIntent,
    SourceIntent,
    StableSubject,
    SubjectPresenceConstraint,
    TerminalKind,
    TerminalReason,
    TerminalState,
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
        history=(),
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

    assert GUIDED_SESSION_SCHEMA_VERSION == 8
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


@pytest.mark.parametrize("bad_version", [7, 9, "8", 8.0, True])
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
        TerminalState(kind=TerminalKind.COMPLETED, reason=TerminalReason.SOLVER_EXHAUSTED, pipeline_yaml="pipeline: {}\n")
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
