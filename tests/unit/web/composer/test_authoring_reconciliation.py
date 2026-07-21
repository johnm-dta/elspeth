"""Exact set-pipeline serialization and authoritative review reconciliation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.hashing import stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.redaction import SetPipelineArgumentsModel
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.composer.tools import ToolContext
from elspeth.web.composer.tools.sessions import _execute_get_pipeline_state, _execute_set_pipeline
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PROMPT_SHIELD_USER_TERM,
    PROMPT_TEMPLATE_PARTS_KEY,
    SOURCE_AUTHORING_KEY,
    WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
    model_choice_artifact_hash,
    pipeline_decision_artifact_hash,
    reconcile_authoritative_reviews,
)
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot


def _requirement(
    *,
    requirement_id: str,
    kind: InterpretationKind,
    user_term: str,
    status: str = "pending",
    draft: str = "review this",
    accepted_value: str | None = None,
    accepted_artifact_hash: str | None = None,
    resolved_prompt_template_hash: str | None = None,
) -> dict[str, object]:
    return {
        "id": requirement_id,
        "kind": kind.value,
        "user_term": user_term,
        "status": status,
        "draft": draft,
        "event_id": "event-1" if status == "resolved" else None,
        "accepted_value": accepted_value,
        "accepted_artifact_hash": accepted_artifact_hash,
        "resolved_prompt_template_hash": resolved_prompt_template_hash,
    }


def _node(
    *,
    node_id: str = "model",
    plugin: str = "llm",
    options: dict[str, object] | None = None,
    input_name: str = "in",
    output_name: str = "out",
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin=plugin,
        input=input_name,
        on_success=output_name,
        on_error="discard",
        options=options or {"schema": {"mode": "observed"}},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _state(*, nodes: tuple[NodeSpec, ...], source_options: dict[str, object] | None = None) -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success=nodes[0].input or "in",
            options=source_options or {"path": "rows.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=nodes,
        edges=(),
        outputs=(
            OutputSpec(
                name=nodes[-1].on_success or "out",
                plugin="json",
                options={
                    "path": "out.jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="reviewed"),
        version=3,
    )


def _exact_arguments(state: CompositionState):
    return _execute_get_pipeline_state(
        {"component": "set_pipeline_arguments"},
        state,
        MagicMock(spec=ToolContext),
    )


def _trained_context() -> ToolContext:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return ToolContext(
        catalog=PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
    )


def test_exact_authoring_payload_validates_and_omits_server_owned_review_fields() -> None:
    model = "bedrock/example.model"
    resolved = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        status="resolved",
        draft=model,
        accepted_value=model,
        resolved_prompt_template_hash=model_choice_artifact_hash(model),
    )
    state = _state(
        nodes=(
            _node(
                options={
                    "model": model,
                    "resolved_prompt_template_hash": "server-owned-node-hash",
                    INTERPRETATION_REQUIREMENTS_KEY: [resolved],
                    "schema": {"mode": "observed"},
                }
            ),
        ),
    )

    result = _exact_arguments(state)

    assert result.success
    assert set(result.data) == {"source", "nodes", "edges", "outputs", "metadata"}
    SetPipelineArgumentsModel.model_validate(result.data)
    assert "version" not in result.data
    assert "inspection" not in result.data
    options = result.data["nodes"][0]["options"]
    assert "resolved_prompt_template_hash" not in options
    shell = options[INTERPRETATION_REQUIREMENTS_KEY][0]
    assert shell == {
        "id": "model_choice_review:model",
        "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
        "user_term": "llm_model_choice:model",
        "status": "pending",
        "draft": model,
    }


def test_default_blob_source_round_trips_as_blob_id_without_storage_metadata() -> None:
    blob_id = "2e9e41eb-e34d-4918-b334-3c1e9ee0f8ff"
    source_options = {
        "path": "/data/blobs/session/input.csv",
        "blob_ref": blob_id,
        "mode": "bind_source",
        SOURCE_AUTHORING_KEY: {
            "modality": "composer_generated",
            "content_hash": "a" * 64,
            "review_event_id": "event-source",
            "resolved_kind": InterpretationKind.INVENTED_SOURCE.value,
        },
        "schema": {"mode": "observed"},
    }
    result = _exact_arguments(_state(nodes=(_node(plugin="passthrough"),), source_options=source_options))

    assert result.success
    assert result.data["source"]["blob_id"] == blob_id
    assert not {"path", "blob_ref", "mode", SOURCE_AUTHORING_KEY} & set(result.data["source"]["options"])


def test_named_blob_source_reports_round_trip_unavailable() -> None:
    state = CompositionState(
        sources={
            "left": SourceSpec(
                plugin="csv",
                on_success="in",
                options={"blob_ref": "2e9e41eb-e34d-4918-b334-3c1e9ee0f8ff", "path": "/data/blobs/left.csv"},
                on_validation_failure="discard",
            )
        },
        nodes=(_node(plugin="passthrough"),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    result = _exact_arguments(state)

    assert not result.success
    assert result.data["error_code"] == "round_trip_unavailable"
    assert result.updated_state is state


@pytest.mark.parametrize(
    "source_options",
    [
        {
            "path": "/data/blobs/session/input.csv",
            "blob_ref": None,
            "mode": "bind_source",
            "schema": {"mode": "observed"},
        },
        {
            "path": "/data/blobs/session/input.csv",
            "blob_ref": {"id": "2e9e41eb-e34d-4918-b334-3c1e9ee0f8ff"},
            "schema": {"mode": "observed"},
        },
    ],
    ids=("missing-identity", "widened-identity"),
)
def test_unsafe_blob_identity_reports_round_trip_unavailable(source_options: dict[str, object]) -> None:
    state = _state(nodes=(_node(plugin="passthrough"),), source_options=source_options)

    result = _exact_arguments(state)

    assert not result.success
    assert result.data["error_code"] == "round_trip_unavailable"
    assert result.updated_state is state


def test_legacy_resolved_vague_term_without_parts_reports_round_trip_unavailable() -> None:
    prompt = "Tone: warm"
    resolved = _requirement(
        requirement_id="vague:tone",
        kind=InterpretationKind.VAGUE_TERM,
        user_term="friendly",
        status="resolved",
        draft="friendly",
        accepted_value="warm",
        resolved_prompt_template_hash=stable_hash(prompt),
    )
    state = _state(
        nodes=(
            _node(
                options={
                    "prompt_template": prompt,
                    INTERPRETATION_REQUIREMENTS_KEY: [resolved],
                }
            ),
        )
    )

    result = _exact_arguments(state)

    assert not result.success
    assert result.data["error_code"] == "round_trip_unavailable"
    assert "warm" not in result.data["error"]
    assert result.updated_state is state


def test_model_choice_review_is_preserved_for_unrelated_drift_and_reopened_for_model_drift() -> None:
    model = "bedrock/example.model"
    resolved = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        status="resolved",
        draft=model,
        accepted_value=model,
        resolved_prompt_template_hash=model_choice_artifact_hash(model),
    )
    previous = _state(nodes=(_node(options={"model": model, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),))
    pending = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        draft=model,
    )
    unrelated = replace(
        _state(nodes=(_node(options={"model": model, INTERPRETATION_REQUIREMENTS_KEY: [pending]}),)),
        metadata=PipelineMetadata(name="unrelated metadata edit"),
    )

    reconciled = reconcile_authoritative_reviews(previous, unrelated)
    carried = reconciled.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0]
    assert carried["status"] == "resolved"
    assert carried["event_id"] == "event-1"

    changed = _state(
        nodes=(
            _node(
                options={
                    "model": "bedrock/different.model",
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {**pending, "draft": "bedrock/different.model"},
                    ],
                }
            ),
        )
    )
    reopened = reconcile_authoritative_reviews(previous, changed)
    assert reopened.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0]["status"] == "pending"


def test_prompt_template_review_is_preserved_only_for_the_same_prompt_artifact() -> None:
    prompt = "Summarise {{ row.text }}"
    resolved = _requirement(
        requirement_id="prompt_template_review:model",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        user_term="llm_prompt_template:model",
        status="resolved",
        draft=prompt,
        accepted_value=prompt,
        resolved_prompt_template_hash=stable_hash(prompt),
    )
    previous = _state(nodes=(_node(options={"prompt_template": prompt, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),))
    shell = _requirement(
        requirement_id="prompt_template_review:model",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        user_term="llm_prompt_template:model",
        draft=prompt,
    )
    unchanged = _state(nodes=(_node(options={"prompt_template": prompt, INTERPRETATION_REQUIREMENTS_KEY: [shell]}),))
    changed = _state(
        nodes=(
            _node(
                options={
                    "prompt_template": "Classify {{ row.text }}",
                    INTERPRETATION_REQUIREMENTS_KEY: [{**shell, "draft": "Classify {{ row.text }}"}],
                }
            ),
        )
    )

    assert reconcile_authoritative_reviews(previous, unchanged).nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0]["status"] == "resolved"
    assert reconcile_authoritative_reviews(previous, changed).nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0]["status"] == "pending"


def test_pipeline_decision_review_uses_kind_and_user_term_hash_authority() -> None:
    options = {
        "url_field": "url",
        "content_field": "page_text",
        "http": {
            "abuse_contact": "review@example.com",
            "scraping_reason": "authoritative reconciliation test",
        },
        "schema": {"mode": "observed"},
    }
    node = _node(node_id="scrape", plugin="web_scrape", options=options)
    artifact_hash = pipeline_decision_artifact_hash(
        node,
        (node,),
        user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
    )
    resolved = _requirement(
        requirement_id="http-identity:scrape",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
        status="resolved",
        accepted_value="approved",
        accepted_artifact_hash=artifact_hash,
    )
    previous = _state(nodes=(replace(node, options={**options, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),))
    shell = _requirement(
        requirement_id="http-identity:scrape",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
    )
    proposed = _state(nodes=(replace(node, options={**options, INTERPRETATION_REQUIREMENTS_KEY: [shell]}),))

    reconciled = reconcile_authoritative_reviews(previous, proposed)

    assert reconciled.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0]["status"] == "resolved"


def test_exact_payload_round_trips_through_real_set_pipeline_with_authoritative_review() -> None:
    options = {
        "url_field": "url",
        "content_field": "page_text",
        "fingerprint_field": "page_fingerprint",
        "format": "text",
        "http": {
            "abuse_contact": "review@foundryside.dev",
            "scraping_reason": "exact authoring reconciliation test",
        },
        "schema": {"mode": "observed"},
    }
    node = _node(node_id="scrape", plugin="web_scrape", options=options)
    resolved = _requirement(
        requirement_id="http-identity:scrape",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
        status="resolved",
        accepted_value="approved",
        accepted_artifact_hash=pipeline_decision_artifact_hash(
            node,
            (node,),
            user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
        ),
    )
    previous = _state(nodes=(replace(node, options={**options, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),))
    exact = _exact_arguments(previous)

    result = _execute_set_pipeline(exact.data, previous, _trained_context())

    assert result.success, result.data
    carried = result.updated_state.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0]
    assert carried["status"] == "resolved"
    assert carried["event_id"] == "event-1"


def test_prompt_shield_recommendation_is_removed_when_effective_shield_is_inserted() -> None:
    llm = _node(
        options={"prompt_template": "Summarise {{ row.text }}", "schema": {"mode": "observed"}},
        input_name="llm_in",
    )
    artifact_hash = pipeline_decision_artifact_hash(llm, (llm,), user_term=PROMPT_SHIELD_USER_TERM)
    resolved = _requirement(
        requirement_id="prompt-shield:model",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=PROMPT_SHIELD_USER_TERM,
        status="resolved",
        draft="Insert a prompt shield",
        accepted_value="continue without shield",
        accepted_artifact_hash=artifact_hash,
    )
    previous = _state(nodes=(replace(llm, options={**llm.options, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),))
    shell = _requirement(
        requirement_id="prompt-shield:model",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=PROMPT_SHIELD_USER_TERM,
        draft="Insert a prompt shield",
    )
    shield = _node(
        node_id="shield",
        plugin="aws_bedrock_prompt_shield",
        input_name="llm_in",
        output_name="shielded",
        options={
            "guardrail_identifier": "gr-123456",
            "guardrail_version": "1",
            "region": "ap-southeast-1",
            "fields": ["text"],
            "schema": {"mode": "observed"},
        },
    )
    proposed_llm = replace(
        llm,
        input="shielded",
        options={**llm.options, INTERPRETATION_REQUIREMENTS_KEY: [shell]},
    )
    proposed = _state(nodes=(shield, proposed_llm))

    reconciled = reconcile_authoritative_reviews(previous, proposed)

    assert INTERPRETATION_REQUIREMENTS_KEY not in reconciled.nodes[1].options


def test_resolved_vague_term_rehydrates_only_with_same_parts_and_renders_from_them() -> None:
    resolved = _requirement(
        requirement_id="vague:tone",
        kind=InterpretationKind.VAGUE_TERM,
        user_term="friendly",
        status="resolved",
        draft="friendly",
        accepted_value="warm",
        resolved_prompt_template_hash=stable_hash("warm"),
    )
    parts = [
        {"kind": "text", "text": "Tone: "},
        {"kind": "interpretation_ref", "requirement_id": "vague:tone"},
    ]
    previous = _state(
        nodes=(
            _node(
                options={
                    "prompt_template": "Tone: warm",
                    PROMPT_TEMPLATE_PARTS_KEY: parts,
                    INTERPRETATION_REQUIREMENTS_KEY: [resolved],
                }
            ),
        )
    )
    shell = _requirement(
        requirement_id="vague:tone",
        kind=InterpretationKind.VAGUE_TERM,
        user_term="friendly",
        draft="friendly",
    )
    proposed = _state(
        nodes=(
            _node(
                options={
                    "prompt_template": "Tone: pending interpretation",
                    PROMPT_TEMPLATE_PARTS_KEY: parts,
                    INTERPRETATION_REQUIREMENTS_KEY: [shell],
                }
            ),
        )
    )

    exact = _exact_arguments(previous)
    exact_options = exact.data["nodes"][0]["options"]
    assert exact_options["prompt_template"] == "Tone: pending interpretation"
    assert "warm" not in repr(exact_options)

    reconciled = reconcile_authoritative_reviews(previous, proposed)

    options = reconciled.nodes[0].options
    assert options["prompt_template"] == "Tone: warm"
    assert options[INTERPRETATION_REQUIREMENTS_KEY][0]["status"] == "resolved"


def _two_vague_term_fixtures(
    *,
    tone_hash: str,
    audience_hash: str,
    previous_prompt: str = "Tone: warm, audience: new users",
) -> tuple[CompositionState, CompositionState]:
    parts = [
        {"kind": "text", "text": "Tone: "},
        {"kind": "interpretation_ref", "requirement_id": "vague:tone"},
        {"kind": "text", "text": ", audience: "},
        {"kind": "interpretation_ref", "requirement_id": "vague:audience"},
    ]
    tone = _requirement(
        requirement_id="vague:tone",
        kind=InterpretationKind.VAGUE_TERM,
        user_term="friendly",
        status="resolved",
        draft="friendly",
        accepted_value="warm",
        resolved_prompt_template_hash=tone_hash,
    )
    audience = _requirement(
        requirement_id="vague:audience",
        kind=InterpretationKind.VAGUE_TERM,
        user_term="everyone",
        status="resolved",
        draft="everyone",
        accepted_value="new users",
        resolved_prompt_template_hash=audience_hash,
    )
    previous = _state(
        nodes=(
            _node(
                options={
                    "prompt_template": previous_prompt,
                    PROMPT_TEMPLATE_PARTS_KEY: parts,
                    INTERPRETATION_REQUIREMENTS_KEY: [tone, audience],
                }
            ),
        )
    )
    shells = [
        _requirement(
            requirement_id="vague:tone",
            kind=InterpretationKind.VAGUE_TERM,
            user_term="friendly",
            draft="friendly",
        ),
        _requirement(
            requirement_id="vague:audience",
            kind=InterpretationKind.VAGUE_TERM,
            user_term="everyone",
            draft="everyone",
        ),
    ]
    proposed = _state(
        nodes=(
            _node(
                options={
                    "prompt_template": "Tone: pending interpretation, audience: pending interpretation",
                    PROMPT_TEMPLATE_PARTS_KEY: parts,
                    INTERPRETATION_REQUIREMENTS_KEY: shells,
                }
            ),
        )
    )
    return previous, proposed


def test_two_resolved_vague_terms_round_trip_regardless_of_resolution_order() -> None:
    """A prompt with two resolved vague terms must reconcile cleanly.

    Each requirement's ``resolved_prompt_template_hash`` attests its own
    accepted value, so it is invariant under the *sibling* term's later
    resolution. The old semantics (hash of the full rendered prompt at that
    requirement's resolution moment) made the first-resolved requirement's
    hash permanently stale once a second term resolved, failing every
    subsequent set_pipeline / splice_transform with hash-drift.
    """
    previous, proposed = _two_vague_term_fixtures(
        tone_hash=stable_hash("warm"),
        audience_hash=stable_hash("new users"),
    )

    reconciled = reconcile_authoritative_reviews(previous, proposed)

    options = reconciled.nodes[0].options
    assert options["prompt_template"] == "Tone: warm, audience: new users"
    statuses = [requirement["status"] for requirement in options[INTERPRETATION_REQUIREMENTS_KEY]]
    assert statuses == ["resolved", "resolved"]


def test_vague_term_accepted_value_hash_mismatch_fails_closed() -> None:
    previous, proposed = _two_vague_term_fixtures(
        tone_hash=stable_hash("chilly"),
        audience_hash=stable_hash("new users"),
    )

    with pytest.raises(ValueError, match="hash drifted"):
        reconcile_authoritative_reviews(previous, proposed)


def test_vague_term_tampered_previous_prompt_fails_closed() -> None:
    """The stored previous prompt must re-render from its own parts and
    requirements; a prompt edited out from under a resolved vague-term
    review is drift, not a benign round-trip."""
    previous, proposed = _two_vague_term_fixtures(
        tone_hash=stable_hash("warm"),
        audience_hash=stable_hash("new users"),
        previous_prompt="Tone: chilly, audience: new users",
    )

    with pytest.raises(ValueError, match="drifted"):
        reconcile_authoritative_reviews(previous, proposed)


def test_invented_source_review_rehydrates_only_for_the_same_content_identity() -> None:
    content_hash = "a" * 64
    source_authoring = {
        "modality": "llm_generated",
        "content_hash": content_hash,
        "review_event_id": "event-1",
        "resolved_kind": InterpretationKind.INVENTED_SOURCE.value,
    }
    resolved = _requirement(
        requirement_id="source_review:inline_source_data",
        kind=InterpretationKind.INVENTED_SOURCE,
        user_term="inline_source_data",
        status="resolved",
        draft="generated rows",
        accepted_value="approved",
        accepted_artifact_hash=content_hash,
    )
    previous = _state(
        nodes=(_node(plugin="passthrough"),),
        source_options={
            "blob_ref": "2e9e41eb-e34d-4918-b334-3c1e9ee0f8ff",
            "path": "/data/blobs/generated.csv",
            SOURCE_AUTHORING_KEY: source_authoring,
            INTERPRETATION_REQUIREMENTS_KEY: [resolved],
        },
    )
    shell = _requirement(
        requirement_id="source_review:inline_source_data",
        kind=InterpretationKind.INVENTED_SOURCE,
        user_term="inline_source_data",
        draft="generated rows",
    )
    proposed = _state(
        nodes=(_node(plugin="passthrough"),),
        source_options={
            "blob_ref": "2e9e41eb-e34d-4918-b334-3c1e9ee0f8ff",
            "path": "/data/blobs/generated.csv",
            SOURCE_AUTHORING_KEY: {**source_authoring, "review_event_id": None, "resolved_kind": None},
            INTERPRETATION_REQUIREMENTS_KEY: [shell],
        },
    )

    reconciled = reconcile_authoritative_reviews(previous, proposed)

    source_options = reconciled.sources["source"].options
    assert source_options[INTERPRETATION_REQUIREMENTS_KEY][0]["status"] == "resolved"
    assert source_options[SOURCE_AUTHORING_KEY] == source_authoring


def test_stale_accepted_hash_is_rejected_instead_of_silently_reopened() -> None:
    resolved = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        status="resolved",
        draft="bedrock/example.model",
        accepted_value="bedrock/example.model",
        resolved_prompt_template_hash="stale-hash",
    )
    previous = _state(
        nodes=(
            _node(
                options={
                    "model": "bedrock/example.model",
                    INTERPRETATION_REQUIREMENTS_KEY: [resolved],
                }
            ),
        )
    )
    shell = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        draft="bedrock/example.model",
    )
    proposed = _state(
        nodes=(
            _node(
                options={
                    "model": "bedrock/example.model",
                    INTERPRETATION_REQUIREMENTS_KEY: [shell],
                }
            ),
        )
    )

    with pytest.raises(ValueError, match="hash drifted"):
        reconcile_authoritative_reviews(previous, proposed)


def test_duplicate_review_identity_and_incoherent_resolved_metadata_fail_closed() -> None:
    model = "bedrock/example.model"
    resolved = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        status="resolved",
        draft=model,
        accepted_value=model,
        resolved_prompt_template_hash=model_choice_artifact_hash(model),
    )
    shell = _requirement(
        requirement_id="model_choice_review:model",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term="llm_model_choice:model",
        draft=model,
    )
    previous = _state(nodes=(_node(options={"model": model, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),))
    duplicate = _state(nodes=(_node(options={"model": model, INTERPRETATION_REQUIREMENTS_KEY: [shell, shell]}),))

    with pytest.raises(ValueError, match="duplicate interpretation requirement identity"):
        reconcile_authoritative_reviews(previous, duplicate)

    incoherent = {**resolved, "event_id": None}
    previous_incoherent = _state(nodes=(_node(options={"model": model, INTERPRETATION_REQUIREMENTS_KEY: [incoherent]}),))
    proposed = _state(nodes=(_node(options={"model": model, INTERPRETATION_REQUIREMENTS_KEY: [shell]}),))

    with pytest.raises(ValueError, match="has no event_id"):
        reconcile_authoritative_reviews(previous_incoherent, proposed)


def test_unknown_pipeline_decision_user_term_fails_closed() -> None:
    bad = _requirement(
        requirement_id="bad-decision",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term="unknown-decision",
        status="resolved",
        accepted_value="approved",
        accepted_artifact_hash="b" * 64,
    )
    previous = _state(
        nodes=(
            _node(
                plugin="passthrough",
                options={"schema": {"mode": "observed"}, INTERPRETATION_REQUIREMENTS_KEY: [bad]},
            ),
        )
    )
    shell = _requirement(
        requirement_id="bad-decision",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term="unknown-decision",
    )
    proposed = _state(
        nodes=(
            _node(
                plugin="passthrough",
                options={"schema": {"mode": "observed"}, INTERPRETATION_REQUIREMENTS_KEY: [shell]},
            ),
        )
    )

    with pytest.raises(ValueError, match="not a registered decision kind"):
        reconcile_authoritative_reviews(previous, proposed)

    result = _execute_set_pipeline(_exact_arguments(previous).data, previous, _trained_context())
    assert not result.success
    assert result.updated_state is previous
    assert result.updated_state.version == previous.version
    assert result.data["error_code"] == "review_reconciliation_failed"
    assert "unknown-decision" not in result.data["error"]
