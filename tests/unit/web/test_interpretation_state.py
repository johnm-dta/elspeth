"""Tests for structured composer interpretation-review authoring state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata, SourceSpec
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PROMPT_SHIELD_USER_TERM,
    PROMPT_TEMPLATE_PARTS_KEY,
    RAW_HTML_CLEANUP_USER_TERM,
    SOURCE_AUTHORING_KEY,
    InterpretationReviewPending,
    interpretation_sites,
    materialize_state_for_authoring,
    materialize_state_for_execution,
    pipeline_decision_artifact_hash,
    prompt_shield_recommendation_warning_pairs,
    prompt_structure_hash_from_options,
    raw_html_cleanup_review_contract_error,
    strip_authoring_options,
    vague_term_wiring_count,
)


def _state_with_llm(options: dict[str, object]) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="rate_coolness",
                node_type="transform",
                plugin="llm",
                input="source",
                on_success="output",
                on_error="stop",
                options=options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_cleanup_node(options: dict[str, object]) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="drop_raw_html",
                node_type="transform",
                plugin="field_mapper",
                input="scored_rows",
                on_success="clean_rows",
                on_error="stop",
                options=options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_web_scrape_cleanup_node(options: dict[str, object]) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="rows",
                on_success="scraped_rows",
                on_error="stop",
                options={
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "content_fingerprint",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="drop_raw_html",
                node_type="transform",
                plugin="field_mapper",
                input="coloured_rows",
                on_success="clean_rows",
                on_error="stop",
                options=options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_web_scrape_gate_to_llm() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="rows",
                on_success="scraped_rows",
                on_error="stop",
                options={"url_field": "url", "content_field": "content"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="interesting_pages",
                node_type="gate",
                plugin=None,
                input="scraped_rows",
                on_success=None,
                on_error=None,
                options={},
                condition="row['interesting'] == true",
                routes={"true": "llm_input", "false": "discard"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="summarise_pages",
                node_type="transform",
                plugin="llm",
                input="llm_input",
                on_success="summaries",
                on_error="stop",
                options={"prompt_template": "Summarise {{ row.content }}."},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_web_scrape_gate_shield_to_llm() -> CompositionState:
    state = _state_with_web_scrape_gate_to_llm()
    shield = NodeSpec(
        id="shield_pages",
        node_type="transform",
        plugin="azure_prompt_shield",
        input="llm_input",
        on_success="shielded_rows",
        on_error="stop",
        options={},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    llm = replace(state.nodes[2], input="shielded_rows")
    return replace(state, nodes=(state.nodes[0], state.nodes[1], shield, llm))


def _pipeline_decision_options(*, status: str = "pending", artifact_hash: str | None = None) -> dict[str, object]:
    options: dict[str, object] = {
        "mapping": {
            "url": "url",
            "agency": "agency",
            "primary_colours": "primary_colours",
        },
        "select_only": True,
    }
    options[INTERPRETATION_REQUIREMENTS_KEY] = [
        {
            "id": "drop_raw_html_review",
            "kind": "pipeline_decision",
            "user_term": "drop_raw_html_fields",
            "status": status,
            "draft": "Drop the scraped raw HTML and fingerprint fields before saving the JSON output.",
            "event_id": "event-raw-html-drop" if status == "resolved" else None,
            "accepted_value": (
                "Drop the scraped raw HTML and fingerprint fields before saving the JSON output." if status == "resolved" else None
            ),
            "accepted_artifact_hash": artifact_hash,
            "resolved_prompt_template_hash": None,
        }
    ]
    return options


def _pending_options() -> dict[str, object]:
    return {
        "prompt_template": "Rate pending interpretation: {{ row.text }}",
        PROMPT_TEMPLATE_PARTS_KEY: [
            {"kind": "text", "text": "Rate "},
            {"kind": "interpretation_ref", "requirement_id": "coolness"},
            {"kind": "text", "text": ": {{ row.text }}"},
        ],
        INTERPRETATION_REQUIREMENTS_KEY: [
            {
                "id": "coolness",
                "kind": "vague_term",
                "user_term": "coolness",
                "status": "pending",
                "draft": "well-designed and useful",
                "event_id": "event-1",
                "accepted_value": None,
                "accepted_artifact_hash": None,
                "resolved_prompt_template_hash": None,
            }
        ],
    }


def test_legacy_placeholder_materializes_for_authoring_without_mutating_source() -> None:
    state = _state_with_llm({"prompt_template": "Rate {{interpretation:coolness}}: {{ row.text }}"})

    authoring = materialize_state_for_authoring(state)

    assert authoring.nodes[0].options["prompt_template"] == "Rate pending interpretation: {{ row.text }}"
    assert state.nodes[0].options["prompt_template"] == "Rate {{interpretation:coolness}}: {{ row.text }}"


def test_pending_structured_requirement_blocks_execution_with_typed_site() -> None:
    state = _state_with_llm(_pending_options())

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    site = result.sites[0]
    assert site.component_id == "rate_coolness"
    assert site.component_type == "transform"
    assert site.user_term == "coolness"
    assert site.kind is InterpretationKind.VAGUE_TERM


def test_interpretation_sites_reports_legacy_and_structured_pending_sites() -> None:
    legacy = _state_with_llm({"prompt_template": "Rate {{interpretation:coolness}}: {{ row.text }}"})
    structured = _state_with_llm(_pending_options())

    legacy_sites = interpretation_sites(legacy)
    structured_sites = interpretation_sites(structured)

    assert len(legacy_sites) == 2
    assert legacy_sites[0].component_id == "rate_coolness"
    assert legacy_sites[0].component_type == "transform"
    assert legacy_sites[0].user_term == "coolness"
    assert legacy_sites[0].kind is InterpretationKind.VAGUE_TERM
    assert legacy_sites[1].component_id == "rate_coolness"
    assert legacy_sites[1].component_type == "transform"
    assert legacy_sites[1].user_term == "llm_prompt_template:rate_coolness"
    assert legacy_sites[1].kind is InterpretationKind.LLM_PROMPT_TEMPLATE

    assert structured_sites[0].component_id == "rate_coolness"
    assert structured_sites[0].component_type == "transform"
    assert structured_sites[0].user_term == "coolness"
    assert structured_sites[0].kind is InterpretationKind.VAGUE_TERM
    assert structured_sites[1].component_id == "rate_coolness"
    assert structured_sites[1].component_type == "transform"
    assert structured_sites[1].user_term == "llm_prompt_template:rate_coolness"
    assert structured_sites[1].kind is InterpretationKind.LLM_PROMPT_TEMPLATE


def test_legacy_interpretation_requirement_missing_kind_defaults_to_vague_term() -> None:
    options = _pending_options()
    requirement = dict(options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    del requirement["kind"]
    options[INTERPRETATION_REQUIREMENTS_KEY] = [requirement]
    state = _state_with_llm(options)

    sites = interpretation_sites(state)

    assert sites[0].kind is InterpretationKind.VAGUE_TERM


def test_interpretation_requirement_non_string_kind_still_fails_closed() -> None:
    options = _pending_options()
    requirement = dict(options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    requirement["kind"] = 123
    options[INTERPRETATION_REQUIREMENTS_KEY] = [requirement]
    state = _state_with_llm(options)

    with pytest.raises(TypeError, match="interpretation requirement kind must be a string"):
        interpretation_sites(state)


def test_resolved_requirement_materializes_prompt_and_hash() -> None:
    options = _pending_options()
    requirement = dict(options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    requirement["status"] = "resolved"
    requirement["accepted_value"] = "well-designed and useful"
    prompt = "Rate well-designed and useful: {{ row.text }}"
    # The prompt-template review anchors to the prompt SKELETON (parts structure),
    # not the substituted text — so it stays valid after the vague term resolves.
    skeleton_hash = prompt_structure_hash_from_options(options)
    assert skeleton_hash is not None
    options[INTERPRETATION_REQUIREMENTS_KEY] = [
        requirement,
        {
            "id": "prompt-template-review",
            "kind": "llm_prompt_template",
            "user_term": "rating prompt",
            "status": "resolved",
            "draft": "Rate pending interpretation: {{ row.text }}",
            "event_id": "event-2",
            "accepted_value": prompt,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": skeleton_hash,
        },
    ]
    state = _state_with_llm(options)

    materialized = materialize_state_for_execution(state)

    assert isinstance(materialized, CompositionState)
    materialized_prompt = materialized.nodes[0].options["prompt_template"]
    assert materialized_prompt == prompt
    # Node-level hash remains the final-prompt-string hash (runtime reads it).
    assert materialized.nodes[0].options["resolved_prompt_template_hash"] == stable_hash(prompt)


def test_pending_invented_source_requirement_blocks_execution() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "pending",
                        "draft": "https://example.gov.au",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].kind is InterpretationKind.INVENTED_SOURCE
    assert result.sites[0].component_id == "source"
    assert result.sites[0].component_type == "source"


def test_pending_llm_prompt_template_requirement_blocks_execution() -> None:
    state = _state_with_llm(
        {
            "prompt_template": "Rate {{ row.text }}",
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "prompt-template-review",
                    "kind": "llm_prompt_template",
                    "user_term": "rating prompt",
                    "status": "pending",
                    "draft": "Rate {{ row.text }}",
                    "event_id": None,
                    "accepted_value": None,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                }
            ],
        }
    )

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].component_id == "rate_coolness"
    assert result.sites[0].component_type == "transform"
    assert result.sites[0].kind is InterpretationKind.LLM_PROMPT_TEMPLATE


def test_pending_pipeline_decision_requirement_blocks_execution_on_non_llm_transform() -> None:
    state = _state_with_cleanup_node(_pipeline_decision_options())

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].component_id == "drop_raw_html"
    assert result.sites[0].component_type == "transform"
    assert result.sites[0].user_term == "drop_raw_html_fields"
    assert result.sites[0].kind is InterpretationKind.PIPELINE_DECISION


def test_unreviewed_field_mapper_drop_of_web_scrape_raw_fields_blocks_execution() -> None:
    state = _state_with_web_scrape_cleanup_node(
        {
            "mapping": {
                "url": "url",
                "primary_colours": "primary_colours",
            },
            "select_only": True,
        }
    )

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].component_id == "drop_raw_html"
    assert result.sites[0].component_type == "transform"
    assert result.sites[0].user_term == "drop_raw_html_fields"
    assert result.sites[0].kind is InterpretationKind.PIPELINE_DECISION


def test_unrelated_pipeline_decision_does_not_satisfy_raw_html_cleanup_review() -> None:
    state = _state_with_web_scrape_cleanup_node(
        {
            "mapping": {
                "url": "url",
                "primary_colours": "primary_colours",
            },
            "select_only": True,
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "unrelated-pipeline-decision",
                    "kind": "pipeline_decision",
                    "user_term": "some_other_pipeline_choice",
                    "status": "pending",
                    "draft": "Approve a different row-shaping decision.",
                    "event_id": None,
                    "accepted_value": None,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                }
            ],
        }
    )

    contract_error = raw_html_cleanup_review_contract_error(state)
    sites = interpretation_sites(state)

    assert contract_error is not None
    assert "drop_raw_html_fields" in contract_error
    assert ("drop_raw_html", "drop_raw_html_fields", InterpretationKind.PIPELINE_DECISION) in (
        (site.component_id, site.user_term, site.kind) for site in sites
    )


def test_gate_routed_web_scrape_into_llm_warns_without_prompt_shield() -> None:
    # The gate topology routes web_scrape output into the LLM via gate routes,
    # exercising the enhanced _producer_by_output_stream (routes/fork_to). The
    # Prompt-shield recommendation is advisory, not blocking:
    # an unshielded LLM-over-scrape surfaces a warning and still composes.
    state = _state_with_web_scrape_gate_to_llm()

    warning_pairs = prompt_shield_recommendation_warning_pairs(state)
    result = materialize_state_for_execution(state)

    assert warning_pairs
    assert any(component == "node:summarise_pages" for component, _message in warning_pairs)
    assert any("prompt-injection shield" in message for _component, message in warning_pairs)
    # Advisory, not blocking: the LLM node still triggers a prompt-template
    # review, but the prompt-shield recommendation is NOT a blocking review site.
    assert isinstance(result, InterpretationReviewPending)
    assert all(site.user_term != PROMPT_SHIELD_USER_TERM for site in result.sites)


def test_gate_routed_web_scrape_through_prompt_shield_emits_no_warning() -> None:
    state = _state_with_web_scrape_gate_shield_to_llm()

    warning_pairs = prompt_shield_recommendation_warning_pairs(state)

    assert warning_pairs == ()


def _state_with_plain_llm_only() -> CompositionState:
    """One llm node with NO upstream producer at all (no web_scrape, no shield)."""
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="rate_node",
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="out",
                on_error="stop",
                options={
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet-4.6",
                    "prompt_template": "Rate {{ row.text }} and return JSON.",
                    "temperature": 0,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def test_plain_unshielded_llm_warns_always_on() -> None:
    # Always-on: an llm node with no upstream producer and no shield still
    # surfaces the advisory (State C default). The pre-change code returned () here.
    state = _state_with_plain_llm_only()
    warning_pairs = prompt_shield_recommendation_warning_pairs(state)
    assert warning_pairs
    assert any(component == "node:rate_node" for component, _message in warning_pairs)


def test_prompt_shield_warning_uses_available_draft_in_state_b() -> None:
    from elspeth.web.interpretation_state import PROMPT_SHIELD_AVAILABLE_DRAFT

    state = _state_with_plain_llm_only()
    pairs_b = prompt_shield_recommendation_warning_pairs(state, shield_available=True)
    assert pairs_b
    assert any(PROMPT_SHIELD_AVAILABLE_DRAFT in message for _component, message in pairs_b)

    pairs_c = prompt_shield_recommendation_warning_pairs(state, shield_available=False)
    assert pairs_c
    assert any("continuing without it is allowed" in message for _component, message in pairs_c)


def test_field_mapper_projection_without_web_scrape_raw_fields_does_not_create_cleanup_review_site() -> None:
    state = _state_with_cleanup_node(
        {
            "mapping": {
                "url": "url",
                "primary_colours": "primary_colours",
            },
            "select_only": True,
        }
    )

    result = materialize_state_for_execution(state)

    assert isinstance(result, CompositionState)


def test_resolved_pipeline_decision_requires_matching_node_hash() -> None:
    reviewed = _state_with_cleanup_node(_pipeline_decision_options())
    artifact_hash = pipeline_decision_artifact_hash(
        reviewed.nodes[0],
        reviewed.nodes,
        user_term=RAW_HTML_CLEANUP_USER_TERM,
    )
    state = _state_with_cleanup_node(_pipeline_decision_options(status="resolved", artifact_hash=artifact_hash))

    materialized = materialize_state_for_execution(state)

    assert isinstance(materialized, CompositionState)
    assert materialized.nodes[0].options["mapping"] == strip_authoring_options(reviewed.nodes[0].options)["mapping"]


def test_resolved_pipeline_decision_hash_drift_fails_closed() -> None:
    state = _state_with_cleanup_node(_pipeline_decision_options(status="resolved", artifact_hash=stable_hash("old node shape")))

    with pytest.raises(ValueError, match="pipeline-decision review hash drifted"):
        materialize_state_for_execution(state)


def test_resolved_raw_html_cleanup_decision_rejects_mapping_that_preserves_raw_fields() -> None:
    reviewed = _state_with_cleanup_node(_pipeline_decision_options())
    bad_options = dict(reviewed.nodes[0].options)
    bad_options["mapping"] = {
        "url": "url",
        "content": "content",
        "content_fingerprint": "content_fingerprint",
        "primary_colours": "primary_colours",
    }
    clean_options = strip_authoring_options(bad_options)
    artifact_hash = stable_hash(
        {
            "id": "drop_raw_html",
            "node_type": "transform",
            "plugin": "field_mapper",
            "input": "scored_rows",
            "on_success": "clean_rows",
            "on_error": "stop",
            "options": clean_options,
        }
    )
    state = _state_with_cleanup_node(bad_options)
    requirement = dict(state.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    requirement["status"] = "resolved"
    requirement["event_id"] = "event-raw-html-drop"
    requirement["accepted_value"] = requirement["draft"]
    requirement["accepted_artifact_hash"] = artifact_hash
    patched_options = dict(state.nodes[0].options)
    patched_options[INTERPRETATION_REQUIREMENTS_KEY] = [requirement]
    state = _state_with_cleanup_node(patched_options)

    with pytest.raises(ValueError, match="preserves raw HTML/fingerprint field"):
        materialize_state_for_execution(state)


def test_resolved_raw_html_cleanup_decision_rejects_custom_raw_field_preservation() -> None:
    options = _pipeline_decision_options(status="resolved")
    options["mapping"] = {
        "url": "url",
        "page_body": "page_body",
        "page_hash": "page_hash",
        "primary_colours": "primary_colours",
    }
    base = _state_with_web_scrape_cleanup_node(options)
    scrape = replace(
        base.nodes[0],
        options={
            "url_field": "url",
            "content_field": "page_body",
            "fingerprint_field": "page_hash",
        },
    )
    mapper = replace(base.nodes[1], id="select_fields")
    state_without_hash = replace(base, nodes=(scrape, mapper))
    artifact_hash = pipeline_decision_artifact_hash(
        state_without_hash.nodes[1],
        state_without_hash.nodes,
        user_term=RAW_HTML_CLEANUP_USER_TERM,
    )
    requirement = dict(options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    requirement["accepted_artifact_hash"] = artifact_hash
    patched_options = dict(options)
    patched_options[INTERPRETATION_REQUIREMENTS_KEY] = [requirement]
    state = replace(state_without_hash, nodes=(scrape, replace(mapper, options=patched_options)))

    with pytest.raises(ValueError, match="page_body"):
        materialize_state_for_execution(state)


def test_resolved_llm_prompt_template_requires_matching_hash() -> None:
    prompt = "Rate {{ row.text }}"
    state = _state_with_llm(
        {
            "prompt_template": prompt,
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "prompt-template-review",
                    "kind": "llm_prompt_template",
                    "user_term": "rating prompt",
                    "status": "resolved",
                    "draft": prompt,
                    "event_id": "event-1",
                    "accepted_value": prompt,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": stable_hash("different prompt"),
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="prompt-template review hash drifted"):
        materialize_state_for_execution(state)


def test_plain_llm_prompt_template_without_review_metadata_blocks_execution() -> None:
    state = _state_with_llm({"prompt_template": "Summarize {{ row.text }}"})

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].component_id == "rate_coolness"
    assert result.sites[0].component_type == "transform"
    assert result.sites[0].user_term == "llm_prompt_template:rate_coolness"
    assert result.sites[0].kind is InterpretationKind.LLM_PROMPT_TEMPLATE


def test_resolved_llm_prompt_template_requirement_without_hash_fails_closed() -> None:
    prompt = "Rate {{ row.text }}"
    state = _state_with_llm(
        {
            "prompt_template": prompt,
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "prompt-template-review",
                    "kind": "llm_prompt_template",
                    "user_term": "rating prompt",
                    "status": "resolved",
                    "draft": prompt,
                    "event_id": "event-1",
                    "accepted_value": prompt,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="prompt-template review hash drifted"):
        materialize_state_for_execution(state)


def test_llm_generated_source_metadata_without_requirement_is_review_site() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                }
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    sites = interpretation_sites(state)
    assert len(sites) == 1
    assert sites[0].component_id == "source"
    assert sites[0].component_type == "source"
    assert sites[0].user_term == "llm_generated_source"
    assert sites[0].kind is InterpretationKind.INVENTED_SOURCE


def test_llm_generated_source_metadata_without_resolved_requirement_blocks_execution() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                }
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].component_id == "source"
    assert result.sites[0].component_type == "source"
    assert result.sites[0].user_term == "llm_generated_source"
    assert result.sites[0].kind is InterpretationKind.INVENTED_SOURCE


def test_resolved_invented_source_requirement_requires_matching_artifact_hash() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": "event-1",
                    "resolved_kind": "invented_source",
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "resolved",
                        "draft": "https://example.gov.au",
                        "event_id": "event-1",
                        "accepted_value": "accepted source artifact",
                        "accepted_artifact_hash": "b" * 64,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    # Drift after review (accepted_artifact_hash != current source content_hash)
    # is a readiness blocker surfaced through the structured interpretation-review
    # machinery, NOT a bare ValueError that leaks to the route layer as a 404/500.
    # (This assertion was previously `pytest.raises(ValueError, ...)`, which
    # codified the defect rather than the desired behaviour — inverted for
    # elspeth-5a94855935.)
    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].component_type == "source"
    assert result.sites[0].component_id == "source"
    assert result.sites[0].kind is InterpretationKind.INVENTED_SOURCE


def test_resolved_invented_source_drift_surfaces_as_pending_review_site() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": "event-1",
                    "resolved_kind": "invented_source",
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "resolved",
                        "draft": "https://example.gov.au",
                        "event_id": "event-1",
                        "accepted_value": "accepted source artifact",
                        "accepted_artifact_hash": "b" * 64,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    # The readiness detector (single source of truth for /validate and /execute)
    # must report the drifted-after-review source as a pending review site, so the
    # existing InterpretationReviewPending path handles it instead of the
    # downstream bare ValueError.
    sites = interpretation_sites(state)

    assert len(sites) == 1
    assert sites[0].component_type == "source"
    assert sites[0].component_id == "source"
    assert sites[0].kind is InterpretationKind.INVENTED_SOURCE
    assert sites[0].user_term == "inline_source_url_list"

    result = materialize_state_for_execution(state)

    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].kind is InterpretationKind.INVENTED_SOURCE


def test_resolved_invented_source_matching_hash_does_not_block_execution() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": "event-1",
                    "resolved_kind": "invented_source",
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "resolved",
                        "draft": "https://example.gov.au",
                        "event_id": "event-1",
                        "accepted_value": "accepted source artifact",
                        "accepted_artifact_hash": "a" * 64,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )

    assert interpretation_sites(state) == ()
    result = materialize_state_for_execution(state)
    assert not isinstance(result, InterpretationReviewPending)


def test_strip_authoring_options_removes_metadata_keys() -> None:
    options = {
        "prompt_template": "Rate {{ row.text }}",
        PROMPT_TEMPLATE_PARTS_KEY: [],
        INTERPRETATION_REQUIREMENTS_KEY: [],
        SOURCE_AUTHORING_KEY: {
            "modality": "llm_generated",
            "content_hash": "abc123",
            "review_event_id": None,
            "resolved_kind": None,
        },
        "resolved_prompt_template_hash": "a" * 64,
    }

    stripped = strip_authoring_options(options)

    assert PROMPT_TEMPLATE_PARTS_KEY not in stripped
    assert INTERPRETATION_REQUIREMENTS_KEY not in stripped
    assert SOURCE_AUTHORING_KEY not in stripped
    assert stripped["resolved_prompt_template_hash"] == "a" * 64


# ---------------------------------------------------------------------------
# pipeline_decision_artifact_hash — material-only scoping
#
# Regression coverage for the staging incident where the composer LLM swapped
# the LLM ``model`` field (claude-3.7-sonnet → claude-3.5-sonnet) after the
# prompt-shield-recommendation review had already resolved. The legacy
# whole-stripped-options hash treated the model change as material drift and
# crashed preflight with a confusing "pipeline-decision review hash drifted"
# message. The narrowed hash binds to topology only, so model edits no longer
# spurious-drift the review.
# ---------------------------------------------------------------------------


def _state_with_web_scrape_llm_pair(llm_options: dict[str, Any]) -> CompositionState:
    """Two-node state: untrusted web_scrape upstream feeding an LLM."""

    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="url_rows",
                on_success="scraped_pages",
                on_error="stop",
                options={
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="identify_colours",
                node_type="transform",
                plugin="llm",
                input="scraped_pages",
                on_success="coloured_pages",
                on_error="stop",
                options=llm_options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _shielded_three_node_state(llm_options: dict[str, Any]) -> CompositionState:
    """Three-node state: web_scrape → azure_prompt_shield → LLM."""

    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="url_rows",
                on_success="scraped_pages",
                on_error="stop",
                options={
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="shield_content",
                node_type="transform",
                plugin="azure_prompt_shield",
                input="scraped_pages",
                on_success="shielded_pages",
                on_error="stop",
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="identify_colours",
                node_type="transform",
                plugin="llm",
                input="shielded_pages",
                on_success="coloured_pages",
                on_error="stop",
                options=llm_options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _shield_review_llm_options(*, model: str, prompt_template: str) -> dict[str, Any]:
    return {
        "provider": "openrouter",
        "model": model,
        "prompt_template": prompt_template,
        "temperature": 0,
        INTERPRETATION_REQUIREMENTS_KEY: [
            {
                "id": "prompt_injection_shield_review:identify_colours",
                "kind": "pipeline_decision",
                "user_term": PROMPT_SHIELD_USER_TERM,
                "status": "resolved",
                "draft": "Public web content from web_scrape flows directly into the LLM. Recommend inserting azure_prompt_shield.",
                "event_id": "shield-resolve-1",
                "accepted_value": "Public web content from web_scrape flows directly into the LLM. Recommend inserting azure_prompt_shield.",
                "accepted_artifact_hash": None,
                "resolved_prompt_template_hash": None,
            }
        ],
    }


def _unshielded_review_llm_options(*, model: str, prompt_template: str) -> dict[str, Any]:
    return {
        "provider": "openrouter",
        "model": model,
        "prompt_template": prompt_template,
        "temperature": 0,
        INTERPRETATION_REQUIREMENTS_KEY: [
            {
                "id": "prompt_template_review:identify_colours",
                "kind": "llm_prompt_template",
                "user_term": "llm_prompt_template:identify_colours",
                "status": "resolved",
                "draft": prompt_template,
                "event_id": "prompt-resolve-1",
                "accepted_value": prompt_template,
                "accepted_artifact_hash": None,
                "resolved_prompt_template_hash": stable_hash(prompt_template),
            },
            {
                "id": "model_choice_review:identify_colours",
                "kind": "llm_model_choice",
                "user_term": "llm_model_choice:identify_colours",
                "status": "resolved",
                "draft": model,
                "event_id": "model-resolve-1",
                "accepted_value": model,
                "accepted_artifact_hash": None,
                "resolved_prompt_template_hash": stable_hash(model),
            },
        ],
    }


def test_prompt_shield_hash_is_stable_across_model_swap() -> None:
    """Regression: changing the LLM model after the shield review must not drift the hash.

    This is the exact staging failure — composer LLM resolved the shield review
    with model=3.7-sonnet, then swapped model=3.5-sonnet after a separate
    value-source validation failure. The narrowed hash binds to topology, not
    model identity, so the swap is immaterial.
    """

    state_v37 = _state_with_web_scrape_llm_pair(
        _shield_review_llm_options(model="anthropic/claude-3.7-sonnet", prompt_template="Identify colours: {{ row.url }} {{ row.content }}")
    )
    state_v35 = _state_with_web_scrape_llm_pair(
        _shield_review_llm_options(model="anthropic/claude-3.5-sonnet", prompt_template="Identify colours: {{ row.url }} {{ row.content }}")
    )

    hash_v37 = pipeline_decision_artifact_hash(state_v37.nodes[1], state_v37.nodes, user_term=PROMPT_SHIELD_USER_TERM)
    hash_v35 = pipeline_decision_artifact_hash(state_v35.nodes[1], state_v35.nodes, user_term=PROMPT_SHIELD_USER_TERM)

    assert hash_v37 == hash_v35


def test_prompt_shield_hash_is_stable_across_prompt_template_edit() -> None:
    """Editing prompt_template doesn't invalidate the shield recommendation."""

    state_a = _state_with_web_scrape_llm_pair(
        _shield_review_llm_options(model="anthropic/claude-3.7-sonnet", prompt_template="Original prompt: {{ row.url }}")
    )
    state_b = _state_with_web_scrape_llm_pair(
        _shield_review_llm_options(
            model="anthropic/claude-3.7-sonnet", prompt_template="Rewritten prompt with extra prose: {{ row.url }} {{ row.content }}"
        )
    )

    hash_a = pipeline_decision_artifact_hash(state_a.nodes[1], state_a.nodes, user_term=PROMPT_SHIELD_USER_TERM)
    hash_b = pipeline_decision_artifact_hash(state_b.nodes[1], state_b.nodes, user_term=PROMPT_SHIELD_USER_TERM)

    assert hash_a == hash_b


def test_prompt_shield_hash_changes_when_authorized_shield_inserted() -> None:
    """Inserting azure_prompt_shield between web_scrape and LLM is material.

    The review's premise was "no shield exists." Adding a shield invalidates
    that premise — the operator needs to confirm whether the prior review is
    still meaningful (it is moot, but the audit trail should reflect that).
    """

    unshielded = _state_with_web_scrape_llm_pair(
        _shield_review_llm_options(model="anthropic/claude-3.7-sonnet", prompt_template="Identify: {{ row.content }}")
    )
    shielded = _shielded_three_node_state(
        _shield_review_llm_options(model="anthropic/claude-3.7-sonnet", prompt_template="Identify: {{ row.content }}")
    )

    hash_unshielded = pipeline_decision_artifact_hash(unshielded.nodes[1], unshielded.nodes, user_term=PROMPT_SHIELD_USER_TERM)
    hash_shielded = pipeline_decision_artifact_hash(shielded.nodes[2], shielded.nodes, user_term=PROMPT_SHIELD_USER_TERM)

    assert hash_unshielded != hash_shielded


def test_prompt_shield_hash_survives_model_swap() -> None:
    """Narrow shield-hash invariant: model swap does not invalidate shield review.

    Mirrors the staging session timeline (b930f0aa-…) — the composer LLM
    resolved the shield recommendation against a v4 state with model=3.7-sonnet
    and then patched the model to 3.5-sonnet at v8. The narrowed
    ``pipeline_decision_artifact_hash`` domain (which deliberately excludes
    ``options.model``) MUST keep the prompt-shield review valid through
    that swap — the shield review's premise is "untrusted content into an
    LLM", not "this specific model".

    Asserts ONLY the hash-domain invariant. The model swap separately
    surfaces a new ``llm_model_choice`` review (algorithmic enforcement
    of the "every model choice surfaced" contract), and the auto-stager
    is responsible for re-staging that requirement when the mutation
    pipeline patches ``options.model``. That behavior is pinned in the
    auto-stager tests; this test focuses on the shield-hash invariant
    alone.
    """

    prompt = "Identify: {{ row.url }} {{ row.content }}"

    pre_swap = _state_with_web_scrape_llm_pair(_shield_review_llm_options(model="anthropic/claude-3.7-sonnet", prompt_template=prompt))
    pre_swap_hash = pipeline_decision_artifact_hash(pre_swap.nodes[1], pre_swap.nodes, user_term=PROMPT_SHIELD_USER_TERM)

    post_swap = _state_with_web_scrape_llm_pair(_shield_review_llm_options(model="anthropic/claude-3.5-sonnet", prompt_template=prompt))
    post_swap_hash = pipeline_decision_artifact_hash(post_swap.nodes[1], post_swap.nodes, user_term=PROMPT_SHIELD_USER_TERM)

    assert pre_swap_hash == post_swap_hash


def test_prompt_shield_warning_is_advisory_not_blocking() -> None:
    state = _state_with_web_scrape_llm_pair(
        _unshielded_review_llm_options(
            model="anthropic/claude-3.7-sonnet", prompt_template="Identify colours: {{ row.url }} {{ row.content }}"
        )
    )

    validation = state.validate()
    warning_text = " ".join(w.message for w in validation.warnings)

    assert "prompt_injection_shield_recommendation" in warning_text
    assert "continuing without it is allowed" in warning_text

    materialized = materialize_state_for_execution(state)
    assert isinstance(materialized, CompositionState)


def test_raw_html_cleanup_hash_includes_upstream_raw_field_set() -> None:
    """Adding a new raw field to upstream web_scrape re-stages cleanup review.

    If the upstream web_scrape suddenly emits a new raw field (e.g. an
    additional fingerprint variant), the prior cleanup review didn't authorise
    dropping it. The hash domain has to surface that as drift so the operator
    re-confirms.
    """

    base = _state_with_web_scrape_cleanup_node({"mapping": {"url": "url", "primary_colours": "primary_colours"}, "select_only": True})
    hash_a = pipeline_decision_artifact_hash(base.nodes[1], base.nodes, user_term=RAW_HTML_CLEANUP_USER_TERM)

    # Upstream web_scrape now also exports a "fingerprint" field via fingerprint_field.
    extended_nodes = list(base.nodes)
    web_scrape = extended_nodes[0]
    extended_options = dict(web_scrape.options)
    extended_options["fingerprint_field"] = "fingerprint_v2"
    extended_nodes[0] = NodeSpec(
        id=web_scrape.id,
        node_type=web_scrape.node_type,
        plugin=web_scrape.plugin,
        input=web_scrape.input,
        on_success=web_scrape.on_success,
        on_error=web_scrape.on_error,
        options=extended_options,
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    extended_state = CompositionState(
        source=None,
        nodes=tuple(extended_nodes),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    hash_b = pipeline_decision_artifact_hash(extended_state.nodes[1], extended_state.nodes, user_term=RAW_HTML_CLEANUP_USER_TERM)

    assert hash_a != hash_b


def test_pipeline_decision_artifact_hash_rejects_unknown_user_term() -> None:
    """Adding a new pipeline-decision kind requires a registered helper.

    We refuse to fall through to a permissive default — every kind needs its
    own material projection or the audit boundary becomes lossy.
    """

    state = _state_with_cleanup_node({"mapping": {"url": "url"}, "select_only": True})

    with pytest.raises(ValueError, match="unknown pipeline_decision user_term"):
        pipeline_decision_artifact_hash(state.nodes[0], state.nodes, user_term="some_new_review_we_havent_implemented")


# --------------------------------------------------------------------------- #
# vague_term_wiring_count — the single resolvability contract shared by the
# tool boundary, the staging repair loop, and (mirrored) the resolver.
# --------------------------------------------------------------------------- #


def _vague_requirement(*, term: str = "cool", status: str = "pending") -> dict[str, object]:
    return {
        "id": term,
        "kind": InterpretationKind.VAGUE_TERM.value,
        "user_term": term,
        "status": status,
        "draft": "visually appealing",
        "event_id": None,
        "accepted_value": None,
        "resolved_prompt_template_hash": None,
    }


def test_wiring_count_structured_with_ref_is_resolvable() -> None:
    options = {
        "prompt_template": "Rate pending: {{ row.text }}",
        PROMPT_TEMPLATE_PARTS_KEY: [
            {"kind": "text", "text": "Rate "},
            {"kind": "interpretation_ref", "requirement_id": "cool"},
            {"kind": "text", "text": ": {{ row.text }}"},
        ],
        INTERPRETATION_REQUIREMENTS_KEY: [_vague_requirement()],
    }
    assert vague_term_wiring_count(options, user_term="cool") == 1


def test_wiring_count_structured_requirement_without_parts_is_unresolvable() -> None:
    """The demo-blocking shape: a requirement with no prompt_template_parts."""
    options = {
        "prompt_template": "Rate pending: {{ row.text }}",
        INTERPRETATION_REQUIREMENTS_KEY: [_vague_requirement()],
    }
    assert vague_term_wiring_count(options, user_term="cool") == 0


def test_wiring_count_structured_parts_without_ref_is_unresolvable() -> None:
    """Parts present but no interpretation_ref → the resolver would silent-drop."""
    options = {
        "prompt_template": "Rate pending: {{ row.text }}",
        PROMPT_TEMPLATE_PARTS_KEY: [{"kind": "text", "text": "Rate this row"}],
        INTERPRETATION_REQUIREMENTS_KEY: [_vague_requirement()],
    }
    assert vague_term_wiring_count(options, user_term="cool") == 0


def test_wiring_count_legacy_placeholder_coexists_with_autostaged_requirements() -> None:
    """A legacy {{interpretation:cool}} placeholder is resolvable even when the
    node carries auto-staged prompt-template / model-choice requirements (which
    are NOT vague_term). This is the production hello-world shape.
    """
    options = {
        "prompt_template": "Rate how {{interpretation:cool}} this row is.",
        INTERPRETATION_REQUIREMENTS_KEY: [
            {
                "id": "prompt_template_review:rate_node",
                "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                "user_term": "llm_prompt_template:rate_node",
                "status": "pending",
            },
            {
                "id": "model_choice_review:rate_node",
                "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
                "user_term": "llm_model_choice:rate_node",
                "status": "pending",
            },
        ],
    }
    assert vague_term_wiring_count(options, user_term="cool") == 1


def test_wiring_count_no_wiring_at_all_is_unresolvable() -> None:
    options = {"prompt_template": "Rate how this row is."}
    assert vague_term_wiring_count(options, user_term="cool") == 0


def test_wiring_count_wrong_term_is_unresolvable() -> None:
    options = {
        "prompt_template": "Rate how {{interpretation:important}} this row is.",
    }
    assert vague_term_wiring_count(options, user_term="cool") == 0


# ---------------------------------------------------------------------------
# prompt_shield_state_for_node — A/B/C state helper
# ---------------------------------------------------------------------------


def test_prompt_shield_state_for_node_returns_A_when_shielded() -> None:
    from elspeth.web.interpretation_state import prompt_shield_state_for_node

    state = _state_with_web_scrape_gate_shield_to_llm()
    llm = next(n for n in state.nodes if n.plugin == "llm")
    assert prompt_shield_state_for_node(llm, state.nodes, shield_available=True) == "A"
    assert prompt_shield_state_for_node(llm, state.nodes, shield_available=False) == "A"


def test_prompt_shield_state_for_node_B_vs_C() -> None:
    from elspeth.web.interpretation_state import prompt_shield_state_for_node

    state = _state_with_plain_llm_only()
    llm = next(n for n in state.nodes if n.plugin == "llm")
    assert prompt_shield_state_for_node(llm, state.nodes, shield_available=True) == "B"
    assert prompt_shield_state_for_node(llm, state.nodes, shield_available=False) == "C"


# refine_prompt_shield_warnings_for_availability — B-vs-C post-processor
# ---------------------------------------------------------------------------


def test_refine_prompt_shield_warnings_rewrites_c_to_b_when_available() -> None:
    from elspeth.web.interpretation_state import (
        PROMPT_SHIELD_AVAILABLE_DRAFT,
        PROMPT_SHIELD_WARNING_DRAFT,
        refine_prompt_shield_warnings_for_availability,
    )

    c_warnings = [
        {"component": "node:rate_node", "message": f"lead {PROMPT_SHIELD_WARNING_DRAFT}", "severity": "medium"},
        {"component": "node:other", "message": "unrelated warning", "severity": "medium"},
    ]
    refined = refine_prompt_shield_warnings_for_availability(c_warnings, shield_available=True)
    shield = [w for w in refined if w["component"] == "node:rate_node"]
    assert shield
    assert PROMPT_SHIELD_AVAILABLE_DRAFT in shield[0]["message"]
    assert PROMPT_SHIELD_WARNING_DRAFT not in shield[0]["message"]
    other = [w for w in refined if w["component"] == "node:other"]
    assert other[0]["message"] == "unrelated warning"
    unchanged = refine_prompt_shield_warnings_for_availability(c_warnings, shield_available=False)
    assert any(PROMPT_SHIELD_WARNING_DRAFT in w["message"] for w in unchanged)
