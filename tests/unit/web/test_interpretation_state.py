"""Tests for structured composer interpretation-review authoring state."""

from __future__ import annotations

import pytest

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata, SourceSpec
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PROMPT_TEMPLATE_PARTS_KEY,
    SOURCE_AUTHORING_KEY,
    InterpretationReviewPending,
    interpretation_sites,
    materialize_state_for_authoring,
    materialize_state_for_execution,
    strip_authoring_options,
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
            "resolved_prompt_template_hash": stable_hash(prompt),
        },
    ]
    state = _state_with_llm(options)

    materialized = materialize_state_for_execution(state)

    assert isinstance(materialized, CompositionState)
    materialized_prompt = materialized.nodes[0].options["prompt_template"]
    assert materialized_prompt == prompt
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
    clean_options = strip_authoring_options(reviewed.nodes[0].options)
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
    state = _state_with_cleanup_node(_pipeline_decision_options(status="resolved", artifact_hash=artifact_hash))

    materialized = materialize_state_for_execution(state)

    assert isinstance(materialized, CompositionState)
    assert materialized.nodes[0].options["mapping"] == clean_options["mapping"]


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

    with pytest.raises(ValueError, match="invented source review drift"):
        materialize_state_for_execution(state)


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
