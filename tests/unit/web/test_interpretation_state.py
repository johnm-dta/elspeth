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

    assert len(legacy_sites) == 1
    assert legacy_sites[0].component_id == "rate_coolness"
    assert legacy_sites[0].component_type == "transform"
    assert legacy_sites[0].user_term == "coolness"
    assert legacy_sites[0].kind is InterpretationKind.VAGUE_TERM

    assert structured_sites[0].component_id == "rate_coolness"
    assert structured_sites[0].component_type == "transform"
    assert structured_sites[0].user_term == "coolness"
    assert structured_sites[0].kind is InterpretationKind.VAGUE_TERM


def test_interpretation_requirement_missing_kind_fails_closed() -> None:
    options = _pending_options()
    requirement = dict(options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    del requirement["kind"]
    options[INTERPRETATION_REQUIREMENTS_KEY] = [requirement]
    state = _state_with_llm(options)

    with pytest.raises(TypeError, match="interpretation requirement kind is required"):
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


def test_plain_llm_prompt_template_without_review_metadata_remains_executable() -> None:
    state = _state_with_llm({"prompt_template": "Summarize {{ row.text }}"})

    result = materialize_state_for_execution(state)

    assert isinstance(result, CompositionState)
    assert result.nodes[0].options["prompt_template"] == "Summarize {{ row.text }}"
    assert "resolved_prompt_template_hash" not in result.nodes[0].options


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


def test_llm_generated_source_metadata_without_requirement_is_not_review_site() -> None:
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

    assert interpretation_sites(state) == ()


def test_llm_generated_source_metadata_without_resolved_requirement_fails_closed() -> None:
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

    with pytest.raises(ValueError, match="invented source review requirement"):
        materialize_state_for_execution(state)


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
