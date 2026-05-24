"""Tests for structured composer interpretation-review authoring state."""

from __future__ import annotations

from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata
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
                "user_term": "coolness",
                "status": "pending",
                "draft": "well-designed and useful",
                "event_id": "event-1",
                "accepted_value": None,
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
    assert result.sites == (("rate_coolness", "coolness"),)


def test_interpretation_sites_reports_legacy_and_structured_pending_sites() -> None:
    legacy = _state_with_llm({"prompt_template": "Rate {{interpretation:coolness}}: {{ row.text }}"})
    structured = _state_with_llm(_pending_options())

    assert interpretation_sites(legacy.nodes) == (("rate_coolness", "coolness"),)
    assert interpretation_sites(structured.nodes) == (("rate_coolness", "coolness"),)


def test_resolved_requirement_materializes_prompt_and_hash() -> None:
    options = _pending_options()
    requirement = dict(options[INTERPRETATION_REQUIREMENTS_KEY][0])  # type: ignore[index]
    requirement["status"] = "resolved"
    requirement["accepted_value"] = "well-designed and useful"
    options[INTERPRETATION_REQUIREMENTS_KEY] = [requirement]
    state = _state_with_llm(options)

    materialized = materialize_state_for_execution(state)

    assert isinstance(materialized, CompositionState)
    prompt = materialized.nodes[0].options["prompt_template"]
    assert prompt == "Rate well-designed and useful: {{ row.text }}"
    assert materialized.nodes[0].options["resolved_prompt_template_hash"] == stable_hash(prompt)


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
