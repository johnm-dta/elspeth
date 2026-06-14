from __future__ import annotations

from uuid import uuid4

import pytest

from elspeth.contracts.composer_interpretation import InterpretationKind, InterpretationSource
from elspeth.contracts.enums import CreationModality
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata, SourceSpec
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, SOURCE_AUTHORING_KEY, SOURCE_COMPONENT_ID
from elspeth.web.sessions.protocol import CompositionStateData
from tests.integration.web.conftest import _make_session


def _state_with_three_review_surfaces() -> CompositionState:
    prompt_template = "Read {{ row.html }} and return JSON."
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={
                "path": "/tmp/generated.csv",
                SOURCE_AUTHORING_KEY: {
                    "modality": CreationModality.LLM_GENERATED.value,
                    "content_hash": "0" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source_review",
                        "kind": InterpretationKind.INVENTED_SOURCE.value,
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
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="rate_node",
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="rated",
                on_error="quarantine",
                options={"prompt_template": "Rate how {{interpretation:cool}} this row is."},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="identify_colour",
                node_type="transform",
                plugin="llm",
                input="rated",
                on_success="out",
                on_error="quarantine",
                options={
                    "prompt_template": prompt_template,
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "prompt_template_review",
                            "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                            "user_term": "llm_prompt_template:identify_colour",
                            "status": "pending",
                            "draft": prompt_template,
                            "event_id": None,
                            "accepted_value": None,
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": None,
                        }
                    ],
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
        metadata=PipelineMetadata(name="opt-out audit fixture", description=""),
        version=1,
    )


@pytest.mark.asyncio
async def test_opted_out_session_still_records_surface_specific_rows(composer_test_client) -> None:
    service = composer_test_client.app.state.phase3_sessions_service
    session_id = uuid4()
    with composer_test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(session_id), user_id="alice")
    state_dict = _state_with_three_review_surfaces().to_dict()
    state = await service.save_composition_state(
        session_id,
        CompositionStateData(
            sources=state_dict["sources"],
            nodes=state_dict["nodes"],
            metadata_=state_dict["metadata"],
            is_valid=True,
        ),
        provenance="tool_call",
    )
    marker = await service.record_session_interpretation_opt_out(session_id=session_id, actor="user:alice")

    requests = [
        (InterpretationKind.INVENTED_SOURCE, SOURCE_COMPONENT_ID, "inline_source_url_list", "https://example.gov.au"),
        (InterpretationKind.VAGUE_TERM, "rate_node", "cool", "A draft definition"),
        (
            InterpretationKind.LLM_PROMPT_TEMPLATE,
            "identify_colour",
            "llm_prompt_template:identify_colour",
            "Read {{ row.html }} and return JSON.",
        ),
    ]
    for index, (kind, affected_node_id, user_term, llm_draft) in enumerate(requests):
        await service.create_pending_interpretation_event(
            session_id=session_id,
            composition_state_id=state.id,
            affected_node_id=affected_node_id,
            tool_call_id=f"call_opt_out_{index}",
            user_term=user_term,
            kind=kind,
            llm_draft=llm_draft,
            model_identifier="anthropic/claude-opus-4-7",
            model_version="2026-05-01",
            provider="anthropic",
            composer_skill_hash="a" * 64,
        )

    events = await service.list_interpretation_events(session_id, status="all")
    surface_opt_outs = [
        event for event in events if event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT and event.kind is not None
    ]
    assert events[0].id == marker.id
    assert [event.kind for event in surface_opt_outs] == [
        InterpretationKind.INVENTED_SOURCE,
        InterpretationKind.VAGUE_TERM,
        InterpretationKind.LLM_PROMPT_TEMPLATE,
    ]
    assert all(event.accepted_value is not None for event in surface_opt_outs)
    assert all(event.hash_domain_version == "v2" for event in surface_opt_outs)
    assert all(event.arguments_hash is not None for event in surface_opt_outs)
    prompt_event = next(event for event in surface_opt_outs if event.kind is InterpretationKind.LLM_PROMPT_TEMPLATE)
    assert prompt_event.resolved_prompt_template_hash == stable_hash(prompt_event.accepted_value)
