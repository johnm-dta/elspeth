from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.config import WebSettings
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    SOURCE_AUTHORING_KEY,
    SOURCE_COMPONENT_ID,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


@pytest.fixture
def sessions_service(engine) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.sessions"),
    )


@pytest.fixture(autouse=True)
def _force_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="anthropic")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


def _composer(tmp_path, sessions_service) -> ComposerServiceImpl:
    from unittest.mock import MagicMock

    from elspeth.web.catalog.protocol import CatalogService

    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = []
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_model="anthropic/claude-opus-4-7",
        shareable_link_signing_key=b"\x00" * 32,
    )
    return ComposerServiceImpl(
        catalog=catalog,
        settings=settings,
        sessions_service=sessions_service,
        session_engine=sessions_service._engine,
    )


def _pt_node() -> NodeSpec:
    prompt = "Read {{ row.html }} and return JSON."
    return NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={
            "prompt_template": prompt,
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "prompt_template_review",
                    "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                    "user_term": "llm_prompt_template:rate_node",
                    "status": "pending",
                    "draft": prompt,
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
    )


async def _persist(sessions_service, state: CompositionState):
    from datetime import UTC, datetime

    from sqlalchemy import insert

    from elspeth.web.sessions.models import sessions_table

    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="u",
                auth_provider_type="local",
                title="surfacer test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    record = await _save_state_for_session(sessions_service, session_id, state)
    return session_id, record.id


async def _save_state_for_session(sessions_service, session_id: UUID, state: CompositionState):
    state_dict = state.to_dict()
    record = await sessions_service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=state_dict["nodes"],
            sources=state_dict["sources"],
            metadata_=state_dict["metadata"],
            is_valid=True,
        ),
        provenance="tool_call",
    )
    return record


@pytest.mark.asyncio
async def test_surfacer_surfaces_prompt_template(tmp_path, sessions_service) -> None:
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        nodes=(_pt_node(),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    kinds = {e.kind for e in events}
    assert InterpretationKind.LLM_PROMPT_TEMPLATE in kinds


def _model_choice_node(model: str = "anthropic/claude-sonnet-4.6") -> NodeSpec:
    return NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={
            "prompt_template": "Rate this row and return JSON.",
            "model": model,
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "prompt_template_review:rate_node",
                    "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                    "user_term": "llm_prompt_template:rate_node",
                    "status": "pending",
                    "draft": "Rate this row and return JSON.",
                    "event_id": None,
                    "accepted_value": None,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                },
                {
                    "id": "model_choice_review:rate_node",
                    "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
                    "user_term": "llm_model_choice:rate_node",
                    "status": "pending",
                    "draft": model,
                    "event_id": None,
                    "accepted_value": None,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                },
            ],
        },
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


@pytest.mark.asyncio
async def test_surfacer_surfaces_model_choice(tmp_path, sessions_service) -> None:
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        nodes=(_model_choice_node(),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    kinds = {e.kind for e in events}
    assert InterpretationKind.LLM_MODEL_CHOICE in kinds
    assert InterpretationKind.LLM_PROMPT_TEMPLATE in kinds


@pytest.mark.asyncio
async def test_surfacer_is_idempotent(tmp_path, sessions_service) -> None:
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        nodes=(_model_choice_node(),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    mc = [e for e in events if e.kind is InterpretationKind.LLM_MODEL_CHOICE]
    assert len(mc) == 1


@pytest.mark.asyncio
async def test_model_choice_dedup_is_draft_aware(tmp_path, sessions_service) -> None:
    """A stale pending event for the same node/kind/user_term but an old draft
    must not deadlock the new staged requirement."""
    composer = _composer(tmp_path, sessions_service)
    old_model = "anthropic/claude-sonnet-4.5"
    new_model = "anthropic/claude-sonnet-4.6"
    old_state = CompositionState(
        source=None,
        nodes=(_model_choice_node(old_model),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, old_state_id = await _persist(sessions_service, old_state)
    await composer.surface_pending_interpretation_reviews(old_state, session_id=str(session_id), current_state_id=str(old_state_id))

    new_state = CompositionState(
        source=None,
        nodes=(_model_choice_node(new_model),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=2,
    )
    new_record = await _save_state_for_session(sessions_service, session_id, new_state)
    await composer.surface_pending_interpretation_reviews(new_state, session_id=str(session_id), current_state_id=str(new_record.id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    drafts = [e.llm_draft for e in events if e.kind is InterpretationKind.LLM_MODEL_CHOICE]
    assert old_model in drafts
    assert new_model in drafts


def _staged_vague_term_node() -> NodeSpec:
    draft = "modern, useful, engaging, and clear for the public."
    return NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={
            "prompt_template": "Rate how {{interpretation:cool}} this is.",
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "vague_term_review:rate_node",
                    "kind": InterpretationKind.VAGUE_TERM.value,
                    "user_term": "cool",
                    "status": "pending",
                    "draft": draft,
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
    )


@pytest.mark.asyncio
async def test_surfacer_surfaces_staged_vague_term(tmp_path, sessions_service) -> None:
    # A staged vague-term requirement with an authored draft is surfaced.
    # This is NOT a backend word-list heuristic and NOT an invented draft.
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        nodes=(_staged_vague_term_node(),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    vt = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
    assert len(vt) == 1
    assert vt[0].user_term == "cool"
    assert vt[0].llm_draft == "modern, useful, engaging, and clear for the public."


@pytest.mark.asyncio
async def test_surfacer_skips_bare_vague_term(tmp_path, sessions_service) -> None:
    # A bare {{interpretation:cool}} token with NO staged requirement is a
    # legacy vague_term site. The surfacer must SKIP it (left fail-closed
    # at the run-time gate) and must not infer a draft from the word "cool".
    node = NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={"prompt_template": "Rate how {{interpretation:cool}} this is."},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        nodes=(node,),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    # Must not raise: the surfacer returns None for a bare token (no staged
    # requirement) BEFORE it ever calls the writer, so the writer boundary is
    # never reached. (The writer would actually ACCEPT a bare vague_term — the
    # node has a non-empty prompt_template, so _find_llm_transform_node passes
    # and the else-branch does no VAGUE_TERM check; the skip is the designed
    # advisory polarity, not protection against a writer rejection.)
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    vt = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
    assert vt == []


def _model_only_node(model: str = "anthropic/claude-sonnet-4.6") -> NodeSpec:
    # W1: an llm node with `model` + a staged llm_model_choice requirement but
    # NO prompt_template. The model_choice SITE emitter fires on `model` alone,
    # so the surfacer's precondition would be met — but the strict writer
    # boundary's else-branch REQUIRES a non-empty prompt_template and would
    # reject the shape with InterpretationResolveError(ValueError). The surfacer
    # must SKIP (precondition guard + try/except backstop), never raise.
    return NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={
            "model": model,
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "model_choice_review:rate_node",
                    "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
                    "user_term": "llm_model_choice:rate_node",
                    "status": "pending",
                    "draft": model,
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
    )


@pytest.mark.asyncio
async def test_surfacer_skips_model_only_node_without_prompt_template(tmp_path, sessions_service) -> None:
    # W1: the surfacer's model_choice precondition is necessary-but-not-sufficient
    # — the writer ALSO requires a non-empty prompt_template. A model-only node
    # must be SKIPPED (left fail-closed at the run-time gate), NOT raised as a 500
    # at the persist seam. Note: if interpretation_sites does not emit a
    # model_choice site for a prompt_template-less node, this assertion still holds
    # (no event surfaced) and the guard is belt-and-suspenders — correct either way.
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        nodes=(_model_only_node(),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    # Must NOT raise even though the writer boundary would reject this shape.
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    mc = [e for e in events if e.kind is InterpretationKind.LLM_MODEL_CHOICE]
    assert mc == []


def _llm_authored_source() -> SourceSpec:
    # An LLM-authored source: modality must be one that requires LLM provenance
    # (CreationModality.LLM_GENERATED) so _pending_source_sites yields an
    # invented_source site, and content_hash must be a populated string for the
    # writer boundary (create_pending_interpretation_event INVENTED_SOURCE
    # branch). The staged requirement's user_term is the verbatim
    # "llm_generated_source" that _pending_source_sites emits for the site.
    content_hash = "a" * 64
    return SourceSpec(
        plugin="inline_blob",
        options={
            SOURCE_AUTHORING_KEY: {
                "modality": "llm_generated",
                "content_hash": content_hash,
            },
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "invented_source_review",
                    "kind": InterpretationKind.INVENTED_SOURCE.value,
                    "user_term": "llm_generated_source",
                    "status": "pending",
                    "draft": "rows: [{url: https://example.gov}]",
                    "event_id": None,
                    "accepted_value": None,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                }
            ],
        },
        on_success="main",
        on_validation_failure="discard",
    )


@pytest.mark.asyncio
async def test_surfacer_surfaces_invented_source(tmp_path, sessions_service) -> None:
    # An LLM-authored default source carrying a staged invented_source
    # requirement surfaces a resolvable pending event at the source-commit
    # writer boundary. The default source lives in sources[SOURCE_COMPONENT_ID];
    # the writer reads source.options.source_authoring.content_hash and the
    # single pending INVENTED_SOURCE requirement.
    composer = _composer(tmp_path, sessions_service)
    state = CompositionState(
        source=None,
        sources={SOURCE_COMPONENT_ID: _llm_authored_source()},
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _persist(sessions_service, state)
    await composer.surface_pending_interpretation_reviews(state, session_id=str(session_id), current_state_id=str(state_id))
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    kinds = {e.kind for e in events}
    assert InterpretationKind.INVENTED_SOURCE in kinds
