from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import insert
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationKind,
)
from elspeth.web.composer import yaml_generator as real_yaml_generator
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.config import WebSettings
from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import sessions_table
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


def _composer(tmp_path: Path, sessions_service: SessionServiceImpl) -> ComposerServiceImpl:
    from unittest.mock import MagicMock

    from elspeth.web.catalog.protocol import CatalogService

    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = []
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    # F22 (same class as F1): WebSettings requires these four composer
    # fields; omitting them raises a 4-error pydantic ValidationError before
    # the service is ever built. Values mirror the guided conftest / F1.
    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="anthropic/claude-opus-4-7",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    return ComposerServiceImpl.for_trained_operator(
        catalog=catalog,
        settings=settings,
        sessions_service=sessions_service,
        session_engine=sessions_service._engine,
    )


def _build_execution_service(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> ExecutionServiceImpl:
    """Real ExecutionServiceImpl over the REAL SessionServiceImpl so execute()'s
    get_current_state(session_id) (~execution/service.py:484) loads the persisted
    state and the interpretation gate (:515-529) sees it.

    The interpretation gate fires BEFORE validate_pipeline / generate_yaml /
    create_run and is fully REAL in BOTH branches — it raises in BLOCK and passes
    in PERMIT. Settings and yaml_generator are REAL here (unlike the earlier
    minimal fixture): a REAL WebSettings (validate_pipeline reads data_dir and
    resolves the audit DB via get_landscape_url) and the REAL yaml_generator, so
    the PERMIT path runs the REAL validate_pipeline (NO patch) against the
    complete LLM pipeline and accepts it. Only _run_pipeline is stubbed by the
    test, to avoid a live engine run; create_run still runs for real, so
    execute() returns a real run_id. Only the loop is mocked, with _call_async
    bridged to a synchronous run (mirror the canonical `service` fixture's
    _call_async bridge in test_service.py) because the real _call_async uses
    run_coroutine_threadsafe, which needs a running loop.

    WebSettings mirrors the sibling `_composer` helper (the uniform-byte
    placeholder signing key is accepted under pytest, where WebSettings allows
    insecure test keys) plus the one field the ExecutionService path needs beyond
    it: landscape_url for create_run's audit DB.
    """
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    broadcaster = ProgressBroadcaster(mock_loop)
    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="anthropic/claude-opus-4-7",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
        landscape_url=f"sqlite:///{tmp_path}/audit.db",
    )
    svc = ExecutionServiceImpl.for_trained_operator(
        loop=mock_loop,
        broadcaster=broadcaster,
        settings=settings,
        session_service=sessions_service,
        yaml_generator=real_yaml_generator,
        telemetry=build_sessions_telemetry(),
    )
    _real_loop = asyncio.new_event_loop()

    def _mock_call_async(coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            return _real_loop.run_until_complete(coro)
        except RuntimeError:
            coro.close()
            return None

    cast(Any, svc)._call_async = _mock_call_async
    return svc


PROMPT = "Rate {{ row.text }} and return JSON."
MODEL = "anthropic/claude-sonnet-4.6"


def _llm_node() -> NodeSpec:
    # A COMPLETE LLM node: provider + model + api_key + a non-empty
    # prompt_template + required_input_fields + observed schema, carrying TWO
    # pending requirements — an llm_prompt_template AND an llm_model_choice.
    # interpretation_sites then yields TWO pending sites, so execute()'s gate
    # raises (BLOCK). Resolving BOTH reviews marks both requirements
    # status="resolved", clearing every site, so the state has zero sites and the
    # REAL validate_pipeline accepts the complete pipeline -> execute() reaches
    # create_run (PERMIT). The completeness is what matters: an incomplete llm
    # node (e.g. no provider/model) fails the real validate_pipeline graph build,
    # which is why the earlier minimal fixture had to patch validate_pipeline; a
    # complete node does not.
    return NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="main",
        on_error="discard",
        options={
            "provider": "openrouter",
            "model": MODEL,
            "api_key": "test-key-literal",
            "prompt_template": PROMPT,
            "required_input_fields": ["text"],
            "schema": {"mode": "observed"},
            INTERPRETATION_REQUIREMENTS_KEY: [
                {
                    "id": "pt",
                    "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                    "user_term": "llm_prompt_template:rate_node",
                    "status": "pending",
                    "draft": PROMPT,
                    "event_id": None,
                    "accepted_value": None,
                    "accepted_artifact_hash": None,
                    "resolved_prompt_template_hash": None,
                },
                {
                    "id": "mc",
                    "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
                    "user_term": "llm_model_choice:rate_node",
                    "status": "pending",
                    "draft": MODEL,
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


async def _persist_state_with_unresolved_node(
    sessions_service: SessionServiceImpl,
    composer: ComposerServiceImpl,
    tmp_path: Path,
) -> tuple[UUID, UUID, UUID, UUID, CompositionState]:
    """Seed a session + a COMPLETE LLM pipeline (text source + llm node + json
    sink + edge) AND the two matching pending, resolvable events
    (llm_prompt_template + llm_model_choice). The complete shape is what lets the
    REAL validate_pipeline pass once both reviews resolve. Returns
    (session_id, state_id, pt_event_id, mc_event_id, state)."""
    session_id = uuid4()
    src = tmp_path / "blobs" / str(session_id) / "input.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("hello world\n", encoding="utf-8")
    out = tmp_path / "outputs" / str(session_id) / "out.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    state = state.with_source(
        SourceSpec(
            plugin="text",
            on_success="rows",
            options={"path": str(src), "column": "text", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        )
    )
    state = state.with_node(_llm_node())
    state = state.with_output(
        OutputSpec(
            name="main",
            plugin="json",
            options={
                "path": str(out),
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
            },
            on_write_failure="discard",
        )
    )
    state = state.with_edge(EdgeSpec(id="e1", from_node="source", to_node="rate_node", edge_type="on_success", label=None))
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="run-backstop test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    state_dict = state.to_dict()
    record = await sessions_service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=state_dict["nodes"],
            sources=state_dict["sources"],
            edges=state_dict["edges"],
            outputs=state_dict["outputs"],
            metadata_=state_dict["metadata"],
            is_valid=True,
        ),
        provenance="tool_call",
    )
    # Create BOTH resolvable pending events through the writer boundary. The PT
    # event goes through the requirement-checked branch (it requires
    # options.prompt_template == llm_draft AND exactly one pending PT requirement
    # matching user_term — both staged on the node); the model_choice event falls
    # through the writer's else-branch (sessions/service.py:2936), which only
    # confirms an llm transform node with a non-empty prompt_template and does no
    # staged-requirement-shape check. Resolving BOTH marks both requirements
    # resolved, clearing every site (PERMIT).
    pt_event = await sessions_service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=record.id,
        affected_node_id="rate_node",
        tool_call_id=f"backend_auto_surface:{uuid4()}",
        user_term="llm_prompt_template:rate_node",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        llm_draft=PROMPT,
        model_identifier=composer._model,
        model_version=composer._model,
        provider="anthropic",
        composer_skill_hash=composer._composer_skill_hash,
    )
    mc_event = await sessions_service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=record.id,
        affected_node_id="rate_node",
        tool_call_id=f"backend_auto_surface:{uuid4()}",
        user_term="llm_model_choice:rate_node",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        llm_draft=MODEL,
        model_identifier=composer._model,
        model_version=composer._model,
        provider="anthropic",
        composer_skill_hash=composer._composer_skill_hash,
    )
    # InterpretationEventRecord.id is a UUID (contracts/composer_interpretation.py:210).
    return session_id, record.id, pt_event.id, mc_event.id, state


@pytest.mark.asyncio
async def test_unresolved_card_blocks_run_resolving_permits(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    composer = _composer(tmp_path, sessions_service)
    execution_service = _build_execution_service(tmp_path, sessions_service)
    session_id, _state_id, pt_event_id, mc_event_id, _state = await _persist_state_with_unresolved_node(
        sessions_service, composer, tmp_path
    )

    # 2. run-time gate BLOCKS: execute() raises on the unresolved placeholder.
    with pytest.raises(UnresolvedInterpretationPlaceholderError):
        await execution_service.execute(session_id=session_id)

    # 3. resolve BOTH pending cards as accepted-as-drafted (the prompt_template
    # and the model_choice). Resolving only one leaves a pending site and the
    # gate keeps raising; both must clear for the state to reach PERMIT.
    for event_id in (pt_event_id, mc_event_id):
        await sessions_service.resolve_interpretation_event(
            session_id=session_id,
            event_id=event_id,
            choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
            amended_value=None,
            actor="user:alice",
        )

    # 4. with BOTH reviews resolved, the interpretation gate PERMITS. The gate
    # (execution/service.py:515-529) runs FIRST and is fully REAL here — it now
    # sees zero sites. The REAL validate_pipeline then runs (NO patch) against the
    # complete LLM pipeline (text source + llm node + json sink + edge) and
    # accepts it; only _run_pipeline is stubbed, to avoid a live engine run.
    # create_run still runs for real, so execute() returns a real run_id.
    with patch.object(execution_service, "_run_pipeline"):
        run_id = await execution_service.execute(session_id=session_id)
    assert run_id is not None
